import csv
import json

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F

from analyze_traffic import inspect_data, read_traffic, start_spark
from congestion import evaluate_levels, fit_thresholds, traffic_level
from download_data import PROJECT_DIR


OUTPUT_DIR = PROJECT_DIR / "outputs" / "forecast"
VALIDATION_START = "2017-01-01 00:00:00"
TEST_START = "2018-01-01 00:00:00"


def with_forecast_features(hourly):
    forecast_hours = hourly.select(
        F.col("date_time").alias("observed_at"),
        F.expr("date_time + INTERVAL 1 HOUR").alias("target_at"),
        F.col("traffic_volume").alias("volume_now"),
    )
    is_weekend = (F.weekday("target_at") >= 5).cast("int")
    return forecast_hours.withColumn("hour_type", F.hour("target_at") + 24 * is_weekend)


def prepare_examples(hourly):
    actual_next_hour = hourly.select(
        F.col("date_time").alias("target_at"),
        F.col("traffic_volume").alias("volume_next"),
    )
    # La jointure exclut les heures sans mesure à h + 1, même si une mesure existe plus tard.
    return with_forecast_features(hourly).join(actual_next_hour, "target_at", "inner")


def split_examples(examples):
    # Les périodes sont définies par l’heure à prédire, y compris aux frontières.
    validation_start = F.lit(VALIDATION_START).cast("timestamp_ntz")
    test_start = F.lit(TEST_START).cast("timestamp_ntz")
    train = examples.where(F.col("target_at") < validation_start)
    validation = examples.where(
        (F.col("target_at") >= validation_start) & (F.col("target_at") < test_start)
    )
    test = examples.where(F.col("target_at") >= test_start)
    return train, validation, test


def fit_forecast(train):
    pipeline = Pipeline(stages=[
        OneHotEncoder(
            inputCols=["hour_type"], outputCols=["calendar_features"],
            handleInvalid="keep", dropLast=False,
        ),
        VectorAssembler(inputCols=["volume_now", "calendar_features"], outputCol="features"),
        LinearRegression(labelCol="volume_next", predictionCol="forecast_raw", regParam=0.001),
    ])
    low, high = fit_thresholds(train)
    return pipeline.fit(train), low, high


def forecast_traffic(model, features, low, high):
    # Un volume de trafic est toujours positif ou nul.
    return model.transform(features).withColumn(
        "forecast", F.greatest(F.lit(0.0), F.col("forecast_raw"))
    ).withColumn("predicted_level", traffic_level(F.col("forecast"), low, high))


def evaluate(predictions):
    stats = {}
    for name, column in (("persistence", "volume_now"), ("linear_regression", "forecast")):
        # RegressionEvaluator attend une prédiction de type double ou float.
        numeric = predictions.withColumn(column, F.col(column).cast("double"))
        evaluator = RegressionEvaluator(labelCol="volume_next", predictionCol=column)
        stats[name] = {
            metric: evaluator.evaluate(numeric, {evaluator.metricName: metric})
            for metric in ("mae", "rmse")
        }
    return stats


def main():
    spark = start_spark("Prévision du trafic I-94 à une heure")
    try:
        raw = read_traffic(spark).cache()
        hourly, quality = inspect_data(raw)
        examples = prepare_examples(hourly).cache()

        train, validation, test = split_examples(examples)
        counts = {name: frame.count() for name, frame in (
            ("train", train), ("validation", validation), ("test", test),
        )}
        if not all(counts.values()):
            raise ValueError("Une des trois périodes ne contient aucune paire horaire.")

        model, low, high = fit_forecast(train)
        report = {
            "validation_start": VALIDATION_START,
            "test_start": TEST_START,
            "hourly_observations": quality["analyzed_hours"],
            "examples_per_split": counts,
            "observations_without_next_hour": quality["analyzed_hours"] - sum(counts.values()),
            "congestion": {"thresholds": {"p50": low, "p85": high}},
        }
        for name, frame in (("validation", validation), ("test", test)):
            predictions = forecast_traffic(model, frame, low, high).select(
                "observed_at", "target_at", "volume_now", "volume_next", "forecast", "predicted_level",
                traffic_level(F.col("volume_next"), low, high).alias("actual_level"),
            ).cache()
            report[name] = evaluate(predictions)
            report["congestion"][name] = evaluate_levels(predictions)
            if name == "test":
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with (OUTPUT_DIR / "predictions_test.csv").open("w", newline="", encoding="utf-8") as out:
                    writer = csv.writer(out)
                    writer.writerow(predictions.columns)
                    writer.writerows(predictions.orderBy("target_at").toLocalIterator())
            predictions.unpersist()

        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        (OUTPUT_DIR / "metrics.json").write_text(report_json + "\n", encoding="utf-8")
        print(report_json)
        print(f"Résultats enregistrés dans {OUTPUT_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
