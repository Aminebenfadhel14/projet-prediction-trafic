import csv
import json

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F

from analyze_traffic import inspect_data, read_traffic, start_spark
from download_data import PROJECT_DIR


OUTPUT_DIR = PROJECT_DIR / "outputs" / "forecast"
VALIDATION_START = "2017-01-01 00:00:00"
TEST_START = "2018-01-01 00:00:00"


def prepare_examples(hourly):
    known_now = hourly.select(
        F.col("date_time").alias("observed_at"),
        F.expr("date_time + INTERVAL 1 HOUR").alias("target_at"),
        F.col("traffic_volume").alias("volume_now"),
    )
    actual_next_hour = hourly.select(
        F.col("date_time").alias("target_at"),
        F.col("traffic_volume").alias("volume_next"),
    )
    # La jointure exclut les heures sans mesure à h + 1, même si une mesure existe plus tard.
    pairs = known_now.join(actual_next_hour, "target_at", "inner")
    is_weekend = (F.weekday("target_at") >= 5).cast("int")
    return pairs.withColumn("hour_type", F.hour("target_at") + 24 * is_weekend)


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

        # Les périodes sont définies par l’heure à prédire, y compris aux frontières.
        validation_start = F.lit(VALIDATION_START).cast("timestamp_ntz")
        test_start = F.lit(TEST_START).cast("timestamp_ntz")
        train = examples.where(F.col("target_at") < validation_start)
        validation = examples.where(
            (F.col("target_at") >= validation_start) & (F.col("target_at") < test_start)
        )
        test = examples.where(F.col("target_at") >= test_start)
        counts = {name: frame.count() for name, frame in (
            ("train", train), ("validation", validation), ("test", test),
        )}
        if not all(counts.values()):
            raise ValueError("Une des trois périodes ne contient aucune paire horaire.")

        pipeline = Pipeline(stages=[
            OneHotEncoder(
                inputCols=["hour_type"], outputCols=["calendar_features"],
                handleInvalid="keep", dropLast=False,
            ),
            VectorAssembler(inputCols=["volume_now", "calendar_features"], outputCol="features"),
            LinearRegression(labelCol="volume_next", predictionCol="forecast_raw", regParam=0.001),
        ])
        model = pipeline.fit(train)
        report = {
            "validation_start": VALIDATION_START,
            "test_start": TEST_START,
            "hourly_observations": quality["analyzed_hours"],
            "examples_per_split": counts,
            "observations_without_next_hour": quality["analyzed_hours"] - sum(counts.values()),
        }
        for name, frame in (("validation", validation), ("test", test)):
            # Un volume de trafic est toujours positif ou nul.
            predictions = model.transform(frame).withColumn(
                "forecast", F.greatest(F.lit(0.0), F.col("forecast_raw"))
            ).select("observed_at", "target_at", "volume_now", "volume_next", "forecast").cache()
            report[name] = evaluate(predictions)
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
