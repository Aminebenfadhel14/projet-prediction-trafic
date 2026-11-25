import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from pyspark.sql import functions as F, types as T

from analyze_traffic import inspect_data, read_traffic, start_spark
from download_data import PROJECT_DIR
from train_traffic import (
    TEST_START,
    fit_forecast,
    forecast_traffic,
    prepare_examples,
    split_examples,
    with_forecast_features,
)


SCHEMA = T.StructType([
    T.StructField("date_time", T.TimestampNTZType()),
    T.StructField("traffic_volume", T.IntegerType()),
])
DEFAULT_START = "2018-01-02 05:00:00"


def publish_reading(staging, inbox, index, reading):
    # Écrire le fichier en dehors du dossier surveillé, puis le déplacer d'un seul coup.
    filename = f"releve-{index:04}.json"
    temporary = staging / filename
    payload = {
        "date_time": reading.date_time.isoformat(sep=" "),
        "traffic_volume": reading.traffic_volume,
    }
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    temporary.replace(inbox / filename)


def replay(spark, start, count, interval):
    raw = read_traffic(spark).cache()
    hourly, _ = inspect_data(raw)
    hourly = hourly.cache()
    train, _, _ = split_examples(prepare_examples(hourly))
    model, low, high = fit_forecast(train)
    readings = (
        hourly.where(F.col("date_time") >= F.lit(start).cast("timestamp_ntz"))
        .orderBy("date_time")
        .limit(count)
        .collect()
    )
    if not readings:
        raise ValueError("Aucun relevé disponible à partir de l’heure choisie.")

    output = PROJECT_DIR / "outputs" / "streaming"
    output.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="rejeu-", dir=output))
    inbox, staging = run / "incoming", run / "staging"
    inbox.mkdir()
    staging.mkdir()

    incoming = (
        spark.readStream.schema(SCHEMA)
        .option("timestampNTZFormat", "yyyy-MM-dd HH:mm:ss")
        .option("mode", "FAILFAST")
        .option("maxFilesPerTrigger", 1)
        .json(inbox.as_posix())
    )
    features = with_forecast_features(
        incoming.where(F.col("date_time").isNotNull() & (F.col("traffic_volume") >= 0))
    )
    predictions = forecast_traffic(model, features, low, high).select(
        "observed_at", "volume_now", "target_at", "forecast", "predicted_level"
    )
    print(f"Rejeu de {len(readings)} relevés. Seuils P50 = {low}, P85 = {high}.", flush=True)
    print(f"Fichiers et checkpoint : {run}", flush=True)
    query = (
        predictions.writeStream.format("console")
        .outputMode("append")
        .option("truncate", False)
        .option("checkpointLocation", (run / "checkpoint").as_posix())
        .start()
    )
    try:
        for index, reading in enumerate(readings, start=1):
            publish_reading(staging, inbox, index, reading)
            # Attendre l'affichage avant d'envoyer le prochain relevé.
            query.processAllAvailable()
            if index < len(readings):
                time.sleep(interval)
    finally:
        query.stop()


def main():
    parser = argparse.ArgumentParser(description="Simuler les relevés de trafic avec Structured Streaming.")
    parser.add_argument("--count", type=int, default=8, help="Nombre maximal de relevés à rejouer.")
    parser.add_argument("--interval", type=float, default=1.0, help="Pause entre les relevés, en secondes.")
    parser.add_argument("--start", default=DEFAULT_START, metavar="DATE_HEURE", help="Première heure à chercher dans le test.")
    args = parser.parse_args()
    try:
        parsed_start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        parser.error("--start doit suivre le format AAAA-MM-JJ HH:MM:SS.")
    start = parsed_start.isoformat(sep=" ")
    if start < TEST_START:
        parser.error("--start doit se situer dans la période de test, à partir de 2018.")
    if args.count < 1 or not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--count doit être positif et --interval doit être un nombre positif ou nul.")

    spark = start_spark("Rejeu du trafic I-94 avec Structured Streaming")
    try:
        replay(spark, start, args.count, args.interval)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
