import csv
import json
import os

from pyspark.sql import SparkSession, functions as F, types as T

from download_data import DATASET_PATH, PROJECT_DIR


OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"
WEEKDAYS = ("Lun.", "Mar.", "Mer.", "Jeu.", "Ven.", "Sam.", "Dim.")
SCHEMA = T.StructType(
    [
        T.StructField("holiday", T.StringType()),
        T.StructField("temp", T.DoubleType()),
        T.StructField("rain_1h", T.DoubleType()),
        T.StructField("snow_1h", T.DoubleType()),
        T.StructField("clouds_all", T.IntegerType()),
        T.StructField("weather_main", T.StringType()),
        T.StructField("weather_description", T.StringType()),
        T.StructField("date_time", T.TimestampNTZType()),
        T.StructField("traffic_volume", T.IntegerType()),
    ]
)


def inspect_data(raw):
    # Un horodatage peut apparaître sur plusieurs lignes météo.
    by_hour = raw.where(F.col("date_time").isNotNull()).groupBy("date_time").agg(
        F.count("*").alias("rows"),
        F.count_if(F.col("traffic_volume") >= 0).alias("valid_volumes"),
        F.min("traffic_volume").alias("min_volume"),
        F.max("traffic_volume").alias("max_volume"),
    )
    # Garder l'heure si ses lignes donnent toutes le même comptage valide.
    hourly = by_hour.where(
        (F.col("min_volume") == F.col("max_volume"))
        & (F.col("valid_volumes") == F.col("rows"))
    ).select("date_time", F.col("min_volume").alias("traffic_volume"))

    raw_stats = raw.agg(
        F.count("*").alias("raw_rows"),
        F.count_if(F.col("temp") == 0).alias("zero_kelvin_rows"),
        F.max("rain_1h").alias("max_hourly_rain_mm"),
        F.count_if(F.col("traffic_volume") < 0).alias("negative_volume_rows"),
        *(F.count_if(F.col(name).isNull()).alias(name) for name in raw.columns),
    ).first().asDict()
    quality = {key: value for key, value in raw_stats.items() if key not in raw.columns}
    quality["nulls_by_column"] = {name: raw_stats[name] for name in raw.columns}

    # Compter les heures de la grille du fichier, indépendamment du fuseau de Spark.
    hour_index = F.unix_date(F.to_date("date_time")) * 24 + F.hour("date_time")
    quality.update(
        by_hour.agg(
            F.min("date_time").cast("string").alias("first_hour"),
            F.max("date_time").cast("string").alias("last_hour"),
            F.count("*").alias("distinct_hours"),
            F.count_if(F.col("rows") > 1).alias("repeated_hours"),
            F.count_if(F.col("min_volume") != F.col("max_volume")).alias("conflicting_hours"),
            F.count_if((F.minute("date_time") != 0) | (F.second("date_time") != 0)).alias(
                "off_grid_hours"
            ),
            (F.max(hour_index) - F.min(hour_index) + 1 - F.count("*")).alias("missing_hours"),
        ).first().asDict()
    )
    if quality["off_grid_hours"]:
        raise ValueError("Certains horodatages ne tombent pas au début d’une heure.")
    quality.update(
        hourly.agg(
            F.count("*").alias("analyzed_hours"),
            F.min("traffic_volume").alias("min_volume"),
            F.max("traffic_volume").alias("max_volume"),
            F.round(F.avg("traffic_volume"), 1).alias("mean_volume"),
            F.percentile_approx("traffic_volume", [0.25, 0.5, 0.75]).alias("volume_quartiles"),
        ).first().asDict()
    )
    return hourly, quality


def write_traffic_profiles(hourly):
    # Spark calcule les agrégats. Seules les petites tables finales sont collectées.
    calendar = hourly.select(
        "traffic_volume",
        F.hour("date_time").alias("hour"),
        F.weekday("date_time").alias("weekday"),
    )
    by_day_and_hour = calendar.groupBy("weekday", "hour").agg(
        F.count("*").alias("observations"),
        F.round(F.avg("traffic_volume"), 1).alias("mean_volume"),
    ).orderBy("weekday", "hour").collect()
    by_day_type = calendar.withColumn(
        "day_type", F.when(F.col("weekday") < 5, "Semaine").otherwise("Week-end")
    ).groupBy("day_type", "hour").agg(
        F.count("*").alias("observations"),
        F.round(F.avg("traffic_volume"), 1).alias("mean_volume"),
    ).orderBy("day_type", "hour").collect()

    for filename, rows, dimension in (
        ("trafic_par_jour_et_heure.csv", by_day_and_hour, "weekday"),
        ("trafic_semaine_weekend.csv", by_day_type, "day_type"),
    ):
        with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow((dimension, "hour", "observations", "mean_volume"))
            writer.writerows(rows)

    save_chart(by_day_and_hour, by_day_type)


def save_chart(by_day_and_hour, by_day_type):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".runtime" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (profile, heatmap) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for day_type, color in (("Semaine", "#126782"), ("Week-end", "#ad501b")):
        rows = [row for row in by_day_type if row.day_type == day_type]
        profile.plot(
            [row.hour for row in rows],
            [row.mean_volume for row in rows],
            color=color,
            linewidth=2.5,
            label=day_type,
        )
    profile.set(
        title="Volume moyen selon l’heure",
        ylabel="Véhicules par heure",
        xticks=range(0, 24, 2),
        ylim=(0, None),
    )
    profile.grid(axis="y", alpha=0.25)
    profile.legend(frameon=False)

    grid = [[float("nan")] * 24 for _ in WEEKDAYS]
    for row in by_day_and_hour:
        grid[row.weekday][row.hour] = row.mean_volume
    image = heatmap.imshow(grid, aspect="auto", cmap="Blues", vmin=0)
    heatmap.set(
        title="Profil par jour de la semaine",
        xlabel="Heure du relevé (heure locale UCI)",
        xticks=range(0, 24, 2),
        yticks=range(7),
        yticklabels=WEEKDAYS,
    )
    fig.colorbar(image, ax=heatmap, label="Véhicules par heure")
    fig.suptitle("I-94 ouest · Minneapolis–Saint Paul · 2012–2018", fontweight="bold")
    fig.savefig(OUTPUT_DIR / "profils_trafic.png", dpi=150)
    plt.close(fig)


def main():
    if not DATASET_PATH.is_file():
        raise SystemExit("Jeu de données absent. Lance d’abord python src/download_data.py.")
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("Analyse du trafic I-94")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = (
            spark.read.schema(SCHEMA)
            .option("header", True)
            .option("enforceSchema", False)
            .option("mode", "FAILFAST")
            .option("timestampNTZFormat", "yyyy-MM-dd HH:mm:ss")
            .csv(DATASET_PATH.as_posix())
            .cache()
        )
        raw.printSchema()
        raw.show(5, truncate=False)
        hourly, quality = inspect_data(raw)
        if not quality["analyzed_hours"]:
            raise SystemExit("Aucun relevé horaire exploitable.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_traffic_profiles(hourly)
        report = json.dumps(quality, ensure_ascii=False, indent=2)
        (OUTPUT_DIR / "qualite_donnees.json").write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"Résultats enregistrés dans {OUTPUT_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
