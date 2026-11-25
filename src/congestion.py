from pyspark.sql import functions as F


LEVELS = ("faible", "modere", "eleve")


def fit_thresholds(train):
    # Utiliser exclusivement les volumes cibles de l'entraînement.
    low, high = train.approxQuantile("volume_next", [0.50, 0.85], 0.0)
    if low >= high:
        raise ValueError("Les deux seuils de trafic doivent être distincts.")
    return int(low), int(high)


def traffic_level(volume, low, high):
    return (
        F.when(volume < low, LEVELS[0])
        .when(volume < high, LEVELS[1])
        .otherwise(LEVELS[2])
    )


def evaluate_levels(predictions):
    # Spark agrège les couples de niveaux. Au plus neuf lignes arrivent en Python.
    counts = {
        (row.actual_level, row.predicted_level): row["count"]
        for row in predictions.groupBy("actual_level", "predicted_level").count().collect()
    }
    matrix = {
        actual: {predicted: counts.get((actual, predicted), 0) for predicted in LEVELS}
        for actual in LEVELS
    }
    high = LEVELS[-1]
    true_positives = matrix[high][high]
    false_positives = sum(matrix[actual][high] for actual in LEVELS[:-1])
    false_negatives = sum(matrix[high][predicted] for predicted in LEVELS[:-1])
    alerts = true_positives + false_positives
    actual_high = true_positives + false_negatives
    return {
        "confusion_matrix": matrix,
        "accuracy": sum(matrix[level][level] for level in LEVELS) / sum(counts.values()),
        "high_traffic_alerts": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": true_positives / alerts if alerts else None,
            "recall": true_positives / actual_high if actual_high else None,
        },
    }
