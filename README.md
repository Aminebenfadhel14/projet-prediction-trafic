# Projet de prédiction du trafic

Cette première étape étudie les comptages horaires de l’I‑94 en direction ouest, entre Minneapolis et Saint Paul. Elle prépare le travail de prévision à une heure.

## Exécuter l’analyse sous Windows

Prérequis pour cette configuration. Python 3.10 ou plus récent et un JDK 17 ou plus récent. Avec le JDK Microsoft 21 placé dans `.runtime/`, ces commandes utilisent le Java local au projet. Si tu as déjà un JDK configuré, conserve ton `JAVA_HOME`.

```powershell
# À lancer depuis le dossier projet-prediction-trafic.
if (-not (Test-Path .venv\Scripts\python.exe)) {
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$projectJdk = Get-ChildItem .runtime -Directory -Filter 'jdk-*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'bin\java.exe') } |
    Select-Object -First 1
if ($projectJdk) { $env:JAVA_HOME = $projectJdk.FullName }

.\.venv\Scripts\python.exe src/download_data.py
.\.venv\Scripts\python.exe src/analyze_traffic.py
```

## Comprendre le code

1. `src/download_data.py` récupère l’archive officielle UCI et place le CSV compressé dans `data/raw/`. La commande conserve par défaut le fichier déjà présent. Ajoute `--force` pour le télécharger à nouveau.
2. `src/analyze_traffic.py` démarre une session Spark locale. Le schéma indique à Spark quels champs sont des textes, des nombres ou des horodatages. Le format `TimestampNTZ` conserve l’heure lue dans le fichier UCI.
3. `inspect_data` vérifie les valeurs nulles, les anomalies météo, les heures répétées et les trous dans la série. Pour les statistiques de trafic, une heure compte une fois si ses lignes donnent toutes le même volume valide. Si une heure contient des valeurs contradictoires ou invalides, elle est écartée des profils. Les observations brutes restent disponibles.
4. `write_traffic_profiles` utilise `groupBy` et `avg` dans Spark pour calculer les profils horaires. Matplotlib reçoit les petits tableaux déjà agrégés et produit le graphique. Les jours de semaine correspondent au lundi–vendredi, jours fériés inclus.

## Résultats générés

| Fichier dans `outputs/eda/` | Contenu |
|---|---|
| `qualite_donnees.json` | Couverture horaire, valeurs nulles, anomalies et statistiques de volume. |
| `trafic_par_jour_et_heure.csv` | Moyenne et nombre de relevés par jour de semaine (lundi = 0) et heure. |
| `trafic_semaine_weekend.csv` | Moyenne et nombre de relevés par heure pour les deux types de jours. |
| `profils_trafic.png` | Courbes semaine et week-end, et carte de chaleur des volumes horaires. |

Le nombre d’heures absentes compare les horodatages disponibles à une grille horaire continue entre le premier et le dernier relevé, dans l’heure locale telle qu’elle est fournie. Une heure absente doit rester distincte d’une heure avec un comptage de zéro. Pour préparer le futur modèle, il faudra associer chaque relevé à celui situé exactement une heure plus tard et écarter les paires dont la cible manque.

Source des données. John Hogue (2019), *Metro Interstate Traffic Volume*, UCI Machine Learning Repository, licence CC BY 4.0. [Fiche UCI](https://doi.org/10.24432/C5X60B)
