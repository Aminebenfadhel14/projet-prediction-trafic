# Projet de prédiction du trafic

Ce projet étudie les comptages horaires de l’I‑94 en direction ouest, entre Minneapolis et Saint Paul, et prévoit le volume de trafic à l’heure suivante.

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

Le nombre d’heures absentes compare les horodatages disponibles à une grille horaire continue entre le premier et le dernier relevé, dans l’heure locale telle qu’elle est fournie. Une heure absente doit rester distincte d’une heure avec un comptage de zéro.

## Entraîner le modèle à une heure

Après avoir configuré Java et téléchargé le jeu de données avec les commandes ci-dessus, lance cette commande depuis le dossier du projet.

```powershell
.\.venv\Scripts\python.exe src/train_traffic.py
```

1. Le script réutilise les relevés horaires validés de l’analyse. Pour chaque relevé, il calcule `target_at = observed_at + 1 heure` et récupère le volume réel via une jointure sur cet horodatage. Si le relevé à l’heure suivante manque, la paire ne sert pas à l’entraînement ou à l’évaluation.
2. Il utilise le volume déjà observé et le créneau calendaire de l’heure à prévoir. Le code du créneau va de 0 à 23 en semaine et de 24 à 47 le week-end. Le pipeline encode ces créneaux en indicateurs binaires et assemble les entrées numériques de la régression. Les variables météo sont écartées de cette première version.
3. Il découpe les paires selon la date cible. L’entraînement utilise les dates avant 2017, la validation utilise 2017 et le test utilise 2018. L’encodage et la régression linéaire sont ajustés sur l’entraînement.
4. Il compare la régression à la persistance, qui reprend le volume observé à l’heure précédente. La MAE est l’erreur absolue moyenne et la RMSE donne davantage de poids aux grosses erreurs.

Le script écrit `outputs/forecast/metrics.json` avec les scores et les effectifs par période. Le fichier `outputs/forecast/predictions_test.csv` contient les horodatages, la prévision, le volume réel et le volume de référence (`volume_now`).

## Déterminer le niveau de trafic prévu

La même commande exécute la détection avec PySpark. `src/congestion.py` calcule les percentiles P50 et P85 sur les volumes cibles de l’entraînement. Les volumes inférieurs au P50 sont classés faibles. Entre le P50 inclus et le P85 exclu, ils sont modérés. À partir du P85 inclus, ils sont élevés.

Le script applique ces seuils au volume prévu pour produire `predicted_level`, puis au volume observé pour produire `actual_level`, utilisé lors de l’évaluation. Les deux colonnes figurent dans `predictions_test.csv`. Dans `metrics.json`, les lignes de la matrice de confusion correspondent aux niveaux observés et les colonnes aux niveaux prévus.

Pour la classe élevée, la précision mesure la part des alertes qui correspondent à un volume réellement élevé. Le rappel mesure la part des heures de trafic élevé que le système signale. Les niveaux reposent sur le nombre de véhicules par heure. Le jeu de données ne fournit ni la vitesse ni le temps de trajet, donc le niveau élevé est un indicateur indirect de congestion.

Source des données. John Hogue (2019), *Metro Interstate Traffic Volume*, UCI Machine Learning Repository, licence CC BY 4.0. [Fiche UCI](https://doi.org/10.24432/C5X60B)
