from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_NAME = "Metro_Interstate_Traffic_Volume.csv.gz"
DATASET_PATH = PROJECT_DIR / "data" / "raw" / DATASET_NAME
DATASET_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"


def main():
    parser = ArgumentParser(description="Télécharger le jeu de données UCI 492.")
    parser.add_argument("--force", action="store_true", help="Remplacer le fichier existant.")
    args = parser.parse_args()
    if DATASET_PATH.is_file() and not args.force:
        print(f"Jeu de données déjà présent : {DATASET_PATH}")
        return

    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        compressed_csv = archive.read(DATASET_NAME)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=DATASET_PATH.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(compressed_csv)
            temporary_file.close()
            temporary_path.replace(DATASET_PATH)
        finally:
            temporary_file.close()
            temporary_path.unlink(missing_ok=True)

    print(f"Jeu de données téléchargé : {DATASET_PATH} ({len(compressed_csv):,} octets)")


if __name__ == "__main__":
    main()
