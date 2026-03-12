from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE = BASE_DIR / "train.csv"
OUTPUT_FILE = INGESTED_DIR / "train.csv"

def data_ingestion():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Dataset is empty")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data ingested successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    data_ingestion()