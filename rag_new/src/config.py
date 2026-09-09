from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = RAW_DIR / "administrative_procedures.xlsx"
CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"

MAX_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 80
