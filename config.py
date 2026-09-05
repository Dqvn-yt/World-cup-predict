from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "processed data" / "csv"
SOURCE_JSON_DIR = ROOT_DIR / "processed data" / "json"
WORLD_CUP_DIR = ROOT_DIR / "processed data" / "worldcup"
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42
SCORE_MATRIX_MAX_GOALS = 6


def ensure_directories() -> None:
    for path in (CLEAN_DIR, FEATURE_DIR, MODEL_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)
