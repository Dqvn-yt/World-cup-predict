import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import CLEAN_DIR, SOURCE_DIR


TWO_ROW_HEADER_FILES = {"shoot.csv", "Miscellaneous.csv", "gk.csv"}


def snake_case(value: object) -> str:
    text = str(value).strip().replace("%", "_pct").replace("/", "_per_")
    text = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def read_source(filename: str) -> pd.DataFrame:
    path = SOURCE_DIR / filename
    header = 1 if filename in TWO_ROW_HEADER_FILES else 0
    frame = pd.read_csv(path, header=header)
    frame.columns = _unique_columns([snake_case(column) for column in frame.columns])
    return frame


def read_clean(filename: str) -> pd.DataFrame:
    return pd.read_csv(CLEAN_DIR / filename)


def _unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        count = counts.get(column, 0)
        output.append(column if count == 0 else f"{column}_{count + 1}")
        counts[column] = count + 1
    return output


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
