import json
from pathlib import Path

import pandas as pd

from config import CLEAN_DIR, METRICS_DIR, SOURCE_DIR, SOURCE_JSON_DIR, ensure_directories
from src.data.loaders import normalize_text, read_source, write_csv


DROP_RAW_COLUMNS = {"rk", "matches", "9999"}

RAW_RENAMES = {
    "shoot.csv": {
        "player": "raw_player_name",
        "pos": "raw_position",
        "squad": "raw_squad",
        "90s": "nineties",
        "gls": "goals",
        "sh": "shots",
        "sot": "shots_on_target",
        "sot_pct": "shots_on_target_pct",
        "sh_per_90": "shots_per90",
        "sot_per_90": "shots_on_target_per90",
        "g_per_sh": "goals_per_shot",
        "g_per_sot": "goals_per_shot_on_target",
    },
    "Miscellaneous.csv": {
        "player": "raw_player_name",
        "pos": "raw_position",
        "squad": "raw_squad",
        "90s": "nineties",
        "crdy": "yellow_cards",
        "crdr": "red_cards",
        "fls": "fouls_committed",
        "fld": "fouled",
        "off": "offsides",
        "crs": "crosses",
        "int": "interceptions",
        "tklw": "tackles_won",
        "og": "own_goals",
    },
    "gk.csv": {
        "player": "raw_player_name",
        "pos": "raw_position",
        "squad": "raw_squad",
        "90s": "nineties",
        "min": "minutes",
        "ga": "goals_against",
        "ga90": "goals_against_per90",
        "sota": "shots_on_target_against",
        "save_pct": "save_pct",
        "cs": "clean_sheets",
        "cs_pct": "clean_sheet_pct",
        "p_katt": "penalties_faced",
        "pka": "penalties_allowed",
        "p_ksv": "penalties_saved",
        "p_km": "penalties_missed",
        "save_pct_1": "penalty_save_pct",
    },
}


def clean_all() -> dict[str, dict[str, int]]:
    ensure_directories()
    report: dict[str, dict[str, int]] = {}
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        frame = read_source(path.name)
        frame = frame.drop(columns=[column for column in frame.columns if column in DROP_RAW_COLUMNS], errors="ignore")
        frame = frame.rename(columns=RAW_RENAMES.get(path.name, {}))
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
        if "date_of_birth" in frame.columns:
            frame["date_of_birth"] = pd.to_datetime(frame["date_of_birth"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "is_knockout" in frame.columns:
            frame["is_knockout"] = frame["is_knockout"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(frame["is_knockout"])
        if "raw_player_name" in frame.columns:
            frame["player_name_key"] = frame["raw_player_name"].map(normalize_text)
        write_csv(frame, CLEAN_DIR / path.name)
        report[path.name] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
        }

    json_path = SOURCE_JSON_DIR / "real_match_details.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as source:
            details = json.load(source)
        with (CLEAN_DIR / "real_match_details.json").open("w", encoding="utf-8") as target:
            json.dump(details, target, ensure_ascii=False, indent=2)
        report[json_path.name] = {"rows": len(details), "columns": 7, "duplicate_rows": 0, "missing_cells": 0}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "data_audit.json").open("w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    return report


if __name__ == "__main__":
    clean_all()
