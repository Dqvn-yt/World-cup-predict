import numpy as np
import pandas as pd

from config import FEATURE_DIR
from src.data.loaders import read_clean, write_csv
from src.data.score_history import load_score_history, neutral_from_hosts


PAIRED_FEATURES = [
    "fifa_rank",
    "elo",
    "is_host",
    "squad_avg_age",
    "squad_total_caps",
    "squad_total_value_eur",
    "squad_avg_value_eur",
    "rest_days",
    "prev_avg_goals_scored",
    "prev_avg_goals_conceded",
    "prev_avg_possession",
    "prev_avg_shots",
    "prev_avg_shots_on_target",
    "prev_avg_saves",
    "prev_avg_corners",
    "prev_avg_fouls",
    "prev_avg_offsides",
    "prev_avg_xg_scored",
    "prev_avg_xg_conceded",
    "team_attack_strength",
    "team_defense_strength",
    "team_xg_strength",
    "team_ppg",
    "team_matches_played",
]

MATCH_MODEL_FEATURES = [
    "is_knockout",
    "venue_capacity",
    "venue_elevation_meters",
    "referee_avg_cards",
    *[f"home_{name}" for name in PAIRED_FEATURES],
    *[f"away_{name}" for name in PAIRED_FEATURES],
    "elo_diff",
    "fifa_rank_diff",
    "squad_value_diff",
    "rest_days_diff",
    "attack_diff",
    "defense_diff",
    "xg_diff",
    "ppg_diff",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
]

PRIOR_MATCHES = 3.0
DEFAULT_PPG = 1.35


def _empty_state() -> dict[str, float]:
    return {"played": 0, "goals_for": 0.0, "goals_against": 0.0, "xg_for": 0.0, "xg_against": 0.0, "points": 0.0}


def _shrunk(total: float, count: int, prior: float) -> float:
    return (total + prior * PRIOR_MATCHES) / (count + PRIOR_MATCHES)


def build_team_history_features(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
    states: dict[int, dict[str, float]] = {}
    league = {"matches": 0, "home_goals": 0.0, "away_goals": 0.0, "home_xg": 0.0, "away_xg": 0.0, "points": 0.0}
    rows: list[dict] = []

    def team_state(team_id: int) -> dict[str, float]:
        return states.setdefault(int(team_id), _empty_state())

    def league_priors() -> dict[str, float]:
        n = max(league["matches"], 1)
        return {
            "goals": (league["home_goals"] + league["away_goals"]) / (2 * n),
            "xg": (league["home_xg"] + league["away_xg"]) / (2 * n),
            "ppg": league["points"] / n,
        }

    def features_for(team_id: int) -> dict[str, float]:
        state = team_state(team_id)
        priors = league_priors()
        n = state["played"]
        return {
            "team_attack_strength": _shrunk(state["goals_for"], n, priors["goals"]) / max(priors["goals"], 1e-6),
            "team_defense_strength": _shrunk(state["goals_against"], n, priors["goals"]) / max(priors["goals"], 1e-6),
            "team_xg_strength": _shrunk(state["xg_for"], n, priors["xg"]) / max(priors["xg"], 1e-6),
            "team_ppg": _shrunk(state["points"], n, DEFAULT_PPG),
            "team_matches_played": float(n),
        }

    for datetime, group in frame.groupby("match_datetime", sort=True):
        for _, match in group.iterrows():
            record: dict = {"match_id": int(match["match_id"])}
            for suffix in ["home", "away"]:
                values = features_for(match[f"{suffix}_team_id"])
                for key, value in values.items():
                    record[f"{suffix}_{key}"] = value
            rows.append(record)
        for _, match in group.iterrows():
            home_id, away_id = int(match["home_team_id"]), int(match["away_team_id"])
            home_score, away_score = float(match["home_score"]), float(match["away_score"])
            home_state, away_state = team_state(home_id), team_state(away_id)
            home_state["played"] += 1
            away_state["played"] += 1
            home_state["goals_for"] += home_score
            home_state["goals_against"] += away_score
            away_state["goals_for"] += away_score
            away_state["goals_against"] += home_score
            home_state["xg_for"] += float(match.get("home_xg") or 0.0)
            home_state["xg_against"] += float(match.get("away_xg") or 0.0)
            away_state["xg_for"] += float(match.get("away_xg") or 0.0)
            away_state["xg_against"] += float(match.get("home_xg") or 0.0)
            home_points, away_points = (3.0, 0.0) if home_score > away_score else ((1.0, 1.0) if home_score == away_score else (0.0, 3.0))
            home_state["points"] += home_points
            away_state["points"] += away_points
            league["matches"] += 1
            league["home_goals"] += home_score
            league["away_goals"] += away_score
            league["home_xg"] += float(match.get("home_xg") or 0.0)
            league["away_xg"] += float(match.get("away_xg") or 0.0)
            league["points"] += home_points + away_points
    return pd.DataFrame(rows)


def build_match_features() -> pd.DataFrame:
    features = read_clean("match_prediction_features_X.csv")
    targets = read_clean("match_prediction_targets_y.csv")
    _, matches, _ = load_score_history()
    matches = matches.copy()
    matches["match_datetime"] = pd.to_datetime(matches["date"] + " " + matches["kickoff_time_utc"])

    history = build_team_history_features(matches)
    home_history = history[["match_id", *[column for column in history.columns if column.startswith("home_")]]]
    away_history = history[["match_id", *[column for column in history.columns if column.startswith("away_")]]]

    frame = features.merge(home_history, on="match_id", how="left", validate="one_to_one").merge(
        away_history, on="match_id", how="left", validate="one_to_one"
    )
    frame = frame.merge(targets, on="match_id", how="left", validate="one_to_one")
    for side in ("home", "away"):
        frame[f"{side}_score"] = frame.match_id.map(matches.set_index("match_id")[f"{side}_score"])
    frame["match_result"] = np.where(frame.home_score > frame.away_score, "H", np.where(frame.home_score == frame.away_score, "D", "A"))
    frame["match_datetime"] = pd.to_datetime(frame["date"] + " " + frame["kickoff_time_utc"])
    frame["neutral"] = neutral_from_hosts(frame["home_is_host"], frame["away_is_host"]).astype(bool)
    frame["elo_diff"] = frame["home_elo"] - frame["away_elo"]
    frame["fifa_rank_diff"] = frame["away_fifa_rank"] - frame["home_fifa_rank"]
    frame["squad_value_diff"] = frame["home_squad_total_value_eur"] - frame["away_squad_total_value_eur"]
    frame["rest_days_diff"] = frame["home_rest_days"] - frame["away_rest_days"]
    frame["attack_diff"] = frame["home_team_attack_strength"] - frame["away_team_attack_strength"]
    frame["defense_diff"] = frame["home_team_defense_strength"] - frame["away_team_defense_strength"]
    frame["xg_diff"] = frame["home_team_xg_strength"] - frame["away_team_xg_strength"]
    frame["ppg_diff"] = frame["home_team_ppg"] - frame["away_team_ppg"]
    frame["home_attack_vs_away_defense"] = frame["home_team_attack_strength"] - frame["away_team_defense_strength"]
    frame["away_attack_vs_home_defense"] = frame["away_team_attack_strength"] - frame["home_team_defense_strength"]
    frame = frame.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)

    missing = [column for column in MATCH_MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing match model features: {missing}")
    frame[MATCH_MODEL_FEATURES] = frame[MATCH_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    write_csv(frame, FEATURE_DIR / "match_model_dataset.csv")
    return frame


if __name__ == "__main__":
    build_match_features()
