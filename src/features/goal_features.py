from collections import defaultdict

import numpy as np
import pandas as pd

from config import FEATURE_DIR
from src.data.loaders import read_clean, write_csv
from src.features.match_features import build_match_features
from src.features.player_form import build_predictive_form
from src.features.team_form import build_team_form


GOAL_NUMERIC_FEATURES = [
    "is_starting_xi",
    "expected_minutes",
    "goals_last3",
    "goals_per90_last5",
    "assists_last3",
    "goal_involvement_per90_last5",
    "predictive_form",
    "start_rate_last5",
    "team_recent_xg",
    "team_attack_form",
    "opponent_recent_xg_conceded",
    "opponent_defense_form",
    "opponent_gk_form",
    "elo_diff",
    "is_home",
    "rest_days",
    "team_predicted_xg",
]
GOAL_CATEGORICAL_FEATURES = ["position"]


def _history_features(frame: pd.DataFrame) -> pd.DataFrame:
    histories: dict[int, list[dict[str, float]]] = defaultdict(list)
    records: list[dict[str, float]] = []
    for row in frame.itertuples():
        history = histories[row.player_id]
        last3 = history[-3:]
        last5 = history[-5:]
        minutes5 = sum(item["minutes"] for item in last5)
        goals5 = sum(item["goals"] for item in last5)
        assists5 = sum(item["assists"] for item in last5)
        records.append({
            "goals_last3": sum(item["goals"] for item in last3),
            "goals_per90_last5": goals5 * 90 / minutes5 if minutes5 else 0.0,
            "assists_last3": sum(item["assists"] for item in last3),
            "goal_involvement_per90_last5": (goals5 + assists5) * 90 / minutes5 if minutes5 else 0.0,
            "start_rate_last5": np.mean([item["started"] for item in last5]) if last5 else 0.0,
            "expected_minutes": np.mean([item["minutes"] for item in last5]) if last5 else 0.0,
        })
        if row.appeared:
            history.append({
                "goals": float(row.goals),
                "assists": float(row.assists),
                "minutes": float(row.minutes_played),
                "started": float(row.is_starting_xi),
            })
    history_frame = pd.DataFrame(records, index=frame.index)
    return pd.concat([frame, history_frame], axis=1)


def build_goal_features() -> pd.DataFrame:
    player_match_path = FEATURE_DIR / "player_match_performance.csv"
    if player_match_path.exists():
        player_match = pd.read_csv(player_match_path)
    else:
        from src.data.build_player_match import build_player_match

        player_match = build_player_match()
    predictive_path = FEATURE_DIR / "player_form_features.csv"
    predictive = pd.read_csv(predictive_path) if predictive_path.exists() else build_predictive_form(player_match)
    team_path = FEATURE_DIR / "team_player_form_features.csv"
    team_form = pd.read_csv(team_path) if team_path.exists() else build_team_form(predictive)
    match_path = FEATURE_DIR / "match_model_dataset.csv"
    match_features = pd.read_csv(match_path) if match_path.exists() else build_match_features()
    expected_path = FEATURE_DIR / "match_expected_goals_oof.csv"
    if not expected_path.exists():
        from src.models.train_score import train_score_models

        train_score_models(match_features)
    expected = pd.read_csv(expected_path)

    frame = player_match.sort_values(["match_datetime", "match_id", "player_id"]).reset_index(drop=True)
    frame = frame.merge(
        predictive[["match_id", "player_id", "predictive_form", "form_trend"]],
        on=["match_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    frame = _history_features(frame)

    team_columns = ["starting_xi_form", "attack_form", "midfield_form", "defense_form", "gk_form"]
    frame = frame.merge(team_form, on=["match_id", "team_id"], how="left", validate="many_to_one")
    opponent_form = team_form.rename(columns={"team_id": "opponent_team_id", **{column: f"opponent_{column}" for column in team_columns}})
    frame = frame.merge(
        opponent_form[["match_id", "opponent_team_id", *[f"opponent_{column}" for column in team_columns]]],
        on=["match_id", "opponent_team_id"],
        how="left",
        validate="many_to_one",
    )
    context_columns = [
        "match_id", "home_elo", "away_elo", "home_rest_days", "away_rest_days",
        "home_prev_avg_xg_scored", "away_prev_avg_xg_scored",
        "home_prev_avg_xg_conceded", "away_prev_avg_xg_conceded",
    ]
    frame = frame.merge(match_features[context_columns], on="match_id", how="left", validate="many_to_one")
    frame = frame.merge(expected, on="match_id", how="left", validate="many_to_one")
    home = frame["is_home"].eq(1)
    frame["team_recent_xg"] = np.where(home, frame["home_prev_avg_xg_scored"], frame["away_prev_avg_xg_scored"])
    frame["opponent_recent_xg_conceded"] = np.where(home, frame["away_prev_avg_xg_conceded"], frame["home_prev_avg_xg_conceded"])
    frame["elo_diff"] = np.where(home, frame["home_elo"] - frame["away_elo"], frame["away_elo"] - frame["home_elo"])
    frame["rest_days"] = np.where(home, frame["home_rest_days"], frame["away_rest_days"])
    frame["team_predicted_xg"] = np.where(home, frame["lambda_home"], frame["lambda_away"])
    frame["team_attack_form"] = frame["attack_form"]
    frame["opponent_defense_form"] = frame["opponent_defense_form"]
    frame["opponent_gk_form"] = frame["opponent_gk_form"]
    frame["position"] = frame["position"].fillna(frame["tactical_position"])

    required = [*GOAL_NUMERIC_FEATURES, *GOAL_CATEGORICAL_FEATURES, "scored_goal"]
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum()
        missing = missing[missing.gt(0)].to_dict()
        raise ValueError(f"Goal feature dataset contains unexpected missing values: {missing}")

    output_columns = [
        "match_id", "match_datetime", "date", "player_id", "player_name", "team_id", "opponent_team_id",
        "scored_goal", "appeared", "form_trend", *GOAL_CATEGORICAL_FEATURES, *GOAL_NUMERIC_FEATURES,
    ]
    output = frame[output_columns].copy()
    write_csv(output, FEATURE_DIR / "goal_probability_dataset.csv")
    return output


if __name__ == "__main__":
    build_goal_features()
