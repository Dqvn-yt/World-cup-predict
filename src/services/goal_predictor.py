from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_DIR, MODEL_DIR


class GoalPredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.bundle = joblib.load(model_path or MODEL_DIR / "goal_probability_model.joblib")
        self.data = pd.read_csv(FEATURE_DIR / "goal_probability_dataset.csv")
        self.feature_columns = [*self.bundle["numeric_features"], *self.bundle["categorical_features"]]

    def predict_match(self, match_id: int, starters_only: bool = False) -> list[dict]:
        frame = self.data[self.data["match_id"].eq(int(match_id))].copy()
        if frame.empty:
            raise ValueError(f"Unknown match_id: {match_id}")
        if starters_only:
            frame = frame[frame["is_starting_xi"].eq(1)].copy()
        probability = self.bundle["model"].predict_proba(frame[self.feature_columns])[:, 1]
        frame["goal_probability"] = probability
        frame = frame.sort_values("goal_probability", ascending=False)
        return [self._serialize(row) for _, row in frame.iterrows()]

    def predict_player(self, match_id: int, player_id: int) -> dict:
        predictions = self.predict_match(match_id)
        for prediction in predictions:
            if prediction["player_id"] == int(player_id):
                return prediction
        raise ValueError(f"Player {player_id} is not registered for match {match_id}")

    @staticmethod
    def _serialize(row: pd.Series) -> dict:
        return {
            "match_id": int(row["match_id"]),
            "player_id": int(row["player_id"]),
            "player_name": str(row["player_name"]),
            "team_id": int(row["team_id"]),
            "position": str(row["position"]),
            "is_starting_xi": bool(row["is_starting_xi"]),
            "goal_probability": float(row["goal_probability"]),
            "predictive_form": float(row["predictive_form"]),
            "form_trend": str(row["form_trend"]),
            "goals_last3": float(row["goals_last3"]),
            "goals_per90_last5": float(row["goals_per90_last5"]),
            "expected_minutes": float(row["expected_minutes"]),
            "team_predicted_xg": float(row["team_predicted_xg"]),
            "opponent_defense_form": float(row["opponent_defense_form"]),
        }
