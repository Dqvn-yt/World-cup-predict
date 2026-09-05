from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_DIR, MODEL_DIR, SCORE_MATRIX_MAX_GOALS
from src.features.match_features import MATCH_MODEL_FEATURES, PAIRED_FEATURES
from src.data.score_history import load_score_history
from src.models.poisson import score_probability_matrix, summarize_score_matrix, ensemble_score_matrix


class MatchPredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.bundle = joblib.load(model_path or MODEL_DIR / "score_models.joblib")
        self.matches = pd.read_csv(FEATURE_DIR / "match_model_dataset.csv")
        self.matches["match_datetime"] = pd.to_datetime(self.matches["match_datetime"])
        self.matches = self.matches.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
        oof_path = FEATURE_DIR / "match_expected_goals_oof.csv"
        self.oof = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame()
        self.team_names = self._team_names()
        self.history, _, _ = load_score_history()
        self._team_state_cache = self._build_team_states()

    def predict_match(self, match_id: int) -> dict:
        selected = self.matches[self.matches["match_id"].eq(int(match_id))]
        if selected.empty:
            raise ValueError(f"Unknown match_id: {match_id}")
        row = selected.iloc[0].copy()
        return self._predict(row, source="oof_walk_forward")

    def predict_teams(self, home_team_id: int, away_team_id: int, neutral: bool = True) -> dict:
        home_team_id, away_team_id = int(home_team_id), int(away_team_id)
        if home_team_id == away_team_id:
            raise ValueError("Home and away teams must be different")
        if home_team_id not in self.team_names or away_team_id not in self.team_names:
            raise ValueError("Unknown team_id")
        medians = self.bundle["feature_medians"]
        row = pd.Series({feature: medians.get(feature, 0.0) for feature in MATCH_MODEL_FEATURES}, dtype=object)
        home_state = self._team_state_cache.get(home_team_id, {})
        away_state = self._team_state_cache.get(away_team_id, {})
        for suffix in PAIRED_FEATURES:
            row[f"home_{suffix}"] = home_state.get(suffix, row.get(f"home_{suffix}", 0.0))
            row[f"away_{suffix}"] = away_state.get(suffix, row.get(f"away_{suffix}", 0.0))
        row["is_knockout"] = 0
        row["home_team_id"] = home_team_id
        row["away_team_id"] = away_team_id
        row["home_team_name"] = self.team_names[home_team_id]
        row["away_team_name"] = self.team_names[away_team_id]
        row["match_id"] = None
        # Hypothetical World Cup fixtures have no known venue, and nearly all
        # tournament games are neutral, so scenarios default to neutral.
        row["neutral"] = bool(neutral)
        self._set_differences(row)
        return self._predict(row, source="team_scenario")

    def _ensemble_lambdas(self, row: pd.Series) -> tuple[float, float]:
        x = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in self.bundle["features"]}])
        weights = self.bundle.get("ensemble_weights", {"dc": 0.7, "ml": 0.3})
        ml_home = float(np.clip(self.bundle["home_model"].predict(x)[0], 0.15, 4.5))
        ml_away = float(np.clip(self.bundle["away_model"].predict(x)[0], 0.15, 4.5))
        dc_home, dc_away = self.bundle["dixon_coles"].lambdas(
            int(row["home_team_id"]), int(row["away_team_id"]), neutral=bool(row.get("neutral", True))
        )
        lambda_home = weights["dc"] * dc_home + weights["ml"] * ml_home
        lambda_away = weights["dc"] * dc_away + weights["ml"] * ml_away
        return float(np.clip(lambda_home, 0.1, 5.0)), float(np.clip(lambda_away, 0.1, 5.0))

    def _predict(self, row: pd.Series, source: str) -> dict:
        if source == "oof_walk_forward":
            oof_row = self.oof[self.oof["match_id"].eq(int(row["match_id"]))] if not self.oof.empty else pd.DataFrame()
            if not oof_row.empty:
                lambda_home = float(oof_row.iloc[0]["lambda_home"])
                lambda_away = float(oof_row.iloc[0]["lambda_away"])
                rho = float(oof_row.iloc[0].get("dc_rho", 0.0))
                result = summarize_score_matrix(ensemble_score_matrix(lambda_home, lambda_away, rho), lambda_home, lambda_away, SCORE_MATRIX_MAX_GOALS)
                result["rho"] = rho
                result["source_model"] = "oof_ensemble" if str(oof_row.iloc[0].get("modeled", False)).lower() == "true" else "oof_prior"
            else:
                prior = self.matches[self.matches["match_datetime"] < row["match_datetime"]]
                prior = pd.concat([self.history[self.history.match_datetime < row.match_datetime], prior])
                age = (row.match_datetime - pd.to_datetime(prior.match_datetime)).dt.total_seconds() / 86400
                weights = np.exp(-np.log(2) * age / 1461.0)
                lambda_home = float(np.clip(np.average(prior.home_score, weights=weights), 0.15, 4.5)) if len(prior) else 1.3
                lambda_away = float(np.clip(np.average(prior.away_score, weights=weights), 0.15, 4.5)) if len(prior) else 1.1
                result = score_probability_matrix(lambda_home, lambda_away, SCORE_MATRIX_MAX_GOALS)
                result["source_model"] = "historical_prior"
        else:
            lambda_home, lambda_away = self._ensemble_lambdas(row)
            result = self._final_matrix(lambda_home, lambda_away)
            result["source_model"] = "final_ensemble"
        result.update({
            "match_id": None if pd.isna(row.get("match_id")) else int(row["match_id"]),
            "home_team_id": int(row["home_team_id"]),
            "away_team_id": int(row["away_team_id"]),
            "home_team_name": str(row["home_team_name"]),
            "away_team_name": str(row["away_team_name"]),
            "source": source,
            "score_duration": "90 minutes including stoppage time; excludes extra time and shootouts",
            "actual_home_score": _optional_int(row.get("home_score")),
            "actual_away_score": _optional_int(row.get("away_score")),
            "comparison": {
                "home_elo": _optional_float(row.get("home_elo")),
                "away_elo": _optional_float(row.get("away_elo")),
                "home_fifa_rank": _optional_float(row.get("home_fifa_rank")),
                "away_fifa_rank": _optional_float(row.get("away_fifa_rank")),
                "home_attack_strength": _optional_float(row.get("home_team_attack_strength")),
                "away_attack_strength": _optional_float(row.get("away_team_attack_strength")),
                "home_defense_strength": _optional_float(row.get("home_team_defense_strength")),
                "away_defense_strength": _optional_float(row.get("away_team_defense_strength")),
                "home_team_ppg": _optional_float(row.get("home_team_ppg")),
                "away_team_ppg": _optional_float(row.get("away_team_ppg")),
            },
        })
        return result

    def _final_matrix(self, lambda_home: float, lambda_away: float) -> dict:
        dc = self.bundle["dixon_coles"]
        weights = self.bundle.get("ensemble_weights", {"dc": 0.7, "ml": 0.3})
        matrix = ensemble_score_matrix(lambda_home, lambda_away, dc.rho, weights["dc"])
        result = summarize_score_matrix(matrix, lambda_home, lambda_away, SCORE_MATRIX_MAX_GOALS)
        result["rho"] = float(dc.rho)
        return result

    def _build_team_states(self) -> dict[int, dict[str, float]]:
        states: dict[int, dict[str, float]] = {}
        for row in self.matches.sort_values("match_datetime").itertuples():
            for role in ["home", "away"]:
                team_id = int(getattr(row, f"{role}_team_id"))
                state = states.setdefault(team_id, {})
                for suffix in PAIRED_FEATURES:
                    value = getattr(row, f"{role}_{suffix}", None)
                    try:
                        if value is not None and not pd.isna(value):
                            state[suffix] = float(value)
                    except TypeError:
                        continue
        return states

    def _team_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for row in self.matches.itertuples():
            names[int(row.home_team_id)] = str(row.home_team_name)
            names[int(row.away_team_id)] = str(row.away_team_name)
        return names

    @staticmethod
    def _set_differences(row: pd.Series) -> None:
        row["elo_diff"] = row["home_elo"] - row["away_elo"]
        row["fifa_rank_diff"] = row["away_fifa_rank"] - row["home_fifa_rank"]
        row["squad_value_diff"] = row["home_squad_total_value_eur"] - row["away_squad_total_value_eur"]
        row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
        row["attack_diff"] = row["home_team_attack_strength"] - row["away_team_attack_strength"]
        row["defense_diff"] = row["home_team_defense_strength"] - row["away_team_defense_strength"]
        row["xg_diff"] = row["home_team_xg_strength"] - row["away_team_xg_strength"]
        row["ppg_diff"] = row["home_team_ppg"] - row["away_team_ppg"]
        row["home_attack_vs_away_defense"] = row["home_team_attack_strength"] - row["away_team_defense_strength"]
        row["away_attack_vs_home_defense"] = row["away_team_attack_strength"] - row["home_team_defense_strength"]


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None or pd.isna(value) else int(value)
