import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FEATURE_DIR, METRICS_DIR, MODEL_DIR, RANDOM_STATE, ensure_directories
from src.features.goal_features import GOAL_CATEGORICAL_FEATURES, GOAL_NUMERIC_FEATURES, build_goal_features
from src.models.calibration import balanced_prior_correction, calibrate_balanced_probabilities
from src.models.evaluate import classification_metrics


MIN_OOF_HISTORY = 20
PRIOR_PREVALENCE = 0.0464


def _pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    processor = ColumnTransformer([
        ("numeric", numeric, GOAL_NUMERIC_FEATURES),
        ("categorical", categorical, GOAL_CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", processor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def _best_threshold(actual: pd.Series, probability: np.ndarray) -> float:
    thresholds = np.arange(0.10, 0.91, 0.02)
    scores = [f1_score(actual, probability >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _top_k_hit_rate(frame: pd.DataFrame, probability: np.ndarray, k: int) -> float:
    ranked = frame[["match_id", "scored_goal"]].copy()
    ranked["probability"] = probability
    hits = []
    for _, match in ranked.groupby("match_id"):
        scorers = match[match["scored_goal"].eq(1)]
        if scorers.empty:
            continue
        top_ids = set(match.nlargest(k, "probability").index)
        hits.append(any(index in top_ids for index in scorers.index))
    return float(np.mean(hits)) if hits else 0.0


def train_goal_model(frame: pd.DataFrame | None = None) -> dict:
    ensure_directories()
    if frame is None:
        path = FEATURE_DIR / "goal_probability_dataset.csv"
        frame = pd.read_csv(path) if path.exists() else build_goal_features()
    frame = frame.sort_values(["match_datetime", "match_id", "player_id"]).reset_index(drop=True)
    ordered_matches = frame[["match_id", "match_datetime"]].drop_duplicates().sort_values(["match_datetime", "match_id"])
    train_end = int(len(ordered_matches) * 0.70)
    val_end = int(len(ordered_matches) * 0.85)
    train_ids = set(ordered_matches.iloc[:train_end]["match_id"])
    val_ids = set(ordered_matches.iloc[train_end:val_end]["match_id"])
    test_ids = set(ordered_matches.iloc[val_end:]["match_id"])
    feature_columns = [*GOAL_NUMERIC_FEATURES, *GOAL_CATEGORICAL_FEATURES]

    train_frame = frame[frame["match_id"].isin(train_ids)]
    validation_frame = frame[frame["match_id"].isin(val_ids)]
    test_frame = frame[frame["match_id"].isin(test_ids)]
    threshold_model = _pipeline().fit(train_frame[feature_columns], train_frame["scored_goal"])
    validation_probability = threshold_model.predict_proba(validation_frame[feature_columns])[:, 1]
    threshold = _best_threshold(validation_frame["scored_goal"], validation_probability)

    final_training = frame[frame["match_id"].isin(train_ids | val_ids)]
    model = _pipeline().fit(final_training[feature_columns], final_training["scored_goal"])
    test_probability = model.predict_proba(test_frame[feature_columns])[:, 1]
    metrics = {
        "algorithm": "Balanced Logistic Regression",
        "threshold": threshold,
        "positive_rate": float(frame["scored_goal"].mean()),
        "split": {
            "train_matches": len(train_ids),
            "validation_matches": len(val_ids),
            "test_matches": len(test_ids),
            "test_rows": len(test_frame),
        },
        **classification_metrics(test_frame["scored_goal"].to_numpy(), test_probability, threshold),
        "top_3_hit_rate": _top_k_hit_rate(test_frame, test_probability, 3),
        "top_5_hit_rate": _top_k_hit_rate(test_frame, test_probability, 5),
    }
    bundle = {
        "model": model,
        "numeric_features": GOAL_NUMERIC_FEATURES,
        "categorical_features": GOAL_CATEGORICAL_FEATURES,
        "threshold": threshold,
        "trained_through": str(final_training["date"].max()),
    }
    joblib.dump(bundle, MODEL_DIR / "goal_probability_model.joblib")
    with (METRICS_DIR / "goal_metrics.json").open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    return metrics


if __name__ == "__main__":
    train_goal_model()
