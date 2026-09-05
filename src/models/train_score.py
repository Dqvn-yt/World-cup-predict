import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_DIR, METRICS_DIR, MODEL_DIR, ensure_directories
from src.data.loaders import write_csv
from src.data.score_history import RESULT_COLUMNS, load_score_history
from src.features.match_features import MATCH_MODEL_FEATURES, build_match_features
from src.models.dixon_coles import DixonColesModel, walk_forward_dixon_coles
from src.models.evaluate import regression_metrics
from src.models.poisson import _total_goals_pmf, ensemble_score_matrix, summarize_score_matrix


DC_WEIGHT = 0.7
ML_WEIGHT = 0.3
MIN_OOF_HISTORY = 20
XG_BOUNDS = (0.15, 4.5)


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=alpha, max_iter=2000)),
    ])


def _select_alpha(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, list[dict]]:
    """Select alpha by validation MAE, keeping bias diagnostics per candidate.

    Bias (mean signed error) is recorded so home/away asymmetries between
    candidates stay visible; selection itself remains MAE-only to avoid
    tuning the protocol around a preferred outcome.
    """
    candidates = [0.1, 1.0, 10.0]
    table = []
    for alpha in candidates:
        model = _pipeline(alpha).fit(x_train, y_train)
        prediction = np.clip(model.predict(x_val), 1e-6, None)
        table.append({
            "alpha": alpha,
            "validation_mae": float(np.mean(np.abs(y_val - prediction))),
            "validation_bias": float(np.mean(prediction - y_val.to_numpy())),
        })
    best = min(table, key=lambda row: row["validation_mae"])
    return best["alpha"], table


def _clip_lambda(value: float) -> float:
    return float(np.clip(value, *XG_BOUNDS))


def _walk_forward_ml(frame: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = frame.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
    x = frame[MATCH_MODEL_FEATURES]
    predictions: list[dict[str, float | int | bool]] = []
    for datetime, group in frame.groupby("match_datetime", sort=True):
        indices = group.index.tolist()
        start = indices[0]
        modeled = start >= MIN_OOF_HISTORY
        if not modeled:
            past = frame.iloc[:start]
            if history is not None:
                past = pd.concat([history[pd.to_datetime(history.match_datetime) < pd.Timestamp(datetime)], past])
            if len(past):
                age = (pd.Timestamp(datetime) - pd.to_datetime(past.match_datetime)).dt.total_seconds() / 86400
                weights = np.exp(-np.log(2) * age / 1461.0)
                home_lambda = _clip_lambda(float(np.average(past.home_score, weights=weights)))
                away_lambda = _clip_lambda(float(np.average(past.away_score, weights=weights)))
            else:
                home_lambda, away_lambda = 1.3, 1.1
            lambdas_home = [home_lambda] * len(indices)
            lambdas_away = [away_lambda] * len(indices)
        else:
            current_history = frame.iloc[:start]
            home_model = _pipeline(1.0).fit(x.iloc[:start], current_history["home_score"])
            away_model = _pipeline(1.0).fit(x.iloc[:start], current_history["away_score"])
            lambdas_home = [_clip_lambda(float(value)) for value in home_model.predict(x.iloc[indices])]
            lambdas_away = [_clip_lambda(float(value)) for value in away_model.predict(x.iloc[indices])]
        for position, index in enumerate(indices):
            predictions.append({
                "match_id": int(frame.iloc[index]["match_id"]),
                "ml_lambda_home": lambdas_home[position],
                "ml_lambda_away": lambdas_away[position],
                "modeled": modeled,
                "ml_history_rows": start,
            })
    return pd.DataFrame(predictions)


def _ensemble_lambdas(dc: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    merged = dc.merge(ml.rename(columns={"modeled": "ml_modeled"}), on="match_id", validate="one_to_one")
    merged["lambda_home"] = DC_WEIGHT * merged["dc_lambda_home"] + ML_WEIGHT * merged["ml_lambda_home"]
    merged["lambda_away"] = DC_WEIGHT * merged["dc_lambda_away"] + ML_WEIGHT * merged["ml_lambda_away"]
    return merged


def ranked_probability_score(actual_index: np.ndarray, probabilities: np.ndarray) -> float:
    cumulative_actual = np.cumsum(np.eye(3)[actual_index], axis=1)
    cumulative_pred = np.cumsum(probabilities, axis=1)
    return float(np.mean(np.sum((cumulative_pred - cumulative_actual) ** 2, axis=1)) / 2.0)


def _evaluate_oof(oof: pd.DataFrame, frame: pd.DataFrame) -> dict:
    merged = oof.merge(
        frame[["match_id", "home_score", "away_score", "home_team_id", "away_team_id"]],
        on="match_id",
        validate="one_to_one",
    )
    modeled = merged[merged["modeled"].eq(True)].reset_index(drop=True)
    if modeled.empty:
        return {}
    actual_home = modeled["home_score"].to_numpy(dtype=float)
    actual_away = modeled["away_score"].to_numpy(dtype=float)
    actual_outcome = np.sign(actual_home - actual_away)
    outcome_index = np.where(actual_outcome > 0, 0, np.where(actual_outcome == 0, 1, 2))

    report: dict = {"matches_modeled": int(len(modeled))}
    for label, prefix in [("ensemble", "lambda"), ("dixon_coles", "dc_lambda"), ("ml_poisson", "ml_lambda")]:
        home_lambda = modeled[f"{prefix}_home"].to_numpy()
        away_lambda = modeled[f"{prefix}_away"].to_numpy()
        matrices = [ensemble_score_matrix(h, a, rho, 0.7 if label == "ensemble" else 1.0 if label == "dixon_coles" else 0.0)
                    for h, a, rho in zip(home_lambda, away_lambda, modeled["dc_rho"])]
        wdl = np.array([[np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()] for m in matrices])
        predicted_outcome = 1 - np.argmax(wdl, axis=1)
        btts_predictions = np.array([
            matrix[1:, 1:].sum() for matrix in matrices
        ])
        over25_predictions = np.array([_total_goals_pmf(m)[3:].sum() for m in matrices])
        report[label] = {
            "home_goals_mae": regression_metrics(actual_home, home_lambda)["mae"],
            "away_goals_mae": regression_metrics(actual_away, away_lambda)["mae"],
            "goals_mae": regression_metrics(actual_home + actual_away, home_lambda + away_lambda)["mae"],
            "home_goals_bias": float(np.mean(home_lambda - actual_home)),
            "away_goals_bias": float(np.mean(away_lambda - actual_away)),
            "goals_bias": float(np.mean(home_lambda + away_lambda - actual_home - actual_away)),
            "scoreline_log_likelihood": float(np.mean([
                np.log(max(m[int(hg), int(ag)], 1e-12))
                for m, hg, ag in zip(matrices, actual_home, actual_away)
            ])),
            "outcome_accuracy": float(np.mean(predicted_outcome == actual_outcome)),
            "ranked_probability_score": ranked_probability_score(outcome_index, wdl),
            "btts_brier": float(np.mean((btts_predictions - ((actual_home > 0) & (actual_away > 0)).astype(float)) ** 2)),
            "over25_brier": float(np.mean((over25_predictions - (actual_home + actual_away > 2.5).astype(float)) ** 2)),
            "over25_rate_actual": float(np.mean(actual_home + actual_away > 2.5)),
        }
    return report


def train_score_models(frame: pd.DataFrame | None = None, history: pd.DataFrame | None = None) -> dict:
    ensure_directories()
    if frame is None:
        frame = build_match_features()
    frame = frame.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
    frame["match_datetime"] = pd.to_datetime(frame["match_datetime"])
    loaded_history, _, audit = load_score_history()
    history = loaded_history if history is None else history.copy()
    history["match_datetime"] = pd.to_datetime(history["match_datetime"])
    if set(history.match_id) & set(frame.match_id):
        raise ValueError("History overlaps current fixtures")
    x = frame[MATCH_MODEL_FEATURES]
    train_end = int(len(frame) * 0.70)
    val_end = int(len(frame) * 0.85)
    while train_end < len(frame) and frame.iloc[train_end].match_datetime == frame.iloc[train_end - 1].match_datetime:
        train_end += 1
    while val_end < len(frame) and frame.iloc[val_end].match_datetime == frame.iloc[val_end - 1].match_datetime:
        val_end += 1
    train, validation, test = slice(0, train_end), slice(train_end, val_end), slice(val_end, len(frame))

    home_alpha, home_alpha_table = _select_alpha(x.iloc[train], frame["home_score"].iloc[train], x.iloc[validation], frame["home_score"].iloc[validation])
    away_alpha, away_alpha_table = _select_alpha(x.iloc[train], frame["away_score"].iloc[train], x.iloc[validation], frame["away_score"].iloc[validation])
    home_model = _pipeline(home_alpha).fit(x.iloc[:val_end], frame["home_score"].iloc[:val_end])
    away_model = _pipeline(away_alpha).fit(x.iloc[:val_end], frame["away_score"].iloc[:val_end])

    # Holdout metrics must not use DC parameters fitted on holdout outcomes.
    holdout_history = history[history.match_datetime < frame.iloc[val_end].match_datetime]
    dc_final = DixonColesModel().fit(pd.concat([holdout_history, frame.iloc[:val_end][RESULT_COLUMNS]], ignore_index=True), reference_time=frame.iloc[val_end].match_datetime)
    home_prediction = np.clip(home_model.predict(x.iloc[test]), *XG_BOUNDS)
    away_prediction = np.clip(away_model.predict(x.iloc[test]), *XG_BOUNDS)
    test_size = len(frame) - val_end
    baseline_home_mean = float(frame["home_score"].iloc[:val_end].mean())
    baseline_away_mean = float(frame["away_score"].iloc[:val_end].mean())
    train_val_outcomes = np.sign(
        frame["home_score"].iloc[:val_end].to_numpy() - frame["away_score"].iloc[:val_end].to_numpy()
    )
    values, counts = np.unique(train_val_outcomes, return_counts=True)
    majority_outcome = int(values[int(np.argmax(counts))])

    dc_lambdas_home, dc_lambdas_away = zip(*[
        dc_final.lambdas(int(row.home_team_id), int(row.away_team_id), neutral=bool(getattr(row, "neutral", True)))
        for row in frame.iloc[test].itertuples()
    ])
    ensemble_home = DC_WEIGHT * np.array(dc_lambdas_home) + ML_WEIGHT * home_prediction
    ensemble_away = DC_WEIGHT * np.array(dc_lambdas_away) + ML_WEIGHT * away_prediction
    predicted_scores = [summarize_score_matrix(ensemble_score_matrix(h, a, dc_final.rho), h, a) for h, a in zip(ensemble_home, ensemble_away)]
    actual_outcomes = np.sign(frame["home_score"].iloc[test].to_numpy() - frame["away_score"].iloc[test].to_numpy())
    predicted_outcomes = np.array([
        1 - np.argmax([result["home_win_probability"], result["draw_probability"], result["away_win_probability"]])
        for result in predicted_scores
    ])
    exact = np.mean([
        result["predicted_home_goals"] == actual_home and result["predicted_away_goals"] == actual_away
        for result, actual_home, actual_away in zip(
            predicted_scores, frame["home_score"].iloc[test], frame["away_score"].iloc[test]
        )
    ])
    metrics = {
        "algorithm": "Dixon-Coles + Poisson Regression ensemble (0.7 DC / 0.3 ML)",
        "split": {"train_rows": train_end, "validation_rows": val_end - train_end, "test_rows": test_size},
        "home_alpha": home_alpha,
        "away_alpha": away_alpha,
        "home_alpha_candidates": home_alpha_table,
        "away_alpha_candidates": away_alpha_table,
        "dc_home_advantage": dc_final.home_advantage,
        "dc_neutral_advantage": dc_final.neutral_advantage,
        "dc_rho": dc_final.rho,
        "test_split": {
            "home_goals": regression_metrics(frame["home_score"].iloc[test].to_numpy(), ensemble_home),
            "away_goals": regression_metrics(frame["away_score"].iloc[test].to_numpy(), ensemble_away),
            "outcome_accuracy": float(np.mean(actual_outcomes == predicted_outcomes)),
            "exact_score_accuracy": float(exact),
        },
        "baselines": {
            "majority_outcome_accuracy": float(np.mean(actual_outcomes == majority_outcome)),
            "home_goals_constant": regression_metrics(
                frame["home_score"].iloc[test].to_numpy(), np.full(test_size, baseline_home_mean)
            ),
            "away_goals_constant": regression_metrics(
                frame["away_score"].iloc[test].to_numpy(), np.full(test_size, baseline_away_mean)
            ),
        },
    }

    dc_oof = walk_forward_dixon_coles(frame, history=history)
    ml_oof = _walk_forward_ml(frame, history=history)
    oof = _ensemble_lambdas(dc_oof, ml_oof)
    write_csv(oof, FEATURE_DIR / "match_expected_goals_oof.csv")
    oof_report = _evaluate_oof(oof, frame)
    if oof_report:
        metrics["out_of_fold"] = oof_report

    baseline_oof = _ensemble_lambdas(walk_forward_dixon_coles(frame), _walk_forward_ml(frame))
    common_ids = baseline_oof.loc[baseline_oof.modeled, "match_id"]
    metrics["comparison"] = {
        "protocol": "Same regulation targets, ridge=1 (incl. venue advantages), half-life=1461 days, two-level venue effect (host/neutral), ML features and splits; history on/off. No tuning on holdout.",
        "oof_common_with_history": _evaluate_oof(oof[oof.match_id.isin(common_ids)], frame),
        "oof_common_without_history": _evaluate_oof(baseline_oof, frame),
    }
    for label, model in [("with_history", dc_final), ("without_history", DixonColesModel().fit(frame.iloc[:val_end], reference_time=frame.iloc[val_end].match_datetime))]:
        pairs = [model.lambdas(int(r.home_team_id), int(r.away_team_id), neutral=bool(getattr(r, "neutral", True)))
                 for r in frame.iloc[test].itertuples()]
        holdout = pd.DataFrame({"match_id": frame.iloc[test].match_id.to_numpy(), "dc_lambda_home": [p[0] for p in pairs],
                                "dc_lambda_away": [p[1] for p in pairs], "dc_rho": model.rho, "modeled": True})
        ml_holdout = pd.DataFrame({"match_id": holdout.match_id, "ml_lambda_home": home_prediction, "ml_lambda_away": away_prediction, "modeled": True})
        metrics["comparison"][f"holdout_{label}"] = _evaluate_oof(_ensemble_lambdas(holdout, ml_holdout), frame)
    serving_history = history[history.match_datetime <= frame.match_datetime.max()]
    metrics["history_audit"] = audit
    metrics["training_counts"] = {
        "ml_train": train_end, "ml_validation": val_end - train_end, "ml_holdout": test_size,
        "dc_train_history": int((history.match_datetime < frame.iloc[train_end].match_datetime).sum()),
        "dc_holdout_current": val_end, "dc_holdout_history": len(holdout_history),
        "dc_serving_current": len(frame), "dc_serving_history": len(serving_history),
        "ml_serving_current": len(frame), "ml_serving_history": 0,
        "oof_first_dc_history": int(oof.iloc[0].dc_history_rows), "oof_last_dc_history": int(oof.iloc[-1].dc_history_rows),
        "oof_ml_fitted_predictions": int(oof.ml_modeled.sum()), "oof_ml_prior_predictions": int((~oof.ml_modeled).sum()),
    }
    # Refit both components on all observed results only after evaluation.
    home_model = _pipeline(home_alpha).fit(x, frame.home_score)
    away_model = _pipeline(away_alpha).fit(x, frame.away_score)

    bundle = {
        "home_model": home_model,
        "away_model": away_model,
        "dixon_coles": DixonColesModel().fit(pd.concat([serving_history, frame[RESULT_COLUMNS]], ignore_index=True)),
        "features": MATCH_MODEL_FEATURES,
        "feature_medians": x.median(numeric_only=True).to_dict(),
        "ensemble_weights": {"dc": DC_WEIGHT, "ml": ML_WEIGHT},
        "trained_through": str(frame.iloc[-1]["match_datetime"]),
        "history_audit": audit,
    }
    joblib.dump(bundle, MODEL_DIR / "score_models.joblib")
    joblib.dump(home_model, MODEL_DIR / "home_goal_model.joblib")
    joblib.dump(away_model, MODEL_DIR / "away_goal_model.joblib")
    with (METRICS_DIR / "score_metrics.json").open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    return metrics


if __name__ == "__main__":
    train_score_models()
