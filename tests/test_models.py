import numpy as np
import pytest

from src.models.dixon_coles import DixonColesModel, dc_score_matrix
from src.models.poisson import score_probability_matrix


def test_poisson_matrix_is_valid_probability_distribution():
    result = score_probability_matrix(1.8, 1.2, max_goals=8)
    matrix = np.asarray(result["matrix"])
    assert matrix.shape == (9, 9)
    assert matrix.sum() <= 1.0
    assert np.isclose(matrix.sum(), result["grid_coverage"])
    assert np.isclose(result["grid_coverage"] + result["tail_probability"], 1.0)
    assert np.isclose(
        result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"],
        1.0,
    )
    assert 0 <= result["predicted_home_goals"] <= 8
    assert 0 <= result["predicted_away_goals"] <= 8


def test_poisson_outcome_probabilities_sum_to_one():
    result = score_probability_matrix(3.4, 0.4, max_goals=6)
    assert np.isclose(
        result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"],
        1.0,
    )


def test_poisson_markets_are_consistent():
    result = score_probability_matrix(1.6, 1.3, max_goals=6)
    totals = result["total_goals_distribution"]
    assert np.isclose(sum(totals.values()), 1.0, atol=1e-6)
    assert np.isclose(result["over_under"]["1.5"], 1.0 - totals["0"] - totals["1"], atol=1e-6)
    assert np.isclose(result["both_teams_to_score"] + result["no_both_teams_to_score"], 1.0)
    double = result["double_chance"]
    assert np.isclose(double["home_or_draw"] + result["away_win_probability"], 1.0, atol=1e-6)
    margins = result["win_margins"]
    assert np.isclose(
        margins["home_by_1"] + margins["home_by_2_plus"] + result["draw_probability"]
        + margins["away_by_1"] + margins["away_by_2_plus"],
        1.0,
        atol=1e-6,
    )
    assert result["confidence"] in {"High", "Medium", "Low"}
    assert result["match_profile"] in {"Low-scoring", "Balanced", "Open / High-scoring"}
    assert np.isclose(result["expected_total_goals"], 1.6 + 1.3, atol=0.15)


def test_dixon_coles_adjusts_low_scores_and_sums_to_one():
    result = dc_score_matrix(1.5, 1.4, rho=-0.1, max_goals=6)
    matrix = np.asarray(result["matrix"])
    assert np.isclose(result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"], 1.0)
    baseline = np.outer(
        np.exp(-1.5) * 1.5 ** np.arange(7) / np.array([float(np.prod(range(1, k + 1))) if k else 1.0 for k in range(7)]),
        np.exp(-1.4) * 1.4 ** np.arange(7) / np.array([float(np.prod(range(1, k + 1))) if k else 1.0 for k in range(7)]),
    )
    baseline = baseline / baseline.sum()
    assert matrix[0, 0] > baseline[0, 0]
    assert np.isclose(result["expected_total_goals"], 2.9, atol=0.2)


def test_dixon_coles_model_fits_and_predicts_positive_lambdas():
    frame = __import__("pandas").DataFrame({
        "match_datetime": __import__("pandas").to_datetime(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]) ,
        "home_team_id": [1, 2, 3, 4],
        "away_team_id": [2, 3, 4, 1],
        "home_score": [2, 1, 0, 3],
        "away_score": [0, 1, 2, 1],
    })
    model = DixonColesModel(max_iter=60).fit(frame)
    lambda_home, lambda_away = model.lambdas(1, 2)
    assert 0.1 <= lambda_home <= 5.0
    assert 0.1 <= lambda_away <= 5.0
    assert -0.2 <= model.rho <= 0.2
    unknown_home, unknown_away = model.lambdas(99, 98)
    assert 0.1 <= unknown_home <= 5.0
    assert 0.1 <= unknown_away <= 5.0


def test_dixon_coles_neutral_venue_gets_no_home_advantage():
    import pandas as pd
    from src.models.dixon_coles import DixonColesModel

    frame = pd.DataFrame({
        "match_datetime": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]),
        "home_team_id": [1, 2, 3, 4],
        "away_team_id": [2, 3, 4, 1],
        "home_score": [2, 1, 0, 3],
        "away_score": [0, 1, 2, 1],
        "neutral": [True, True, True, False],
    })
    model = DixonColesModel(max_iter=60).fit(frame)
    assert model.neutral_matches == 3
    # Unknown teams isolate the venue mechanism: attack/defense are zero.
    unknown_neutral_home, unknown_neutral_away = model.lambdas(99, 98, neutral=True)
    unknown_host_home, unknown_host_away = model.lambdas(99, 98, neutral=False)
    assert unknown_neutral_home == pytest.approx(np.exp(model.neutral_advantage))
    assert unknown_host_home == pytest.approx(np.exp(model.home_advantage))
    assert unknown_neutral_away == unknown_host_away == pytest.approx(1.0)
    legacy = DixonColesModel(max_iter=60).fit(frame.drop(columns=["neutral"]))
    assert legacy.neutral_matches == 0


def test_inference_probabilities_are_bounded(app):
    match = app.extensions["match_predictor"].predict_match(1)
    assert match["lambda_home"] > 0
    assert match["lambda_away"] > 0
    assert isinstance(match["predicted_home_goals"], int)
    assert isinstance(match["predicted_away_goals"], int)
    assert 0 <= match["home_win_probability"] <= 1
    assert 0 <= match["both_teams_to_score"] <= 1
    assert 0 <= match["over_under"]["2.5"] <= 1
    goals = app.extensions["goal_predictor"].predict_match(1)
    assert len(goals) == 52
    assert all(0 <= player["goal_probability"] <= 1 for player in goals)


def test_score_holdout_dc_fit_excludes_test_rows(tmp_path, monkeypatch):
    import pandas as pd
    from config import FEATURE_DIR
    from src.models import train_score

    frame = pd.read_csv(FEATURE_DIR / "match_model_dataset.csv").iloc[:40].copy()
    fits = []
    class StubDC:
        home_advantage = 0.0
        neutral_advantage = 0.0
        rho = -0.05

        def fit(self, history, reference_time=None):
            fits.append(history["match_id"].tolist())
            return self

        def lambdas(self, *args, **kwargs):
            return 1.3, 1.1

    monkeypatch.setattr(train_score, "DixonColesModel", StubDC)
    monkeypatch.setattr(train_score, "ensure_directories", lambda: None)
    for name in ["FEATURE_DIR", "MODEL_DIR", "METRICS_DIR"]:
        monkeypatch.setattr(train_score, name, tmp_path)
    monkeypatch.setattr(train_score.joblib, "dump", lambda *args: None)
    monkeypatch.setattr(train_score, "walk_forward_dixon_coles", lambda frame, history=None: pd.DataFrame({
        "match_id": frame.match_id, "dc_lambda_home": 1.3, "dc_lambda_away": 1.1,
        "dc_rho": -0.05, "modeled": True, "dc_history_rows": 128}))
    monkeypatch.setattr(train_score, "_walk_forward_ml", lambda frame, history=None: pd.DataFrame({
        "match_id": frame.match_id, "ml_lambda_home": 1.3, "ml_lambda_away": 1.1, "modeled": True}))
    monkeypatch.setattr(train_score, "_evaluate_oof", lambda *args: {})
    report = train_score.train_score_models(frame)
    assert report["split"]["test_rows"] == 6
    historical, _, _ = train_score.load_score_history()
    assert fits == [historical.match_id.tolist() + frame.iloc[:34].match_id.tolist(),
                    frame.iloc[:34].match_id.tolist(), historical.match_id.tolist() + frame.match_id.tolist()]
    assert not set(frame.iloc[34:].match_id) & set(fits[0])
