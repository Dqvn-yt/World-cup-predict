import joblib
import numpy as np
import pandas as pd
import pytest

from config import FEATURE_DIR, MODEL_DIR
from src.data.score_history import RESULT_COLUMNS, load_score_history
from src.models.dixon_coles import DixonColesModel, walk_forward_dixon_coles
from src.models.poisson import ensemble_score_matrix
from src.models.train_score import _walk_forward_ml


def test_identity_and_regulation_contract():
    history, current, audit = load_score_history()
    assert audit["shared_teams"] == 30
    assert audit["historical_only_teams"] == 10
    assert audit["regulation_adjusted_match_ids"] == [81, 87, 99, 100, 104]
    assert len(history) == 128 and len(current) == 104
    assert pd.concat([history, current]).match_id.nunique() == 232
    assert list(history.columns) == RESULT_COLUMNS
    final = history.set_index("match_id").loc[2022064]
    assert (final.home_score, final.away_score) == (2, 2)
    assert current.set_index("match_id").loc[104, "home_score"] == 0
    frame = pd.read_csv(FEATURE_DIR / "match_model_dataset.csv").set_index("match_id")
    pd.testing.assert_frame_equal(frame[["home_score", "away_score"]].sort_index(), current.set_index("match_id")[["home_score", "away_score"]].sort_index(), check_dtype=False)


def test_all_history_teams_participate_in_serving_and_oof():
    history, current, _ = load_score_history()
    bundle = joblib.load(MODEL_DIR / "score_models.joblib")
    dc = bundle["dixon_coles"]
    assert dc.matches_used == 232
    assert len(dc.team_index) == 58
    assert set(history.home_team_id).union(history.away_team_id) <= dc.team_index.keys()
    oof = pd.read_csv(FEATURE_DIR / "match_expected_goals_oof.csv")
    assert len(oof) == 104 and oof.modeled.all()
    assert oof.iloc[0].dc_history_rows == 128
    assert oof.iloc[-1].dc_history_rows == 231
    assert oof.iloc[0].ml_history_rows == 0
    assert oof.iloc[-1].ml_history_rows == 103
    first = walk_forward_dixon_coles(current.iloc[:1], history=history)
    np.testing.assert_allclose(first[["dc_lambda_home", "dc_lambda_away", "dc_rho"]], oof.iloc[:1][["dc_lambda_home", "dc_lambda_away", "dc_rho"]])
    no_history = walk_forward_dixon_coles(current.iloc[:1])
    assert not np.allclose(first[["dc_lambda_home", "dc_lambda_away"]], no_history[["dc_lambda_home", "dc_lambda_away"]])


def test_oof_excludes_simultaneous_and_future_history_and_rho():
    history, current, _ = load_score_history()
    fixtures = current.iloc[:3].copy()
    fixtures.loc[fixtures.index[:2], "match_datetime"] = fixtures.iloc[0].match_datetime
    injected = history.iloc[:2].copy()
    injected["match_id"] = [9000001, 9000002]
    injected["match_datetime"] = [fixtures.iloc[0].match_datetime, fixtures.iloc[-1].match_datetime + pd.Timedelta(days=1)]
    baseline = walk_forward_dixon_coles(fixtures, history=history)
    changed = fixtures.copy()
    changed.loc[changed.index[:2], ["home_score", "away_score"]] = [8, 7]
    with_future = pd.concat([history, injected])
    prediction = walk_forward_dixon_coles(changed, history=with_future)
    columns = ["dc_lambda_home", "dc_lambda_away", "dc_rho", "dc_history_rows"]
    pd.testing.assert_frame_equal(baseline.iloc[:2][columns], prediction.iloc[:2][columns])
    injected.loc[:, ["home_score", "away_score"]] = 99
    again = walk_forward_dixon_coles(fixtures.iloc[:2], history=pd.concat([history, injected]))
    pd.testing.assert_frame_equal(baseline.iloc[:2][columns], again[columns])


def test_cold_ml_prior_uses_history_but_not_future_results():
    history, _, _ = load_score_history()
    frame = pd.read_csv(FEATURE_DIR / "match_model_dataset.csv").iloc[:2].copy()
    frame["match_datetime"] = pd.to_datetime(frame.match_datetime)
    baseline = _walk_forward_ml(frame, history)
    assert not np.isclose(baseline.iloc[0].ml_lambda_home, 1.3)
    future = history.iloc[:1].copy()
    future["match_datetime"] = frame.match_datetime.max() + pd.Timedelta(days=1)
    future[["home_score", "away_score"]] = 99
    pd.testing.assert_frame_equal(baseline, _walk_forward_ml(frame, pd.concat([history, future])))


def test_history_changes_dc_scenario_parameters():
    history, current, _ = load_score_history()
    with_history = joblib.load(MODEL_DIR / "score_models.joblib")["dixon_coles"]
    without_history = DixonColesModel().fit(current)
    assert not np.allclose(with_history.lambdas(37, 33), without_history.lambdas(37, 33))


def test_historical_prediction_uses_own_rho_not_final_model(app, monkeypatch):
    predictor = app.extensions["match_predictor"]
    row = predictor.oof.iloc[0]
    expected = ensemble_score_matrix(row.lambda_home, row.lambda_away, row.dc_rho)
    monkeypatch.setattr(predictor.bundle["dixon_coles"], "rho", 0.2)
    result = predictor.predict_match(int(row.match_id))
    np.testing.assert_allclose(result["matrix"], expected[:7, :7])
    assert result["rho"] == row.dc_rho
    assert result["both_teams_to_score"] == pytest.approx(expected[1:, 1:].sum())


def test_archive_is_not_fabricated_feature_api(client):
    assert client.post("/api/predict/match", json={"match_id": 2018001}).status_code == 400
    assert client.post("/api/predict/match", json={"home_team_id": 10000, "away_team_id": 33}).status_code == 400
    result = client.post("/api/predict/match", json={"match_id": 104}).get_json()
    assert (result["actual_home_score"], result["actual_away_score"]) == (0, 0)
    assert "90 minutes" in result["score_duration"]


def test_ml_fit_never_receives_historical_rows(monkeypatch):
    from src.models import train_score

    history, _, _ = load_score_history()
    frame = pd.read_csv(FEATURE_DIR / "match_model_dataset.csv").iloc[:22].copy()
    calls = []

    class StubML:
        def fit(self, x, y):
            assert list(x.columns) == train_score.MATCH_MODEL_FEATURES
            pd.testing.assert_frame_equal(x, frame.iloc[:len(x)][train_score.MATCH_MODEL_FEATURES])
            calls.append(len(x))
            return self

        def predict(self, x):
            return np.ones(len(x))

    monkeypatch.setattr(train_score, "_pipeline", lambda alpha: StubML())
    train_score._walk_forward_ml(frame, history)
    assert calls == [20, 20, 21, 21]


def test_decay_includes_gap_before_first_current_fixture():
    history, current, _ = load_score_history()
    reference = current.iloc[0].match_datetime
    model = DixonColesModel().fit(history, reference_time=reference)
    age = (reference - history.match_datetime).dt.total_seconds() / 86400
    assert model.effective_matches == pytest.approx(np.exp(-np.log(2) * age / 1461).sum())
    assert 45 < model.effective_matches < 55


def test_oof_metric_probabilities_match_api(app):
    from src.models.train_score import _evaluate_oof

    predictor = app.extensions["match_predictor"]
    row = predictor.oof.iloc[:1]
    result = predictor.predict_match(int(row.iloc[0].match_id))
    metrics = _evaluate_oof(row, predictor.matches)["ensemble"]
    actual_btts = float(result["actual_home_score"] > 0 and result["actual_away_score"] > 0)
    actual_over = float(result["actual_home_score"] + result["actual_away_score"] > 2.5)
    assert metrics["btts_brier"] == pytest.approx((result["both_teams_to_score"] - actual_btts) ** 2)
    assert metrics["over25_brier"] == pytest.approx((result["over_under"]["2.5"] - actual_over) ** 2)
    assert metrics["scoreline_log_likelihood"] == pytest.approx(np.log(result["matrix"][result["actual_home_score"]][result["actual_away_score"]]))


def test_identity_audit_rejects_changed_clean_code(monkeypatch):
    from pathlib import Path
    from src.data import score_history

    read_csv = pd.read_csv

    def corrupt_clean(path, *args, **kwargs):
        frame = read_csv(path, *args, **kwargs)
        if Path(path).parent.name == "clean" and Path(path).name == "teams.csv":
            frame.loc[0, "fifa_code"] = "ZZZ"
        return frame

    monkeypatch.setattr(score_history.pd, "read_csv", corrupt_clean)
    with pytest.raises(ValueError, match="Source/clean team identities differ"):
        score_history.load_score_history()
