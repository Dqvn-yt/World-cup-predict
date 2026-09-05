import numpy as np
import pandas as pd
import pytest

from src.services.match_predictor import MatchPredictor


@pytest.mark.parametrize("home,away", [(1, 2), (37, 33), (9, 29), (48, 18), (33, 37)])
def test_scenario_api_and_detail(client, home, away):
    response = client.post("/api/predict/match", json={"home_team_id": home, "away_team_id": away})
    assert response.status_code == 200
    result = response.get_json()
    matrix = np.array(result["matrix"])
    assert matrix.shape == (7, 7)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(result["grid_coverage"])
    assert matrix.sum() + result["tail_probability"] == pytest.approx(1)
    assert result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"] == pytest.approx(1)
    for key, visible in [("home_win_probability", np.tril(matrix, -1).sum()), ("draw_probability", np.trace(matrix)), ("away_win_probability", np.triu(matrix, 1).sum()), ("both_teams_to_score", matrix[1:, 1:].sum())]:
        assert visible - 1e-12 <= result[key] <= visible + result["tail_probability"] + 1e-12
    for score in result["top_scorelines"]:
        assert score["probability"] == pytest.approx(matrix[score["home_goals"], score["away_goals"]])
    best = result["top_scorelines"][0]
    assert (result["predicted_home_goals"], result["predicted_away_goals"]) == (best["home_goals"], best["away_goals"])
    assert client.get("/match/104").status_code == 200


def test_team_scenario_defaults_to_neutral_venue():
    predictor = MatchPredictor()
    neutral = predictor.predict_teams(1, 2)
    non_neutral = predictor.predict_teams(1, 2, neutral=False)
    assert neutral["lambda_home"] <= non_neutral["lambda_home"]
    assert neutral["lambda_away"] == non_neutral["lambda_away"]


def test_historical_fallback_never_calls_final_model(monkeypatch):
    predictor = MatchPredictor()
    def forbidden(*args):
        raise AssertionError("Future-trained model used for historical prediction")
    monkeypatch.setattr(predictor, "_ensemble_lambdas", forbidden)
    first = predictor.matches.iloc[0]
    predictor.oof = pd.DataFrame([{"match_id": first.match_id, "modeled": False, "lambda_home": 1.3, "lambda_away": 1.1}])
    assert predictor.predict_match(first.match_id)["source_model"] == "oof_prior"
    predictor.oof = pd.DataFrame()
    before = predictor.predict_match(first.match_id)
    predictor.matches.loc[:, "home_score"] = 99
    assert predictor.predict_match(first.match_id)["matrix"] == before["matrix"]
    assert before["source_model"] == "historical_prior"


def test_all_existing_match_predictions():
    predictor = MatchPredictor()
    for match_id in predictor.matches["match_id"]:
        result = predictor.predict_match(match_id)
        assert result["source_model"] != "final_ensemble"
        assert np.isfinite(result["matrix"]).all()


@pytest.mark.parametrize("home,away,rho", [(0.1, 0.1, -0.2), (5, 5, 0.2), (5, 0.1, -0.2), (1.5, 1.4, -0.1)])
def test_blended_markets_share_full_distribution(home, away, rho):
    from types import SimpleNamespace
    from src.models.dixon_coles import dixon_coles_tau
    from src.models.poisson import FULL_SUPPORT, _pmf, _markets

    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.bundle = {"dixon_coles": SimpleNamespace(rho=rho)}
    poisson = np.outer(_pmf(home, FULL_SUPPORT), _pmf(away, FULL_SUPPORT))
    dc = poisson * dixon_coles_tau(home, away, rho, FULL_SUPPORT)
    expected = 0.3 * poisson / poisson.sum() + 0.7 * dc / dc.sum()
    result = predictor._final_matrix(home, away)
    np.testing.assert_allclose(result["matrix"], expected[:7, :7])
    for key, value in _markets(expected, home, away).items():
        assert result[key] == (value if isinstance(value, str) else pytest.approx(value))


def test_missing_oof_uses_only_strictly_earlier_results():
    predictor = MatchPredictor()
    predictor.oof = pd.DataFrame()
    row = predictor.matches.iloc[20]
    before = predictor.predict_match(row.match_id)
    predictor.matches.loc[predictor.matches["match_datetime"] >= row.match_datetime, ["home_score", "away_score"]] = 99
    after = predictor.predict_match(row.match_id)
    assert after["matrix"] == before["matrix"]
