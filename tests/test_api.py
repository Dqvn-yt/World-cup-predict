import pytest


@pytest.mark.parametrize("url", [
    "/", "/players", "/player/1", "/form", "/matches", "/match/1",
    "/predict/match", "/predict/match/scenario?home_team_id=1&away_team_id=2",
    "/predict/goals", "/compare", "/models",
])
def test_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_match_prediction_api(client):
    response = client.post("/api/predict/match", json={"match_id": 1})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["home_team_name"] == "Mexico"
    assert abs(payload["home_win_probability"] + payload["draw_probability"] + payload["away_win_probability"] - 1) < 1e-9


def test_goal_prediction_api(client):
    response = client.post("/api/predict/goal", json={"match_id": 1, "player_id": 16})
    assert response.status_code == 200
    assert response.get_json()["player_id"] == 16


def test_scenario_detail_matches_fixture_output_sections(client):
    scenario = client.get("/predict/match/scenario?home_team_id=1&away_team_id=2")
    fixture = client.get("/match/1")
    assert scenario.status_code == 200
    assert fixture.status_code == 200
    scenario_html = scenario.get_data(as_text=True)
    fixture_html = fixture.get_data(as_text=True)
    shared_sections = [
        "Score probability matrix",
        "Most likely scores",
        "Total goals distribution",
        "Winning margins",
        "Team comparison",
    ]
    for section in shared_sections:
        assert section in scenario_html
        assert section in fixture_html
    assert "Team scenario" in scenario_html
    assert "No registered lineup" in scenario_html
    assert "Top goal threats" not in scenario_html
    assert "Starting XI" not in scenario_html


def test_scenario_detail_rejects_invalid_teams(client):
    assert client.get("/predict/match/scenario?home_team_id=1&away_team_id=1").status_code == 400
    assert client.get("/predict/match/scenario?home_team_id=999&away_team_id=2").status_code == 400
    assert client.get("/predict/match/scenario").status_code == 400


def test_api_rejects_invalid_input(client):
    assert client.post("/api/predict/match", json={"home_team_id": 1, "away_team_id": 1}).status_code == 400
    assert client.post("/api/predict/goal", json={"match_id": 999}).status_code == 400
