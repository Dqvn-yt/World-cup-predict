import pandas as pd

from config import CLEAN_DIR, FEATURE_DIR


def test_player_match_has_expected_grain_and_valid_teams():
    frame = pd.read_csv(FEATURE_DIR / "player_match_performance.csv")
    assert len(frame) == 5408
    assert not frame.duplicated(["match_id", "player_id"]).any()
    assert ((frame["team_id"] == frame["home_team_id"]) | (frame["team_id"] == frame["away_team_id"])).all()


def test_goal_target_excludes_own_goals_and_shootouts():
    frame = pd.read_csv(FEATURE_DIR / "player_match_performance.csv")
    assert frame["goals"].sum() == 294
    assert frame["scored_goal"].sum() == 251
    assert ((frame["scored_goal"] == 1) == (frame["goals"] > 0)).all()


def test_events_reconcile_with_match_scores():
    frame = pd.read_csv(FEATURE_DIR / "player_match_performance.csv")
    matches = pd.read_csv(CLEAN_DIR / "matches.csv")
    event_goals = frame.groupby("match_id")[["goals", "own_goals"]].sum().sum(axis=1)
    match_goals = matches.set_index("match_id")[["home_score", "away_score"]].sum(axis=1)
    pd.testing.assert_series_equal(event_goals.sort_index(), match_goals.sort_index(), check_names=False)


def test_no_targets_in_match_model_features():
    from src.features.match_features import MATCH_MODEL_FEATURES

    forbidden = {"home_score", "away_score", "home_xg", "away_xg", "result_type", "match_result"}
    assert forbidden.isdisjoint(MATCH_MODEL_FEATURES)
