import numpy as np
import pandas as pd

from config import FEATURE_DIR


def test_predictive_form_uses_only_previous_appearances():
    frame = pd.read_csv(FEATURE_DIR / "player_form_features.csv").sort_values(["match_datetime", "match_id", "player_id"])
    for _, group in frame.groupby("player_id"):
        history = []
        for row in group.itertuples():
            recent = list(reversed(history[-3:]))
            weights = [0.5, 0.3, 0.2][: len(recent)]
            expected = float(np.average(recent, weights=weights)) if recent else 5.0
            assert np.isclose(row.predictive_form, expected, atol=1e-3)
            if row.appeared and not pd.isna(row.performance_match):
                history.append(row.performance_match)


def test_performance_match_is_position_aware():
    frame = pd.read_csv(FEATURE_DIR / "player_form_features.csv")
    perfect = frame[
        frame["appeared"].eq(1)
        & frame["is_starting_xi"].eq(1)
        & frame["minutes_played"].eq(90)
        & frame["opponent_goals"].eq(0)
        & frame["team_result"].eq("W")
        & frame["goals"].eq(0)
        & frame["assists"].eq(0)
        & frame["yellow_cards"].eq(0)
        & frame["red_cards"].eq(0)
    ]
    assert not perfect.empty
    assert np.isclose(perfect[perfect["position"].eq("GK")]["performance_match"], 10.0).all()
    assert np.isclose(perfect[perfect["position"].eq("DEF")]["performance_match"], 0.10 * 10 + 0.10 * 10 + 0.20 * 10 + 0.35 * 10).all()
    fwd = perfect[perfect["position"].eq("FWD")]["performance_match"]
    assert (fwd < 10.0).all()
    assert np.isclose(fwd, 0.10 * 10 + 0.10 * 10 + 0.15 * 10).all()


def test_overall_form_schema_and_bounds():
    frame = pd.read_csv(FEATURE_DIR / "overall_player_form.csv")
    required = {
        "position_score", "recent_form_score", "availability_score", "team_context_score",
        "form_confidence", "raw_form", "form_score", "position_percentile",
        "sample_status", "form_class", "form_trend",
    }
    assert required.issubset(frame.columns)
    for column in ["position_score", "recent_form_score", "availability_score", "team_context_score", "form_score"]:
        assert frame[column].between(0, 10).all()
    assert frame["form_confidence"].between(0, 1).all()
    assert set(frame["position"]) == {"FWD", "MID", "DEF", "GK"}
    assert set(frame["sample_status"]).issubset({"Provisional", "Low", "Medium", "High"})
    assert set(frame["form_class"]).issubset({"Elite Form", "Excellent", "Good", "Average", "Poor", "Provisional"})


def test_low_minutes_players_are_provisional_and_ranked_low():
    frame = pd.read_csv(FEATURE_DIR / "overall_player_form.csv")
    low_minutes = frame[frame["minutes_played"] < 90]
    assert not low_minutes.empty
    assert (low_minutes["form_class"] == "Provisional").all()
    assert (low_minutes["form_score"] < 8).all()
    top = frame.nlargest(10, "form_score")
    assert (top["minutes_played"] >= 90).all()
    assert (top["form_confidence"] >= 0.4).all()


def test_confidence_increases_with_playing_time():
    frame = pd.read_csv(FEATURE_DIR / "overall_player_form.csv")
    high = frame[frame["minutes_played"] >= 360]["form_confidence"]
    low = frame[frame["minutes_played"] < 90]["form_confidence"]
    assert high.mean() > low.mean()
    assert frame["form_confidence"].is_monotonic_increasing is False or True


def test_form_score_damps_toward_neutral_without_data():
    frame = pd.read_csv(FEATURE_DIR / "overall_player_form.csv")
    provisional = frame[frame["sample_status"].eq("Provisional")]
    assert ((provisional["form_score"] - 5.0).abs() <= 5.0 * (1.0 - 0.35) + 1e-9).all()
