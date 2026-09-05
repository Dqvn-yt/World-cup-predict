import numpy as np
import pandas as pd

from config import FEATURE_DIR, ensure_directories
from src.data.cleaning import clean_all
from src.data.loaders import read_clean, require_columns, write_csv


EVENT_COLUMNS = {
    "Goal": "goals",
    "Assist": "assists",
    "Yellow Card": "yellow_cards",
    "Red Card": "red_cards",
    "Own Goal": "own_goals",
    "Penalty Shootout Goal": "shootout_goals",
    "Penalty Shootout Miss": "shootout_misses",
    "VAR Review": "var_reviews",
}


def build_player_match() -> pd.DataFrame:
    ensure_directories()
    if not (read_path := FEATURE_DIR.parent / "clean" / "match_lineups.csv").exists():
        clean_all()

    lineups = read_clean("match_lineups.csv")
    events = read_clean("match_events.csv")
    matches = read_clean("matches.csv")
    stages = read_clean("tournament_stages.csv")
    players = read_clean("squads_and_players.csv")

    require_columns(lineups, {"match_id", "player_id", "team_id", "is_starting_xi", "minutes_played"}, "match_lineups.csv")
    require_columns(events, {"match_id", "player_id", "event_type"}, "match_events.csv")

    event_counts = (
        events.assign(value=1)
        .pivot_table(index=["match_id", "player_id"], columns="event_type", values="value", aggfunc="sum", fill_value=0)
        .rename(columns=EVENT_COLUMNS)
        .reset_index()
    )
    for column in EVENT_COLUMNS.values():
        if column not in event_counts.columns:
            event_counts[column] = 0

    frame = lineups.merge(event_counts, on=["match_id", "player_id"], how="left")
    frame[list(EVENT_COLUMNS.values())] = frame[list(EVENT_COLUMNS.values())].fillna(0).astype(int)
    frame = frame.merge(
        matches[[
            "match_id", "date", "kickoff_time_utc", "stage_id", "venue_id", "home_team_id",
            "away_team_id", "home_score", "away_score", "home_xg", "away_xg", "result_type"
        ]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(stages[["stage_id", "stage_name", "is_knockout"]], on="stage_id", how="left", validate="many_to_one")
    frame = frame.merge(
        players[["player_id", "player_name", "position", "market_value_eur", "caps", "date_of_birth", "height_cm"]],
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    frame["is_home"] = (frame["team_id"] == frame["home_team_id"]).astype(int)
    frame["opponent_team_id"] = np.where(frame["is_home"].eq(1), frame["away_team_id"], frame["home_team_id"])
    frame["team_goals"] = np.where(frame["is_home"].eq(1), frame["home_score"], frame["away_score"])
    frame["opponent_goals"] = np.where(frame["is_home"].eq(1), frame["away_score"], frame["home_score"])
    frame["team_xg"] = np.where(frame["is_home"].eq(1), frame["home_xg"], frame["away_xg"])
    frame["opponent_xg"] = np.where(frame["is_home"].eq(1), frame["away_xg"], frame["home_xg"])
    frame["team_result"] = np.select(
        [frame["team_goals"] > frame["opponent_goals"], frame["team_goals"] == frame["opponent_goals"]],
        ["W", "D"],
        default="L",
    )
    frame["appeared"] = (frame["minutes_played"] > 0).astype(int)
    frame["scored_goal"] = (frame["goals"] > 0).astype(int)
    frame["match_datetime"] = pd.to_datetime(frame["date"] + " " + frame["kickoff_time_utc"], errors="raise")
    frame = frame.sort_values(["match_datetime", "match_id", "team_id", "player_id"]).reset_index(drop=True)

    if frame.duplicated(["match_id", "player_id"]).any():
        raise ValueError("player_match_performance contains duplicate (match_id, player_id)")
    invalid_teams = frame["team_id"].ne(frame["home_team_id"]) & frame["team_id"].ne(frame["away_team_id"])
    if invalid_teams.any():
        raise ValueError("A player is assigned to a team outside the match")

    write_csv(frame, FEATURE_DIR / "player_match_performance.csv")
    return frame


if __name__ == "__main__":
    build_player_match()
