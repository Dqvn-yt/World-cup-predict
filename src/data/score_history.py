"""Validated result-only history; never manufacture historical ML features."""

import pandas as pd

from config import CLEAN_DIR, SOURCE_DIR, WORLD_CUP_DIR


RESULT_COLUMNS = ["match_id", "match_datetime", "home_team_id", "away_team_id", "home_score", "away_score", "neutral"]


def neutral_from_hosts(home_is_host: pd.Series, away_is_host: pd.Series) -> pd.Series:
    """Neutral venue when neither side is a tournament host.

    World Cup venues sit in host countries, so only host-involved fixtures
    keep a genuine home advantage. Everything else is neutral.
    """
    return ~(home_is_host.fillna(0).astype(bool) | away_is_host.fillna(0).astype(bool))


def load_score_history() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = pd.read_csv(SOURCE_DIR / "teams.csv")
    clean = pd.read_csv(CLEAN_DIR / "teams.csv")
    mapping = pd.read_csv(WORLD_CUP_DIR / "team_mapping.csv")
    registry = pd.read_csv(WORLD_CUP_DIR / "historical_team_registry.csv")
    for table, key in [(source, "fifa_code"), (clean, "fifa_code"), (mapping, "fifa_code"), (registry, "fifa_code")]:
        if table[key].isna().any() or table[key].duplicated().any() or table.team_id.isna().any() or table.team_id.duplicated().any():
            raise ValueError("Invalid team identity")
    ids = source.set_index("fifa_code").team_id.to_dict()
    if ids != clean.set_index("fifa_code").team_id.to_dict():
        raise ValueError("Source/clean team identities differ")
    for row in registry.itertuples():
        if row.fifa_code in ids or row.team_id in ids.values():
            raise ValueError("Historical registry collision")
        ids[row.fifa_code] = row.team_id
    for row in mapping.itertuples():
        if ids.get(row.fifa_code) != row.team_id:
            raise ValueError("Historical mapping mismatch")
        current_id = source.set_index("fifa_code").team_id.get(row.fifa_code)
        if (current_id is None and pd.notna(row.team_id_2026)) or (current_id is not None and row.team_id_2026 != current_id):
            raise ValueError("Current mapping mismatch")
    old = pd.read_csv(WORLD_CUP_DIR / "worldcup_history_matches.csv")
    for side in (1, 2):
        if not old[f"team{side}_code"].map(ids).eq(old[f"team{side}_id"]).all():
            raise ValueError("Historical fixture identity mismatch")
    current = pd.read_csv(CLEAN_DIR / "matches.csv")
    original = pd.read_csv(SOURCE_DIR / "matches.csv")
    keys = ["match_id", "home_team_id", "away_team_id", "home_score", "away_score", "date", "kickoff_time_utc"]
    pd.testing.assert_frame_equal(current[keys].sort_values("match_id").reset_index(drop=True), original[keys].sort_values("match_id").reset_index(drop=True), check_dtype=False)
    for directory in (SOURCE_DIR, CLEAN_DIR):
        features = pd.read_csv(directory / "match_prediction_features_X.csv")
        feature_keys = ["match_id", "home_team_id", "away_team_id", "date", "kickoff_time_utc"]
        pd.testing.assert_frame_equal(features[feature_keys].sort_values("match_id").reset_index(drop=True), current[feature_keys].sort_values("match_id").reset_index(drop=True), check_dtype=False)
        for side in ("home", "away"):
            if not features[f"{side}_fifa_code"].map(ids).eq(features[f"{side}_team_id"]).all():
                raise ValueError("Feature team FIFA code mismatch")
        targets = pd.read_csv(directory / "match_prediction_targets_y.csv")
        target_keys = ["match_id", "home_score", "away_score"]
        pd.testing.assert_frame_equal(targets[target_keys].sort_values("match_id").reset_index(drop=True), current[target_keys].sort_values("match_id").reset_index(drop=True), check_dtype=False)
    clean_features = pd.read_csv(CLEAN_DIR / "match_prediction_features_X.csv")
    current["neutral"] = current.match_id.map(
        neutral_from_hosts(clean_features.set_index("match_id")["home_is_host"],
                           clean_features.set_index("match_id")["away_is_host"])
    ).fillna(True).astype(bool)
    current_ids = set(source.team_id)
    if not set(current.home_team_id).union(current.away_team_id) <= current_ids:
        raise ValueError("Unknown current fixture team")
    if old.match_id.duplicated().any() or current.match_id.duplicated().any() or set(old.match_id) & set(current.match_id):
        raise ValueError("Match identity collision")
    events = pd.read_csv(CLEAN_DIR / "match_events.csv")
    goals = events[events.event_type.isin(["Goal", "Own Goal"])].merge(
        current[["match_id", "home_team_id", "away_team_id"]], on="match_id", validate="many_to_one")
    if not (goals.team_id.eq(goals.home_team_id) | goals.team_id.eq(goals.away_team_id)).all():
        raise ValueError("Unknown scoring team")
    goals["credited_team_id"] = goals.team_id
    own = goals.event_type.eq("Own Goal")
    goals.loc[own, "credited_team_id"] = goals.loc[own].apply(
        lambda row: row.away_team_id if row.team_id == row.home_team_id else row.home_team_id, axis=1)
    totals = goals.groupby(["match_id", "credited_team_id"]).size()
    regulation = goals[pd.to_numeric(goals.minute, errors="raise") <= 90].groupby(["match_id", "credited_team_id"]).size()
    changed = []
    for index, row in current.iterrows():
        for side in ("home", "away"):
            key = (row.match_id, row[f"{side}_team_id"])
            if totals.get(key, 0) != row[f"{side}_score"]:
                raise ValueError(f"Goal events do not reconcile: {key}")
        if row.result_type in ("AET", "Penalties"):
            scores = [int(regulation.get((row.match_id, row[f"{side}_team_id"]), 0)) for side in ("home", "away")]
            if scores[0] != scores[1]:
                raise ValueError(f"Ambiguous regulation score: {row.match_id}")
            if scores != [row.home_score, row.away_score]:
                changed.append(int(row.match_id))
            current.loc[index, ["home_score", "away_score"]] = scores
    current["match_datetime"] = pd.to_datetime(current.date + " " + current.kickoff_time_utc)
    history = old.rename(columns={"team1_id": "home_team_id", "team2_id": "away_team_id", "team1_ft": "home_score", "team2_ft": "away_score"}).copy()
    # Kickoff timezone is unverified: make results available only on the next day.
    history["match_datetime"] = pd.to_datetime(history.date) + pd.Timedelta(days=1)
    # Missing archive flags default to neutral: never invent a home advantage.
    history["neutral"] = old["neutral"].fillna(1).astype(bool) if "neutral" in old.columns else True
    history = history[RESULT_COLUMNS]
    if history[RESULT_COLUMNS].isna().any().any() or (history[["home_score", "away_score"]] < 0).any().any():
        raise ValueError("Invalid historical results")
    audit = {"current_teams": len(source), "shared_teams": int(mapping.team_id_2026.notna().sum()),
             "historical_only_teams": len(registry), "historical_teams": len(mapping),
             "historical_matches": len(history), "current_matches": len(current), "id_collisions": 0,
             "neutral_history_matches": int(history["neutral"].sum()),
             "neutral_current_matches": int(current["neutral"].sum()),
             "regulation_adjusted_match_ids": changed, "score_target": "90 minutes including regulation stoppage time; no ET or shootouts"}
    return history, current, audit
