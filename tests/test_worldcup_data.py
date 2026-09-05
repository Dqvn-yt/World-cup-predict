import json

import pandas as pd

from config import ROOT_DIR, SOURCE_DIR

WORLD_CUP_DIR = ROOT_DIR / "processed data" / "worldcup"


def test_worldcup_history_files_exist():
    for filename in [
        "worldcup_2018_matches.csv",
        "worldcup_2022_matches.csv",
        "worldcup_history_matches.csv",
        "worldcup_goalscorers.csv",
        "team_mapping.csv",
        "worldcup_audit.json",
    ]:
        assert (WORLD_CUP_DIR / filename).exists(), filename


def test_worldcup_match_counts_and_champions():
    frames = {
        2018: pd.read_csv(WORLD_CUP_DIR / "worldcup_2018_matches.csv"),
        2022: pd.read_csv(WORLD_CUP_DIR / "worldcup_2022_matches.csv"),
    }
    for year, frame in frames.items():
        assert len(frame) == 64
        assert frame[["team1_ft", "team2_ft"]].notna().all().all()
        assert (frame[["team1_ft", "team2_ft"]] >= 0).all().all()
        assert (frame["stage"] == "Final").sum() == 1
    final_2018 = frames[2018][frames[2018]["stage"].eq("Final")].iloc[0]
    assert {final_2018["team1"], final_2018["team2"]} == {"France", "Croatia"}
    assert (final_2018["team1_ft"], final_2018["team2_ft"]) == (4, 2)
    final_2022 = frames[2022][frames[2022]["stage"].eq("Final")].iloc[0]
    assert {final_2022["team1"], final_2022["team2"]} == {"Argentina", "France"}
    assert int(final_2022["went_extra_time"]) == 1
    assert (int(final_2022["team1_et"]), int(final_2022["team2_et"])) == (3, 3)
    assert (int(final_2022["team1_pens"]), int(final_2022["team2_pens"])) == (4, 2)


def test_worldcup_team_mapping_coverage():
    mapping = pd.read_csv(WORLD_CUP_DIR / "team_mapping.csv")
    assert mapping["team_id_2026"].notna().sum() == 30
    history = pd.read_csv(WORLD_CUP_DIR / "worldcup_history_matches.csv")
    assert len(history) == 128
    assert history[["team1_id", "team2_id"]].notna().all(axis=1).sum() == 128
    assert mapping["team_id"].nunique() == 40
    current = pd.read_csv(SOURCE_DIR / "teams.csv")
    common = mapping.merge(current, on="fifa_code", suffixes=("_history", "_current"))
    assert (common["team_id_history"] == common["team_id_current"]).all()
    registry = pd.read_csv(WORLD_CUP_DIR / "historical_team_registry.csv")
    assert len(registry) == registry["team_id"].nunique() == 10
    assert not set(registry["team_id"]) & set(current["team_id"])
    assert history["match_id"].is_unique
    assert not set(history["match_id"]) & set(pd.read_csv(SOURCE_DIR / "matches.csv")["match_id"])


def test_worldcup_audit_is_consistent():
    audit = json.loads((WORLD_CUP_DIR / "worldcup_audit.json").read_text(encoding="utf-8"))
    assert audit["history_total_matches"] == 128
    for year in ["2018", "2022"]:
        checks = audit["tournaments"][year]
        assert checks["stage_counts_ok"] is True
        assert checks["champion_ok"] is True
        assert checks["missing_ft_scores"] == 0
        assert checks["scorer_list_mismatches"] == []
        assert checks["credited_team_score_mismatches"] == []


def test_score_time_and_player_semantics():
    history = pd.read_csv(WORLD_CUP_DIR / "worldcup_history_matches.csv")
    goals = pd.read_csv(WORLD_CUP_DIR / "worldcup_goalscorers.csv")
    assert history["kickoff_utc"].isna().all()
    assert goals["player_id"].isna().all()
    assert goals["goal_event_id"].is_unique
    own = goals["own_goal"].eq(1)
    assert own.sum() == 14
    assert (goals.loc[own, "credited_team_id"] != goals.loc[own, "scorer_team_id"]).all()
    assert (goals.loc[~own, "credited_team_id"] == goals.loc[~own, "scorer_team_id"]).all()
    assert goals.groupby("tournament").size().to_dict() == {2018: 169, 2022: 172}
    final = history[(history["tournament"] == 2022) & (history["stage"] == "Final")].iloc[0]
    assert (final.team1_ft, final.team2_ft) == (2, 2)
    assert (final.team1_extra_time_goals, final.team2_extra_time_goals) == (1, 1)


def test_normalization_is_repeatable(tmp_path, monkeypatch):
    import shutil
    from src.data import collect_worldcup as collector

    shutil.copytree(collector.RAW_DIR, tmp_path / "raw")
    monkeypatch.setattr(collector, "WORLD_CUP_DIR", tmp_path)
    monkeypatch.setattr(collector, "RAW_DIR", tmp_path / "raw")
    collector.collect_worldcup(download=False)
    first = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
    collector.collect_worldcup(download=False)
    assert first == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
