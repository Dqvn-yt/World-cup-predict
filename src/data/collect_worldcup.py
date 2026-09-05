import json
import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import WORLD_CUP_DIR, SOURCE_DIR
from src.data.loaders import write_csv

RAW_DIR = WORLD_CUP_DIR / "raw"

SOURCES = {
    2018: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
}

NAME_TO_FIFA = {
    "Argentina": "ARG", "Australia": "AUS", "Belgium": "BEL", "Brazil": "BRA",
    "Cameroon": "CMR", "Canada": "CAN", "Colombia": "COL", "Costa Rica": "CRC",
    "Croatia": "CRO", "Denmark": "DEN", "Ecuador": "ECU", "Egypt": "EGY",
    "England": "ENG", "France": "FRA", "Germany": "GER", "Ghana": "GHA",
    "Iceland": "ISL", "Iran": "IRN", "Japan": "JPN", "Mexico": "MEX",
    "Morocco": "MAR", "Netherlands": "NED", "Nigeria": "NGA", "Panama": "PAN",
    "Peru": "PER", "Poland": "POL", "Portugal": "POR", "Qatar": "QAT",
    "Russia": "RUS", "Saudi Arabia": "KSA", "Senegal": "SEN", "Serbia": "SRB",
    "South Korea": "KOR", "Spain": "ESP", "Sweden": "SWE", "Switzerland": "SUI",
    "Tunisia": "TUN", "USA": "USA", "Uruguay": "URU", "Wales": "WAL",
}

EXPECTED_CHAMPIONS = {2018: ("France", "Croatia", 4, 2), 2022: ("Argentina", "France", 3, 3)}
EXPECTED_COUNTS = {"Matchday": 48, "Round of 16": 8, "Quarter-finals": 4, "Semi-finals": 2, "Match for third place": 1, "Final": 1}


def _stage_group(round_name: str) -> str:
    if round_name.startswith("Matchday"):
        return "Matchday"
    return round_name


def _download(year: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"{year}_worldcup.json"
    if not target.exists():
        request = urllib.request.Request(SOURCES[year], headers={"User-Agent": "football-analytics-ai/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    return target


def _team_ids() -> dict[str, int]:
    teams = pd.read_csv(SOURCE_DIR / "teams.csv")
    if teams["fifa_code"].duplicated().any() or teams["team_id"].duplicated().any():
        raise ValueError("Duplicate current team identity")
    ids = dict(zip(teams["fifa_code"], teams["team_id"].astype(int)))
    registry_path = WORLD_CUP_DIR / "historical_team_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame(columns=["fifa_code", "team_id"])
    if registry["fifa_code"].duplicated().any() or registry["team_id"].duplicated().any():
        raise ValueError("Duplicate historical identity")
    for code, team_id in zip(registry["fifa_code"], registry["team_id"]):
        team_id = int(team_id)
        if (code in ids and ids[code] != team_id) or (team_id in ids.values() and ids.get(code) != team_id):
            raise ValueError(f"Historical identity collision: {code}")
        if code in ids and code not in set(teams["fifa_code"]):
            raise ValueError(f"Duplicate historical identity: {code}")
        ids[code] = team_id
    for code in sorted(set(NAME_TO_FIFA.values()) - ids.keys()):
        ids[code] = max(10000, max(ids.values()) + 1)
    write_csv(pd.DataFrame([
        {"fifa_code": code, "team_id": ids[code]}
        for code in sorted(set(registry["fifa_code"]) | (set(NAME_TO_FIFA.values()) - set(teams["fifa_code"])))
    ]), registry_path)
    return ids


def _parse_match(year: int, match_no: int, match: dict, team_ids: dict[str, int]) -> tuple[dict, list[dict]]:
    raw_score = match.get("score", {})
    score = {"ft": raw_score} if isinstance(raw_score, list) else raw_score
    team1, team2 = match["team1"], match["team2"]
    code1, code2 = NAME_TO_FIFA[team1], NAME_TO_FIFA[team2]
    ft = score["ft"]
    et = score.get("et")
    pens = score.get("p")
    final1, final2 = (et if et else ft)
    for pair in [ft, score.get("ht"), et, pens]:
        if pair is not None and (len(pair) != 2 or any(type(value) is not int or value < 0 for value in pair)):
            raise ValueError(f"Invalid score: {year}/{match_no}")
    if et is not None and (ft[0] != ft[1] or any(et[i] < ft[i] for i in range(2))):
        raise ValueError(f"Invalid extra-time score: {year}/{match_no}")
    if pens is not None and (final1 != final2 or pens[0] == pens[1]):
        raise ValueError(f"Invalid shootout: {year}/{match_no}")
    if pens:
        winner = team1 if pens[0] > pens[1] else team2
    elif final1 != final2:
        winner = team1 if final1 > final2 else team2
    else:
        winner = "Draw"
    row = {
        "match_id": year * 1000 + match_no,
        "tournament": year,
        "match_no": match_no,
        "date": match["date"],
        "kickoff_source_time": match.get("time", ""),
        "kickoff_timezone": "unverified source local time",
        "kickoff_utc": "",
        "source_url": SOURCES[year],
        "stage": match.get("round", ""),
        "group_name": match.get("group", ""),
        "team1": team1,
        "team2": team2,
        "team1_code": code1,
        "team2_code": code2,
        "team1_id": team_ids.get(code1),
        "team2_id": team_ids.get(code2),
        "neutral": int({2018: "Russia", 2022: "Qatar"}[year] not in {team1, team2}),
        "team1_ht": score.get("ht", ["", ""])[0],
        "team2_ht": score.get("ht", ["", ""])[1],
        "team1_ft": ft[0],
        "team2_ft": ft[1],
        "went_extra_time": int(et is not None),
        "team1_et": et[0] if et else "",
        "team2_et": et[1] if et else "",
        "team1_extra_time_goals": et[0] - ft[0] if et else "",
        "team2_extra_time_goals": et[1] - ft[1] if et else "",
        "team1_final": final1,
        "team2_final": final2,
        "team1_pens": pens[0] if pens else "",
        "team2_pens": pens[1] if pens else "",
        "winner": winner,
        "venue": match.get("ground", ""),
    }
    scorers = []
    for team, goals in [(team1, match.get("goals1", [])), (team2, match.get("goals2", []))]:
        for goal in goals:
            scorers.append({
                "match_id": row["match_id"],
                "goal_event_id": f"wc{year}-{match_no}-{len(scorers) + 1}",
                "credited_team_id": team_ids[NAME_TO_FIFA[team]],
                "scorer_team_id": team_ids[NAME_TO_FIFA[(team2 if team == team1 else team1) if goal.get("owngoal") else team]],
                "player_id": None,
                "player_identity_status": "unresolved_source_name",
                "source_url": SOURCES[year],
                "tournament": year,
                "match_no": match_no,
                "date": match["date"],
                "team": team,
                "scorer": goal.get("name", ""),
                "minute": str(goal.get("minute", "")),
                "penalty": int(bool(goal.get("penalty"))),
                "own_goal": int(bool(goal.get("owngoal"))),
            })
    return row, scorers


def _verify(year: int, matches: pd.DataFrame, scorers: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["total_matches"] = int(len(matches))
    stage_counts = matches["stage"].map(_stage_group).value_counts().to_dict()
    checks["stage_counts"] = {str(stage): int(count) for stage, count in stage_counts.items()}
    checks["stage_counts_ok"] = all(stage_counts.get(stage) == count for stage, count in EXPECTED_COUNTS.items())
    final = matches[matches["stage"].eq("Final")].iloc[0]
    champion, runner_up, goals_for, goals_against = EXPECTED_CHAMPIONS[year]
    if int(final["went_extra_time"]):
        final_home, final_away = int(final["team1_et"]), int(final["team2_et"])
    else:
        final_home, final_away = int(final["team1_ft"]), int(final["team2_ft"])
    checks["final"] = f"{final['team1']} {final['team1_ft']}-{final['team2_ft']} {final['team2']}"
    checks["champion_ok"] = bool(
        ((final["team1"] == champion) and (final["team2"] == runner_up) and (final_home == goals_for) and (final_away == goals_against))
        or ((final["team1"] == runner_up) and (final["team2"] == champion) and (final_home == goals_against) and (final_away == goals_for))
    )
    checks["champion_ok"] = checks["champion_ok"] and final["winner"] == champion
    checks["missing_ft_scores"] = int(matches[["team1_ft", "team2_ft"]].isna().any(axis=1).sum())
    checks["negative_scores"] = int(((matches["team1_ft"] < 0) | (matches["team2_ft"] < 0)).sum())
    checks["matches_with_both_teams_mapped"] = int(matches[["team1_id", "team2_id"]].notna().all(axis=1).sum())
    listed = scorers.groupby("match_no").size()
    expected = matches.set_index("match_no").apply(
        lambda row: (row["team1_et"] if row["went_extra_time"] else row["team1_ft"])
        + (row["team2_et"] if row["went_extra_time"] else row["team2_ft"]),
        axis=1,
    )
    mismatched = [int(match_no) for match_no in matches["match_no"] if listed.get(match_no, 0) != expected.get(match_no)]
    checks["scorer_list_mismatches"] = mismatched
    checks["credited_team_score_mismatches"] = [
        int(row.match_no) for row in matches.itertuples()
        if any(len(scorers[(scorers["match_no"] == row.match_no) & (scorers["credited_team_id"] == team_id)]) != goals
               for team_id, goals in [(row.team1_id, row.team1_final), (row.team2_id, row.team2_final)])
    ]
    return checks


def collect_worldcup(download: bool = True) -> dict:
    WORLD_CUP_DIR.mkdir(parents=True, exist_ok=True)
    team_ids = _team_ids()
    all_matches: list[dict] = []
    all_scorers: list[dict] = []
    audit: dict[str, object] = {
        "sources": SOURCES,
        "license": "Open data, public domain (openfootball worldcup.json)",
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieved_at_utc": None,
        "retrieval_note": "Raw cache retained; original retrieval timestamp unknown. Hashes identify exact input bytes.",
        "raw_files": {},
        "training_integration": "Result-only 90-minute Dixon-Coles training and chronological OOF/cold-start priors via score_history; no historical ML/player features",
        "tournaments": {},
    }
    for year in (2018, 2022):
        raw_path = _download(year) if download else RAW_DIR / f"{year}_worldcup.json"
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        audit["raw_files"][str(year)] = {"path": str(raw_path.relative_to(WORLD_CUP_DIR)), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
        rows, scorers = [], []
        for match_no, match in enumerate(data["matches"], start=1):
            row, match_scorers = _parse_match(year, match_no, match, team_ids)
            rows.append(row)
            scorers.extend(match_scorers)
        matches = pd.DataFrame(rows).sort_values(["date", "match_no"]).reset_index(drop=True)
        scorers_frame = pd.DataFrame(scorers)
        all_matches.append(matches)
        all_scorers.append(scorers_frame)
        audit["tournaments"][str(year)] = _verify(year, matches, scorers_frame)
    history = pd.concat(all_matches, ignore_index=True)
    existing_ids = set(pd.read_csv(SOURCE_DIR / "matches.csv")["match_id"])
    if history["match_id"].duplicated().any() or existing_ids & set(history["match_id"]):
        raise ValueError("Historical match ID collision")
    for year, checks in audit["tournaments"].items():
        if (checks["total_matches"] != 64 or not checks["stage_counts_ok"] or not checks["champion_ok"]
                or checks["scorer_list_mismatches"] or checks["credited_team_score_mismatches"]
                or checks["missing_ft_scores"] or checks["negative_scores"]):
            raise ValueError(f"Historical audit failed: {year}: {checks}")
        write_csv(history[history["tournament"].eq(int(year))], WORLD_CUP_DIR / f"worldcup_{year}_matches.csv")
    write_csv(history, WORLD_CUP_DIR / "worldcup_history_matches.csv")
    write_csv(pd.concat(all_scorers, ignore_index=True), WORLD_CUP_DIR / "worldcup_goalscorers.csv")
    current_ids = pd.read_csv(SOURCE_DIR / "teams.csv").set_index("fifa_code")["team_id"].to_dict()
    mapping = pd.DataFrame([
        {"team_name": name, "fifa_code": NAME_TO_FIFA[name], "team_id": team_ids[NAME_TO_FIFA[name]], "team_id_2026": current_ids.get(NAME_TO_FIFA[name])}
        for name in sorted(NAME_TO_FIFA)
    ])
    write_csv(mapping, WORLD_CUP_DIR / "team_mapping.csv")
    audit["mapped_teams"] = int(mapping["team_id_2026"].notna().sum())
    audit["historical_identity_coverage"] = int(mapping["team_id"].notna().sum())
    audit["new_historical_ids"] = int(mapping["team_id_2026"].isna().sum())
    audit["unmapped_teams"] = mapping[mapping["team_id_2026"].isna()]["team_name"].tolist()
    audit["history_matches_both_mapped"] = int(history[["team1_id", "team2_id"]].notna().all(axis=1).sum())
    audit["history_total_matches"] = int(len(history))
    audit["unique_match_ids"] = int(history["match_id"].nunique())
    audit["existing_match_id_collisions"] = 0
    audit["goal_events"] = int(sum(len(frame) for frame in all_scorers))
    audit["unresolved_player_events"] = audit["goal_events"]
    (WORLD_CUP_DIR / "worldcup_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


if __name__ == "__main__":
    result = collect_worldcup()
    print(json.dumps(result, indent=2, ensure_ascii=False))
