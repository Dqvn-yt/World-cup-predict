from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from config import FEATURE_DIR, ensure_directories
from src.data.build_player_match import build_player_match
from src.data.loaders import normalize_text, read_clean, write_csv


POSITION_WEIGHTS = {
    "FWD": {
        "goals_per90": 0.35,
        "assists_per90": 0.20,
        "shots_on_target_per90": 0.25,
        "shots_on_target_pct": 0.10,
        "goals_per_shot": 0.10,
    },
    "MID": {
        "goals_per90": 0.15,
        "assists_per90": 0.30,
        "crosses_per90": 0.20,
        "interceptions_per90": 0.15,
        "tackles_won_per90": 0.10,
        "fouled_per90": 0.10,
    },
    "DEF": {
        "interceptions_per90": 0.30,
        "tackles_won_per90": 0.30,
        "clean_sheet_rate": 0.25,
        "discipline": 0.10,
        "goal_contribution_per90": 0.05,
    },
    "GK": {
        "save_pct": 0.35,
        "saves_per90": 0.20,
        "clean_sheet_pct": 0.20,
        "negative_goals_against_per90": 0.15,
        "penalty_save_pct": 0.10,
    },
}

FORM_COMPONENT_WEIGHTS = {
    "position_score": 0.45,
    "recent_form_score": 0.35,
    "availability_score": 0.10,
    "team_context_score": 0.10,
}

PROVISIONAL_MINUTES = 90
RELIABLE_MINUTES = 180
HIGH_CONFIDENCE_MINUTES = 360
PSEUDO_NINETIES = 2.0
CONFIDENCE_FLOOR = 0.35

RATE_METRICS = {
    "goals_per90": "match_nineties",
    "assists_per90": "match_nineties",
    "goal_contribution_per90": "match_nineties",
    "shots_on_target_per90": "match_nineties",
    "shots_on_target_pct": "match_nineties",
    "goals_per_shot": "match_nineties",
    "crosses_per90": "misc_nineties",
    "interceptions_per90": "misc_nineties",
    "tackles_won_per90": "misc_nineties",
    "fouled_per90": "misc_nineties",
    "save_pct": "gk_nineties",
    "saves_per90": "gk_nineties",
    "clean_sheet_pct": "gk_nineties",
    "penalty_save_pct": "gk_nineties",
    "negative_goals_against_per90": "gk_nineties",
    "clean_sheet_rate": "match_nineties",
}

TEAM_ALIASES = {
    "bosniaherz": "bosniaandherzegovina",
    "korearepublic": "southkorea",
}


def _tokens(value: object) -> set[str]:
    text = "".join(char if char.isalnum() else " " for char in str(value).lower())
    return {normalize_text(token) for token in text.split() if token}


def _attach_player_ids(raw: pd.DataFrame, profile: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["team_name_key"] = raw["raw_squad"].str.replace(r"^[a-z]{2,3}\s+", "", regex=True).map(normalize_text).replace(TEAM_ALIASES)
    roster = profile[["player_id", "player_name", "team_id", "position"]].merge(
        teams[["team_id", "team_name"]], on="team_id", how="left", validate="many_to_one"
    )
    roster["team_name_key"] = roster["team_name"].map(normalize_text)
    roster["name_key"] = roster["player_name"].map(normalize_text)
    assignments: dict[int, int] = {}
    for team_key, raw_team in raw.groupby("team_name_key"):
        candidates = roster[roster["team_name_key"].eq(team_key)]
        if candidates.empty:
            continue
        used_players: set[int] = set()
        scored_pairs: list[tuple[float, int, int]] = []
        for raw_index, raw_player in raw_team.iterrows():
            raw_tokens = _tokens(raw_player["raw_player_name"])
            raw_position = str(raw_player.get("raw_position", ""))
            for candidate in candidates.itertuples():
                candidate_tokens = _tokens(candidate.player_name)
                coverage = len(raw_tokens & candidate_tokens) / max(len(raw_tokens), 1)
                sequence = SequenceMatcher(None, raw_player["player_name_key"], candidate.name_key).ratio()
                position_match = (
                    (candidate.position == "FWD" and "FW" in raw_position)
                    or (candidate.position == "MID" and "MF" in raw_position)
                    or (candidate.position == "DEF" and "DF" in raw_position)
                    or (candidate.position == "GK" and "GK" in raw_position)
                )
                score = 0.6 * coverage + 0.4 * sequence + (0.08 if position_match else 0.0)
                scored_pairs.append((score, int(raw_index), int(candidate.player_id)))
        used_raw: set[int] = set()
        for score, raw_index, player_id in sorted(scored_pairs, reverse=True):
            if score < 0.58 or raw_index in used_raw or player_id in used_players:
                continue
            assignments[raw_index] = player_id
            used_raw.add(raw_index)
            used_players.add(player_id)
    raw["player_id"] = raw.index.to_series().map(assignments).astype("Int64")
    return raw.dropna(subset=["player_id"]).assign(player_id=lambda value: value["player_id"].astype(int))


def _safe_per90(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    return np.where(minutes.gt(0), numerator * 90 / minutes, 0.0)


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    std = numeric.std(ddof=0)
    return (numeric - numeric.mean()) / std if std > 0 else pd.Series(0.0, index=series.index)


def _shrink_rate(values: pd.Series, sample: pd.Series, prior: float) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").fillna(prior)
    size = pd.to_numeric(sample, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (raw * size + prior * PSEUDO_NINETIES) / (size + PSEUDO_NINETIES)


def _rate_prior(frame: pd.DataFrame, metric: str, reliable: pd.Series) -> float:
    prior = pd.to_numeric(frame.loc[reliable, metric], errors="coerce").mean()
    if pd.isna(prior):
        prior = pd.to_numeric(frame[metric], errors="coerce").mean()
    return 0.0 if pd.isna(prior) else float(prior)


def _recent_form_summary(predictive: pd.DataFrame) -> pd.DataFrame:
    appeared = predictive[predictive["appeared"].eq(1)].sort_values(["match_datetime", "match_id", "player_id"])
    summary = appeared.groupby("player_id").agg(
        recent_form_score=("predictive_form", "last"),
        form_delta=("form_delta", "last"),
        appearances=("appeared", "sum"),
    )
    return summary


def _team_context(player_match: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    appeared = player_match[player_match["appeared"].eq(1)].copy()
    points = appeared["team_result"].map({"W": 3.0, "D": 1.0, "L": 0.0})
    ppg = appeared.assign(points=points).groupby("player_id")["points"].mean() / 3.0
    team_contribution = appeared.groupby("team_id")[["goals", "assists"]].sum().sum(axis=1)
    player_contribution = appeared.groupby("player_id")[["goals", "assists"]].sum().sum(axis=1)
    team_of_player = appeared.groupby("player_id")["team_id"].first()
    share = (player_contribution / team_of_player.map(team_contribution)).clip(0, 1).fillna(0.0)
    ppg = ppg.reindex(players["player_id"]).fillna(0.5).to_numpy()
    share = share.reindex(players["player_id"]).fillna(0.0).to_numpy()
    context = 10.0 * (0.6 * ppg + 0.4 * share)
    return pd.Series(context, index=players.index), pd.Series(share, index=players.index)


def build_overall_form(
    player_match: pd.DataFrame | None = None,
    predictive: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ensure_directories()
    if player_match is None:
        player_match = _load_or_build_player_match()
    if predictive is None:
        predictive = _load_or_build_predictive(player_match)
    stats = read_clean("player_stats.csv")
    profile = read_clean("squads_and_players.csv")
    teams = read_clean("teams.csv")
    shoot = _attach_player_ids(read_clean("shoot.csv"), profile, teams)
    misc = _attach_player_ids(read_clean("Miscellaneous.csv"), profile, teams)
    gk = _attach_player_ids(read_clean("gk.csv"), profile, teams)

    frame = stats.merge(
        profile[["player_id", "club_team", "market_value_eur", "caps", "date_of_birth", "height_cm"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    shoot_columns = ["player_id", "shots_on_target_pct", "shots_per90", "shots_on_target_per90", "goals_per_shot"]
    misc_columns = ["player_id", "nineties", "crosses", "interceptions", "tackles_won", "fouled"]
    gk_columns = ["player_id", "save_pct", "goals_against_per90", "saves", "nineties", "clean_sheet_pct", "penalty_save_pct"]
    frame = frame.merge(shoot[shoot_columns], on="player_id", how="left", validate="one_to_one")
    frame = frame.merge(misc[misc_columns].rename(columns={"nineties": "misc_nineties"}), on="player_id", how="left", validate="one_to_one")
    frame = frame.merge(gk[gk_columns].rename(columns={"nineties": "gk_nineties", "saves": "raw_saves"}), on="player_id", how="left", validate="one_to_one")

    minutes = frame["minutes_played"].clip(lower=0)
    frame["match_nineties"] = minutes / 90
    frame["goals_per90"] = _safe_per90(frame["goals"], minutes)
    frame["assists_per90"] = _safe_per90(frame["assists"], minutes)
    frame["goal_contribution_per90"] = _safe_per90(frame["goals"] + frame["assists"], minutes)
    denominator = frame["misc_nineties"].replace(0, np.nan)
    for source, target in [
        ("crosses", "crosses_per90"),
        ("interceptions", "interceptions_per90"),
        ("tackles_won", "tackles_won_per90"),
        ("fouled", "fouled_per90"),
    ]:
        frame[target] = frame[source] / denominator
    frame["saves_per90"] = frame["raw_saves"] / frame["gk_nineties"].replace(0, np.nan)
    frame["negative_goals_against_per90"] = -frame["goals_against_per90"]
    frame["discipline"] = -(frame["yellow_cards"] + 3 * frame["red_cards"])

    appearance = player_match[player_match["appeared"].eq(1)]
    clean_sheet = appearance.assign(clean_sheet=appearance["opponent_goals"].eq(0).astype(int)).groupby("player_id")["clean_sheet"].mean()
    frame["clean_sheet_rate"] = frame["player_id"].map(clean_sheet).fillna(0.0)
    frame["clean_sheet_pct"] = frame["clean_sheet_pct"].fillna(frame["clean_sheet_rate"] * 100)
    goals_conceded = frame["goals_against_per90"].fillna(0) * frame["gk_nineties"].fillna(0)
    save_faced = frame["saves"].fillna(0) + goals_conceded
    frame["save_pct_fallback"] = pd.Series(
        np.where(save_faced.gt(0), frame["saves"].fillna(0) / save_faced * 100, np.nan),
        index=frame.index,
    )
    frame["save_pct"] = frame["save_pct"].fillna(frame["save_pct_fallback"])

    frame["weighted_z"] = 0.0
    for position, weights in POSITION_WEIGHTS.items():
        mask = frame["position"].eq(position)
        reliable = mask & frame["minutes_played"].ge(RELIABLE_MINUTES)
        score = pd.Series(0.0, index=frame.index[mask])
        for metric, weight in weights.items():
            if metric in RATE_METRICS:
                prior = _rate_prior(frame.loc[mask], metric, reliable.loc[mask])
                metric_values = _shrink_rate(
                    frame.loc[mask, metric],
                    frame.loc[mask, RATE_METRICS[metric]],
                    prior,
                )
            else:
                metric_values = frame.loc[mask, metric]
            score = score + weight * _zscore(metric_values)
        frame.loc[mask, "weighted_z"] = score

    frame["position_score"] = (10.0 / (1.0 + np.exp(-frame["weighted_z"]))).round(3)

    summary = _recent_form_summary(predictive)
    frame["appearances"] = frame["player_id"].map(summary["appearances"]).fillna(0).astype(int)
    frame["recent_form_score"] = frame["player_id"].map(summary["recent_form_score"]).fillna(5.0).round(3)
    frame["form_delta"] = frame["player_id"].map(summary["form_delta"])

    team_matches = player_match.groupby("team_id")["match_id"].nunique()
    max_minutes = frame["team_id"].map(team_matches).fillna(1) * 90
    frame["availability_score"] = (10.0 * minutes / max_minutes).clip(upper=10.0).round(3)

    context, share = _team_context(player_match, frame)
    frame["team_context_score"] = context.round(3)
    frame["team_contribution_share"] = share.round(3)

    minutes_reliability = minutes / (minutes + RELIABLE_MINUTES)
    appearance_reliability = frame["appearances"] / (frame["appearances"] + 3)
    frame["form_confidence"] = (0.7 * minutes_reliability + 0.3 * appearance_reliability).round(3)

    raw_form = sum(weight * frame[column] for column, weight in FORM_COMPONENT_WEIGHTS.items())
    frame["raw_form"] = raw_form
    frame["form_score"] = (
        5.0 + (raw_form - 5.0) * (CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * frame["form_confidence"])
    ).clip(0, 10).round(2)

    frame["position_percentile"] = frame.groupby("position")["raw_form"].rank(pct=True, method="average") * 100
    frame["sample_status"] = np.select(
        [
            (frame["minutes_played"] < PROVISIONAL_MINUTES) | frame["appearances"].eq(0),
            frame["minutes_played"] < RELIABLE_MINUTES,
            frame["minutes_played"] < HIGH_CONFIDENCE_MINUTES,
        ],
        ["Provisional", "Low", "Medium"],
        default="High",
    )
    frame["form_trend"] = np.select(
        [frame["form_delta"] > 0.35, frame["form_delta"] < -0.35],
        ["Improving", "Declining"],
        default="Stable",
    )
    frame.loc[frame["appearances"].eq(0), "form_trend"] = "Unknown"
    frame["form_class"] = np.where(
        frame["sample_status"].eq("Provisional"),
        "Provisional",
        np.select(
            [
                (frame["form_score"] >= 8.5) & frame["sample_status"].eq("High"),
                frame["form_score"] >= 7.5,
                frame["form_score"] >= 6.0,
                frame["form_score"] >= 4.5,
            ],
            ["Elite Form", "Excellent", "Good", "Average"],
            default="Poor",
        ),
    )

    output_columns = [
        "player_id", "player_name", "team_id", "position", "club_team", "market_value_eur", "caps",
        "matches_played", "matches_started", "minutes_played", "appearances", "goals", "assists",
        "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90", "shots_on_target_pct",
        "goals_per_shot", "crosses_per90", "interceptions_per90", "tackles_won_per90", "fouled_per90",
        "save_pct", "saves_per90", "clean_sheet_pct", "goals_against_per90",
        "position_score", "recent_form_score", "availability_score", "team_context_score",
        "team_contribution_share", "form_confidence", "raw_form", "form_score", "position_percentile",
        "sample_status", "form_class", "form_trend", "form_delta",
    ]
    output = frame[output_columns].copy()
    write_csv(output, FEATURE_DIR / "overall_player_form.csv")
    return output


def build_predictive_form(player_match: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_directories()
    if player_match is None:
        player_match = _load_or_build_player_match()
    frame = player_match.sort_values(["match_datetime", "match_id", "player_id"]).copy()
    result_value = frame["team_result"].map({"W": 10.0, "D": 5.0, "L": 0.0})
    minutes_score = frame["minutes_played"].clip(0, 90) / 90 * 10
    start_score = frame["is_starting_xi"] * 10
    goal_score = frame["goals"].clip(upper=2) / 2 * 10
    assist_score = frame["assists"].clip(upper=2) / 2 * 10
    clean_sheet_score = (frame["opponent_goals"].eq(0) & frame["appeared"].eq(1)) * 10
    conceded_score = (10.0 * (1.0 - frame["opponent_goals"].clip(lower=0) / 3.0)).clip(lower=0)

    frame["performance_match"] = np.select(
        [frame["position"].eq(position) for position in POSITION_WEIGHTS],
        [
            (
                0.10 * minutes_score + 0.10 * start_score + 0.15 * result_value
                + 0.45 * goal_score + 0.20 * assist_score
                - 0.5 * frame["yellow_cards"] - 2.0 * frame["red_cards"]
            ),
            (
                0.10 * minutes_score + 0.10 * start_score + 0.20 * result_value
                + 0.20 * goal_score + 0.40 * assist_score
                - 0.5 * frame["yellow_cards"] - 2.0 * frame["red_cards"]
            ),
            (
                0.10 * minutes_score + 0.10 * start_score + 0.20 * result_value
                + 0.35 * clean_sheet_score + 0.15 * goal_score + 0.10 * assist_score
                - 0.75 * frame["yellow_cards"] - 2.5 * frame["red_cards"]
            ),
            (
                0.10 * minutes_score + 0.10 * start_score + 0.20 * result_value
                + 0.35 * clean_sheet_score + 0.25 * conceded_score
                - 0.75 * frame["yellow_cards"] - 2.5 * frame["red_cards"]
            ),
        ],
        default=(
            0.10 * minutes_score + 0.10 * start_score + 0.15 * result_value
            + 0.45 * goal_score + 0.20 * assist_score
            - 0.5 * frame["yellow_cards"] - 2.0 * frame["red_cards"]
        ),
    ).clip(0, 10)
    frame["performance_match"] = frame["performance_match"].where(frame["appeared"].eq(1), np.nan).round(3)

    histories: dict[int, list[float]] = defaultdict(list)
    predictive: list[float] = []
    trend_values: list[float] = []
    for row in frame.itertuples():
        history = histories[row.player_id]
        weights = [0.5, 0.3, 0.2][: len(history[-3:])]
        recent = list(reversed(history[-3:]))
        current_form = float(np.average(recent, weights=weights)) if recent else 5.0
        previous_recent = list(reversed(history[-4:-1]))
        previous_weights = [0.5, 0.3, 0.2][: len(previous_recent)]
        previous_form = float(np.average(previous_recent, weights=previous_weights)) if previous_recent else 5.0
        predictive.append(current_form)
        trend_values.append(current_form - previous_form)
        if row.appeared and not pd.isna(row.performance_match):
            history.append(float(row.performance_match))

    frame["predictive_form"] = np.round(predictive, 3)
    frame["form_delta"] = np.round(trend_values, 3)
    frame["form_trend"] = np.select(
        [frame["form_delta"] > 0.35, frame["form_delta"] < -0.35],
        ["Improving", "Declining"],
        default="Stable",
    )
    columns = [
        "match_id", "match_datetime", "date", "player_id", "player_name", "team_id", "opponent_team_id",
        "position", "tactical_position", "is_starting_xi", "minutes_played", "appeared", "goals", "assists",
        "yellow_cards", "red_cards", "scored_goal", "team_result", "opponent_goals",
        "performance_match", "predictive_form", "form_delta", "form_trend",
    ]
    output = frame[columns].copy()
    write_csv(output, FEATURE_DIR / "player_form_features.csv")
    return output


def build_player_forms() -> tuple[pd.DataFrame, pd.DataFrame]:
    player_match = _load_or_build_player_match()
    predictive = build_predictive_form(player_match)
    overall = build_overall_form(player_match, predictive)
    return overall, predictive


def _load_or_build_player_match() -> pd.DataFrame:
    path = FEATURE_DIR / "player_match_performance.csv"
    if not path.exists():
        return build_player_match()
    return pd.read_csv(path)


def _load_or_build_predictive(player_match: pd.DataFrame) -> pd.DataFrame:
    path = FEATURE_DIR / "player_form_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return build_predictive_form(player_match)


if __name__ == "__main__":
    build_player_forms()
