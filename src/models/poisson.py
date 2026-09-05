from math import factorial

import numpy as np


FULL_SUPPORT = 25


def ensemble_score_matrix(lambda_home: float, lambda_away: float, rho: float, dc_weight: float = 0.7) -> np.ndarray:
    from src.models.dixon_coles import dixon_coles_tau

    poisson = np.outer(_pmf(lambda_home, FULL_SUPPORT), _pmf(lambda_away, FULL_SUPPORT))
    corrected = poisson * dixon_coles_tau(lambda_home, lambda_away, rho, FULL_SUPPORT)
    return (1 - dc_weight) * poisson / poisson.sum() + dc_weight * corrected / corrected.sum()


def poisson_probability(goals: int, expected_goals: float) -> float:
    expected_goals = max(float(expected_goals), 1e-9)
    return float(np.exp(-expected_goals) * expected_goals**goals / factorial(goals))


def _pmf(expected_goals: float, support: int) -> np.ndarray:
    expected_goals = float(expected_goals)
    goals = np.arange(support + 1)
    return np.exp(-expected_goals) * np.power(expected_goals, goals) / np.array([factorial(k) for k in goals], dtype=float)


def _market_confidence(probabilities: np.ndarray) -> str:
    top = float(probabilities.max())
    if top >= 0.55:
        return "High"
    if top >= 0.42:
        return "Medium"
    return "Low"


def _match_profile(lambda_home: float, lambda_away: float) -> str:
    total = lambda_home + lambda_away
    if total < 2.2:
        return "Low-scoring"
    if total > 3.4:
        return "Open / High-scoring"
    return "Balanced"


def _team_goal_probabilities(pmf: np.ndarray) -> dict:
    return {
        "scores_1_plus": float(1.0 - pmf[0]),
        "scores_2_plus": float(1.0 - pmf[0] - pmf[1]),
        "scores_3_plus": float(1.0 - pmf[0] - pmf[1] - pmf[2]),
    }


def _total_goals_pmf(matrix: np.ndarray) -> np.ndarray:
    support = matrix.shape[0]
    totals = np.zeros(2 * support - 1)
    for i in range(support):
        for j in range(support):
            totals[i + j] += matrix[i, j]
    return totals / matrix.sum()


def _markets(full_matrix: np.ndarray, lambda_home: float, lambda_away: float) -> dict:
    total = float(full_matrix.sum())
    home_pmf = full_matrix.sum(axis=1) / total
    away_pmf = full_matrix.sum(axis=0) / total
    goal_totals = _total_goals_pmf(full_matrix)
    home_win = float(np.tril(full_matrix, k=-1).sum() / total)
    draw = float(np.trace(full_matrix) / total)
    away_win = float(np.triu(full_matrix, k=1).sum() / total)
    both_score = float(full_matrix[1:, 1:].sum() / total)
    return {
        "home_win_probability": home_win,
        "draw_probability": draw,
        "away_win_probability": away_win,
        "double_chance": {
            "home_or_draw": float(home_win + draw),
            "home_or_away": float(home_win + away_win),
            "draw_or_away": float(draw + away_win),
        },
        "win_margins": {
            "home_by_1": float(np.trace(full_matrix, offset=-1) / total),
            "home_by_2_plus": float(np.tril(full_matrix, k=-2).sum() / total),
            "away_by_1": float(np.trace(full_matrix, offset=1) / total),
            "away_by_2_plus": float(np.triu(full_matrix, k=2).sum() / total),
        },
        "home_clean_sheet_probability": float(away_pmf[0]),
        "away_clean_sheet_probability": float(home_pmf[0]),
        "home_goals_market": _team_goal_probabilities(home_pmf),
        "away_goals_market": _team_goal_probabilities(away_pmf),
        "total_goals_distribution": {
            "0": float(goal_totals[0]),
            "1": float(goal_totals[1]),
            "2": float(goal_totals[2]),
            "3": float(goal_totals[3]),
            "4+": float(goal_totals[4:].sum()),
        },
        "over_under": {
            "1.5": float(goal_totals[2:].sum()),
            "2.5": float(goal_totals[3:].sum()),
            "3.5": float(goal_totals[4:].sum()),
        },
        "both_teams_to_score": both_score,
        "no_both_teams_to_score": float(1.0 - both_score),
        "expected_total_goals": float(sum(index * value for index, value in enumerate(goal_totals))),
        "match_profile": _match_profile(lambda_home, lambda_away),
        "confidence": _market_confidence(np.array([home_win, draw, away_win])),
    }


def score_probability_matrix(lambda_home: float, lambda_away: float, max_goals: int = 6) -> dict:
    home_full = _pmf(lambda_home, FULL_SUPPORT)
    away_full = _pmf(lambda_away, FULL_SUPPORT)
    full_matrix = np.outer(home_full, away_full)
    return summarize_score_matrix(full_matrix, lambda_home, lambda_away, max_goals)


def summarize_score_matrix(full_matrix: np.ndarray, lambda_home: float, lambda_away: float, max_goals: int = 6) -> dict:
    full_matrix = full_matrix / full_matrix.sum()
    best_home, best_away = np.unravel_index(np.argmax(full_matrix), full_matrix.shape)

    display = full_matrix[: max_goals + 1, : max_goals + 1]
    grid_coverage = float(display.sum())
    scorelines = [
        {"home_goals": int(i), "away_goals": int(j), "probability": float(display[i, j])}
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]
    scorelines.sort(key=lambda item: item["probability"], reverse=True)
    result = {
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "predicted_home_goals": int(best_home),
        "predicted_away_goals": int(best_away),
        "tail_probability": max(0.0, 1.0 - grid_coverage),
        "grid_coverage": grid_coverage,
        "matrix": display.tolist(),
        "top_scorelines": scorelines[:5],
        "max_goals": max_goals,
    }
    result.update(_markets(full_matrix, lambda_home, lambda_away))
    return result
