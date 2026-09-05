import numpy as np
import pandas as pd
from scipy.optimize import minimize


DC_SUPPORT = 12
RIDGE = 1.0
SHRINK_MATCHES = 3.0
DEFAULT_RHO = -0.05


def dixon_coles_tau(lambda_home: float, lambda_away: float, rho: float, support: int = DC_SUPPORT) -> np.ndarray:
    tau = np.ones((support + 1, support + 1))
    tau[0, 0] = max(1.0 - lambda_home * lambda_away * rho, 1e-6)
    tau[1, 0] = max(1.0 + lambda_away * rho, 1e-6)
    tau[0, 1] = max(1.0 + lambda_home * rho, 1e-6)
    tau[1, 1] = max(1.0 - rho, 1e-6)
    return tau


def dc_score_matrix(lambda_home: float, lambda_away: float, rho: float, max_goals: int = 6) -> dict:
    from src.models.poisson import _markets, _pmf

    home_pmf = _pmf(lambda_home, DC_SUPPORT)
    away_pmf = _pmf(lambda_away, DC_SUPPORT)
    matrix = np.outer(home_pmf, away_pmf) * dixon_coles_tau(lambda_home, lambda_away, rho)
    matrix = matrix / matrix.sum()
    best_home, best_away = np.unravel_index(np.argmax(matrix), matrix.shape)
    display = matrix[: max_goals + 1, : max_goals + 1]
    scorelines = [
        {"home_goals": int(i), "away_goals": int(j), "probability": float(display[i, j])}
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]
    scorelines.sort(key=lambda item: item["probability"], reverse=True)
    result = {
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "rho": float(rho),
        "predicted_home_goals": int(best_home),
        "predicted_away_goals": int(best_away),
        "tail_probability": max(0.0, 1.0 - float(display.sum())),
        "grid_coverage": float(display.sum()),
        "matrix": display.tolist(),
        "top_scorelines": scorelines[:5],
        "max_goals": max_goals,
    }
    result.update(_markets(matrix, lambda_home, lambda_away))
    return result


class DixonColesModel:
    def __init__(
        self,
        decay_half_life_days: float = 1461.0,
        shrink_matches: float = SHRINK_MATCHES,
        rho_bound: float = 0.2,
        max_iter: int = 120,
    ) -> None:
        self.decay_half_life_days = decay_half_life_days
        self.shrink_matches = shrink_matches
        self.rho_bound = rho_bound
        self.max_iter = max_iter
        self.team_index: dict[int, int] = {}
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_advantage = 0.0
        self.neutral_advantage = 0.0
        self.rho = DEFAULT_RHO
        self.matches_used = 0

    def fit(self, matches: pd.DataFrame, initial_params: np.ndarray | None = None, reference_time: object = None) -> "DixonColesModel":
        frame = matches.sort_values("match_datetime").reset_index(drop=True)
        teams = sorted(set(frame["home_team_id"]).union(set(frame["away_team_id"])))
        self.team_index = {int(team): index for index, team in enumerate(teams)}
        n_teams = len(teams)
        home_idx = frame["home_team_id"].map(self.team_index).to_numpy()
        away_idx = frame["away_team_id"].map(self.team_index).to_numpy()
        home_goals = frame["home_score"].to_numpy(dtype=float)
        away_goals = frame["away_score"].to_numpy(dtype=float)
        if "neutral" in frame.columns:
            neutral = frame["neutral"].fillna(True).to_numpy(dtype=bool)
        else:
            neutral = np.zeros(len(frame), dtype=bool)
        self.neutral_matches = int(neutral.sum())
        dates = pd.to_datetime(frame["match_datetime"])
        reference = dates.max() if reference_time is None else pd.Timestamp(reference_time)
        if reference < dates.max():
            raise ValueError("Fit reference precedes observed results")
        age_days = (reference - dates).dt.total_seconds().to_numpy(dtype=float) / 86400
        weights = np.exp(-np.log(2.0) * age_days / self.decay_half_life_days)
        self.effective_matches = float(weights.sum())

        init = np.zeros(2 * n_teams + 3)
        init[-3] = 0.2
        init[-2] = 0.0
        init[-1] = DEFAULT_RHO
        if initial_params is not None and len(initial_params) == len(init):
            init = np.asarray(initial_params, dtype=float)
        bounds = [(-2.0, 2.0)] * (2 * n_teams) + [(0.0, 1.2), (-0.5, 0.5), (-self.rho_bound, self.rho_bound)]
        max_goal = int(max(home_goals.max(), away_goals.max()))
        log_factorial = np.array([sum(np.log(range(1, k + 1))) if k else 0.0 for k in range(max_goal + 1)])
        log_fact_home = log_factorial[home_goals.astype(int)]
        log_fact_away = log_factorial[away_goals.astype(int)]

        def negative_log_likelihood(params: np.ndarray) -> float:
            attack, defense = params[:n_teams], params[n_teams : 2 * n_teams]
            host_adv, neutral_adv, rho = params[-3], params[-2], params[-1]
            # Two-level venue effect: genuine host advantage on non-neutral
            # fixtures, residual nominal-home edge on neutral ones. The mask
            # keeps each level estimated from its own fixtures only.
            venue_adv = host_adv * (1.0 - neutral) + neutral_adv * neutral
            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + venue_adv)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])
            lambda_home = np.clip(lambda_home, 1e-6, 10.0)
            lambda_away = np.clip(lambda_away, 1e-6, 10.0)
            log_pmf_home = -lambda_home + home_goals * np.log(lambda_home) - log_fact_home
            log_pmf_away = -lambda_away + away_goals * np.log(lambda_away) - log_fact_away
            tau = np.ones(len(frame))
            zero_home = home_goals == 0
            zero_away = away_goals == 0
            tau[zero_home & zero_away] = 1.0 - lambda_home[zero_home & zero_away] * lambda_away[zero_home & zero_away] * rho
            tau[(~zero_home) & zero_away & (home_goals == 1)] = 1.0 + lambda_away[(~zero_home) & zero_away & (home_goals == 1)] * rho
            tau[zero_home & (~zero_away) & (away_goals == 1)] = 1.0 + lambda_home[zero_home & (~zero_away) & (away_goals == 1)] * rho
            tau[(home_goals == 1) & (away_goals == 1)] = 1.0 - rho
            tau = np.clip(tau, 1e-6, None)
            log_likelihood = weights * (log_pmf_home + log_pmf_away + np.log(tau))
            # Venue advantages share the ridge prior toward zero: host_adv is
            # identified from few host fixtures and must not explode.
            penalty = RIDGE * (np.sum(attack**2) + np.sum(defense**2) + host_adv**2 + neutral_adv**2)
            return -float(log_likelihood.sum()) + penalty

        result = minimize(
            negative_log_likelihood,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        params = result.x
        self.converged = bool(result.success)
        self.params_ = params
        self.attack = params[:n_teams]
        self.defense = params[n_teams : 2 * n_teams]
        self.home_advantage = float(params[-3])
        self.neutral_advantage = float(params[-2])
        self.rho = float(params[-1])
        self.matches_used = int(len(frame))
        return self

    def lambdas(self, home_team_id: int, away_team_id: int, matches_played: tuple[int, int] | None = None, neutral: bool = False) -> tuple[float, float]:
        if self.attack is None:
            raise RuntimeError("DixonColesModel must be fitted before predicting")
        shrink_home = self._shrink(matches_played[0] if matches_played else None)
        shrink_away = self._shrink(matches_played[1] if matches_played else None)
        attack_home = self.attack[self.team_index[home_team_id]] * shrink_home if home_team_id in self.team_index else 0.0
        defense_home = self.defense[self.team_index[home_team_id]] * shrink_home if home_team_id in self.team_index else 0.0
        attack_away = self.attack[self.team_index[away_team_id]] * shrink_away if away_team_id in self.team_index else 0.0
        defense_away = self.defense[self.team_index[away_team_id]] * shrink_away if away_team_id in self.team_index else 0.0
        advantage = self.neutral_advantage if neutral else self.home_advantage
        lambda_home = float(np.exp(attack_home + defense_away + advantage))
        lambda_away = float(np.exp(attack_away + defense_home))
        return float(np.clip(lambda_home, 0.1, 5.0)), float(np.clip(lambda_away, 0.1, 5.0))

    def _shrink(self, matches_played: int | None) -> float:
        if matches_played is None:
            return 1.0
        return float(matches_played / (matches_played + self.shrink_matches))


def walk_forward_dixon_coles(
    matches: pd.DataFrame,
    half_life_days: float = 1461.0,
    min_history: int = 8,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = matches.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
    predictions: list[dict[str, float | int | bool]] = []
    model: DixonColesModel | None = None
    fitted_through = -1
    pool = pd.concat([history, frame], ignore_index=True) if history is not None else frame
    pool = pool.copy()
    pool["match_datetime"] = pd.to_datetime(pool["match_datetime"])
    frame["match_datetime"] = pd.to_datetime(frame["match_datetime"])
    for _, row in frame.iterrows():
        prior = pool[pool["match_datetime"] < row["match_datetime"]]
        if len(prior) >= min_history and fitted_through < len(prior) - 1:
            model = DixonColesModel(decay_half_life_days=half_life_days).fit(prior, reference_time=row["match_datetime"])
            fitted_through = len(prior) - 1
        if model is None:
            past = prior
            home_lambda = float(past["home_score"].mean()) if len(past) else 1.3
            away_lambda = float(past["away_score"].mean()) if len(past) else 1.1
            predictions.append({
                "match_id": int(row["match_id"]),
                "dc_lambda_home": float(np.clip(home_lambda, 0.1, 5.0)),
                "dc_lambda_away": float(np.clip(away_lambda, 0.1, 5.0)),
                "modeled": False,
                "dc_rho": 0.0,
                "dc_history_rows": len(prior),
                "neutral": bool(row["neutral"]) if "neutral" in row and pd.notna(row["neutral"]) else True,
            })
        else:
            row_neutral = bool(row["neutral"]) if "neutral" in row and pd.notna(row["neutral"]) else True
            home_lambda, away_lambda = model.lambdas(int(row["home_team_id"]), int(row["away_team_id"]), neutral=row_neutral)
            predictions.append({
                "match_id": int(row["match_id"]),
                "dc_lambda_home": home_lambda,
                "dc_lambda_away": away_lambda,
                "modeled": True,
                "dc_rho": model.rho,
                "dc_history_rows": len(prior),
                "neutral": row_neutral,
            })
    return pd.DataFrame(predictions)
