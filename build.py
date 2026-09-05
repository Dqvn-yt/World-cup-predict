from src.data.cleaning import clean_all
from src.data.build_player_match import build_player_match
from src.features.goal_features import build_goal_features
from src.features.match_features import build_match_features
from src.features.player_form import build_overall_form, build_predictive_form
from src.features.team_form import build_team_form
from src.models.train_goal import train_goal_model
from src.models.train_score import train_score_models


def build_all() -> None:
    print("[1/7] Cleaning source data")
    clean_all()
    print("[2/7] Building player-match dataset")
    player_match = build_player_match()
    print("[3/7] Building player form features")
    build_overall_form(player_match)
    player_form = build_predictive_form(player_match)
    build_team_form(player_form)
    print("[4/7] Building match features")
    match_features = build_match_features()
    print("[5/7] Training Dixon-Coles + Poisson score models")
    score_metrics = train_score_models(match_features)
    print("[6/7] Building goal probability features")
    goal_features = build_goal_features()
    print("[7/7] Training logistic goal model")
    goal_metrics = train_goal_model(goal_features)
    print("Build complete")
    print(f"Score outcome accuracy (test): {score_metrics['test_split']['outcome_accuracy']:.3f}")
    print(f"Goal PR-AUC: {goal_metrics['average_precision']:.3f}")


if __name__ == "__main__":
    build_all()
