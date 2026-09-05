import pandas as pd

from config import FEATURE_DIR
from src.data.loaders import write_csv
from src.features.player_form import build_predictive_form


def build_team_form(player_form: pd.DataFrame | None = None) -> pd.DataFrame:
    if player_form is None:
        path = FEATURE_DIR / "player_form_features.csv"
        player_form = pd.read_csv(path) if path.exists() else build_predictive_form()
    starters = player_form[player_form["is_starting_xi"].eq(1)].copy()
    overall = starters.groupby(["match_id", "team_id"], as_index=False)["predictive_form"].mean().rename(
        columns={"predictive_form": "starting_xi_form"}
    )
    by_position = starters.pivot_table(
        index=["match_id", "team_id"], columns="tactical_position", values="predictive_form", aggfunc="mean"
    ).reset_index()
    by_position = by_position.rename(columns={"FWD": "attack_form", "MID": "midfield_form", "DEF": "defense_form", "GK": "gk_form"})
    output = overall.merge(by_position, on=["match_id", "team_id"], how="left", validate="one_to_one")
    for column in ["attack_form", "midfield_form", "defense_form", "gk_form"]:
        output[column] = output[column].fillna(output["starting_xi_form"])
    write_csv(output, FEATURE_DIR / "team_player_form_features.csv")
    return output


if __name__ == "__main__":
    build_team_form()
