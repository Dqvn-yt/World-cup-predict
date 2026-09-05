import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from config import FEATURE_DIR, METRICS_DIR, MODEL_DIR
from src.services.goal_predictor import GoalPredictor
from src.services.match_predictor import MatchPredictor


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.update(TESTING=testing, JSON_SORT_KEYS=False)
    _check_artifacts()
    match_predictor = MatchPredictor()
    goal_predictor = GoalPredictor()
    players = _player_data()
    matches = match_predictor.matches.copy()
    teams = sorted(match_predictor.team_names.items(), key=lambda item: item[1])
    app.extensions["match_predictor"] = match_predictor
    app.extensions["goal_predictor"] = goal_predictor

    @app.template_filter("pct")
    def pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    @app.template_filter("num")
    def num(value: float | None, decimals: int = 2) -> str:
        return "N/A" if value is None or pd.isna(value) else f"{float(value):.{decimals}f}"

    @app.get("/")
    def home():
        featured_match_id = int(matches.iloc[-1]["match_id"])
        prediction = match_predictor.predict_match(featured_match_id)
        threats = goal_predictor.predict_match(featured_match_id, starters_only=True)[:5]
        ranked = players[players["sample_status"].ne("Provisional")]
        return render_template("home.html", top_players=ranked.nlargest(5, "form_score").to_dict("records"), prediction=prediction, threats=threats)

    @app.get("/players")
    def player_list():
        filtered = players.copy()
        query = request.args.get("q", "").strip()
        position = request.args.get("position", "").strip().upper()
        team_id = request.args.get("team_id", type=int)
        sort = request.args.get("sort", "form_score")
        if query:
            filtered = filtered[filtered["player_name"].str.contains(query, case=False, na=False)]
        if position:
            filtered = filtered[filtered["position"].eq(position)]
        if team_id:
            filtered = filtered[filtered["team_id"].eq(team_id)]
        if sort not in {"form_score", "recent_form_score", "goals", "goals_per90", "minutes_played"}:
            sort = "form_score"
        filtered = filtered.sort_values(sort, ascending=False).head(100)
        return render_template("players.html", players=filtered.to_dict("records"), teams=teams, query=query, position=position, selected_team=team_id, sort=sort)

    @app.get("/player/<int:player_id>")
    def player_detail(player_id: int):
        selected = players[players["player_id"].eq(player_id)]
        if selected.empty:
            return render_template("error.html", message="Player not found"), 404
        history = pd.read_csv(FEATURE_DIR / "player_form_features.csv")
        history = history[history["player_id"].eq(player_id) & history["appeared"].eq(1)].sort_values("match_datetime")
        return render_template("player_detail.html", player=selected.iloc[0].to_dict(), history=history.tail(10).to_dict("records"))

    @app.get("/form")
    def form_leaderboard():
        position = request.args.get("position", "").upper()
        status = request.args.get("status", "").title()
        min_minutes = request.args.get("min_minutes", type=int) or 0
        ranked = players.copy()
        if position:
            ranked = ranked[ranked["position"].eq(position)]
        if status in {"Provisional", "Low", "Medium", "High"}:
            ranked = ranked[ranked["sample_status"].eq(status)]
        if min_minutes > 0:
            ranked = ranked[ranked["minutes_played"].ge(min_minutes)]
        return render_template(
            "form.html",
            players=ranked.nlargest(50, "form_score").to_dict("records"),
            position=position,
            status=status,
            min_minutes=min_minutes,
        )

    @app.get("/matches")
    def match_list():
        stage = request.args.get("stage", "")
        filtered = matches[matches["stage_id"].astype(str).eq(stage)] if stage else matches
        records = filtered.sort_values("match_datetime", ascending=False).to_dict("records")
        stages = matches[["stage_id", "is_knockout"]].drop_duplicates().sort_values("stage_id").to_dict("records")
        return render_template("matches.html", matches=records, stages=stages, selected_stage=stage)

    @app.get("/match/<int:match_id>")
    def match_detail(match_id: int):
        try:
            prediction = match_predictor.predict_match(match_id)
            lineups = goal_predictor.predict_match(match_id, starters_only=True)
            threats = lineups[:10]
        except ValueError as error:
            return render_template("error.html", message=str(error)), 404
        home_lineup = [player for player in lineups if player["team_id"] == prediction["home_team_id"]]
        away_lineup = [player for player in lineups if player["team_id"] == prediction["away_team_id"]]
        return render_template(
            "match_detail.html",
            prediction=prediction,
            threats=threats,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
        )

    @app.get("/predict/match/scenario")
    def scenario_detail():
        home_team_id = request.args.get("home_team_id", type=int)
        away_team_id = request.args.get("away_team_id", type=int)
        try:
            prediction = match_predictor.predict_teams(home_team_id, away_team_id)
        except (TypeError, ValueError) as exception:
            return render_template("error.html", message=str(exception)), 400
        return render_template(
            "match_detail.html",
            prediction=prediction,
            threats=[],
            home_lineup=[],
            away_lineup=[],
            is_scenario=True,
        )

    @app.get("/predict")
    def predict_index():
        return redirect(url_for("predict_match_page"))

    @app.route("/predict/match", methods=["GET", "POST"])
    def predict_match_page():
        prediction = None
        error = None
        if request.method == "POST":
            try:
                match_id = request.form.get("match_id", type=int)
                home_id = request.form.get("home_team_id", type=int)
                away_id = request.form.get("away_team_id", type=int)
                prediction = match_predictor.predict_match(match_id) if match_id else match_predictor.predict_teams(home_id, away_id)
            except (TypeError, ValueError) as exception:
                error = str(exception)
        return render_template("predict_match.html", matches=matches.to_dict("records"), teams=teams, prediction=prediction, error=error)

    @app.route("/predict/goals", methods=["GET", "POST"])
    def predict_goals_page():
        match_id = request.values.get("match_id", type=int) or int(matches.iloc[-1]["match_id"])
        try:
            predictions = goal_predictor.predict_match(match_id)
        except ValueError:
            predictions = []
        return render_template("predict_goals.html", matches=matches.to_dict("records"), selected_match=match_id, predictions=predictions)

    @app.get("/compare")
    def compare():
        first_id = request.args.get("player_a", type=int)
        second_id = request.args.get("player_b", type=int)
        selected = []
        for player_id in [first_id, second_id]:
            row = players[players["player_id"].eq(player_id)] if player_id else pd.DataFrame()
            if not row.empty:
                selected.append(row.iloc[0].to_dict())
        home_team_id = request.args.get("home_team_id", type=int)
        away_team_id = request.args.get("away_team_id", type=int)
        team_prediction = None
        team_error = None
        if home_team_id or away_team_id:
            try:
                team_prediction = match_predictor.predict_teams(home_team_id, away_team_id)
            except (TypeError, ValueError) as exception:
                team_error = str(exception)
        return render_template(
            "compare.html",
            players=players.sort_values("player_name").to_dict("records"),
            selected=selected,
            player_a=first_id,
            player_b=second_id,
            teams=teams,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_prediction=team_prediction,
            team_error=team_error,
        )

    @app.get("/models")
    def models_page():
        return render_template("models.html", metrics=_load_metrics())

    @app.get("/about")
    def about_page():
        return redirect(url_for("models_page"))

    @app.get("/api/players")
    def api_players():
        filtered = players.copy()
        team_id = request.args.get("team_id", type=int)
        position = request.args.get("position", "").upper()
        if team_id:
            filtered = filtered[filtered["team_id"].eq(team_id)]
        if position:
            filtered = filtered[filtered["position"].eq(position)]
        return jsonify(filtered.to_dict("records"))

    @app.get("/api/player/<int:player_id>/form")
    def api_player_form(player_id: int):
        history = pd.read_csv(FEATURE_DIR / "player_form_features.csv")
        history = history[history["player_id"].eq(player_id)]
        if history.empty:
            return jsonify({"error": "Player not found"}), 404
        columns = ["match_id", "date", "predictive_form", "performance_match", "form_delta", "form_trend"]
        return jsonify(history[columns].where(pd.notna(history[columns]), None).to_dict("records"))

    @app.post("/api/predict/match")
    def api_predict_match():
        payload = request.get_json(silent=True) or {}
        try:
            if payload.get("match_id") is not None:
                result = match_predictor.predict_match(int(payload["match_id"]))
            else:
                result = match_predictor.predict_teams(int(payload["home_team_id"]), int(payload["away_team_id"]))
            return jsonify(result)
        except (KeyError, TypeError, ValueError) as exception:
            return jsonify({"error": str(exception)}), 400

    @app.post("/api/predict/goal")
    def api_predict_goal():
        payload = request.get_json(silent=True) or {}
        try:
            match_id = int(payload["match_id"])
            if payload.get("player_id") is not None:
                result = goal_predictor.predict_player(match_id, int(payload["player_id"]))
            else:
                result = goal_predictor.predict_match(match_id)
            return jsonify(result)
        except (KeyError, TypeError, ValueError) as exception:
            return jsonify({"error": str(exception)}), 400

    @app.get("/api/model/metrics")
    def api_metrics():
        return jsonify(_load_metrics())

    return app


def _check_artifacts() -> None:
    required = [
        MODEL_DIR / "score_models.joblib",
        MODEL_DIR / "goal_probability_model.joblib",
        FEATURE_DIR / "overall_player_form.csv",
        FEATURE_DIR / "match_model_dataset.csv",
        FEATURE_DIR / "goal_probability_dataset.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing generated artifacts. Run `python build.py` first: {missing}")


def _player_data() -> pd.DataFrame:
    players = pd.read_csv(FEATURE_DIR / "overall_player_form.csv")
    history = pd.read_csv(FEATURE_DIR / "player_form_features.csv").sort_values(["match_datetime", "match_id"])
    latest = history.groupby("player_id", as_index=False).tail(1)[["player_id", "predictive_form"]]
    latest = latest.rename(columns={"predictive_form": "latest_predictive_form"})
    return players.merge(latest, on="player_id", how="left", validate="one_to_one")


def _load_metrics() -> dict:
    output = {}
    for key, filename in [("score", "score_metrics.json"), ("goal", "goal_metrics.json"), ("data", "data_audit.json")]:
        path = METRICS_DIR / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as source:
                output[key] = json.load(source)
    return output


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
