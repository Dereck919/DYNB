# model.py
#
# ML-powered next-game stat predictor.
#
# Trains a GradientBoostingRegressor per stat that learns the optimal
# way to combine three inputs:
#   - Season averages
#   - Last 3 game averages (recent form)
#   - vs Opponent averages (matchup history)
#   - Home/away, days of rest, games played
#
# Instead of hardcoded 50/30/20 weights, the model figures out
# the best blend from every game played this season.
#
# Also computes confidence % and range for each prediction.

import math
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from data.fetcher import parse_opponent
from config import STATS, PLAYERS

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

FEATURE_COLS = (
    [f"savg_{s}" for s in STATS]
    + [f"last3_{s}" for s in STATS]
    + [f"vs_opp_{s}" for s in STATS]
    + ["is_home", "days_rest", "games_played"]
)

SHRINKAGE = 0.40  # tightens range to match betting spreads


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_over_line(prediction, line, std_dev):
    if std_dev <= 0:
        return 1.0 if prediction >= line else 0.0
    z = (line - 0.5 - prediction) / std_dev
    return 1.0 - _normal_cdf(z)


def _build_features_for_game(history, opponent, is_home, game_date=None):
    """
    Build one feature row from a player's game history.
    Uses only games that happened BEFORE the target game.
    """
    features = {}

    # Season averages
    for s in STATS:
        features[f"savg_{s}"] = float(history[s].mean()) if s in history.columns else 0.0

    # Last 3 games
    last3 = history.tail(3)
    for s in STATS:
        features[f"last3_{s}"] = float(last3[s].mean()) if s in last3.columns else 0.0

    # Most recent game vs this opponent
    if "OPPONENT" in history.columns:
        vs = history[history["OPPONENT"] == opponent]
    else:
        vs = pd.DataFrame()

    if not vs.empty:
        last_vs = vs.iloc[-1]
        for s in STATS:
            features[f"vs_opp_{s}"] = float(last_vs[s]) if s in last_vs.index else 0.0
    else:
        # No matchup data: fall back to season avg
        for s in STATS:
            features[f"vs_opp_{s}"] = features[f"savg_{s}"]

    features["is_home"] = int(is_home)
    features["games_played"] = len(history)

    if game_date is not None and len(history) > 0:
        dates = pd.to_datetime(history["GAME_DATE"])
        delta = (pd.to_datetime(game_date) - dates.max()).days
        features["days_rest"] = max(0, delta)
    else:
        features["days_rest"] = 2

    return features


def train_models(game_logs):
    """
    Train one GradientBoostingRegressor per stat.

    Uses every game from every default player as training data.
    For each game, features = what we knew before it happened,
    target = what actually happened.
    """
    all_X = []
    all_y = []

    for name in PLAYERS:
        pdf = game_logs[game_logs["PLAYER_NAME"] == name].copy()
        if len(pdf) < 6:
            continue

        pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
        pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
        pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)
        pdf["IS_HOME"] = pdf["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)

        for i in range(5, len(pdf)):
            current = pdf.iloc[i]
            history = pdf.iloc[:i]

            features = _build_features_for_game(
                history,
                opponent=current["OPPONENT"],
                is_home=current["IS_HOME"],
                game_date=current["GAME_DATE"],
            )

            target = {}
            for s in STATS:
                target[s] = float(current[s]) if s in current.index else 0.0

            all_X.append(features)
            all_y.append(target)

    if not all_X:
        print("  No training data.")
        return {}, {}

    X = pd.DataFrame(all_X)[FEATURE_COLS]
    y = pd.DataFrame(all_y)[STATS]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Training on {len(X)} samples from {len(PLAYERS)} players")

    models = {}
    metrics = {}
    for s in STATS:
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X, y[s])
        models[s] = model

        preds = model.predict(X)
        mae = float(np.mean(np.abs(y[s].values - preds)))
        metrics[s] = {"MAE": round(mae, 2), "samples": len(X)}
        print(f"    {s:>8s}  MAE={mae:.2f}")

    return models, metrics


def predict_games(game_logs, models, player_name, upcoming_games):
    """
    Predict stats for upcoming games using the trained models.

    Each prediction includes value, rounded line, confidence %,
    range, and a breakdown of the three input components.
    """
    if not models or not upcoming_games:
        return []

    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name].copy()
    if pdf.empty:
        return []

    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
    pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)

    # Compute std devs for confidence calculation
    season_std = {}
    for s in STATS:
        season_std[s] = float(pdf[s].std()) if s in pdf.columns and len(pdf) > 1 else 0.0

    last_5 = pdf.tail(5)
    recent_std = {}
    for s in STATS:
        if s in last_5.columns and len(last_5) > 1:
            recent_std[s] = float(last_5[s].std())
        else:
            recent_std[s] = season_std.get(s, 0.0)

    # Precompute season and last-3 averages for breakdown display
    season_avg = {}
    for s in STATS:
        season_avg[s] = round(float(pdf[s].mean()), 1) if s in pdf.columns else 0.0

    last3 = pdf.tail(3)
    last3_avg = {}
    for s in STATS:
        last3_avg[s] = round(float(last3[s].mean()), 1) if s in last3.columns else 0.0

    predictions = []
    for game in upcoming_games:
        opp = game["opponent"]
        is_home = game["home"]

        # Build feature vector
        features = _build_features_for_game(pdf, opp, is_home)
        X = pd.DataFrame([features])[FEATURE_COLS]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # vs opponent averages for breakdown
        vs_games = pdf[pdf["OPPONENT"] == opp]
        has_matchup = len(vs_games) > 0
        vs_avg = {}
        if has_matchup:
            for s in STATS:
                vs_avg[s] = round(float(vs_games[s].mean()), 1) if s in vs_games.columns else 0.0

        pred = {
            "opponent": opp,
            "home": is_home,
            "date": game.get("date", ""),
            "display_date": game.get("display_date", ""),
        }

        breakdown = {
            "season_avg": dict(season_avg),
            "last_3_avg": dict(last3_avg),
            "vs_opponent_avg": vs_avg if has_matchup else {s: None for s in STATS},
            "has_matchup_data": has_matchup,
            "vs_games_count": len(vs_games),
        }

        for s in STATS:
            val = float(models[s].predict(X)[0])

            if "PCT" in s:
                val = max(0.0, min(1.0, val))
            else:
                val = max(0.0, val)

            rounded = round(val)

            # Confidence and range
            raw_std = (season_std[s] * 0.4) + (recent_std[s] * 0.6)
            effective_std = raw_std * SHRINKAGE

            if "PCT" in s:
                confidence = None
                range_val = round(effective_std * 100, 1)
            else:
                confidence = round(_prob_over_line(val, rounded, effective_std), 2)
                confidence = max(0.01, min(0.99, confidence))
                range_val = round(effective_std, 1)

            pred[s] = {
                "value": round(val, 1),
                "rounded": rounded,
                "confidence": confidence,
                "range": range_val,
            }

        pred["breakdown"] = breakdown
        predictions.append(pred)

    return predictions