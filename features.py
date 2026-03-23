# features.py
#
# Builds feature vectors for training and prediction.
#
# For each game, the features are:
#   - Season averages up to (but not including) that game
#   - Last 3 game averages
#   - Stats from the most recent game vs the same opponent
#   - Home/away, days of rest, games played count
#
# All features are computed using ONLY past data.

import pandas as pd
import numpy as np
from data.fetcher import parse_opponent, is_home_game
from config import STATS

# Columns the model uses as input
FEATURE_COLS = (
    # Season averages
    [f"savg_{s}" for s in STATS]
    # Last 3 game averages
    + [f"last3_{s}" for s in STATS]
    # Most recent game vs this opponent
    + [f"vs_opp_{s}" for s in STATS]
    + ["is_home", "days_rest", "games_played"]
)


def build_training_data(game_logs, player_name):
    """
    Turn one player's game log into (X, y) training pairs.

    Each game becomes one sample where:
      X = features from games BEFORE it
      y = what actually happened in that game

    Skips the first 5 games (not enough history for features).

    Returns (X DataFrame, y DataFrame) or (None, None) if not enough data.
    """
    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name].copy()
    if len(pdf) < 6:
        return None, None

    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
    pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)
    pdf["IS_HOME"] = pdf["MATCHUP"].apply(is_home_game).astype(int)

    rows_X = []
    rows_y = []

    for i in range(5, len(pdf)):
        current = pdf.iloc[i]
        history = pdf.iloc[:i]  # all games before this one

        features = _build_features(
            history=history,
            opponent=current["OPPONENT"],
            is_home=current["IS_HOME"],
            game_date=current["GAME_DATE"],
        )

        target = {}
        for s in STATS:
            target[s] = float(current[s]) if s in current.index else 0.0

        rows_X.append(features)
        rows_y.append(target)

    if not rows_X:
        return None, None

    X = pd.DataFrame(rows_X)[FEATURE_COLS]
    y = pd.DataFrame(rows_y)[STATS]
    return X, y


def build_prediction_features(game_logs, player_name, opponent, is_home):
    """
    Build a feature vector for predicting a future game.

    Uses all games in game_logs as history, then constructs
    features for a game against `opponent`.

    Returns a single-row DataFrame with columns matching FEATURE_COLS.
    """
    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name].copy()
    if pdf.empty:
        return None

    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
    pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)

    features = _build_features(
        history=pdf,
        opponent=opponent,
        is_home=1 if is_home else 0,
        game_date=None,  # future game, no date for rest calc
    )

    return pd.DataFrame([features])[FEATURE_COLS]


def _build_features(history, opponent, is_home, game_date=None):
    """
    Internal: construct the feature dict from game history.
    """
    features = {}

    # 1. Season averages (all games in history)
    for s in STATS:
        if s in history.columns:
            features[f"savg_{s}"] = history[s].mean()
        else:
            features[f"savg_{s}"] = 0.0

    # 2. Last 3 game averages
    last3 = history.tail(3)
    for s in STATS:
        if s in last3.columns:
            features[f"last3_{s}"] = last3[s].mean()
        else:
            features[f"last3_{s}"] = 0.0

    # 3. Most recent game vs this opponent
    if "OPPONENT" in history.columns:
        vs_games = history[history["OPPONENT"] == opponent]
    else:
        vs_games = pd.DataFrame()

    if not vs_games.empty:
        last_vs = vs_games.iloc[-1]  # most recent matchup
        for s in STATS:
            features[f"vs_opp_{s}"] = float(last_vs[s]) if s in last_vs.index else 0.0
    else:
        # Never faced this opponent: fall back to season avg
        for s in STATS:
            features[f"vs_opp_{s}"] = features[f"savg_{s}"]

    # 4. Context features
    features["is_home"] = int(is_home)
    features["games_played"] = len(history)

    # Days of rest (0 if we don't know)
    if game_date is not None and len(history) > 0:
        history_dates = pd.to_datetime(history["GAME_DATE"])
        last_game = history_dates.max()
        delta = (pd.to_datetime(game_date) - last_game).days
        features["days_rest"] = max(0, delta)
    else:
        features["days_rest"] = 2  # typical default

    return features


def get_season_averages(game_logs, player_name):
    """Simple season averages for display."""
    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name]
    if pdf.empty:
        return {}
    avgs = {}
    for s in STATS:
        if s in pdf.columns:
            avgs[s] = round(float(pdf[s].mean()), 1)
    avgs["GP"] = len(pdf)
    return avgs


def get_last_n_games(game_logs, player_name, n=3):
    """Last N games for the recent form display."""
    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name].copy()
    if pdf.empty:
        return []
    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").tail(n)
    pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)
    pdf["IS_HOME"] = pdf["MATCHUP"].apply(is_home_game)

    games = []
    for _, row in pdf.iterrows():
        g = {
            "date": row["GAME_DATE"].strftime("%b %d"),
            "opponent": row["OPPONENT"],
            "home": bool(row["IS_HOME"]),
            "result": row.get("WL", "?"),
        }
        for s in STATS:
            if s in row.index:
                g[s] = round(float(row[s]), 1)
        games.append(g)
    return games
