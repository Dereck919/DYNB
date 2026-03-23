# model.py
#
# Predicts next-game stats using a weighted blend of three factors:
#
#   50%  Season average   (what the player does overall)
#   30%  Last 3 games     (are they hot or cold right now?)
#   20%  vs Opponent      (how do they play against this team?)
#
# If the player hasn't faced the opponent this season,
# the weights shift to 60% season / 40% last 3.
#
# Also computes:
#   - Confidence %: probability of hitting at or above the rounded line
#   - Range (±): one standard deviation from the prediction
#
# The probability uses a normal distribution centered on the
# prediction with the player's actual game-to-game variance.

import math
import pandas as pd
from data.fetcher import parse_opponent
from config import STATS

# ── Weights ──
W_SEASON  = 0.50
W_RECENT  = 0.30
W_MATCHUP = 0.20

# If no opponent history:
W_SEASON_NO_OPP = 0.60
W_RECENT_NO_OPP = 0.40


def _normal_cdf(x):
    """
    Approximate the cumulative distribution function of a
    standard normal distribution. No scipy needed.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_over_line(prediction, line, std_dev):
    """
    Probability that the actual stat >= line,
    given a normal distribution centered on `prediction`
    with standard deviation `std_dev`.

    Returns a value between 0 and 1.
    """
    if std_dev <= 0:
        return 1.0 if prediction >= line else 0.0

    # P(X >= line) = 1 - P(X < line)
    # P(X < line) = CDF((line - prediction) / std_dev)
    # We use line - 0.5 for continuity correction since
    # we're comparing to a whole number
    z = (line - 0.5 - prediction) / std_dev
    return 1.0 - _normal_cdf(z)


def predict_games(game_logs, player_name, upcoming_games):
    """
    Predict stats for upcoming games.

    Each prediction includes:
      - value: the decimal prediction (e.g. 34.8)
      - rounded: the whole number line (e.g. 35)
      - confidence: probability of hitting >= rounded (e.g. 0.93)
      - range: one std dev spread (e.g. 5.2)
      - breakdown: season avg, last 3, vs opponent components
    """
    if not upcoming_games:
        return []

    pdf = game_logs[game_logs["PLAYER_NAME"] == player_name].copy()
    if pdf.empty:
        return []

    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)
    pdf["OPPONENT"] = pdf["MATCHUP"].apply(parse_opponent)

    # Season averages and std devs
    season_avg = {}
    season_std = {}
    for s in STATS:
        if s in pdf.columns:
            season_avg[s] = float(pdf[s].mean())
            season_std[s] = float(pdf[s].std()) if len(pdf) > 1 else 0.0
        else:
            season_avg[s] = 0.0
            season_std[s] = 0.0

    # Last 5 games std dev (captures current-form volatility)
    last_5 = pdf.tail(5)
    recent_std = {}
    for s in STATS:
        if s in last_5.columns and len(last_5) > 1:
            recent_std[s] = float(last_5[s].std())
        else:
            recent_std[s] = season_std.get(s, 0.0)

    # Last 3 games
    last_3 = pdf.tail(3)
    last3_avg = {}
    for s in STATS:
        if s in last_3.columns:
            last3_avg[s] = float(last_3[s].mean())
        else:
            last3_avg[s] = 0.0

    # Build a prediction for each upcoming game
    predictions = []
    for game in upcoming_games:
        opp = game["opponent"]

        # vs opponent history
        vs_games = pdf[pdf["OPPONENT"] == opp]
        has_matchup = len(vs_games) > 0

        vs_avg = {}
        vs_std = {}
        if has_matchup:
            for s in STATS:
                if s in vs_games.columns:
                    vs_avg[s] = float(vs_games[s].mean())
                    vs_std[s] = float(vs_games[s].std()) if len(vs_games) > 1 else season_std.get(s, 0.0)
                else:
                    vs_avg[s] = 0.0
                    vs_std[s] = 0.0

        pred = {
            "opponent": opp,
            "home": game["home"],
            "date": game.get("date", ""),
            "display_date": game.get("display_date", ""),
        }

        breakdown = {
            "season_avg": {},
            "last_3_avg": {},
            "vs_opponent_avg": {},
            "has_matchup_data": has_matchup,
            "vs_games_count": len(vs_games),
        }

        for s in STATS:
            sa = season_avg[s]
            l3 = last3_avg[s]
            breakdown["season_avg"][s] = round(sa, 1)
            breakdown["last_3_avg"][s] = round(l3, 1)

            if has_matchup:
                vo = vs_avg[s]
                breakdown["vs_opponent_avg"][s] = round(vo, 1)
                blended = (sa * W_SEASON) + (l3 * W_RECENT) + (vo * W_MATCHUP)
            else:
                breakdown["vs_opponent_avg"][s] = None
                blended = (sa * W_SEASON_NO_OPP) + (l3 * W_RECENT_NO_OPP)

            # ── Effective std dev ──
            # Blend season-wide variance (40%) with recent-form variance (60%)
            # since recent games better reflect current state (hot streak, injury, etc.)
            raw_std = (season_std[s] * 0.4) + (recent_std[s] * 0.6)

            # Shrinkage factor: our prediction is informed by 3 sources,
            # so the effective uncertainty is lower than raw game-to-game noise.
            # 0.40 brings PTS range to ~3 (full spread ~6), matching
            # typical betting lines. Higher = wider range, lower confidence.
            SHRINKAGE = 0.40
            effective_std = raw_std * SHRINKAGE

            # Clamp prediction
            if "PCT" in s:
                blended = max(0.0, min(1.0, blended))
            else:
                blended = max(0.0, blended)

            rounded = round(blended)

            # Probability of hitting at or above the rounded line
            if "PCT" in s:
                confidence = None
                range_val = round(effective_std * 100, 1)
            else:
                confidence = round(_prob_over_line(blended, rounded, effective_std), 2)
                confidence = max(0.01, min(0.99, confidence))
                range_val = round(effective_std, 1)

            pred[s] = {
                "value": round(blended, 1),
                "rounded": rounded,
                "confidence": confidence,
                "range": range_val,
            }

        pred["breakdown"] = breakdown
        predictions.append(pred)

    return predictions
