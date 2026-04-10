# api.py
#
# Run with: python3 -m uvicorn api:app --reload
# Open: http://localhost:8000
#
# No ML training step. Just fetches game logs and computes
# weighted predictions on the fly. Starts in ~20 seconds.

from pathlib import Path
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from fetcher import (
    fetch_all_default_players,
    fetch_current_season,
    search_players,
    get_upcoming_games,
)
from features import get_season_averages, get_last_n_games
from model import train_models, predict_games
from config import PLAYERS, TEAM_INFO, STATS

app = FastAPI(title="Do You Know Ball?", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {"game_logs": None}


_cache = {"game_logs": None, "models": None}


def _load():
    if _cache["models"] is not None:
        return

    print("Fetching current season game logs ...")
    _cache["game_logs"] = fetch_all_default_players()
    print()

    print("Training models ...")
    models, metrics = train_models(_cache["game_logs"])
    _cache["models"] = models
    print(f"\nReady! Open http://localhost:8000\n")


@app.on_event("startup")
def startup():
    _load()


def _player_response(name, game_logs, info=None):
    """Build the full JSON response for one player."""
    if info is None:
        info = TEAM_INFO.get(name, {"team": "???", "pos": "??", "number": 0})

    # Season averages
    season_avg = get_season_averages(game_logs, name)

    # Last 3 games
    recent = get_last_n_games(game_logs, name, n=3)

    # Upcoming schedule
    team = info.get("team", "???")
    upcoming = get_upcoming_games(team, n=5)

    # Predictions for upcoming games
    predictions = predict_games(game_logs, _cache["models"], name, upcoming)
    # Format season averages
    formatted_avg = {}
    for s in STATS:
        val = season_avg.get(s, 0.0)
        formatted_avg[s] = {"value": round(val, 1), "rounded": round(val)}

    return {
        "team": info.get("team", "???"),
        "pos": info.get("pos", "??"),
        "number": info.get("number", 0),
        "games_played": season_avg.get("GP", 0),
        "season_avg": formatted_avg,
        "recent_games": recent,
        "upcoming": predictions,
    }


@app.get("/api/players")
def get_players():
    _load()
    result = {}
    for name in PLAYERS:
        info = TEAM_INFO.get(name)
        result[name] = _player_response(name, _cache["game_logs"], info)
    return result


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=2)):
    return search_players(q)


@app.get("/api/predict")
def api_predict(
    id: str = Query(...),
    name: str = Query("Unknown"),
):
    """On-demand prediction for any player."""
    _load()
    logs = _cache["game_logs"]

    # Fetch if not already cached
    if name not in logs["PLAYER_NAME"].values:
        print(f"  Fetching {name} on demand ...")
        new = fetch_current_season(id, name)
        if new.empty:
            return {"error": f"No games found for {name} this season"}
        _cache["game_logs"] = pd.concat([logs, new], ignore_index=True)
        logs = _cache["game_logs"]

    # Guess team from most recent game
    pdf = logs[logs["PLAYER_NAME"] == name]
    team = "???"
    if not pdf.empty:
        last_matchup = pdf.iloc[0]["MATCHUP"]  # game log is reverse chron
        if "vs." in str(last_matchup):
            team = str(last_matchup).split("vs.")[0].strip().split()[-1]
        elif "@" in str(last_matchup):
            team = str(last_matchup).split("@")[0].strip().split()[-1]

    info = {"team": team, "pos": "??", "number": 0}
    result = _player_response(name, logs, info)
    result["_player_id"] = id
    return result

@app.get("/api/refresh")
def refresh():
    global _cache
    _cache = {"game_logs": None, "models": None}
    _load()
    return {"status": "refreshed", "games": len(_cache["game_logs"])}

@app.get("/api/health")
def health():
    n = len(_cache["game_logs"]) if _cache["game_logs"] is not None else 0
    return {"status": "ok", "games": n}


@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")
