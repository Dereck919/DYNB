# data/fetcher.py
#
# Three jobs:
#   1. Fetch current-season game logs (for training + features)
#   2. Fetch upcoming schedule (next 5 games for a team)
#   3. Search any NBA player by name

import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as player_db
from config import PLAYERS, CURRENT_SEASON, TEAM_IDS, TEAM_ID_TO_ABBR


# ── Player search ──

def search_players(query):
    """Search NBA players by name. Returns up to 15 results."""
    query = query.strip().lower()
    if len(query) < 2:
        return []
    matches = []
    for p in player_db.get_players():
        if query in p["full_name"].lower():
            matches.append({
                "id": str(p["id"]),
                "name": p["full_name"],
                "active": p["is_active"],
            })
    matches.sort(key=lambda x: (not x["active"], x["name"]))
    return matches[:15]


# ── Game logs ──

def fetch_current_season(player_id, player_name="Unknown"):
    """
    Fetch this season's game log for one player.
    Pulls both Regular Season and Playoffs so playoff games
    automatically appear once they begin.
    """
    frames = []
    for stype in ["Regular Season", "Playin", "Playoffs"]:
        try:
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=CURRENT_SEASON,
                season_type_all_star=stype,
            )
            df = log.get_data_frames()[0]
            if not df.empty:
                frames.append(df)
        except Exception:
            # Playoffs call will fail/return empty before postseason starts
            pass

    if not frames:
        print(f"  Could not fetch {player_name}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["PLAYER_NAME"] = player_name
    return combined


def fetch_all_default_players():
    """Fetch current season game logs for all players in config."""
    frames = []
    for name, pid in PLAYERS.items():
        print(f"  {name} ...", end=" ", flush=True)
        df = fetch_current_season(pid, name)
        if not df.empty:
            frames.append(df)
            print(f"{len(df)} games")
        else:
            print("no data")
        time.sleep(0.6)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Opponent parsing ──

def parse_opponent(matchup):
    """'DEN vs. LAL' -> 'LAL', 'DEN @ LAL' -> 'LAL'"""
    if not isinstance(matchup, str):
        return "UNK"
    if "vs." in matchup:
        return matchup.split("vs.")[-1].strip()
    if "@" in matchup:
        return matchup.split("@")[-1].strip()
    return "UNK"


def is_home_game(matchup):
    """'DEN vs. LAL' -> True, 'DEN @ LAL' -> False"""
    return "vs." in str(matchup)


# ── Schedule ──

_schedule_cache = None

def _fetch_nba_schedule():
    """
    Pull the full NBA schedule from the public CDN.
    Cached so we only call it once per server session.
    """
    global _schedule_cache
    if _schedule_cache is not None:
        return _schedule_cache

    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    try:
        print("  Fetching NBA schedule ...", end=" ", flush=True)
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        resp.raise_for_status()
        data = resp.json()
        _schedule_cache = data
        print("done")
        return data
    except Exception as e:
        print(f"failed: {e}")
        return None


def get_upcoming_games(team_abbr, n=5):
    """
    Get the next N upcoming games for a team.

    Returns a list of dicts:
    [
        {"date": "2025-03-25", "opponent": "LAL", "home": True},
        {"date": "2025-03-27", "opponent": "GSW", "home": False},
        ...
    ]

    Falls back to an empty list if the schedule can't be fetched.
    """
    schedule_data = _fetch_nba_schedule()
    if not schedule_data:
        return []

    team_id = TEAM_IDS.get(team_abbr)
    if not team_id:
        return []

    now = datetime.utcnow()
    upcoming = []

    try:
        game_dates = schedule_data.get("leagueSchedule", {}).get("gameDates", [])
        for date_block in game_dates:
            games = date_block.get("games", [])
            for game in games:
                # Parse game date
                game_date_str = game.get("gameDateTimeUTC", "")
                if not game_date_str:
                    continue
                try:
                    game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                    game_dt = game_dt.replace(tzinfo=None)
                except Exception:
                    continue

                # Only future games
                if game_dt <= now:
                    continue

                home_id = game.get("homeTeam", {}).get("teamId")
                away_id = game.get("awayTeam", {}).get("teamId")

                if home_id == team_id:
                    opp_id = away_id
                    is_home = True
                elif away_id == team_id:
                    opp_id = home_id
                    is_home = False
                else:
                    continue

                opp_abbr = TEAM_ID_TO_ABBR.get(opp_id, "???")
                upcoming.append({
                    "date": game_dt.strftime("%Y-%m-%d"),
                    "display_date": game_dt.strftime("%b %d"),
                    "opponent": opp_abbr,
                    "home": is_home,
                })

                if len(upcoming) >= n:
                    return upcoming

    except Exception as e:
        print(f"  Schedule parsing error: {e}")

    return upcoming
