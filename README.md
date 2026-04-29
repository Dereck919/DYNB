# Do You Know Ball?

Next-game NBA stat predictor built for sports betting and fantasy leagues. Uses machine learning to predict player stats for upcoming games based on season performance, recent form, and opponent matchup history.

## Quick Start
```bash
cd dynb
pip3 install -r requirements.txt
python3 -m uvicorn api:app --reload
```

Open **http://localhost:8000**

## How It Works

A Gradient Boosting model (scikit-learn) is trained on every game from the current season. For each game, the model learns from what we knew before it happened:

- **Season averages** up to that point
- **Last 3 game averages** (recent hot/cold streaks)
- **Most recent game vs that specific opponent** (matchup tendencies)
- **Home/away**, days of rest, games played

One model is trained per stat (PTS, AST, REB, STL, BLK, FG%, 3P%). The model learns the optimal way to weight these inputs rather than using hardcoded ratios.

For upcoming games, the app pulls the real NBA schedule, identifies the next 5 opponents, and generates a prediction for each. The prediction for "vs LAL" differs from "@ GSW" because the opponent feature changes.

## Predictions Format

Each prediction shows:

- **35 (34.8) PTS** -- rounded line first, precise value in parentheses
- **93%** -- confidence the player hits at or above the rounded line (green 70%+ = strong, yellow 45-69% = coin flip, red <45% = unlikely)
- **±3.8** -- expected range based on player consistency

At standard -110 odds you need 52.4% accuracy to profit. Anything showing 55%+ is a real lean.

Click any prediction row to expand the breakdown showing season average, last 3 average, and vs-opponent average that fed into the model.

## Features

- **Player search** -- search any NBA player by name, data fetched on demand
- **Real schedule** -- pulls upcoming games from NBA's public schedule API
- **ML predictions** -- GradientBoostingRegressor trained per stat on current season data
- **Opponent-specific** -- each game uses matchup history as a model feature
- **Confidence + range** -- probability of clearing the line with consistency spread
- **Expandable breakdown** -- see what inputs drove each prediction
- **Refresh button** -- re-fetch latest game data without restarting the server

## Project Structure
```
dynb/
    config.py           <- Players, season, team IDs, model params
    api.py              <- FastAPI web server
    model.py            <- ML training + prediction + confidence calc
    features.py         <- Season averages, recent form, matchup history
    data/fetcher.py     <- NBA API calls, schedule, player search
    frontend/index.html <- Dashboard (single file, no build tools)
    requirements.txt
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard |
| `GET /api/players` | Default player predictions |
| `GET /api/search?q=curry` | Search any player |
| `GET /api/predict?id=201939&name=Stephen Curry` | On-demand prediction |
| `GET /api/refresh` | Re-fetch game data and retrain models |
| `GET /api/health` | Server status |

## Tech Stack

- **Python 3.9+** -- no modern type hint syntax
- **scikit-learn** -- GradientBoostingRegressor for per-stat models
- **nba_api** -- current season game logs
- **FastAPI + Uvicorn** -- API server
- **React 18 (CDN)** -- frontend, no Node.js required
- **NBA CDN** -- real upcoming schedule data

## Adding Players

Edit `config.py`:
```python
PLAYERS["Luka Doncic"] = "1629029"
TEAM_INFO["Luka Doncic"] = {"team": "DAL", "pos": "PG", "number": 77}
```

Or just search for them in the dashboard.
## Live Demo
 **https://dynb.onrender.com/**
## Notes

- First launch takes ~30 seconds (fetch game logs + train models)
- Game data from nba_api may lag a few hours after games end
- Use the refresh button to pull latest data without restarting
- Season is set to 2025-26 in config.py (update each year)
- Model retrains automatically on refresh with latest game data