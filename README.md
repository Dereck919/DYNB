# Do You Know Ball?

Next-game NBA stat predictor. Blends season averages, recent form, and opponent matchup history to predict stats for upcoming games.

## Quick Start

```bash
cd DYNB
pip3 install -r requirements.txt
python3 -m uvicorn api:app --reload
```

Open **http://localhost:8000**. Ready in ~20 seconds.

## How Predictions Work

Each prediction blends three factors:

```
50%  Season average    (what the player does overall)
30%  Last 3 games      (are they hot or cold right now?)
20%  vs Opponent       (how do they play against this team?)
```

If no matchup history exists: 60% season + 40% last 3.

Example: Luka averages 33 PTS on the season, dropped 44.3 over his last 3, and averages 38 vs the Lakers.

```
prediction = 33 * 0.5 + 44.3 * 0.3 + 38 * 0.2 = 37.4 PTS
```

Click any prediction card to expand the breakdown and see all three components.

Weights are in `model.py` if you want to tweak them.

## Project Structure

```
nba_predictor/
    config.py           <- Players, season, team IDs
    api.py              <- Web server
    model.py            <- Weighted blend prediction
    features.py         <- Season averages, recent form, matchup history
    data/fetcher.py     <- NBA API + schedule + search
    frontend/index.html <- Dashboard
    requirements.txt
```

## Features

- **Player search** -- type any name, data is fetched on demand
- **Real schedule** -- pulls upcoming games from NBA's public schedule API
- **Opponent-specific predictions** -- each game's prediction factors in matchup history
- **Expandable breakdown** -- click a prediction to see how it was calculated
- **Recent form** -- last 3 games shown with opponent, W/L, and stats
- **Rounded values** -- 37.4 (37) for betting line reference

## API

| Endpoint | What |
|---|---|
| `GET /` | Dashboard |
| `GET /api/players` | Default player predictions |
| `GET /api/search?q=curry` | Player search |
| `GET /api/predict?id=201939&name=Stephen Curry` | Predict for any player |

## Notes

- Python 3.9+ compatible
- No Node.js, no scikit-learn, no ML training step
- Season set to 2025-26 in config.py (update each year)
- Schedule from cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
