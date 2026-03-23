# config.py

PLAYERS = {
    "Nikola Jokic":             "203999",
    "Shai Gilgeous-Alexander":  "1628983",
    "Kawhi Leonard":            "202695",
    "LeBron James":             "2544",
    "Stephen Curry":            "201939",
    "Kevin Durant":             "201142",
    "Victor Wembanyama":        "1641705",
}

TEAM_INFO = {
    "Nikola Jokic":             {"team": "DEN", "pos": "C",  "number": 15},
    "Shai Gilgeous-Alexander":  {"team": "OKC", "pos": "PG", "number": 2},
    "Kawhi Leonard":            {"team": "LAC", "pos": "SF", "number": 2},
    "LeBron James":             {"team": "LAL", "pos": "SF", "number": 23},
    "Stephen Curry":            {"team": "GSW", "pos": "PG", "number": 30},
    "Kevin Durant":             {"team": "PHX", "pos": "PF", "number": 35},
    "Victor Wembanyama":        {"team": "SAS", "pos": "C",  "number": 1},
}

STATS = ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT"]
CURRENT_SEASON = "2025-26"

# NBA team ID mapping (used for schedule lookup)
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

# Reverse: team_id -> abbreviation
TEAM_ID_TO_ABBR = {v: k for k, v in TEAM_IDS.items()}

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}
