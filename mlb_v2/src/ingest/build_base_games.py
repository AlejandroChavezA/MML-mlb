"""
Descarga partidos históricos MLB desde statsapi.mlb.com y genera
data/processed/mlb_games_base.csv con scores, equipos y target HOME_WIN.

Uso:
    python src/ingest/build_base_games.py
    python src/ingest/build_base_games.py --seasons 2022 2023 2024
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
API_BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1
GAME_TYPE = "R"  # Regular Season


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    time.sleep(0.15)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_season(season: int) -> pd.DataFrame:
    print(f"  Descargando schedule {season}...")
    data = _get("/schedule", {
        "sportId": SPORT_ID,
        "season": season,
        "gameType": GAME_TYPE,
        "hydrate": "probablePitcher",
    })

    rows = []
    for date_entry in data.get("dates", []):
        date_str = date_entry["date"]
        for g in date_entry.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            finished = status == "Final"

            home = g["teams"]["home"]
            away = g["teams"]["away"]

            home_pp = home.get("probablePitcher") or {}
            away_pp = away.get("probablePitcher") or {}

            row = {
                "game_pk": g["gamePk"],
                "date": date_str,
                "season": season,
                "status": "FINISHED" if finished else "SCHEDULED",
                "home_team": home["team"]["name"],
                "away_team": away["team"]["name"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_team_code": home["team"].get("abbreviation", ""),
                "away_team_code": away["team"].get("abbreviation", ""),
                "venue": g.get("venue", {}).get("name", ""),
                "day_night": g.get("dayNight", ""),
                "doubleheader": g.get("doubleHeader", "N") == "Y",
                "home_pitcher_id": home_pp.get("id"),
                "away_pitcher_id": away_pp.get("id"),
                "home_score": home.get("score") if finished else np.nan,
                "away_score": away.get("score") if finished else np.nan,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"    {len(df)} juegos ({df['status'].value_counts().to_dict()})")
    return df


def build(seasons: list[int]) -> pd.DataFrame:
    frames = [fetch_season(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)

    finished = df["status"] == "FINISHED"
    df["total_runs"] = np.where(
        finished,
        df["home_score"] + df["away_score"],
        np.nan,
    )
    df["run_diff"] = np.where(
        finished,
        df["home_score"] - df["away_score"],
        np.nan,
    )
    df["HOME_WIN"] = np.where(
        finished,
        (df["home_score"] > df["away_score"]).astype(int),
        np.nan,
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "mlb_games_base.csv"
    df.to_csv(out, index=False)
    print(f"\n  Guardado: {out}  ({len(df)} filas)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=list(range(2015, 2026)),
                    help="Temporadas a descargar (default 2015-2025)")
    args = ap.parse_args()

    print("=== build_base_games.py ===")
    print(f"Temporadas: {args.seasons}")
    build(args.seasons)


if __name__ == "__main__":
    main()
