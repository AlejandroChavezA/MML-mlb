"""
Genera data/processed/mlb_team_games_base.csv:
una fila por equipo por partido con box score completo.

Requiere: data/processed/mlb_games_base.csv

Uso:
    python src/ingest/build_team_games_base.py                 # con boxscores (~4 min)
    python src/ingest/build_team_games_base.py --no-boxscore   # solo scores (~5 seg)
    python src/ingest/build_team_games_base.py --workers 20    # más threads (default 15)
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_FILE = PROCESSED_DIR / ".boxscore_cache.json"
API_BASE = "https://statsapi.mlb.com/api/v1"

# un session por thread (thread-local)
import threading
_local = threading.local()

def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = "MML-MLB/2.0"
        _local.session = s
    return _local.session


def _get(endpoint: str) -> dict:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    r = _session().get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_boxscore(game_pk: int) -> dict:
    try:
        data = _get(f"/game/{game_pk}/boxscore")
        result = {}
        for side in ("home", "away"):
            td = data.get("teams", {}).get(side, {})
            batting = td.get("teamStats", {}).get("batting", {})
            pitching = td.get("teamStats", {}).get("pitching", {})
            fielding = td.get("teamStats", {}).get("fielding", {})
            result[side] = {
                "hits": batting.get("hits", np.nan),
                "doubles": batting.get("doubles", np.nan),
                "triples": batting.get("triples", np.nan),
                "home_runs": batting.get("homeRuns", np.nan),
                "rbi": batting.get("rbi", np.nan),
                "walks": batting.get("baseOnBalls", np.nan),
                "strikeouts_bat": batting.get("strikeOuts", np.nan),
                "left_on_base": batting.get("leftOnBase", np.nan),
                "era": pitching.get("era", np.nan),
                "strikeouts_pit": pitching.get("strikeOuts", np.nan),
                "walks_allowed": pitching.get("baseOnBalls", np.nan),
                "errors": fielding.get("errors", np.nan),
            }
        return result
    except Exception:
        return {}


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def fetch_all_boxscores(game_pks: list[int], workers: int = 15) -> dict:
    cache = _load_cache()
    pending = [pk for pk in game_pks if str(pk) not in cache]

    if not pending:
        print(f"  Cache completo ({len(cache)} boxscores)")
        return cache

    print(f"  Cache: {len(cache)} guardados. Descargando {len(pending)} restantes "
          f"con {workers} threads...")

    done = 0
    save_every = 500

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_boxscore, pk): pk for pk in pending}
        for future in as_completed(futures):
            pk = futures[future]
            cache[str(pk)] = future.result()
            done += 1
            if done % save_every == 0:
                _save_cache(cache)
                print(f"    {done}/{len(pending)} ({done/len(pending):.0%})")

    _save_cache(cache)
    print(f"  Descarga completada: {len(pending)} boxscores")
    return cache


def build(use_boxscore: bool = True, workers: int = 15) -> pd.DataFrame:
    base_path = PROCESSED_DIR / "mlb_games_base.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Ejecuta primero build_base_games.py — no encontrado: {base_path}")

    games = pd.read_csv(base_path, parse_dates=["date"])
    finished = games[games["status"] == "FINISHED"].copy()
    print(f"  {len(finished)} partidos FINISHED")

    box_cache: dict = {}

    if use_boxscore:
        game_pks = finished["game_pk"].astype(int).tolist()
        box_cache = fetch_all_boxscores(game_pks, workers=workers)

    rows = []
    for _, g in finished.iterrows():
        pk = int(g["game_pk"])
        box = box_cache.get(str(pk), {}) if use_boxscore else {}

        for side, opp_side in (("home", "away"), ("away", "home")):
            is_home = side == "home"
            b = box.get(side, {})
            score = g[f"{side}_score"]
            opp_score = g[f"{opp_side}_score"]

            rows.append({
                "game_pk": pk,
                "date": g["date"],
                "season": g["season"],
                "team": g[f"{side}_team"],
                "team_id": g[f"{side}_team_id"],
                "team_code": g[f"{side}_team_code"],
                "opponent": g[f"{opp_side}_team"],
                "opponent_id": g[f"{opp_side}_team_id"],
                "IS_HOME": int(is_home),
                "WIN": int(score > opp_score),
                "RUNS_FOR": score,
                "RUNS_AGAINST": opp_score,
                "RUN_DIFF": score - opp_score,
                "HITS": b.get("hits", np.nan),
                "DOUBLES": b.get("doubles", np.nan),
                "TRIPLES": b.get("triples", np.nan),
                "HR": b.get("home_runs", np.nan),
                "RBI": b.get("rbi", np.nan),
                "WALKS": b.get("walks", np.nan),
                "SO_BAT": b.get("strikeouts_bat", np.nan),
                "LOB": b.get("left_on_base", np.nan),
                "SO_PIT": b.get("strikeouts_pit", np.nan),
                "BB_ALLOWED": b.get("walks_allowed", np.nan),
                "ERRORS": b.get("errors", np.nan),
                "pitcher_id": g[f"{side}_pitcher_id"],
                "venue": g["venue"],
                "day_night": g["day_night"],
            })

    df = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    out = PROCESSED_DIR / "mlb_team_games_base.csv"
    df.to_csv(out, index=False)
    print(f"\n  Guardado: {out}  ({len(df)} filas)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-boxscore", action="store_true",
                    help="Saltar descarga de boxscores (solo scores, ~5 seg)")
    ap.add_argument("--workers", type=int, default=15,
                    help="Threads concurrentes para descargar boxscores (default 15)")
    args = ap.parse_args()

    print("=== build_team_games_base.py ===")
    build(use_boxscore=not args.no_boxscore, workers=args.workers)


if __name__ == "__main__":
    main()
