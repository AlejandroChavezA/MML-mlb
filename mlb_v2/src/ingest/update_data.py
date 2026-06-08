"""
Actualización incremental de datos MLB.

Descarga solo los partidos nuevos desde el último FINISHED en el CSV
hasta ayer, los agrega a mlb_games_base.csv y regenera los archivos
procesados aguas abajo (team_games_base, rolling features, game features).

Uso:
    python src/ingest/update_data.py            # descarga hasta ayer
    python src/ingest/update_data.py --full     # re-descarga todo (lento)
    python src/ingest/update_data.py --dry-run  # muestra qué descargaría
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import local as thread_local

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_FILE = PROCESSED_DIR / ".boxscore_cache.json"
API_BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1

_local = thread_local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "MML-MLB/2.0"
        _local.s = s
    return _local.s


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    time.sleep(0.1)
    r = _session().get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────
# Descarga de schedule
# ──────────────────────────────────────────────

def fetch_games_range(start: date, end: date) -> pd.DataFrame:
    """Descarga todos los juegos entre start y end (inclusive)."""
    data = _get("/schedule", {
        "sportId": SPORT_ID,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "gameType": "R,E",
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

            rows.append({
                "game_pk":         g["gamePk"],
                "date":            date_str,
                "season":          int(g.get("season", date_str[:4])),
                "status":          "FINISHED" if finished else "SCHEDULED",
                "home_team":       home["team"]["name"],
                "away_team":       away["team"]["name"],
                "home_team_id":    home["team"]["id"],
                "away_team_id":    away["team"]["id"],
                "home_team_code":  home["team"].get("abbreviation", ""),
                "away_team_code":  away["team"].get("abbreviation", ""),
                "venue":           g.get("venue", {}).get("name", ""),
                "day_night":       g.get("dayNight", ""),
                "doubleheader":    g.get("doubleHeader", "N") == "Y",
                "home_pitcher_id": home_pp.get("id"),
                "away_pitcher_id": away_pp.get("id"),
                "home_score":      home.get("score") if finished else np.nan,
                "away_score":      away.get("score") if finished else np.nan,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df["total_runs"] = np.where(
            df["status"] == "FINISHED",
            df["home_score"] + df["away_score"], np.nan,
        )
        df["run_diff"] = np.where(
            df["status"] == "FINISHED",
            df["home_score"] - df["away_score"], np.nan,
        )
        df["HOME_WIN"] = np.where(
            df["status"] == "FINISHED",
            (df["home_score"] > df["away_score"]).astype(float), np.nan,
        )
    return df


# ──────────────────────────────────────────────
# Boxscores paralelos con caché
# ──────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def fetch_boxscore(game_pk: int) -> dict:
    try:
        data = _get(f"/game/{game_pk}/boxscore")
        result = {}
        for side in ("home", "away"):
            td = data.get("teams", {}).get(side, {})
            batting  = td.get("teamStats", {}).get("batting", {})
            pitching = td.get("teamStats", {}).get("pitching", {})
            fielding = td.get("teamStats", {}).get("fielding", {})
            result[side] = {
                "hits":           batting.get("hits", np.nan),
                "doubles":        batting.get("doubles", np.nan),
                "triples":        batting.get("triples", np.nan),
                "home_runs":      batting.get("homeRuns", np.nan),
                "rbi":            batting.get("rbi", np.nan),
                "walks":          batting.get("baseOnBalls", np.nan),
                "strikeouts_bat": batting.get("strikeOuts", np.nan),
                "left_on_base":   batting.get("leftOnBase", np.nan),
                "strikeouts_pit": pitching.get("strikeOuts", np.nan),
                "walks_allowed":  pitching.get("baseOnBalls", np.nan),
                "errors":         fielding.get("errors", np.nan),
            }
        return result
    except Exception:
        return {}


def enrich_boxscores(game_pks: list[int], workers: int = 15) -> dict:
    cache = _load_cache()
    pending = [pk for pk in game_pks if str(pk) not in cache]

    if not pending:
        return cache

    print(f"  Descargando {len(pending)} boxscores nuevos ({workers} threads)...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_boxscore, pk): pk for pk in pending}
        for fut in as_completed(futures):
            pk = futures[fut]
            cache[str(pk)] = fut.result()
            done += 1
            if done % 200 == 0:
                _save_cache(cache)
                print(f"    {done}/{len(pending)}")

    _save_cache(cache)
    return cache


# ──────────────────────────────────────────────
# Merge incremental
# ──────────────────────────────────────────────

def merge_into_base(new_df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """Une nuevos partidos al CSV base evitando duplicados."""
    if base_path.exists():
        existing = pd.read_csv(base_path, parse_dates=["date"])
    else:
        existing = pd.DataFrame()

    if len(existing) == 0:
        combined = new_df
    else:
        existing_pks = set(existing["game_pk"].tolist())
        truly_new = new_df[~new_df["game_pk"].isin(existing_pks)]

        # Actualizar partidos que antes eran SCHEDULED y ahora son FINISHED
        scheduled_mask = existing["status"] == "SCHEDULED"
        became_finished = new_df[
            new_df["game_pk"].isin(existing[scheduled_mask]["game_pk"])
            & (new_df["status"] == "FINISHED")
        ]
        if len(became_finished) > 0:
            existing = existing[~existing["game_pk"].isin(became_finished["game_pk"])]
            existing = pd.concat([existing, became_finished], ignore_index=True)
            print(f"  Actualizados {len(became_finished)} partidos SCHEDULED→FINISHED")

        combined = pd.concat([existing, truly_new], ignore_index=True)

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(base_path, index=False)
    return combined


def build_team_games(games_df: pd.DataFrame, box_cache: dict) -> pd.DataFrame:
    finished = games_df[games_df["status"] == "FINISHED"].copy()
    rows = []
    for _, g in finished.iterrows():
        pk = str(int(g["game_pk"]))
        box = box_cache.get(pk, {})
        for side, opp in (("home", "away"), ("away", "home")):
            is_home = side == "home"
            b = box.get(side, {})
            score = g[f"{side}_score"]
            opp_score = g[f"{opp}_score"]
            rows.append({
                "game_pk":    int(g["game_pk"]),
                "date":       g["date"],
                "season":     g["season"],
                "team":       g[f"{side}_team"],
                "team_id":    g[f"{side}_team_id"],
                "team_code":  g[f"{side}_team_code"],
                "opponent":   g[f"{opp}_team"],
                "opponent_id":g[f"{opp}_team_id"],
                "IS_HOME":    int(is_home),
                "WIN":        int(score > opp_score),
                "RUNS_FOR":   score,
                "RUNS_AGAINST": opp_score,
                "RUN_DIFF":   score - opp_score,
                "HITS":       b.get("hits", np.nan),
                "DOUBLES":    b.get("doubles", np.nan),
                "TRIPLES":    b.get("triples", np.nan),
                "HR":         b.get("home_runs", np.nan),
                "RBI":        b.get("rbi", np.nan),
                "WALKS":      b.get("walks", np.nan),
                "SO_BAT":     b.get("strikeouts_bat", np.nan),
                "LOB":        b.get("left_on_base", np.nan),
                "SO_PIT":     b.get("strikeouts_pit", np.nan),
                "BB_ALLOWED": b.get("walks_allowed", np.nan),
                "ERRORS":     b.get("errors", np.nan),
                "pitcher_id": g[f"{side}_pitcher_id"],
                "venue":      g["venue"],
                "day_night":  g["day_night"],
            })
    return pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────
# Pipeline downstream (rolling + game features)
# ──────────────────────────────────────────────

def _import_from(rel_path: str, attr: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        attr, ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def rebuild_features():
    """Regenera rolling y game-level features desde cero."""
    print("\n  Regenerando rolling features...")
    compute_rolling = _import_from("src/features/build_team_rolling_features.py", "compute_rolling")
    build_game_feats = _import_from("src/features/build_game_level_features.py", "build")

    team_base = pd.read_csv(PROCESSED_DIR / "mlb_team_games_base.csv", parse_dates=["date"])
    rolling = compute_rolling(team_base)
    rolling.to_csv(PROCESSED_DIR / "mlb_team_games_features.csv", index=False)
    print(f"    mlb_team_games_features.csv: {len(rolling)} filas")

    print("  Regenerando game-level features...")
    game_feats = build_game_feats()
    print(f"    mlb_games_features.csv: {len(game_feats)} filas")


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────

def update(full: bool = False, dry_run: bool = False, workers: int = 15):
    yesterday = date.today() - timedelta(days=1)
    base_path = PROCESSED_DIR / "mlb_games_base.csv"

    if full or not base_path.exists():
        start = date(2015, 4, 1)
        print(f"  Modo completo: {start} → {yesterday}")
    else:
        existing = pd.read_csv(base_path, parse_dates=["date"])
        finished = existing[existing["status"] == "FINISHED"]
        if len(finished) == 0:
            start = date(2015, 4, 1)
        else:
            last_finished = finished["date"].max().date()
            # Descargamos desde el día siguiente al último FINISHED
            # más un margen de 3 días para capturar cualquier SCHEDULED→FINISHED
            start = last_finished - timedelta(days=3)
        print(f"  Modo incremental: {start} → {yesterday}")
        print(f"  Último FINISHED en CSV: {last_finished}")

    if dry_run:
        print(f"\n  [DRY-RUN] Descargaría: {start} → {yesterday}")
        total_days = (yesterday - start).days + 1
        print(f"  Aproximadamente {total_days} días (~{total_days * 15 / 30:.0f} juegos)")
        return

    # 1. Descargar schedule nuevo
    print(f"\n  Descargando schedule {start} → {yesterday}...")
    new_games = fetch_games_range(start, yesterday)
    finished_new = new_games[new_games["status"] == "FINISHED"]
    print(f"  Descargados: {len(new_games)} juegos ({len(finished_new)} FINISHED)")

    if len(finished_new) == 0:
        print("  Sin juegos nuevos FINISHED. Nada que hacer.")
        return

    # 2. Merge en base
    print("\n  Actualizando mlb_games_base.csv...")
    all_games = merge_into_base(new_games, base_path)
    all_finished = all_games[all_games["status"] == "FINISHED"]
    print(f"  Total en CSV: {len(all_games)} ({len(all_finished)} FINISHED)")
    print(f"  Cobertura: {all_games['date'].min().date()} → {all_games['date'].max().date()}")

    # 3. Boxscores de partidos nuevos
    new_pks = finished_new["game_pk"].astype(int).tolist()
    box_cache = enrich_boxscores(new_pks, workers=workers)

    # 4. Reconstruir team_games_base completo
    print("\n  Reconstruyendo mlb_team_games_base.csv...")
    full_cache = _load_cache()
    team_df = build_team_games(all_finished, full_cache)
    team_df.to_csv(PROCESSED_DIR / "mlb_team_games_base.csv", index=False)
    print(f"  mlb_team_games_base.csv: {len(team_df)} filas")

    # 5. Rolling + game features
    rebuild_features()

    print(f"\n  Datos actualizados hasta: {all_finished['date'].max().date()}")
    print("  Listo. Puedes reentrenar con: python src/models/train.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full",    action="store_true", help="Re-descarga todo desde 2015")
    ap.add_argument("--dry-run", action="store_true", help="Solo muestra qué descargaría")
    ap.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()

    print("=== update_data.py — MLB Incremental Update ===")
    update(full=args.full, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
