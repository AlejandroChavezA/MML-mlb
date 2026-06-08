"""
Genera data/processed/mlb_games_features.csv:
una fila por partido con prefijos HOME_ / AWAY_ y diferenciales DIFF_*.

También agrega:
  - TARGET_HOME_WIN (para modelo ganador)
  - TARGET_TOTAL_RUNS (para modelo totales)
  - TARGET_OVER_8_5  (binario O/U 8.5 carreras)

Requiere:
  data/processed/mlb_games_base.csv
  data/processed/mlb_team_games_features.csv

Uso:
    python src/features/build_game_level_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

OU_LINE = 8.5

# columnas rolling que queremos duplicar como HOME_/AWAY_
ROLL_COLS_PATTERN = [
    "WIN_RATE_roll{w}", "RUN_DIFF_roll{w}", "RUNS_FOR_roll{w}", "RUNS_AGAINST_roll{w}",
    "HITS_roll{w}", "HR_roll{w}", "WALKS_roll{w}", "SO_BAT_roll{w}",
    "SO_PIT_roll{w}", "BB_ALLOWED_roll{w}", "ERRORS_roll{w}",
]
WINDOWS = [5, 10, 20]

CONTEXT_COLS = ["DAYS_REST", "BACK_TO_BACK", "WIN_STREAK", "IS_HOME"]


def expand_patterns(patterns: list[str], windows: list[int]) -> list[str]:
    return [p.format(w=w) for w in windows for p in patterns]


def build() -> pd.DataFrame:
    games = pd.read_csv(PROCESSED_DIR / "mlb_games_base.csv", parse_dates=["date"])
    team_feat = pd.read_csv(PROCESSED_DIR / "mlb_team_games_features.csv", parse_dates=["date"])

    finished = games[games["status"] == "FINISHED"].copy()
    print(f"  {len(finished)} partidos FINISHED para cruzar")

    roll_cols = expand_patterns(ROLL_COLS_PATTERN, WINDOWS) + CONTEXT_COLS

    # columnas que efectivamente existen
    available = [c for c in roll_cols if c in team_feat.columns]

    def get_team_features(team_col: str, game_pk_col: str = "game_pk") -> pd.DataFrame:
        merged = finished[["game_pk", "date", team_col]].merge(
            team_feat[["game_pk", "team"] + available],
            left_on=["game_pk", team_col],
            right_on=["game_pk", "team"],
            how="left",
        ).drop(columns=["team"])
        return merged

    home_feats = get_team_features("home_team")
    away_feats = get_team_features("away_team")

    def prefix(df: pd.DataFrame, pfx: str) -> pd.DataFrame:
        rename = {c: f"{pfx}{c}" for c in available}
        return df[["game_pk"] + available].rename(columns=rename)

    df = finished.copy()
    df = df.merge(prefix(home_feats, "HOME_"), on="game_pk", how="left")
    df = df.merge(prefix(away_feats, "AWAY_"), on="game_pk", how="left")

    # diferenciales
    for w in WINDOWS:
        for pat in ROLL_COLS_PATTERN:
            col = pat.format(w=w)
            h, a = f"HOME_{col}", f"AWAY_{col}"
            if h in df.columns and a in df.columns:
                df[f"DIFF_{col}"] = df[h] - df[a]

    df["DIFF_DAYS_REST"] = df["HOME_DAYS_REST"] - df["AWAY_DAYS_REST"]

    # targets
    df["TARGET_HOME_WIN"] = (df["home_score"] > df["away_score"]).astype(int)
    df["TARGET_TOTAL_RUNS"] = df["home_score"] + df["away_score"]
    df["TARGET_OVER_8_5"] = (df["TARGET_TOTAL_RUNS"] > OU_LINE).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    out = PROCESSED_DIR / "mlb_games_features.csv"
    df.to_csv(out, index=False)
    print(f"  Guardado: {out}  ({len(df)} filas, {len(df.columns)} columnas)")
    return df


def main():
    print("=== build_game_level_features.py ===")
    build()


if __name__ == "__main__":
    main()
