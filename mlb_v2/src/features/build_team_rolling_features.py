"""
Genera data/processed/mlb_team_games_features.csv con rolling stats
calculadas ANTES de cada partido (shift(1) garantiza no data leakage).

Ventanas: 5, 10, 20 partidos.

Métricas rolling:
  WIN_RATE, MARGIN, RUNS_FOR, RUNS_AGAINST
  HITS, HR, WALKS, SO_BAT (ofensiva)
  SO_PIT, BB_ALLOWED, ERRORS (pitcheo/defensa)
  DAYS_REST, BACK_TO_BACK

Requiere: data/processed/mlb_team_games_base.csv

Uso:
    python src/features/build_team_rolling_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

WINDOWS = [5, 10, 20]

NUMERIC_COLS = [
    "WIN", "RUN_DIFF", "RUNS_FOR", "RUNS_AGAINST",
    "HITS", "HR", "WALKS", "SO_BAT",
    "SO_PIT", "BB_ALLOWED", "ERRORS",
]


def compute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team", "date"]).copy()

    # días de descanso
    df["DAYS_REST"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .fillna(3)
        .clip(0, 15)
    )
    df["BACK_TO_BACK"] = (df["DAYS_REST"] == 1).astype(int)

    feature_frames = [df[["game_pk", "team", "date", "season",
                           "IS_HOME", "WIN", "RUNS_FOR", "RUNS_AGAINST",
                           "RUN_DIFF", "DAYS_REST", "BACK_TO_BACK",
                           "pitcher_id", "venue", "day_night"]].copy()]

    for w in WINDOWS:
        grp = df.groupby("team")
        rolled = {}
        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            # shift(1) → solo datos anteriores al partido actual
            rolled[f"{col}_roll{w}"] = (
                grp[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        # win_rate explícita (media de WIN ya la tenemos, pero nombramos aparte)
        rolled[f"WIN_RATE_roll{w}"] = rolled.pop(f"WIN_roll{w}")

        feature_frames.append(pd.DataFrame(rolled, index=df.index))

    # racha activa
    def streak(wins: pd.Series) -> pd.Series:
        s = wins.shift(1).fillna(0).astype(int)
        result = []
        current = 0
        for v in s:
            if v == 1:
                current = max(1, current + 1)
            else:
                current = min(-1, current - 1)
            result.append(current)
        return pd.Series(result, index=wins.index)

    df["WIN_STREAK"] = df.groupby("team")["WIN"].transform(streak)
    feature_frames.append(df[["WIN_STREAK"]])

    out = pd.concat(feature_frames, axis=1)
    # quitar columnas duplicadas
    out = out.loc[:, ~out.columns.duplicated()]
    return out.reset_index(drop=True)


def main():
    src = PROCESSED_DIR / "mlb_team_games_base.csv"
    if not src.exists():
        raise FileNotFoundError(f"Ejecuta primero build_team_games_base.py — no encontrado: {src}")

    print("=== build_team_rolling_features.py ===")
    df = pd.read_csv(src, parse_dates=["date"])
    print(f"  {len(df)} filas leídas")

    out_df = compute_rolling(df)

    dst = PROCESSED_DIR / "mlb_team_games_features.csv"
    out_df.to_csv(dst, index=False)
    print(f"  Guardado: {dst}  ({len(out_df)} filas, {len(out_df.columns)} columnas)")


if __name__ == "__main__":
    main()
