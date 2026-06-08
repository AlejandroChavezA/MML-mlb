"""
Verifica que los archivos procesados tienen forma y valores esperados.

Uso:
    python tests/verified_values.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

CHECKS = [
    ("mlb_games_base.csv", ["game_pk", "date", "home_team", "away_team",
                             "home_score", "away_score", "HOME_WIN"]),
    ("mlb_team_games_base.csv", ["game_pk", "date", "team", "IS_HOME",
                                  "WIN", "RUNS_FOR", "RUNS_AGAINST"]),
    ("mlb_team_games_features.csv", ["game_pk", "team", "WIN_RATE_roll10",
                                      "RUNS_FOR_roll10", "DAYS_REST"]),
    ("mlb_games_features.csv", ["HOME_WIN_RATE_roll10", "AWAY_WIN_RATE_roll10",
                                  "DIFF_WIN_RATE_roll10",
                                  "TARGET_HOME_WIN", "TARGET_TOTAL_RUNS"]),
]

passed = 0
failed = 0

for filename, required_cols in CHECKS:
    path = PROCESSED / filename
    print(f"\n  {filename}")
    if not path.exists():
        print(f"    FALTA: {path}")
        failed += 1
        continue

    df = pd.read_csv(path, nrows=5000)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"    COLUMNAS FALTANTES: {missing}")
        failed += 1
        continue

    null_pct = df[required_cols].isnull().mean().max() * 100
    print(f"    ✓ {len(df)} filas  {len(df.columns)} columnas  max_null={null_pct:.1f}%")

    if filename == "mlb_games_base.csv":
        finished = df[df.get("status", df.get("HOME_WIN", pd.Series(dtype=float)).notna()).astype(bool)]
        hw_rate = df["HOME_WIN"].mean() if "HOME_WIN" in df else None
        if hw_rate is not None:
            if not (0.45 <= hw_rate <= 0.60):
                print(f"    ADVERTENCIA: home win rate={hw_rate:.3f} (esperado 0.50-0.57)")
            else:
                print(f"    home win rate={hw_rate:.3f} OK")

    passed += 1

print(f"\n  {'='*40}")
print(f"  Pasados: {passed}  Fallados: {failed}")
sys.exit(0 if failed == 0 else 1)
