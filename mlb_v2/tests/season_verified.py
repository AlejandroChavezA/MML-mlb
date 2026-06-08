"""
Verifica cobertura por temporada en mlb_games_base.csv.

Uso:
    python tests/season_verified.py
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

MIN_GAMES_PER_SEASON = 1200  # Regular season MLB ≈ 2430 por temporada completa


def main():
    path = PROCESSED / "mlb_games_base.csv"
    if not path.exists():
        print(f"  No encontrado: {path}")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["status"] == "FINISHED"]

    if "season" not in df.columns:
        df["season"] = df["date"].dt.year

    print(f"\n  Cobertura por temporada ({len(df)} juegos FINISHED total)")
    print(f"  {'─'*40}")
    print(f"  {'Temporada':>10}  {'Juegos':>8}  {'Estado':>10}")
    print(f"  {'─'*40}")

    failed = 0
    for season in sorted(df["season"].unique()):
        n = (df["season"] == season).sum()
        ok = n >= MIN_GAMES_PER_SEASON
        status = "✓ OK" if ok else f"⚠ BAJO (min {MIN_GAMES_PER_SEASON})"
        if not ok:
            failed += 1
        print(f"  {season:>10}  {n:>8}  {status:>10}")

    print(f"  {'─'*40}")
    hw = df["HOME_WIN"].mean() if "HOME_WIN" in df.columns else None
    if hw:
        print(f"  Home win rate global: {hw:.3f}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
