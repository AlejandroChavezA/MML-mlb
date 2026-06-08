"""
Evalúa el modelo de ganador filtrando por umbral de confianza.

Carga mlb_games_features.csv, corre predicciones sobre el set de test
temporal (último 20%) y muestra accuracy / log-loss por banda de confianza.

Uso:
    python src/evaluation/eval_confidence.py
    python src/evaluation/eval_confidence.py --thresholds 0.55 0.60 0.65 0.70
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "TARGET_HOME_WIN"
EXCLUDE = {
    "game_pk", "date", "season", "status",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_team_code", "away_team_code", "venue", "day_night",
    "doubleheader", "home_pitcher_id", "away_pitcher_id",
    "home_score", "away_score", "total_runs", "run_diff",
    "HOME_WIN",
    "TARGET_HOME_WIN", "TARGET_TOTAL_RUNS", "TARGET_OVER_8_5",
}


def evaluate(thresholds: list[float] = None):
    if thresholds is None:
        thresholds = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]

    winner_path = MODELS_DIR / "mlb_logreg_rolling.joblib"
    if not winner_path.exists():
        print(f"  Modelo no encontrado: {winner_path}")
        return

    bundle = joblib.load(winner_path)
    model = bundle["model"]
    feat_cols = bundle["features"]

    src = PROCESSED_DIR / "mlb_games_features.csv"
    df = pd.read_csv(src, parse_dates=["date"])
    df = df[df[TARGET].notna()].copy()

    # test temporal: último 20%
    n = int(len(df) * 0.8)
    test = df.iloc[n:].copy()
    X_test = test[feat_cols].fillna(0)
    y_test = test[TARGET].astype(int)

    proba = model.predict_proba(X_test)[:, 1]
    conf = np.maximum(proba, 1 - proba)  # confianza = max(p, 1-p)

    print(f"\n  EVALUACIÓN POR UMBRAL DE CONFIANZA")
    print(f"  Set de test: {len(test)} partidos  "
          f"({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"  {'─'*60}")
    print(f"  {'Umbral':>8}  {'Picks':>6}  {'%Total':>7}  {'Accuracy':>9}  {'Log-loss':>9}")
    print(f"  {'─'*60}")

    for thresh in thresholds:
        mask = conf >= thresh
        n_picks = mask.sum()
        if n_picks < 10:
            print(f"  {thresh:>8.2f}  {n_picks:>6}  {'<10 muestras':>17}")
            continue
        pct = n_picks / len(test) * 100
        acc = accuracy_score(y_test[mask], (proba[mask] >= 0.5).astype(int))
        ll = log_loss(y_test[mask], proba[mask])
        marker = " ◄" if acc >= 0.57 else ""
        print(f"  {thresh:>8.2f}  {n_picks:>6}  {pct:>6.1f}%  {acc:>9.4f}  {ll:>9.4f}{marker}")

    print(f"  {'─'*60}")

    # baseline
    base = y_test.mean()
    print(f"  Baseline (siempre LOCAL): {base:.4f}")

    # top picks
    test_copy = test.copy()
    test_copy["proba"] = proba
    test_copy["conf"] = conf
    test_copy["correct"] = ((proba >= 0.5).astype(int) == y_test.values).astype(int)

    print(f"\n  Top 10 picks más confiables:")
    top = test_copy.nlargest(10, "conf")[
        ["date", "home_team", "away_team", "proba", "conf", "correct"]
    ]
    for _, row in top.iterrows():
        winner = row["home_team"] if row["proba"] >= 0.5 else row["away_team"]
        ok = "✓" if row["correct"] else "✗"
        print(f"    {ok} {row['date'].date()}  {row['away_team'][:15]:>15} @ {row['home_team'][:15]:<15}  "
              f"→ {winner[:12]:<12}  ({row['conf']:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.52, 0.55, 0.58, 0.60, 0.65, 0.70])
    args = ap.parse_args()
    print("=== eval_confidence.py ===")
    evaluate(args.thresholds)


if __name__ == "__main__":
    main()
