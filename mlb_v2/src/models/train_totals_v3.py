"""
Entrena el modelo de TOTALES v3: selección automática entre
Ridge, XGBoost y LightGBM según MAE en test temporal.

Predice:
  - total de carreras (regresión)
  - over/under 8.5 (threshold aplicado al total predicho)

Guarda: models/mlb_totals_v3.joblib

Uso:
    python src/models/train_totals_v3.py
    python src/models/train_totals_v3.py --rolling-years 5 --no-interactions
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET_RUNS = "TARGET_TOTAL_RUNS"
TARGET_OU = "TARGET_OVER_8_5"
OU_LINE = 8.5

EXCLUDE = {
    "game_pk", "date", "season", "status",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_team_code", "away_team_code", "venue", "day_night",
    "doubleheader", "home_pitcher_id", "away_pitcher_id",
    "home_score", "away_score", "total_runs", "run_diff",
    "HOME_WIN",
    "TARGET_HOME_WIN", "TARGET_TOTAL_RUNS", "TARGET_OVER_8_5",
}

# features enfocadas en ofensiva y ritmo (más relevantes para totales)
TOTALS_FOCUS = [
    "RUNS_FOR", "RUNS_AGAINST", "HR", "WALKS", "SO_BAT",
    "SO_PIT", "BB_ALLOWED", "HITS",
]


def get_features(df: pd.DataFrame, interactions: bool = True) -> list[str]:
    base = [c for c in df.columns
            if c not in EXCLUDE and df[c].dtype in (float, int, np.float64, np.int64)]

    if not interactions:
        return base

    # features de interacción: producto de runs_for_home * runs_against_away, etc.
    interact = []
    for w in [5, 10, 20]:
        h_off = f"HOME_RUNS_FOR_roll{w}"
        a_def = f"AWAY_RUNS_AGAINST_roll{w}"
        a_off = f"AWAY_RUNS_FOR_roll{w}"
        h_def = f"HOME_RUNS_AGAINST_roll{w}"
        for col_pair in [(h_off, a_def), (a_off, h_def)]:
            c1, c2 = col_pair
            if c1 in df.columns and c2 in df.columns:
                name = f"INTER_{c1.replace('HOME_','').replace('AWAY_','')}_x_{c2.replace('HOME_','').replace('AWAY_','')}"
                df[name] = df[c1] * df[c2]
                interact.append(name)

    return base + [i for i in interact if i not in base]


def temporal_split(df: pd.DataFrame, ratio: float = 0.8):
    n = int(len(df) * ratio)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def ou_accuracy(y_true_runs: pd.Series, y_pred_runs: np.ndarray, line: float = OU_LINE) -> float:
    pred_over = (y_pred_runs > line).astype(int)
    true_over = (y_true_runs > line).astype(int)
    return (pred_over == true_over).mean()


def try_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
    except ImportError:
        return None


def try_lightgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    except ImportError:
        return None


def train(rolling_years: int = 3, interactions: bool = True):
    src = PROCESSED_DIR / "mlb_games_features.csv"
    if not src.exists():
        raise FileNotFoundError(f"Ejecuta el pipeline primero: {src}")

    df = pd.read_csv(src, parse_dates=["date"])
    df = df[df[TARGET_RUNS].notna()].copy()

    if rolling_years:
        cutoff = df["date"].max() - pd.DateOffset(years=rolling_years)
        df = df[df["date"] >= cutoff].copy()

    print(f"  Dataset: {len(df)} partidos  ({df['date'].min().date()} → {df['date'].max().date()})")

    feat_cols = get_features(df, interactions)
    X = df[feat_cols].fillna(0)
    y = df[TARGET_RUNS]

    train_df, test_df = temporal_split(df)
    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df[TARGET_RUNS]
    X_test = test_df[feat_cols].fillna(0)
    y_test = test_df[TARGET_RUNS]

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Features: {len(feat_cols)}")

    candidates = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
    }

    xgb = try_xgboost()
    if xgb:
        candidates["xgboost"] = xgb

    lgb = try_lightgbm()
    if lgb:
        candidates["lightgbm"] = lgb

    results = {}
    for name, model in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        ou_acc = ou_accuracy(y_test, pred)
        results[name] = {"model": model, "mae": mae, "ou_acc": ou_acc}
        print(f"  {name:12} MAE={mae:.3f}  O/U acc={ou_acc:.4f}")

    best_name = min(results, key=lambda n: results[n]["mae"])
    best = results[best_name]
    print(f"\n  Mejor modelo: {best_name}  (MAE={best['mae']:.3f}, O/U={best['ou_acc']:.4f})")

    # reentrenar en 100%
    best["model"].fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "mlb_totals_v3.joblib"
    joblib.dump({
        "model": best["model"],
        "model_name": best_name,
        "features": feat_cols,
        "test_mae": best["mae"],
        "test_ou_acc": best["ou_acc"],
        "ou_line": OU_LINE,
    }, out)
    print(f"  Guardado: {out}")
    return best["model"], feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolling-years", type=int, default=3)
    ap.add_argument("--no-interactions", action="store_true")
    args = ap.parse_args()
    print("=== train_totals_v3.py — MLB Totals Model ===")
    train(args.rolling_years, not args.no_interactions)


if __name__ == "__main__":
    main()
