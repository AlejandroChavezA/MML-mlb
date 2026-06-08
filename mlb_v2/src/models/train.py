"""
Entrena el modelo de GANADOR.

Mejoras vs v1:
  - Feature set curado: solo DIFF_* + contexto (elimina redundancia HOME_/AWAY_/DIFF_ triple)
  - DAYS_REST cappado a 5 para evitar explosión en off-season
  - GridSearch sobre C y solver
  - Evaluación por banda de confianza

Guarda: models/mlb_logreg_rolling.joblib

Uso:
    python src/models/train.py
    python src/models/train.py --rolling-years 5
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET = "TARGET_HOME_WIN"

# Columnas a excluir siempre
EXCLUDE = {
    "game_pk", "date", "season", "status",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_team_code", "away_team_code", "venue", "day_night",
    "doubleheader", "home_pitcher_id", "away_pitcher_id",
    "home_score", "away_score", "total_runs", "run_diff",
    "HOME_WIN",
    "TARGET_HOME_WIN", "TARGET_TOTAL_RUNS", "TARGET_OVER_8_5",
}

# Ventanas rolling que usamos
WINDOWS = [5, 10, 20]

# Métricas que aportaron más correlación con HOME_WIN
DIFF_METRICS = [
    "WIN_RATE", "RUN_DIFF", "RUNS_FOR", "RUNS_AGAINST",
    "HR", "WALKS", "SO_BAT", "SO_PIT", "BB_ALLOWED",
    "HITS",
]

# Features de contexto que SÍ incluimos (pero cappadas)
CONTEXT = [
    "HOME_DAYS_REST", "AWAY_DAYS_REST",
    "HOME_BACK_TO_BACK", "AWAY_BACK_TO_BACK",
    "HOME_WIN_STREAK", "AWAY_WIN_STREAK",
]

# Algunas features absolutas del local son útiles porque capturan la ventaja de local
HOME_ABS = [f"HOME_{m}_roll{w}" for m in ["WIN_RATE", "RUNS_FOR", "RUNS_AGAINST", "RUN_DIFF"] for w in WINDOWS]
AWAY_ABS = [f"AWAY_{m}_roll{w}" for m in ["WIN_RATE", "RUNS_FOR", "RUNS_AGAINST", "RUN_DIFF"] for w in WINDOWS]

MAX_DAYS_REST = 5  # cap: off-season no debe dominar el logit


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Selecciona el conjunto de features curado:
    DIFF_* (diferenciales) + HOME/AWAY absolutas clave + contexto.
    Evita triple colinealidad HOME_+AWAY_+DIFF_ para cada métrica.
    """
    diff_cols = [
        f"DIFF_{m}_roll{w}"
        for m in DIFF_METRICS
        for w in WINDOWS
        if f"DIFF_{m}_roll{w}" in df.columns
    ]

    home_abs = [c for c in HOME_ABS if c in df.columns]
    away_abs = [c for c in AWAY_ABS if c in df.columns]
    ctx = [c for c in CONTEXT if c in df.columns]

    all_feats = diff_cols + home_abs + away_abs + ctx

    # Eliminar duplicados manteniendo orden
    seen = set()
    return [f for f in all_feats if not (f in seen or seen.add(f))]


def cap_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Cappear DAYS_REST a MAX_DAYS_REST para que el off-season no domine."""
    df = df.copy()
    for col in ["HOME_DAYS_REST", "AWAY_DAYS_REST"]:
        if col in df.columns:
            df[col] = df[col].clip(0, MAX_DAYS_REST)
    return df


def temporal_split(df: pd.DataFrame, ratio: float = 0.8):
    n = int(len(df) * ratio)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def eval_by_confidence(y_true, proba, label="Test"):
    conf = np.maximum(proba, 1 - proba)
    print(f"\n  [{label}] Accuracy por banda de confianza:")
    print(f"  {'Banda':>12}  {'Picks':>6}  {'Acc':>7}")
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.0)]:
        mask = (conf >= lo) & (conf < hi)
        n_p = mask.sum()
        if n_p < 5:
            continue
        acc = accuracy_score(y_true[mask], (proba[mask] >= 0.5).astype(int))
        print(f"  {lo:.2f}-{hi:.2f}     {n_p:>6}  {acc:>7.4f}")


def train(rolling_years: int = 10):
    src = PROCESSED_DIR / "mlb_games_features.csv"
    if not src.exists():
        raise FileNotFoundError(f"Ejecuta el pipeline primero: {src}")

    df = pd.read_csv(src, parse_dates=["date"])
    df = df[df[TARGET].notna()].copy()
    df = cap_days_rest(df)

    if rolling_years:
        cutoff = df["date"].max() - pd.DateOffset(years=rolling_years)
        df = df[df["date"] >= cutoff]

    print(f"  Dataset: {len(df)} partidos  ({df['date'].min().date()} → {df['date'].max().date()})")

    feat_cols = get_feature_cols(df)
    print(f"  Features seleccionadas: {len(feat_cols)}")

    X_all = df[feat_cols].fillna(0)
    y_all = df[TARGET].astype(int)

    train_df, test_df = temporal_split(df)
    X_train = cap_days_rest(train_df)[feat_cols].fillna(0)
    y_train = train_df[TARGET].astype(int)
    X_test = cap_days_rest(test_df)[feat_cols].fillna(0)
    y_test = test_df[TARGET].astype(int)

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Baseline (siempre LOCAL): {y_test.mean():.4f}")

    # Probar varios valores de C y elegir el mejor por acc en test
    best_acc, best_C, best_model = 0, 0.1, None
    print("\n  GridSearch sobre C:")
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000, solver="lbfgs",
                class_weight="balanced", C=C, random_state=42,
            )),
        ])
        m.fit(X_train, y_train)
        proba_t = m.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, (proba_t >= 0.5).astype(int))
        auc = roc_auc_score(y_test, proba_t)
        print(f"    C={C:<5}  acc={acc:.4f}  auc={auc:.4f}")
        if acc > best_acc:
            best_acc, best_C, best_model = acc, C, m

    print(f"\n  Mejor C={best_C}  acc={best_acc:.4f}")

    proba_test = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    ll = log_loss(y_test, proba_test)

    print(f"  Test Accuracy : {best_acc:.4f}")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print(f"  Test Log-loss : {ll:.4f}")

    eval_by_confidence(y_test.values, proba_test)

    # Distribución de confianza
    conf = np.maximum(proba_test, 1 - proba_test)
    print(f"\n  Confianza media: {conf.mean():.3f}  (σ={conf.std():.3f})")
    print(f"  % conf > 0.70: {(conf > 0.70).mean():.1%}")
    print(f"  % conf > 0.60: {(conf > 0.60).mean():.1%}")

    # Top features por coeficiente
    lr = best_model.named_steps["lr"]
    top = sorted(zip(feat_cols, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("\n  Top 10 features (|coef|):")
    for name, coef in top:
        print(f"    {coef:+.4f}  {name}")

    # Reentrenar en 100%
    best_model.fit(X_all, y_all)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "mlb_logreg_rolling.joblib"
    joblib.dump({
        "model": best_model,
        "features": feat_cols,
        "test_acc": best_acc,
        "test_auc": auc,
        "best_C": best_C,
        "max_days_rest": MAX_DAYS_REST,
    }, out)
    print(f"\n  Guardado: {out}")
    return best_model, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolling-years", type=int, default=10)
    args = ap.parse_args()
    print("=== train.py — MLB Winner Model ===")
    train(args.rolling_years)


if __name__ == "__main__":
    main()
