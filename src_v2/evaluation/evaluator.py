"""
Model Evaluator (MLB)
=====================
Evalúa rendimiento de modelos MLB.

Métricas:
- Winner: Accuracy, AUC-ROC
- Runs: MAE, O/U 8.5 accuracy
- Overfitting detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class ModelEvaluator:
    """Evalúa modelos de predicción MLB"""

    def __init__(self, data_dir: str = "data", models_dir: str = "models_mlb"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

    def evaluate_test_set(self, feature_engineer, winner, runs, year: int = 2024):
        """Evaluar modelos en un año de test (cronológico)"""
        games = feature_engineer.games.get(year)
        if games is None:
            print(f"  No hay datos para {year}")
            return

        finished = games[games["status"] == "FINISHED"].copy()
        print(f"\n  Evaluando en {year} ({len(finished)} juegos)...")

        winner_correct = 0
        winner_total = 0
        winner_probas = []
        winner_actual = []

        runs_mae_total = 0
        runs_ou_correct = 0
        runs_total = 0

        errors = 0

        for _, m in finished.iterrows():
            try:
                date = m["date"]
                if hasattr(date, "tzinfo") and date.tzinfo:
                    date = date.replace(tzinfo=None)

                home_team = m["home_team"]
                away_team = m["away_team"]
                actual_winner = 1 if m["home_runs"] > m["away_runs"] else 0
                actual_runs = m["total_runs"]

                w_pred = winner.predict(home_team, away_team, date, feature_engineer)
                r_pred = runs.predict(home_team, away_team, date, feature_engineer)

                if "error" not in w_pred:
                    winner_total += 1
                    if w_pred["code"] == actual_winner:
                        winner_correct += 1
                    winner_probas.append(w_pred["probabilities"]["LOCAL"])
                    winner_actual.append(actual_winner)

                if "error" not in r_pred:
                    runs_total += 1
                    er = r_pred.get("expected_runs")
                    if er is not None:
                        runs_mae_total += abs(er - actual_runs)
                    ou_pred = r_pred["markets"]["over_8.5"]["prediction"]
                    actual_ou = "OVER" if actual_runs > 8.5 else "UNDER"
                    if ou_pred == actual_ou:
                        runs_ou_correct += 1

            except Exception:
                errors += 1
                continue

        if winner_total > 0:
            from sklearn.metrics import roc_auc_score
            winner_acc = winner_correct / winner_total
            try:
                winner_auc = roc_auc_score(winner_actual, winner_probas)
            except Exception:
                winner_auc = 0.5

            print(f"\n  WINNER PREDICTOR:")
            print(f"    Accuracy: {winner_correct}/{winner_total} ({winner_acc:.1%})")
            print(f"    AUC-ROC:  {winner_auc:.3f}")

        if runs_total > 0:
            runs_ou_acc = runs_ou_correct / runs_total

            print(f"\n  RUNS PREDICTOR:")
            if runs_mae_total > 0:
                runs_mae = runs_mae_total / runs_total
                print(f"    MAE:           {runs_mae:.2f} runs")
            print(f"    O/U 8.5 acc:   {runs_ou_correct}/{runs_total} ({runs_ou_acc:.1%})")

        if errors > 0:
            print(f"\n  Errores: {errors}")

    def check_overfitting(self, train_acc: float, test_acc: float,
                          cv_mean: float) -> Dict:
        """Detectar overfitting"""
        gap = train_acc - test_acc

        status = "OK"
        if gap > 0.30:
            status = "SEVERE_OVERFIT"
        elif gap > 0.15:
            status = "MODERATE_OVERFIT"

        recommendations = []
        if gap > 0.15:
            recommendations.append("Reducir max_depth")
            recommendations.append("Aumentar min_samples_split")

        if abs(cv_mean - test_acc) > 0.1:
            recommendations.append("Verificar data leak (time-series split)")

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "cv_mean": cv_mean,
            "gap": gap,
            "status": status,
            "recommendations": recommendations,
        }

    def print_summary(self):
        """Imprimir resumen de evaluación"""
        print("\n MLB EVALUACIÓN")
        print("=" * 50)


def get_evaluator(data_dir: str = "data",
                  models_dir: str = "models_mlb") -> ModelEvaluator:
    return ModelEvaluator(data_dir, models_dir)
