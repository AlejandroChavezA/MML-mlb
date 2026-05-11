#!/usr/bin/env python3
"""
Train MLB
=========
Script de entrenamiento para modelos MLB.

Uso:
    python src_v2/train.py

Pipeline:
    1. Cargar datos limpios (2021-2023 train, 2024 test)
    2. Crear features + targets (winner + runs en un solo pase)
    3. Entrenar WinnerPredictor (binario)
    4. Entrenar RunsPredictor (O/U 8.5)
    5. Evaluar en test set 2024
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.features.mlb_feature_engineer import get_mlb_feature_engineer
from src_v2.features.competitiveness import get_competitiveness
from src_v2.models.winner_predictor import get_winner_predictor
from src_v2.models.runs_predictor import get_runs_predictor
from src_v2.evaluation.evaluator import get_evaluator


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO MLB v2")
    print("  Train: 2021-2023 | Test: 2024")
    print("=" * 60)

    fe = get_mlb_feature_engineer("data")
    if not fe.load_data(years=[2021, 2022, 2023, 2024]):
        print(" Error cargando datos")
        return 1

    print("\n[1/4] Creando dataset de entrenamiento (2021-2023)...")
    features_df, targets_df, runs_targets = fe.create_training_dataset(
        years=[2021, 2022, 2023]
    )

    if features_df.empty:
        print(" Dataset vacío")
        return 1

    print(f"  Samples: {len(features_df)}, Features: {len(features_df.columns)}")

    local_count = targets_df.sum()
    visitante_count = len(targets_df) - local_count
    print(f"  LOCAL: {local_count} ({local_count/len(targets_df)*100:.1f}%)")
    print(f"  VISITANTE: {visitante_count} ({visitante_count/len(targets_df)*100:.1f}%)")
    print(f"  Avg runs: {runs_targets.mean():.2f}")

    comp = get_competitiveness("data")
    if comp.load_and_calculate([2021, 2022, 2023, 2024]):
        comp.print_summary()

    print("\n[2/4] Entrenando WinnerPredictor (binario)...")
    winner = get_winner_predictor("models_mlb")
    winner.train(features_df, targets_df)

    print("\n[3/4] Entrenando RunsPredictor (O/U 8.5)...")
    runs = get_runs_predictor("models_mlb")
    runs.train(features_df, runs_targets)

    print("\n[4/4] Evaluación en test set 2024...")
    evaluator = get_evaluator("data", "models_mlb")
    evaluator.evaluate_test_set(fe, winner, runs, year=2024)

    print("\n" + "=" * 60)
    print("  Entrenamiento completo!")
    print(f"  Modelos guardados en: models_mlb/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
