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


def _detect_years():
    cleaned_dir = Path("data") / "cleaned"
    years = set()
    for f in cleaned_dir.glob("games_*_cleaned.csv"):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)


def main():
    all_years = _detect_years()
    if len(all_years) < 2:
        print(f" Se necesitan al menos 2 temporadas, encontradas: {all_years}")
        return 1

    print("=" * 60)
    print("  ENTRENAMIENTO MLB v2")
    print(f"  Datos: {all_years[0]}-{all_years[-1]} ({len(all_years)} temporadas)")
    print("  Split: 80/20 cronológico (primeros 80% train, último 20% test)")
    print("  Final: re-entrena con 100% de los datos")
    print("=" * 60)

    fe = get_mlb_feature_engineer("data")
    if not fe.load_data(years=all_years):
        print(" Error cargando datos")
        return 1

    print(f"\n[1/4] Creando dataset con TODOS los años ({all_years[0]}-{all_years[-1]})...")
    features_df, targets_df, runs_targets = fe.create_training_dataset(
        years=all_years
    )

    if features_df.empty:
        print(" Dataset vacío")
        return 1

    print(f"  Total: {len(features_df)} muestras, {len(features_df.columns)} features")

    local_count = targets_df.sum()
    visitante_count = len(targets_df) - local_count
    print(f"  LOCAL: {local_count} ({local_count/len(targets_df)*100:.1f}%)")
    print(f"  VISITANTE: {visitante_count} ({visitante_count/len(targets_df)*100:.1f}%)")
    print(f"  Avg runs: {runs_targets.mean():.2f}")

    # Split cronológico 80/20
    split_idx = int(len(features_df) * 0.8)
    train_feat = features_df.iloc[:split_idx]
    train_win = targets_df.iloc[:split_idx]
    train_runs = runs_targets.iloc[:split_idx]
    test_feat = features_df.iloc[split_idx:]
    test_win = targets_df.iloc[split_idx:]
    test_runs = runs_targets.iloc[split_idx:]

    print(f"\n  Split cronológico: {len(train_feat)} train | {len(test_feat)} test")

    comp = get_competitiveness("data")
    if comp.load_and_calculate(all_years):
        comp.print_summary()

    print("\n[2/4] Entrenando WinnerPredictor (binario)...")
    winner = get_winner_predictor("models_mlb")
    winner.train(train_feat, train_win, X_test=test_feat, y_test=test_win)

    print("\n[3/4] Entrenando RunsPredictor (O/U 8.5)...")
    runs = get_runs_predictor("models_mlb")
    runs.train(train_feat, train_runs, X_test=test_feat, y_test=test_runs)

    winner.print_comparison()
    runs.print_comparison()

    print(f"\n[4/4] Entrenando modelo final con 100% de datos...")
    winner.train_final(features_df, targets_df)
    runs.train_final(features_df, runs_targets)

    print("\n" + "=" * 60)
    print("  Entrenamiento completo!")
    print(f"  Train: {len(train_feat)} (80% cronológico)")
    print(f"  Test:  {len(test_feat)} (20% más recientes)")
    print(f"  Final: {len(features_df)} (100%)")
    print(f"  Modelos guardados en: models_mlb/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
