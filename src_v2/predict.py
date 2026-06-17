#!/usr/bin/env python3
"""
Predict MLB
===========
Script para hacer predicciones con modelo entrenado.

Uso:
    python src_v2/predict.py                        # Predicción interactiva
    python src_v2/predict.py --test                 # Test con resultados conocidos
    python src_v2/predict.py --game "NYY" "BOS"     # Juego específico
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.features.mlb_feature_engineer import get_mlb_feature_engineer
from src_v2.models.winner_predictor import get_winner_predictor
from src_v2.models.runs_predictor import get_runs_predictor

# Reuse detailed display from main
from main import _display_game_detail, SEPARADOR

TEAM_CODES = {
    "AZ": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYY": "New York Yankees",
    "NYM": "New York Mets", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}


def predict_game(home_code: str, away_code: str, date=None,
                  w_model: str = None, r_model: str = None):
    """Predecir un juego MLB"""
    home = TEAM_CODES.get(home_code.upper())
    away = TEAM_CODES.get(away_code.upper())

    if not home or not away:
        print(f" Códigos inválidos. Usa: NYY, BOS, LAD, CHC, etc.")
        return None

    print(f"\n {home} vs {away}")
    print("-" * 50)

    from pathlib import Path
    cleaned_dir = Path("data") / "cleaned"
    all_years = sorted(set(
        int(f.stem.split("_")[1])
        for f in cleaned_dir.glob("games_*_cleaned.csv")
    )) or [2024, 2025]

    fe = get_mlb_feature_engineer("data")
    fe.load_data(years=all_years)

    winner = get_winner_predictor("models_mlb")
    if not winner.load():
        print(" Modelo no encontrado. Ejecuta: python src_v2/train.py")
        return None

    runs = get_runs_predictor("models_mlb")
    runs.load()

    if date is None:
        date = datetime.now()

    if w_model is None:
        w_model = winner.best_model
    if r_model is None:
        r_model = runs.best_model

    print(f"  Winner model: {w_model} | Runs model: {r_model}")

    pred = winner.predict(home, away, date, fe, model_name=w_model)
    if "error" in pred:
        print(f" Error: {pred['error']}")
        return None

    rpred = runs.predict(home, away, date, fe, model_name=r_model)

    dummy_game = {"date": date}
    _display_game_detail(home_code.upper(), away_code.upper(), home, away, dummy_game, pred, rpred, fe, winner)
    print()

    return pred


def test_with_known_results():
    """Test con últimos juegos de 2024 que ya conocemos"""
    print("\n  TEST CON RESULTADOS CONOCIDOS")
    print("=" * 50)

    from pathlib import Path
    cleaned_dir = Path("data") / "cleaned"
    all_years = sorted(set(
        int(f.stem.split("_")[1])
        for f in cleaned_dir.glob("games_*_cleaned.csv")
    )) or [2024]

    fe = get_mlb_feature_engineer("data")
    fe.load_data(years=all_years)

    winner = get_winner_predictor("models_mlb")
    if not winner.load():
        print(" Modelo no encontrado")
        return
    runs = get_runs_predictor("models_mlb")
    runs.load()

    year = all_years[-1]
    games = fe.games[year]
    finished = games[games["status"] == "FINISHED"].tail(20).copy()

    correct = 0
    total = 0

    print(f"\nÚltimos 20 juegos de 2024:")
    for _, m in finished.iterrows():
        date = m["date"]
        if hasattr(date, "tzinfo") and date.tzinfo:
            date = date.replace(tzinfo=None)

        pred = winner.predict(m["home_team"], m["away_team"], date, fe)
        if "error" in pred:
            continue

        actual = "LOCAL" if m["home_runs"] > m["away_runs"] else "VISITANTE"
        correct_flag = pred["predicted"] == actual
        if correct_flag:
            correct += 1
        total += 1

        mark = "+" if correct_flag else "-"
        print(f"  {mark} {pred['predicted']:10} vs {actual:10} | {m['home_team'][:20]:20} vs {m['away_team']}")

    if total > 0:
        print(f"\n  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLB Predictor")
    parser.add_argument("--test", action="store_true", help="Test con resultados conocidos")
    parser.add_argument("--game", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predecir juego (códigos: NYY BOS)")
    parser.add_argument("--date", type=str, help="Fecha (YYYY-MM-DD)")
    parser.add_argument("--w-model", type=str, default=None,
                        choices=["random_forest", "gradient_boosting", "logistic_regression"],
                        help="Modelo winner (default: best)")
    parser.add_argument("--r-model", type=str, default=None,
                        choices=["random_forest", "gradient_boosting", "logistic_regression"],
                        help="Modelo runs (default: best)")

    args = parser.parse_args()

    if args.test:
        test_with_known_results()
        return

    if args.game:
        date = None
        if args.date:
            date = datetime.fromisoformat(args.date)
        predict_game(args.game[0], args.game[1], date, args.w_model, args.r_model)
        return

    print("\n MLB PREDICTOR v2")
    print("=" * 40)
    print("\nUso:")
    print("  python src_v2/predict.py --test")
    print('  python src_v2/predict.py --game NYY BOS')
    print('  python src_v2/predict.py --game LAD SF --date 2025-06-15')
    print("\nCódigos de equipos:")
    for code, name in sorted(TEAM_CODES.items()):
        print(f"  {code:5} {name}")


if __name__ == "__main__":
    main()
