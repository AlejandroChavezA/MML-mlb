"""
Tracking de predicciones MLB
============================
Guarda predicciones y las compara con resultados reales.

Uso:
    from src_v2.tracking import save_prediction, print_report
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


def _load_predictions() -> list:
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE) as f:
            data = json.load(f)
            return data.get("predictions", [])
    except (json.JSONDecodeError, Exception):
        return []


def _save_predictions(predictions: list):
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)


def save_prediction(game_pk: int, data: Dict):
    preds = _load_predictions()

    existing = [i for i, p in enumerate(preds) if p.get("game_pk") == game_pk]
    if existing:
        preds[existing[0]].update(data)
    else:
        preds.append(data)

    _save_predictions(preds)


def get_report(year: int) -> Dict:
    import pandas as pd

    games_file = DATA_DIR / "cleaned" / f"games_{year}_cleaned.csv"
    if not games_file.exists():
        return {"error": f"No hay datos de juegos para {year}"}

    games = pd.read_csv(games_file)
    finished = games[games["status"] == "FINISHED"]

    if len(finished) == 0:
        return {"error": f"No hay juegos terminados en {year}"}

    preds = _load_predictions()
    year_preds = [p for p in preds if str(p.get("date", "")).startswith(str(year))]

    if len(year_preds) == 0:
        return {"error": f"No hay predicciones guardadas para {year}"}

    results = {}
    for p in year_preds:
        pk = p.get("game_pk")
        match = finished[finished["game_pk"] == pk]
        if len(match) == 0:
            continue

        g = match.iloc[0]
        actual_winner_code = 1 if g["home_runs"] > g["away_runs"] else 0
        actual_total_runs = g["total_runs"]

        pred_winner_code = p.get("predicted_winner_code")
        winner_correct = pred_winner_code == actual_winner_code

        actual_ou = "OVER" if actual_total_runs > 8.5 else "UNDER"
        pred_ou = p.get("over_under")
        ou_correct = pred_ou == actual_ou

        results[pk] = {
            "date": p.get("date"),
            "home": p.get("home_full", p.get("home_team")),
            "away": p.get("away_full", p.get("away_team")),
            "predicted_winner": p.get("predicted_winner"),
            "actual_winner": g["home_team_code"] if actual_winner_code == 1 else g["away_team_code"],
            "winner_correct": winner_correct,
            "confidence": p.get("confidence", 0),
            "predicted_ou": pred_ou,
            "actual_ou": actual_ou,
            "ou_correct": ou_correct,
            "over_prob": p.get("over_prob", 0),
        }

    total = len(results)
    if total == 0:
        return {"error": f"Ninguna predicción coincide con juegos terminados en {year}"}

    winner_ok = sum(1 for r in results.values() if r["winner_correct"])
    ou_ok = sum(1 for r in results.values() if r["ou_correct"])

    high_conf = [r for r in results.values() if r["confidence"] >= 0.70]
    med_conf = [r for r in results.values() if 0.55 <= r["confidence"] < 0.70]
    low_conf = [r for r in results.values() if r["confidence"] < 0.55]

    def acc(group):
        if not group:
            return 0, 0, 0
        ok = sum(1 for r in group if r["winner_correct"])
        return ok, len(group), ok / len(group)

    team_stats = {}
    for r in results.values():
        for team in [r["home"], r["away"]]:
            if team not in team_stats:
                team_stats[team] = {"predicted": 0, "correct": 0}
            team_stats[team]["predicted"] += 1
            if r["winner_correct"]:
                team_stats[team]["correct"] += 1

    team_ranking = sorted(
        [{"team": t, **s} for t, s in team_stats.items()],
        key=lambda x: x["correct"] / max(1, x["predicted"]),
        reverse=True,
    )

    return {
        "year": year,
        "total": total,
        "winner_accuracy": winner_ok / total,
        "winner_correct": winner_ok,
        "ou_accuracy": ou_ok / total,
        "ou_correct": ou_ok,
        "confidence_bands": {
            "high": {"correct": acc(high_conf)[0], "total": acc(high_conf)[1], "acc": acc(high_conf)[2]},
            "medium": {"correct": acc(med_conf)[0], "total": acc(med_conf)[1], "acc": acc(med_conf)[2]},
            "low": {"correct": acc(low_conf)[0], "total": acc(low_conf)[1], "acc": acc(low_conf)[2]},
        },
        "best_teams": team_ranking[:5],
        "worst_teams": team_ranking[-5:] if len(team_ranking) >= 5 else team_ranking,
    }


def print_report(year: int):
    report = get_report(year)

    if "error" in report:
        print(f"\n  {report['error']}")
        return

    print(f"\n  REPORTE DE RENDIMIENTO {report['year']}")
    print(f"  {'=' * 45}")
    print(f"  Total predicciones cotejadas: {report['total']}")
    print()
    print(f"  WINNER PREDICTOR")
    print(f"    Accuracy: {report['winner_correct']}/{report['total']} ({report['winner_accuracy']:.1%})")
    print()
    print(f"  O/U 8.5 PREDICTOR")
    print(f"    Accuracy: {report['ou_correct']}/{report['total']} ({report['ou_accuracy']:.1%})")
    print()
    print(f"  POR CONFIANZA")
    bands = report["confidence_bands"]
    print(f"    Alta   (>=70%):  {bands['high']['correct']}/{bands['high']['total']} ({bands['high']['acc']:.1%})")
    print(f"    Media  (55-70%): {bands['medium']['correct']}/{bands['medium']['total']} ({bands['medium']['acc']:.1%})")
    print(f"    Baja   (<55%):   {bands['low']['correct']}/{bands['low']['total']} ({bands['low']['acc']:.1%})")
    print()
    print(f"  MEJORES EQUIPOS (accuracy)")
    for t in report["best_teams"]:
        pct = t["correct"] / max(1, t["predicted"])
        print(f"    + {t['team']:<25} {t['correct']}/{t['predicted']} ({pct:.1%})")
    print()
    print(f"  PEORES EQUIPOS (accuracy)")
    for t in report["worst_teams"]:
        pct = t["correct"] / max(1, t["predicted"])
        print(f"    - {t['team']:<25} {t['correct']}/{t['predicted']} ({pct:.1%})")


def get_summary():
    import pandas as pd

    preds = _load_predictions()
    if not preds:
        return {"error": "No hay predicciones guardadas"}

    years = sorted(set(str(p.get("date", ""))[:4] for p in preds if p.get("date")))
    return {"total_predicciones": len(preds), "temporadas": years}
