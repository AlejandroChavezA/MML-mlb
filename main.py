#!/usr/bin/env python3
"""
MML-MLB: MLB Match Prediction System
=====================================
Sistema de predicción de juegos de MLB usando Machine Learning.

Uso:
    python main.py                    # Menú interactivo
    python main.py --train            # Entrenar modelos
    python main.py --update           # Actualizar datos
    python main.py --predict NYY BOS  # Predecir juego
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.features.mlb_feature_engineer import get_mlb_feature_engineer
from src_v2.features.competitiveness import get_competitiveness
from src_v2.tracking import save_prediction
from src_v2.models.winner_predictor import get_winner_predictor
from src_v2.models.runs_predictor import get_runs_predictor


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

SEPARADOR = "\n" + "─" * 80

from src_v2.export_to_panel import _analyze_features as _panel_analyze


def _get_date_str(game):
    d = game["date"]
    if hasattr(d, "strftime"):
        dias = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
                "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
                "Sunday": "Domingo"}
        s = d.strftime("%A %I:%M %p").replace(" 0", " ")
        for en, es in dias.items():
            s = s.replace(en, es)
        return s
    return str(d)


def _display_game_detail(home_short, away_short, home, away, game, wpred, rpred, fe, winner):
    date_str = _get_date_str(game)
    winner_is_home = wpred.get("code") == 1
    winner_team = home_short if winner_is_home else away_short
    confidence = wpred.get("confidence", 0)
    probs = wpred.get("probabilities", {})

    icon = "🏠" if winner_is_home else "✈️"
    print(SEPARADOR)
    print(f" ⚾ {home_short} @ {away_short}")
    print(f" 📅 {date_str}")
    print(SEPARADOR)
    print(f" 🎯 EL MODELO DICE: {icon} GANA {winner_team}")
    print(f"    Confianza: {confidence:.1%}")
    print(f"    {home_short}: {probs.get('LOCAL', 0):.1%} | {away_short}: {probs.get('VISITANTE', 0):.1%}")

    # Señales
    favor, contra = _panel_analyze(fe, home, away, game["date"], winner, wpred)
    if favor or contra:
        winner_team_full = home_short if winner_is_home else away_short
        loser_team_full = away_short if winner_is_home else home_short

        if favor:
            print(f"\n ✅ ¿POR QUÉ FAVORECE A {winner_team_full}?")
            print(f" {'─' * 78}")
            for i, s in enumerate(favor, 1):
                print(f"  {i}. Señal del modelo: {s} ⭐")

        if contra:
            print(f"\n ❌ ¿QUÉ FAVORECE A {loser_team_full}?")
            print(f" {'─' * 78}")
            for i, s in enumerate(contra, 1):
                print(f"  {i}. Señal del modelo: {s} ⭐")

    # O/U
    if "error" not in (rpred or {}):
        er = rpred.get("expected_runs")
        ou = rpred.get("markets", {}).get("over_8.5", {})
        ou_pred = ou.get("prediction", "?")
        ou_pct = ou.get("over_prob", 0)
        ou_icon = "📈" if ou_pred == "OVER" else "📉"
        print(f"\n 📊 O/U 8.5: {ou_icon} {ou_pred} ({ou_pct:.1%})")
        if er is not None:
            print(f"    Carreras totales esperadas: {er:.1f}")


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def update_data():
    """Actualizar datos desde APIs"""
    print("\n Actualizando datos MLB...")
    print("=" * 40)

    from datetime import datetime as dt
    current = dt.now().year

    from src_v2.data.mlb_collector import get_mlb_collector
    from src_v2.data.mlb_pybaseball import get_pybaseball_wrapper

    collector = get_mlb_collector("data")
    try:
        schedule = collector.get_schedule(current)
        schedule.to_csv(f"data/games_{current}.csv", index=False)
        print(f"  Games {current}: {len(schedule)} guardados")
        collector.get_standings(current)  # guarda y reporta standings_{year}.csv
        collector.get_teams()
    except Exception as e:
        print(f"  ⚠️ No se pudo recolectar {current}: {e}")

    pb = get_pybaseball_wrapper("data")
    try:
        pb.collect_season(current)
    except Exception as e:
        print(f"  ⚠️ Pybaseball {current}: {e}")

    print("\n Limpiando datos...")
    from src_v2.data.cleaner import get_cleaner
    cleaner = get_cleaner("data")
    data_dir = Path("data")
    candidate_years = sorted(set([current, current - 1]))
    years = [
        y for y in candidate_years
        if (data_dir / f"games_{y}.csv").exists()
        and (data_dir / f"standings_{y}.csv").exists()
    ]
    missing = [y for y in candidate_years if y not in years]
    if missing:
        print(f"  ⚠️ Sin datos crudos para {missing} — omite limpieza de esos años")
        print("     Ejecuta download_mlb_historical.py para temporadas históricas")
    if years:
        cleaner.run_cleaning(years=years)
    else:
        print("  ⚠️ No hay temporadas listas para limpiar")

    print("\n Datos actualizados!")


def _detect_years():
    cleaned_dir = Path("data") / "cleaned"
    years = set()
    for f in cleaned_dir.glob("games_*_cleaned.csv"):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)


def train_models():
    """Entrenar modelos desde cero con split cronológico y 100% final"""
    all_years = _detect_years()
    if len(all_years) < 2:
        print(f" Se necesitan al menos 2 temporadas, encontradas: {all_years}")
        return

    print("\n Entrenando modelos MLB...")
    print("=" * 40)
    print(f"  Datos: {all_years[0]}-{all_years[-1]} ({len(all_years)} temporadas)")
    print(f"  Split: 80/20 cronológico")
    print(f"  Final: 100% de los datos")

    fe = get_mlb_feature_engineer("data")
    if not fe.load_data(years=all_years):
        print(" Error cargando datos")
        return

    print("\n Creando dataset con TODOS los años...")
    features_df, targets_df, runs_targets = fe.create_training_dataset(
        years=all_years
    )

    if features_df.empty:
        print(" Dataset vacío")
        return

    print(f"  {len(features_df)} muestras, {len(features_df.columns)} features")

    # Split cronológico 80/20
    split_idx = int(len(features_df) * 0.8)
    train_feat = features_df.iloc[:split_idx]
    train_win = targets_df.iloc[:split_idx]
    train_runs = runs_targets.iloc[:split_idx]
    test_feat = features_df.iloc[split_idx:]
    test_win = targets_df.iloc[split_idx:]
    test_runs = runs_targets.iloc[split_idx:]

    print(f"  Train: {len(train_feat)} | Test: {len(test_feat)}")

    comp = get_competitiveness("data")
    if comp.load_and_calculate(all_years):
        comp.print_summary()

    winner = get_winner_predictor("models_mlb")
    winner.train(train_feat, train_win, X_test=test_feat, y_test=test_win)

    runs = get_runs_predictor("models_mlb")
    runs.train(train_feat, train_runs, X_test=test_feat, y_test=test_runs)

    winner.print_comparison()
    runs.print_comparison()

    print("\n Entrenando modelo final con 100% de datos...")
    winner.train_final(features_df, targets_df)
    runs.train_final(features_df, runs_targets)

    print("\n Modelos guardados en models_mlb/")


def predict_interactive():
    """Modo predicción interactiva"""
    all_years = _detect_years()

    fe = get_mlb_feature_engineer("data")
    fe.load_data(years=all_years)

    winner = get_winner_predictor("models_mlb")
    if not winner.load():
        print(" Modelos no encontrados. Ejecuta: python main.py --train")
        return

    runs = get_runs_predictor("models_mlb")
    runs.load()

    comp = get_competitiveness("data")
    comp.load_and_calculate(all_years)

    print("\n MLB PREDICTOR")
    print(f"  Winner best: {winner.best_model}")
    print(f"  Runs best:   {runs.best_model}")
    print(f"  Competitiveness: {comp.global_level} ({comp.global_score:.2f})")
    print()

    w_model = winner.best_model
    r_model = runs.best_model

    print(" Códigos disponibles:")
    print("  ", "  ".join(sorted(TEAM_CODES.keys())))
    print("\n Comandos:")
    print("  NYY BOS       -> predecir")
    print("  model         -> cambiar modelo")
    print("  q             -> salir")

    from datetime import datetime

    while True:
        try:
            inp = input("\nLocal Visitante: ").strip().upper()
            if inp == "Q":
                break
            if inp == "MODEL":
                print(f"\n  Winner models: {', '.join(winner.MODEL_NAMES)}")
                print(f"  Actual: {w_model}")
                m = input("  Winner model: ").strip().lower()
                if m in winner.MODEL_NAMES:
                    w_model = m
                print(f"\n  Runs models: {', '.join(runs.MODEL_NAMES)}")
                print(f"  Actual: {r_model}")
                m = input("  Runs model: ").strip().lower()
                if m in runs.MODEL_NAMES:
                    r_model = m
                continue

            parts = inp.split()
            if len(parts) != 2:
                print(" Formato: NYY BOS")
                continue

            home = TEAM_CODES.get(parts[0])
            away = TEAM_CODES.get(parts[1])

            if not home or not away:
                print(f" Código inválido: {parts[0] if not home else parts[1]}")
                continue

            pred = winner.predict(home, away, datetime.now(), fe, model_name=w_model)
            if "error" in pred:
                print(f" {pred['error']}")
                continue

            r = runs.predict(home, away, datetime.now(), fe, model_name=r_model)

            home_short = parts[0]
            away_short = parts[1]

            dummy_game = {"date": datetime.now()}
            _display_game_detail(home_short, away_short, home, away, dummy_game, pred, r, fe, winner)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f" Error: {e}")


def predict_date_games(date_str=None):
    """Predecir todos los juegos de una fecha y opcionalmente enviarlos al panel"""
    from datetime import datetime as dt

    if not date_str:
        date_str = input("Fecha (YYYY-MM-DD) [Enter = hoy]: ").strip()

    try:
        if date_str:
            target_date = dt.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = dt.now().date()
    except ValueError:
        print(" Formato inválido. Usa YYYY-MM-DD")
        return

    year = target_date.year

    import pandas as pd
    all_years = _detect_years()
    data_year = year if year in all_years else (all_years[-1] if all_years else 2025)

    fe = get_mlb_feature_engineer("data")
    fe.load_data(years=all_years)

    day_games = pd.DataFrame()
    games_df = fe.games.get(data_year)
    if games_df is not None and len(games_df) > 0:
        day_games = games_df[
            (pd.to_datetime(games_df["date"]).dt.date == target_date)
            & (games_df["status"] == "SCHEDULED")
        ]

    if len(day_games) == 0:
        from src_v2.data.mlb_collector import get_mlb_collector
        collector = get_mlb_collector("data")
        live = collector.get_schedule(year, hydrate=False)
        if live is not None and len(live) > 0:
            live["date"] = pd.to_datetime(live["date"])
            day_games = live[live["date"].dt.date == target_date].copy()

    if len(day_games) == 0:
        print(f" No hay juegos programados para {target_date}")
        return

    winner = get_winner_predictor("models_mlb")
    if not winner.load():
        print(" Modelos no entrenados. Ejecuta: python main.py --train")
        return

    runs = get_runs_predictor("models_mlb")
    runs.load()

    teams_df = fe.teams
    team_id_map = {}
    if teams_df is not None and len(teams_df) > 0:
        for _, r in teams_df.iterrows():
            team_id_map[r["name"]] = {"code": r.get("team_code", ""), "id": r.get("team_id", 0)}

    name_to_code = {v: k for k, v in TEAM_CODES.items()}

    comp = get_competitiveness("data")
    comp.load_and_calculate(all_years)

    print(f"\n PREDICCIÓN {target_date}")
    print("=" * 80)

    predictions = []
    total_conf = 0
    home_wins_count = 0
    away_wins_count = 0
    over_count = 0

    for _, game in day_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        date_val = game["date"]
        if hasattr(date_val, "tzinfo") and date_val.tzinfo:
            date_val = date_val.replace(tzinfo=None)

        home_info = team_id_map.get(home, {"code": "", "id": 0})
        away_info = team_id_map.get(away, {"code": "", "id": 0})

        wpred = winner.predict(home, away, date_val, fe)
        if "error" in wpred:
            print(f"\n ⚠️ Error: {home} vs {away} - {wpred['error']}")
            continue

        rpred = runs.predict(home, away, date_val, fe)

        home_short = name_to_code.get(home, home_info.get("code", home[:3].upper() or ""))
        away_short = name_to_code.get(away, away_info.get("code", away[:3].upper() or ""))

        _display_game_detail(home_short, away_short, home, away, game, wpred, rpred, fe, winner)

        confidence = wpred.get("confidence", 0)
        total_conf += confidence
        if wpred.get("code") == 1:
            home_wins_count += 1
        else:
            away_wins_count += 1

        ou_market = rpred.get("markets", {}).get("over_8.5", {})
        ou_pred = ou_market.get("prediction", "UNDER")
        ou_pct = ou_market.get("over_prob", 0)
        if ou_pred == "OVER":
            over_count += 1

        game_pk = game.get("game_pk")
        if game_pk:
            save_prediction(int(game_pk), {
                "game_pk": int(game_pk),
                "date": str(date_val.date()),
                "home_team": home_info.get("code", ""),
                "away_team": away_info.get("code", ""),
                "home_full": home,
                "away_full": away,
                "predicted_winner": home_short if wpred.get("code") == 1 else away_short,
                "predicted_winner_code": wpred.get("code"),
                "confidence": confidence,
                "winner_model": wpred.get("model", ""),
                "over_under": ou_pred,
                "over_prob": ou_pct,
                "runs_model": rpred.get("model", ""),
                "timestamp": target_date.isoformat(),
            })

        from src_v2.export_to_panel import transform_to_panel_format
        panel_pred = transform_to_panel_format(
            home, away,
            home_info["code"], away_info["code"],
            home_info["id"], away_info["id"],
            date_val, wpred, rpred,
            feature_engineer=fe, winner_model=winner,
        )
        predictions.append(panel_pred)

    under_count = len(predictions) - over_count

    # Resumen
    print("\n" + "=" * 80)
    print(" 📊 RESUMEN".upper())
    print("=" * 80)
    liga = comp.global_level if hasattr(comp, "global_level") else "N/A"
    score = comp.global_score if hasattr(comp, "global_score") else 0
    if score > 0.6:
        nota = "Liga balanceada — esperar mayor variabilidad"
    elif score > 0.4:
        nota = "Liga moderadamente competitiva"
    else:
        nota = "Liga con pocos equipos dominantes"
    print(f" 🏆 Competitividad MLB: {liga} ({score:.2f})")
    print(f"    {nota}")
    print(f" Victorias locales: {home_wins_count}")
    print(f" Victorias visitantes: {away_wins_count}")
    print(f" OVER: {over_count} | UNDER: {under_count}")
    print(f" Total partidos: {len(predictions)}")
    print(f" Confianza promedio: {total_conf / max(len(predictions), 1):.1%}")
    print(f" Modelo utilizado: {winner.best_model}")
    print(SEPARADOR)
    print()

    print("=" * 80)
    print(" ACCIONES")
    print("=" * 80)
    send = input("\n  Enviar todo al panel? (s/n): ").strip().lower()
    if send == "s":
        from src_v2.export_to_panel import send_predictions, get_api_key

        api_key = get_api_key()
        if api_key is None:
            print("  No se pudo obtener API key")
            return

        result = send_predictions(predictions, api_key)
        if result.get("success"):
            print(f"  Exportación exitosa!")
            print(f"    Importadas: {result.get('imported', 0)}")
            print(f"    Saltadas: {result.get('skipped', 0)}")
            print(f"    Total: {result.get('total', 0)}")
        else:
            print(f"  Error: {result.get('message', 'desconocido')}")


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("""
 MML-MLB - MLB Match Prediction System

 Uso:
   python main.py                    Menu interactivo
   python main.py --train            Entrenar modelos
   python main.py --update           Actualizar datos
   python main.py --predict H A      Predecir (ej: NYY BOS)
   python main.py --bolo [YYYY-MM-DD]  Predecir todos los juegos de una fecha
        """)
        return

    if "--train" in args:
        train_models()
        return

    if "--update" in args:
        update_data()
        return

    if "--bolo" in args:
        date_str = None
        if len(args) > 1:
            idx = args.index("--bolo")
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                date_str = args[idx + 1]
        predict_date_games(date_str)
        return

    if "--predict" in args:
        if len(args) >= 3:
            wm = args[3] if len(args) >= 4 else None
            rm = args[4] if len(args) >= 5 else None
            from src_v2.predict import predict_game
            predict_game(args[1], args[2], w_model=wm, r_model=rm)
        else:
            print(" Uso: python main.py --predict HOME AWAY [W_MODEL] [R_MODEL]")
            print("  Modelos: random_forest, gradient_boosting, logistic_regression")
        return

    if "--v2" in sys.argv:
        print(" Para usar el sistema MLB:")
        print("   python3 main.py")
        return

    menu_items = [
        ("Predecir juegos (por fecha)", predict_date_games),
        ("Predecir juego (interactivo)", predict_interactive),
        ("Entrenar modelos", train_models),
        ("Actualizar datos", update_data),
        ("Exportar predicciones al panel", export_to_panel),
        ("Ver rendimiento de predicciones", view_performance),
        ("Salir", None),
    ]

    while True:
        clear()
        print(" MML-MLB: MLB MATCH PREDICTOR")
        print("=" * 40)
        print(f" Modelos: models_mlb/")
        print()

        for i, (label, _) in enumerate(menu_items, 1):
            print(f"  {i}. {label}")

        print()
        try:
            n_items = len(menu_items)
            choice = input(f"Selecciona (1-{n_items}): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == str(n_items) or choice.lower() == "q":
            break

        idx = int(choice) - 1 if choice.isdigit() else -1
        if 0 <= idx < len(menu_items):
            clear()
            _, func = menu_items[idx]
            if func:
                func()
            input("\nPresiona Enter para continuar...")
        else:
            input("Opción inválida. Presiona Enter...")

    print("\n MLB Predictor finalizado.")


def view_performance():
    """Ver rendimiento de predicciones vs resultados reales"""
    print("\n RENDIMIENTO DE PREDICCIONES")
    print("=" * 40)

    from src_v2.tracking import print_report, get_summary

    summary = get_summary()
    if "error" in summary:
        print(f"  {summary['error']}")
        return

    print(f"  Total predicciones guardadas: {summary['total_predicciones']}")
    print(f"  Temporadas: {', '.join(summary['temporadas'])}")
    print()

    year = input("  Año a evaluar (Enter = año más reciente): ").strip()
    if not year and summary["temporadas"]:
        year = summary["temporadas"][-1]
    if not year or not year.isdigit():
        print("  Año inválido")
        return

    print_report(int(year))


def export_to_panel():
    """Exportar predicciones de hoy al safesports-panel"""
    print("\n Exportando predicciones al panel...")
    print("=" * 40)

    from src_v2.export_to_panel import export_todays_games

    fe = get_mlb_feature_engineer("data")
    if not fe.load_data(years=_detect_years()):
        print(" No se pudieron cargar datos")
        return

    winner = get_winner_predictor("models_mlb")
    if not winner.load():
        print(" Modelos no encontrados. Ejecuta: python main.py --train")
        return

    runs = get_runs_predictor("models_mlb")
    runs.load()

    result = export_todays_games(fe, winner, runs)

    if result.get("success"):
        print(f"\n Exportación exitosa!")
        print(f"  Importadas: {result.get('imported', 0)}")
        print(f"  Saltadas (ya existían): {result.get('skipped', 0)}")
        print(f"  Total: {result.get('total', 0)}")
    else:
        print(f"\n Error: {result.get('message', 'desconocido')}")

    if result.get("errors"):
        print(f"  Errores: {len(result['errors'])}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n MLB Predictor finalizado.")
