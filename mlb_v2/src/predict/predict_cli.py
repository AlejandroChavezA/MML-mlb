"""
CLI interactivo de predicciones MLB v2.

Modos:
  1. Manual      — formato "AWAY @ HOME" (ej. BOS @ NYY)
  2. Por fecha   — busca los juegos del día en la API y los predice todos
  3. Salir

Uso:
    python src/predict/predict_cli.py
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))  # para imports relativos si se usa como módulo

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
API_BASE = "https://statsapi.mlb.com/api/v1"

OU_LINE = 8.5

# Mapa código → nombre completo
TEAM_CODES: dict[str, str] = {
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
NAME_TO_CODE = {v: k for k, v in TEAM_CODES.items()}


# ─────────────────────────────────────────────
# Carga de modelos
# ─────────────────────────────────────────────

def load_models() -> tuple:
    winner_path = MODELS_DIR / "mlb_logreg_rolling.joblib"
    totals_path = MODELS_DIR / "mlb_totals_v3.joblib"

    if not winner_path.exists():
        print(f"  Modelo ganador no encontrado: {winner_path}")
        print("  Ejecuta: python src/models/train.py")
        sys.exit(1)

    winner = joblib.load(winner_path)
    totals = joblib.load(totals_path) if totals_path.exists() else None

    print(f"  Winner model cargado  (test_acc={winner.get('test_acc', '?'):.4f})")
    if totals:
        print(f"  Totals model cargado  ({totals.get('model_name','?')}  "
              f"MAE={totals.get('test_mae','?'):.3f}  "
              f"O/U={totals.get('test_ou_acc','?'):.4f})")
    return winner, totals


# ─────────────────────────────────────────────
# Construcción de features en tiempo real
# ─────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> dict:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    time.sleep(0.15)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


class LiveFeatureBuilder:
    """Calcula el vector de features para un partido futuro usando datos históricos."""

    def __init__(self):
        self._team_feats: Optional[pd.DataFrame] = None
        self._games_base: Optional[pd.DataFrame] = None

    def _load_cached(self):
        if self._team_feats is None:
            p = PROCESSED_DIR / "mlb_team_games_features.csv"
            if not p.exists():
                raise FileNotFoundError(
                    f"Ejecuta el pipeline de datos primero: {p}\n"
                    "  python src/ingest/build_base_games.py\n"
                    "  python src/ingest/build_team_games_base.py\n"
                    "  python src/features/build_team_rolling_features.py"
                )
            self._team_feats = pd.read_csv(p, parse_dates=["date"])
        if self._games_base is None:
            p2 = PROCESSED_DIR / "mlb_games_base.csv"
            if p2.exists():
                self._games_base = pd.read_csv(p2, parse_dates=["date"])

    def _get_latest_team_row(self, team_name: str) -> pd.Series:
        df = self._team_feats
        rows = df[df["team"] == team_name].sort_values("date")
        if len(rows) == 0:
            return pd.Series(dtype=float)
        return rows.iloc[-1]

    def _days_rest(self, team_name: str, match_date: date) -> tuple[int, int]:
        """Retorna (days_rest, back_to_back)."""
        if self._games_base is None:
            return 3, 0
        df = self._games_base
        team_games = df[
            ((df["home_team"] == team_name) | (df["away_team"] == team_name))
            & (df["status"] == "FINISHED")
            & (df["date"].dt.date < match_date)
        ].sort_values("date")

        if len(team_games) == 0:
            return 3, 0
        last_date = team_games.iloc[-1]["date"].date()
        rest = (match_date - last_date).days
        # cap igual que en entrenamiento: off-season no debe dominar el modelo
        rest = min(rest, 5)
        return rest, int(rest == 1)

    def build(self, home_name: str, away_name: str,
              match_date: date, feat_cols: list[str]) -> dict:
        self._load_cached()

        home_row = self._get_latest_team_row(home_name)
        away_row = self._get_latest_team_row(away_name)

        home_rest, home_b2b = self._days_rest(home_name, match_date)
        away_rest, away_b2b = self._days_rest(away_name, match_date)

        feat: dict = {}

        # prefixar todas las columnas rolling disponibles
        roll_cols = [c for c in home_row.index
                     if any(c.endswith(f"_roll{w}") for w in [5, 10, 20])
                     or c in ("WIN_STREAK",)]

        for col in roll_cols:
            feat[f"HOME_{col}"] = home_row.get(col, 0) if len(home_row) else 0
            feat[f"AWAY_{col}"] = away_row.get(col, 0) if len(away_row) else 0

        # contexto
        feat["HOME_DAYS_REST"] = home_rest
        feat["HOME_BACK_TO_BACK"] = home_b2b
        feat["HOME_IS_HOME"] = 1
        feat["AWAY_DAYS_REST"] = away_rest
        feat["AWAY_BACK_TO_BACK"] = away_b2b
        feat["AWAY_IS_HOME"] = 0

        feat["DIFF_DAYS_REST"] = home_rest - away_rest

        # diferenciales automáticos
        for w in [5, 10, 20]:
            for pattern in [
                "WIN_RATE", "RUN_DIFF", "RUNS_FOR", "RUNS_AGAINST",
                "HITS", "HR", "WALKS", "SO_BAT", "SO_PIT", "BB_ALLOWED", "ERRORS",
            ]:
                col = f"{pattern}_roll{w}"
                h = feat.get(f"HOME_{col}", 0)
                a = feat.get(f"AWAY_{col}", 0)
                feat[f"DIFF_{col}"] = h - a

        # interacciones (igual que en train_totals_v3)
        for w in [5, 10, 20]:
            h_off = feat.get(f"HOME_RUNS_FOR_roll{w}", 0)
            a_def = feat.get(f"AWAY_RUNS_AGAINST_roll{w}", 0)
            a_off = feat.get(f"AWAY_RUNS_FOR_roll{w}", 0)
            h_def = feat.get(f"HOME_RUNS_AGAINST_roll{w}", 0)
            feat[f"INTER_RUNS_FOR_roll{w}_x_RUNS_AGAINST_roll{w}"] = h_off * a_def
            feat[f"INTER_RUNS_FOR_roll{w}_x_RUNS_AGAINST_roll{w}_away"] = a_off * h_def

        # alinear con feat_cols del modelo
        row = {c: feat.get(c, 0) for c in feat_cols}
        return feat, row


# ─────────────────────────────────────────────
# Predicción de un partido
# ─────────────────────────────────────────────

def predict_game(
    home_code: str, away_code: str,
    winner_bundle: dict, totals_bundle: Optional[dict],
    builder: LiveFeatureBuilder,
    match_date: date,
    ou_line: float = OU_LINE,
    verbose: bool = True,
) -> dict:
    # import directo por path para soportar tanto "python src/predict/predict_cli.py"
    # como "python -m src.predict.predict_cli"
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "explain_natural",
        Path(__file__).parent / "explain_natural.py"
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_signals = _mod.build_signals

    home_name = TEAM_CODES.get(home_code.upper(), home_code)
    away_name = TEAM_CODES.get(away_code.upper(), away_code)

    w_model = winner_bundle["model"]
    w_feats = winner_bundle["features"]
    feat_dict, w_row = builder.build(home_name, away_name, match_date, w_feats)

    X_w = pd.DataFrame([w_row])[w_feats]  # orden exacto del entrenamiento
    home_win_prob = float(w_model.predict_proba(X_w)[0][1])
    predicted_winner = home_code if home_win_prob >= 0.5 else away_code
    winner_conf = home_win_prob if home_win_prob >= 0.5 else 1 - home_win_prob

    result = {
        "home": home_code,
        "away": away_code,
        "home_name": home_name,
        "away_name": away_name,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
        "predicted_winner": predicted_winner,
        "winner_confidence": winner_conf,
        "predicted_total": None,
        "over_prob": None,
        "ou_prediction": None,
        "ou_line": ou_line,
    }

    if totals_bundle:
        t_model = totals_bundle["model"]
        t_feats = totals_bundle["features"]
        _, t_row = builder.build(home_name, away_name, match_date, t_feats)
        X_t = pd.DataFrame([t_row])[t_feats]  # orden exacto del entrenamiento
        predicted_total = float(t_model.predict(X_t)[0])
        over_prob = min(0.99, max(0.01, _poisson_over(predicted_total, ou_line)))
        ou_prediction = "OVER" if predicted_total > ou_line else "UNDER"

        result.update({
            "predicted_total": predicted_total,
            "over_prob": over_prob,
            "ou_prediction": ou_prediction,
        })

    if verbose:
        signals = build_signals(home_name, away_name, feat_dict)
        result["signals"] = signals

    return result


def _poisson_over(lamb: float, threshold: float) -> float:
    from math import exp, factorial
    lamb = max(3.0, lamb)
    k = int(threshold)
    prob_under = sum(
        exp(-lamb) * (lamb ** i) / factorial(i)
        for i in range(k + 1)
    )
    return 1 - prob_under


# ─────────────────────────────────────────────
# Formato de salida
# ─────────────────────────────────────────────

SEP = "─" * 80

def print_prediction(r: dict, ou_line: float):
    home, away = r["home"], r["away"]
    winner = r["predicted_winner"]
    loser  = home if winner == away else away
    conf   = r["winner_confidence"]
    loser_conf = 1 - conf

    winner_name = r["home_name"] if winner == home else r["away_name"]
    loser_name  = r["home_name"] if loser  == home else r["away_name"]

    print(f"\n{SEP}")
    print(f"⚾  PARTIDO: {away} @ {home}")
    print(f"   {r['away_name']} @ {r['home_name']}")
    print(SEP)

    # ── Veredicto ──────────────────────────────────────────────
    print(f"\n🎯 EL MODELO DICE: {winner} GANA")
    print(f"   Confianza: {conf:.0%}")
    print(f"   {away}: {r['away_win_prob']:.0%} chance  |  {home}: {r['home_win_prob']:.0%} chance")

    if "signals" not in r:
        # modo verbose=False, solo imprimir cabecera
        if r["predicted_total"] is not None:
            ou_rec = r["ou_prediction"]
            over_p = r["over_prob"]
            print(f"\n🧾 Total: {r['predicted_total']:.1f} carreras  →  {ou_rec} {ou_line}  (over prob: {over_p:.0%})")
            print(f"\nDATA|{away}|{home}||{conf:.4f}|{r['predicted_total']:.1f}|{ou_line}|{ou_rec}")
        return

    sig = r["signals"]

    # ── Por qué favorece al ganador ─────────────────────────────
    print(f"\n✅ ¿POR QUÉ FAVORECE A {winner}?")
    print(SEP)
    lines_winner = sig["home_lines"] if winner == home else sig["away_lines"]
    for line in lines_winner[:5]:
        print(f"  • {line}")

    # ── Qué favorece al otro equipo ─────────────────────────────
    print(f"\n❌ ¿QUÉ FAVORECE A {loser}?")
    print(SEP)
    lines_loser = sig["home_lines"] if loser == home else sig["away_lines"]
    for line in lines_loser[:4]:
        print(f"  • {line}")

    # ── Predicción de carreras ──────────────────────────────────
    if r["predicted_total"] is not None:
        total = r["predicted_total"]
        over_p = r["over_prob"]
        ou_rec = r["ou_prediction"]

        # estimación por equipo (split simple basado en offensive vs defensive strength)
        h_off = sig["home_stats"]["runs_for_10"]
        a_def = sig["away_stats"]["runs_against_10"]
        a_off = sig["away_stats"]["runs_for_10"]
        h_def = sig["home_stats"]["runs_against_10"]

        raw_h = (h_off + a_def) / 2
        raw_a = (a_off + h_def) / 2
        scale = total / (raw_h + raw_a) if (raw_h + raw_a) > 0 else 1
        est_h = raw_h * scale
        est_a = raw_a * scale

        print(f"\n🎯 PREDICCIÓN DE CARRERAS:")
        print(f"   {home}: {est_h:.1f} carreras")
        print(f"   {away}: {est_a:.1f} carreras")
        print(f"   Total: {total:.1f} carreras")

        # contexto de totales
        print(f"\n🧾 O/U {ou_line}: {ou_rec}  (over prob: {over_p:.0%})")
        for ctx in sig["totals_context"][:3]:
            print(f"   → {ctx}")

        # línea DATA para copiar/pegar
        print(f"\nDATA|{away}|{home}||{conf:.4f}|{total:.1f}|{ou_line}|{ou_rec}")
    else:
        print(f"\nDATA|{away}|{home}||{conf:.4f}|N/A|{ou_line}|N/A")


# ─────────────────────────────────────────────
# Obtener juegos del día desde la API
# ─────────────────────────────────────────────

def fetch_todays_games(target_date: date) -> list[dict]:
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        data = _api_get("/schedule", {
            "sportId": 1,
            "date": date_str,
            "gameType": "R,E",   # Regular + Exhibition (cubre todos los tipos del día)
            "hydrate": "team",   # "team" trae abbreviation; "teams" no la incluye
        })
    except Exception as e:
        print(f"  Error consultando API: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            home_code = home.get("abbreviation", "")
            away_code = away.get("abbreviation", "")
            # ignorar juegos sin código de equipo reconocido
            if home_code not in TEAM_CODES or away_code not in TEAM_CODES:
                continue
            games.append({
                "home_code": home_code,
                "away_code": away_code,
                "home_name": home["name"],
                "away_name": away["name"],
                "game_pk": g["gamePk"],
                "time": g.get("gameDate", "")[11:16] + " UTC",
            })
    return games


# ─────────────────────────────────────────────
# Menú principal
# ─────────────────────────────────────────────

def parse_game_line(line: str) -> tuple[str, str, float] | None:
    """Parsea 'AWAY @ HOME' o 'AWAY @ HOME | 8.5' → (away, home, ou_line)"""
    line = line.strip()
    ou_line = OU_LINE
    if "|" in line:
        parts = line.split("|")
        line = parts[0].strip()
        try:
            ou_line = float(parts[1].strip())
        except ValueError:
            pass

    if "@" not in line:
        return None
    parts = [p.strip().upper() for p in line.split("@")]
    if len(parts) != 2:
        return None
    away, home = parts[0], parts[1]
    if away not in TEAM_CODES or home not in TEAM_CODES:
        # intentar por nombre parcial
        away = _fuzzy_code(away)
        home = _fuzzy_code(home)
        if not away or not home:
            return None
    return away, home, ou_line


def _fuzzy_code(query: str) -> str | None:
    query = query.upper()
    if query in TEAM_CODES:
        return query
    for code, name in TEAM_CODES.items():
        if query in name.upper() or query in code:
            return code
    return None


def mode_manual(winner, totals, builder):
    print("\n  Formato: AWAY @ HOME  (ej. BOS @ NYY)")
    print("  Línea O/U opcional: BOS @ NYY | 8.5")
    print("  Escribe DONE para terminar.\n")

    default_ou = OU_LINE
    first = input("  Línea O/U por defecto [Enter=8.5]: ").strip()
    if first:
        try:
            default_ou = float(first)
        except ValueError:
            pass

    today = date.today()
    while True:
        raw = input("\n  Partido (AWAY @ HOME): ").strip()
        if raw.upper() in ("DONE", "Q", "SALIR", ""):
            break
        parsed = parse_game_line(raw)
        if parsed is None:
            print(f"  Formato inválido. Usa: BOS @ NYY")
            print(f"  Códigos: {', '.join(sorted(TEAM_CODES.keys()))}")
            continue
        away_code, home_code, ou_line = parsed
        if ou_line == OU_LINE:
            ou_line = default_ou
        r = predict_game(home_code, away_code, winner, totals, builder, today,
                         ou_line=ou_line)
        print_prediction(r, ou_line)


def mode_by_date(winner, totals, builder):
    raw = input("\n  Fecha (YYYY-MM-DD) [Enter=hoy]: ").strip()
    if not raw:
        target = date.today()
    else:
        try:
            target = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print("  Formato inválido.")
            return

    print(f"\n  Buscando juegos del {target}...")
    games = fetch_todays_games(target)

    if not games:
        print("  No se encontraron juegos programados.")
        return

    ou_raw = input(f"  Línea O/U para todos [Enter={OU_LINE}]: ").strip()
    ou_line = float(ou_raw) if ou_raw else OU_LINE

    print(f"\n  {len(games)} juegos encontrados para {target}")

    for g in games:
        home_code = g["home_code"]
        away_code = g["away_code"]
        if home_code not in TEAM_CODES or away_code not in TEAM_CODES:
            print(f"  Saltando {away_code} @ {home_code} (código no reconocido)")
            continue
        r = predict_game(home_code, away_code, winner, totals, builder, target,
                         ou_line=ou_line)
        print_prediction(r, ou_line)


def main():
    print("\n  MLB v2 — Predictor de Partidos")
    print("  " + "=" * 40)

    winner, totals = load_models()
    builder = LiveFeatureBuilder()

    menu = [
        ("Predicción manual (AWAY @ HOME)", mode_manual),
        ("Predicciones por fecha (busca en API)", mode_by_date),
        ("Salir", None),
    ]

    while True:
        print("\n")
        for i, (label, _) in enumerate(menu, 1):
            print(f"  {i}. {label}")
        choice = input("\n  Selecciona (1-3): ").strip()
        if choice == "3" or choice.lower() in ("q", "salir"):
            print("\n  MLB v2 finalizado.")
            break
        if choice == "1":
            mode_manual(winner, totals, builder)
        elif choice == "2":
            mode_by_date(winner, totals, builder)
        else:
            print("  Opción inválida.")


if __name__ == "__main__":
    main()
