import os
import json
import math
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from src_v2.tracking import save_prediction

load_dotenv(Path(__file__).parent.parent / ".env.local")

PANEL_URL = os.getenv("SAFESPORTS_PANEL_URL", "http://localhost:3000")
PANEL_EMAIL = os.getenv("SAFESPORTS_PANEL_EMAIL", "admin@sudo.com")
PANEL_PASSWORD = os.getenv("SAFESPORTS_PANEL_PASSWORD", "Admin123!")
IMPORT_API_SECRET = os.getenv("IMPORT_API_SECRET")
USER_API_KEY = os.getenv("SAFESPORTS_USER_API_KEY")

TEAM_LOGO_URLS = {
    108: "https://www.mlbstatic.com/team-logos/108.svg",
    109: "https://www.mlbstatic.com/team-logos/109.svg",
    110: "https://www.mlbstatic.com/team-logos/110.svg",
    111: "https://www.mlbstatic.com/team-logos/111.svg",
    112: "https://www.mlbstatic.com/team-logos/112.svg",
    113: "https://www.mlbstatic.com/team-logos/113.svg",
    114: "https://www.mlbstatic.com/team-logos/114.svg",
    115: "https://www.mlbstatic.com/team-logos/115.svg",
    116: "https://www.mlbstatic.com/team-logos/116.svg",
    117: "https://www.mlbstatic.com/team-logos/117.svg",
    118: "https://www.mlbstatic.com/team-logos/118.svg",
    119: "https://www.mlbstatic.com/team-logos/119.svg",
    120: "https://www.mlbstatic.com/team-logos/120.svg",
    121: "https://www.mlbstatic.com/team-logos/121.svg",
    133: "https://www.mlbstatic.com/team-logos/133.svg",
    134: "https://www.mlbstatic.com/team-logos/134.svg",
    135: "https://www.mlbstatic.com/team-logos/135.svg",
    136: "https://www.mlbstatic.com/team-logos/136.svg",
    137: "https://www.mlbstatic.com/team-logos/137.svg",
    138: "https://www.mlbstatic.com/team-logos/138.svg",
    139: "https://www.mlbstatic.com/team-logos/139.svg",
    140: "https://www.mlbstatic.com/team-logos/140.svg",
    141: "https://www.mlbstatic.com/team-logos/141.svg",
    142: "https://www.mlbstatic.com/team-logos/142.svg",
    143: "https://www.mlbstatic.com/team-logos/143.svg",
    144: "https://www.mlbstatic.com/team-logos/144.svg",
    145: "https://www.mlbstatic.com/team-logos/145.svg",
    146: "https://www.mlbstatic.com/team-logos/146.svg",
    147: "https://www.mlbstatic.com/team-logos/147.svg",
    158: "https://www.mlbstatic.com/team-logos/158.svg",
}

TEAM_COLORS = {
    108: {"primary": "#BA0021", "secondary": "#003263", "code": "LAA"},
    109: {"primary": "#A71930", "secondary": "#E3D4AD", "code": "AZ"},
    110: {"primary": "#DF4601", "secondary": "#000000", "code": "BAL"},
    111: {"primary": "#BD3039", "secondary": "#0C2340", "code": "BOS"},
    112: {"primary": "#0E3386", "secondary": "#CC3433", "code": "CHC"},
    113: {"primary": "#C6011F", "secondary": "#000000", "code": "CIN"},
    114: {"primary": "#E31937", "secondary": "#0C2340", "code": "CLE"},
    115: {"primary": "#33006F", "secondary": "#C4CED4", "code": "COL"},
    116: {"primary": "#0C2340", "secondary": "#FA4616", "code": "DET"},
    117: {"primary": "#002D62", "secondary": "#EB6E1F", "code": "HOU"},
    118: {"primary": "#004687", "secondary": "#BD9B60", "code": "KC"},
    119: {"primary": "#005A9C", "secondary": "#EF3E42", "code": "LAD"},
    120: {"primary": "#AB0003", "secondary": "#14225A", "code": "WSH"},
    121: {"primary": "#002D72", "secondary": "#FF5910", "code": "NYM"},
    133: {"primary": "#003831", "secondary": "#FFB81C", "code": "ATH"},
    134: {"primary": "#27251F", "secondary": "#FFB81C", "code": "PIT"},
    135: {"primary": "#2F241D", "secondary": "#FFC425", "code": "SD"},
    136: {"primary": "#005C5C", "secondary": "#C4CED4", "code": "SEA"},
    137: {"primary": "#FD5A1E", "secondary": "#27251F", "code": "SF"},
    138: {"primary": "#C41E3A", "secondary": "#0C2340", "code": "STL"},
    139: {"primary": "#092C5C", "secondary": "#8FBCE6", "code": "TB"},
    140: {"primary": "#003278", "secondary": "#C0111F", "code": "TEX"},
    141: {"primary": "#134A8E", "secondary": "#E8291C", "code": "TOR"},
    142: {"primary": "#002B5C", "secondary": "#D31145", "code": "MIN"},
    143: {"primary": "#E81828", "secondary": "#003278", "code": "PHI"},
    144: {"primary": "#0F437C", "secondary": "#CE1141", "code": "ATL"},
    145: {"primary": "#27251F", "secondary": "#C4CED4", "code": "CWS"},
    146: {"primary": "#00A3E0", "secondary": "#EF3340", "code": "MIA"},
    147: {"primary": "#003087", "secondary": "#E4002B", "code": "NYY"},
    158: {"primary": "#0A2351", "secondary": "#FFC52F", "code": "MIL"},
}

SIGNAL_NL = {
    "form_win_rate_diff": (
        lambda v, h, a: f"{h} llegan con mejor racha que {a} en los últimos juegos" if v > 0
        else f"{a} llegan con mejor racha que {h} en los últimos juegos"
    ),
    "form_runs_diff": (
        lambda v, h, a: f"{h} han anotado {abs(v):.0f} carreras más que {a} en los últimos 5 juegos" if v > 0
        else f"{a} han anotado {abs(v):.0f} carreras más que {h} en los últimos 5 juegos"
    ),
    "venue_advantage": (
        lambda v, h, a: f"{h} rinden mejor como locales que {a}" if v > 0
        else f"{a} rinden mejor como visitantes que {h}"
    ),
    "rest_diff": (
        lambda v, h, a: f"{h} tienen {abs(v):.0f} día(s) más de descanso que {a}" if v > 0
        else f"{a} tienen {abs(v):.0f} día(s) más de descanso que {h}"
    ),
    "h2h_home_win_rate": (
        lambda v, h, a: f"{h} dominan el historial en casa contra {a}" if v > 0
        else f"{a} han tenido éxito en casa de {h} históricamente"
    ),
    "win_pct_diff": (
        lambda v, h, a: f"{h} tienen mejor % de victorias que {a}" if v > 0
        else f"{a} tienen mejor % de victorias que {h}"
    ),
    "rolling_win_pct_diff": (
        lambda v, h, a: f"El abridor de {h} está en mejor forma (3 últimas salidas) que el de {a}" if v > 0
        else f"El abridor de {a} está en mejor forma (3 últimas salidas) que el de {h}"
    ),
    "rolling_runs_diff": (
        lambda v, h, a: f"El abridor de {h} permite {abs(v):.1f} carreras menos por salida que el de {a}" if v > 0
        else f"El abridor de {a} permite {abs(v):.1f} carreras menos por salida que el de {h}"
    ),
    "era_s_diff": (
        lambda v, h, a: f"El abridor de {h} tiene mejor ERA ({abs(v):.2f}) que el de {a}" if v > 0
        else f"El abridor de {a} tiene mejor ERA ({abs(v):.2f}) que el de {h}"
    ),
}

SIGNAL_PAIRS_NL = [
    ("home_ops", "away_ops", lambda v, h, a: f"{h} tienen mejor OPS ({v:+.3f}) que {a}" if v > 0 else f"{a} tienen mejor OPS ({-v:.3f}) que {h}"),
    ("home_avg", "away_avg", lambda v, h, a: f"{h} batean mejor ({v:+.3f} AVG) que {a}" if v > 0 else f"{a} batean mejor ({-v:.3f} AVG) que {h}"),
    ("home_runs_per_game", "away_runs_per_game", lambda v, h, a: f"{h} anotan {abs(v):.2f} carreras más por juego que {a}" if v > 0 else f"{a} anotan {abs(v):.2f} carreras más por juego que {h}"),
    ("home_team_era", "away_team_era", lambda v, h, a: f"{h} tienen mejor ERA ({abs(v):.2f}) que {a}" if v > 0 else f"{a} tienen mejor ERA ({abs(v):.2f}) que {h}"),
    ("home_team_whip", "away_team_whip", lambda v, h, a: f"{h} tienen mejor WHIP ({abs(v):.2f}) que {a}" if v > 0 else f"{a} tienen mejor WHIP ({abs(v):.2f}) que {h}"),
    ("home_bullpen_era", "away_bullpen_era", lambda v, h, a: f"El bullpen de {h} es mejor ({abs(v):.2f} ERA) que el de {a}" if v > 0 else f"El bullpen de {a} es mejor ({abs(v):.2f} ERA) que el de {h}"),
    ("home_team_k9", "away_team_k9", lambda v, h, a: f"{h} ponchan más ({abs(v):.2f} K/9) que {a}" if v > 0 else f"{a} ponchan más ({abs(v):.2f} K/9) que {h}"),
    ("home_k9_s", "away_k9_s", lambda v, h, a: f"El abridor de {h} poncha más ({abs(v):.2f} K/9) que el de {a}" if v > 0 else f"El abridor de {a} poncha más ({abs(v):.2f} K/9) que el de {h}"),
    ("home_whip_s", "away_whip_s", lambda v, h, a: f"El abridor de {h} tiene mejor control ({abs(v):.2f} WHIP) que el de {a}" if v > 0 else f"El abridor de {a} tiene mejor control ({abs(v):.2f} WHIP) que el de {h}"),
    ("home_rolling_avg_runs", "away_rolling_avg_runs", lambda v, h, a: f"El abridor de {h} permite {abs(v):.2f} carreras menos por salida que el de {a}" if v > 0 else f"El abridor de {a} permite {abs(v):.2f} carreras menos por salida que el de {h}"),
    ("home_games_14d", "away_games_14d", lambda v, h, a: f"{h} han jugado {abs(v):.0f} juegos más que {a} en los últimos 14 días" if v > 0 else f"{a} han jugado {abs(v):.0f} juegos más que {h} en los últimos 14 días"),
]

SIGNAL_CONFIG = {
    "form_win_rate_diff":      ("racha reciente", lambda v: v, "%+.3f"),
    "form_runs_diff":          ("diff carreras recientes", lambda v: v, "%+.1f"),
    "venue_advantage":         ("rendimiento como local", lambda v: v, "%+.3f"),
    "rest_diff":               ("días de descanso", lambda v: v, "%+d"),
    "h2h_home_win_rate":       ("historial H2H como local", lambda v: v, "%+.2f"),
    "win_pct_diff":            ("porcentaje de victorias", lambda v: v, "%+.3f"),
    "rolling_win_pct_diff":    ("forma del pitcher (3 salidas)", lambda v: v, "%+.3f"),
    "rolling_runs_diff":       ("carreras permitidas pitcher rival", lambda v: v, "%+.1f"),
    "era_s_diff":              ("ERA del pitcher", lambda v: -v, "%+.2f"),
}

SIGNAL_PAIRS = [
    ("home_ops", "away_ops", "OPS del equipo", False, "%.3f"),
    ("home_avg", "away_avg", "AVG del equipo", False, "%.3f"),
    ("home_runs_per_game", "away_runs_per_game", "carreras por juego", False, "%.2f"),
    ("home_team_era", "away_team_era", "ERA del equipo", True, "%.2f"),
    ("home_team_whip", "away_team_whip", "WHIP del equipo", True, "%.2f"),
    ("home_bullpen_era", "away_bullpen_era", "ERA del bullpen", True, "%.2f"),
    ("home_team_k9", "away_team_k9", "K/9 del equipo", False, "%.2f"),
    ("home_k9_s", "away_k9_s", "K/9 del pitcher titular", False, "%.2f"),
    ("home_whip_s", "away_whip_s", "WHIP del pitcher titular", True, "%.2f"),
    ("home_rolling_avg_runs", "away_rolling_avg_runs", "carreras/salida pitcher", False, "%.2f"),
    ("home_games_14d", "away_games_14d", "juegos en 14 días", False, "%.0f"),
]




def _natural_reasons(fe, home_team, away_team, date_val, winner_model, wpred):
    """Generate natural language reasons for a prediction."""
    winner_is_home = wpred.get("code") == 1
    features = fe.create_match_features(home_team, away_team, date_val)
    fi = winner_model.get_feature_importance()
    if not fi:
        return [], []
    fi_dict = dict(fi)
    h = home_team.split()[-1] if len(home_team.split()) > 1 else home_team
    a = away_team.split()[-1] if len(away_team.split()) > 1 else away_team

    signals = []
    for key, nl_func in SIGNAL_NL.items():
        if key not in features:
            continue
        raw = features[key]
        if key == "era_s_diff":
            ha = -raw
        else:
            ha = raw
        if ha == 0:
            continue
        imp = fi_dict.get(key, 0)
        text = nl_func(ha, h, a)
        signals.append((ha, text, imp))

    for hk, ak, nl_func in SIGNAL_PAIRS_NL:
        if hk not in features or ak not in features:
            continue
        diff = features[hk] - features[ak]
        if diff == 0:
            continue
        imp = max(fi_dict.get(hk, 0), fi_dict.get(ak, 0))
        text = nl_func(diff, h, a)
        signals.append((diff, text, imp))

    signals.sort(key=lambda x: abs(x[0]) * x[2], reverse=True)

    favor, contra = [], []
    seen = set()
    for ha, text, imp in signals:
        key = text.split("(")[0].strip()
        if key in seen:
            continue
        seen.add(key)
        favors_winner = (ha > 0 and winner_is_home) or (ha < 0 and not winner_is_home)
        if favors_winner:
            favor.append(text)
        else:
            contra.append(text)

    return favor[:4], contra[:4]


def _analyze_features(fe, home, away, date_val, winner, wpred):
    winner_is_home = wpred.get("code") == 1
    features = fe.create_match_features(home, away, date_val)
    fi = winner.get_feature_importance()
    if not fi:
        return [], []
    fi_dict = dict(fi)

    signals = []
    for key, (label, transform, fmt) in SIGNAL_CONFIG.items():
        if key not in features:
            continue
        raw = features[key]
        ha = transform(raw)
        if ha == 0:
            continue
        imp = fi_dict.get(key, 0)
        signals.append((ha, label, fmt, imp))

    for hk, ak, label, invert, fmt in SIGNAL_PAIRS:
        if hk not in features or ak not in features:
            continue
        diff = features[hk] - features[ak]
        if invert:
            diff = -diff
        if diff == 0:
            continue
        imp = max(fi_dict.get(hk, 0), fi_dict.get(ak, 0))
        signals.append((diff, label, fmt, imp))

    signals.sort(key=lambda x: abs(x[0]) * x[3], reverse=True)

    favoring_winner = []
    favoring_loser = []
    seen_topics = set()

    for ha, label, fmt, imp in signals:
        topic = label.split("(")[0].strip()
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        favors_winner = (ha > 0 and winner_is_home) or (ha < 0 and not winner_is_home)
        val_str = fmt % ha
        if ha < 0 and fmt.startswith("%+"):
            val_str = "--" + val_str[1:]
        entry = f"{label} ({val_str})"
        if favors_winner:
            favoring_winner.append(entry)
        else:
            favoring_loser.append(entry)

    return favoring_winner[:4], favoring_loser[:4]


def get_api_key() -> Optional[str]:
    if USER_API_KEY:
        print(f"  Usando API key existente")
        return USER_API_KEY

    if not PANEL_EMAIL or not PANEL_PASSWORD:
        print("  No hay credenciales configuradas en .env.local")
        return None

    try:
        url = f"{PANEL_URL}/api/auth/api-key/generate"
        resp = requests.post(url, json={
            "email": PANEL_EMAIL,
            "password": PANEL_PASSWORD,
        }, timeout=10)
        data = resp.json()
        if data.get("success") and data.get("apiKey"):
            print(f"  API key generada: {data['apiKey'][:12]}...")
            return data["apiKey"]
        print(f"  Error obteniendo API key: {data.get('message', 'unknown')}")
        return None
    except Exception as e:
        print(f"  Error conectando al panel: {e}")
        return None


def confidence_to_risk(confidence: float) -> str:
    if confidence >= 0.70:
        return "low"
    elif confidence >= 0.55:
        return "medium"
    return "high"


def _over_under_thresholds(runs_pred: Dict) -> Dict:
    er = runs_pred.get("expected_runs")
    ou8 = runs_pred.get("markets", {}).get("over_8.5", {})
    over_prob = ou8.get("over_prob", 0.5)

    if er is not None:
        thresholds = {}
        for t in [7.5, 8.5, 9.5, 10.5]:
            mu = max(er, 0.1)
            p = 1 - sum(math.exp(-mu) * (mu ** k) / math.factorial(k) for k in range(int(t) + 1))
            prob = max(0.0, min(1.0, p))
            thresholds[f"over_{t}"] = {
                "prob": round(prob, 4),
                "prediction": "OVER" if prob > 0.5 else "UNDER",
            }
        return thresholds

    return {
        "over_8.5": {
            "prob": round(over_prob, 4),
            "prediction": "OVER" if over_prob > 0.5 else "UNDER",
        },
    }


def transform_to_panel_format(
    home_team: str, away_team: str, home_code: str, away_code: str,
    home_id: int, away_id: int, game_date: datetime,
    winner_pred: Dict, runs_pred: Dict,
    feature_engineer=None, winner_model=None,
) -> Dict:
    confidence_pct = round(winner_pred.get("confidence", 0) * 100)
    risk = confidence_to_risk(winner_pred.get("confidence", 0))
    winner_is_home = winner_pred.get("code") == 1
    predicted_winner_code = home_code if winner_is_home else away_code

    probs = winner_pred.get("probabilities", {})
    prob_home = probs.get("LOCAL", 0)
    prob_away = probs.get("VISITANTE", 0)

    prediction_str = "Home Win" if winner_is_home else "Away Win"

    home_img = TEAM_LOGO_URLS.get(home_id, "")
    away_img = TEAM_LOGO_URLS.get(away_id, "")

    reasons_for = []
    reasons_against = []

    if feature_engineer is not None and winner_model is not None:
        fav, ag = _natural_reasons(feature_engineer, home_team, away_team, game_date, winner_model, winner_pred)
        reasons_for = fav
        reasons_against = ag

    return {
        "sport": "mlb",
        "gameDate": game_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "homeTeam": home_code,
        "homeTeamFullName": home_team,
        "homeTeamLogo": home_img,
        "awayTeam": away_code,
        "awayTeamFullName": away_team,
        "awayTeamLogo": away_img,
        "predictedWinner": predicted_winner_code,
        "prediction": prediction_str,
        "confidence": confidence_pct,
        "riskLevel": risk,
        "status": "active",
        "reasonsFor": reasons_for,
        "reasonsAgainst": reasons_against,
        "overUnder": _over_under_thresholds(runs_pred),
    }


def send_predictions(predictions: List[Dict], api_key: str = None,
                     matchday: int = 1) -> Dict:
    if api_key is None:
        api_key = get_api_key()
    if api_key is None:
        return {"success": False, "message": "No API key available"}

    try:
        url = f"{PANEL_URL}/api/predictions/import"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "matchday": matchday,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_matches": len(predictions),
            "predictions": predictions,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        return resp.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def export_todays_games(feature_engineer, winner, runs,
                        year: int = None, model_name: str = None) -> Dict:
    if year is None:
        year = datetime.now().year

    games_df = feature_engineer.games.get(year)
    if games_df is None or len(games_df) == 0:
        return {"success": False, "message": f"No hay datos para {year}"}

    today = datetime.now().date()
    todays_games = games_df[
        (games_df["date"].dt.date == today) & (games_df["status"] == "SCHEDULED")
    ]

    if len(todays_games) == 0:
        return {"success": False, "message": f"No hay juegos programados para hoy ({today})"}

    teams_df = feature_engineer.teams
    team_id_map = {}
    if teams_df is not None and len(teams_df) > 0:
        for _, r in teams_df.iterrows():
            team_id_map[r["name"]] = {"code": r.get("team_code", ""), "id": r.get("team_id", 0)}

    predictions = []
    for _, game in todays_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        date_val = game["date"]
        if hasattr(date_val, "tzinfo") and date_val.tzinfo:
            date_val = date_val.replace(tzinfo=None)

        home_info = team_id_map.get(home, {"code": "", "id": 0})
        away_info = team_id_map.get(away, {"code": "", "id": 0})

        wpred = winner.predict(home, away, date_val, feature_engineer, model_name)
        rpred = runs.predict(home, away, date_val, feature_engineer, model_name)

        if "error" in wpred:
            print(f"  Error prediciendo {home} vs {away}: {wpred['error']}")
            continue

        game_pk = game.get("game_pk")
        if game_pk:
            home_code = home_info.get("code", "")
            away_code = away_info.get("code", "")
            ou = rpred.get("markets", {}).get("over_8.5", {})
            save_prediction(int(game_pk), {
                "game_pk": int(game_pk),
                "date": str(date_val.date()),
                "home_team": home_code,
                "away_team": away_code,
                "home_full": home,
                "away_full": away,
                "predicted_winner": home_code if wpred.get("code") == 1 else away_code,
                "predicted_winner_code": wpred.get("code"),
                "confidence": wpred.get("confidence", 0),
                "winner_model": wpred.get("model", ""),
                "over_under": ou.get("prediction", "UNDER"),
                "over_prob": ou.get("over_prob", 0),
                "runs_model": rpred.get("model", ""),
                "exported": True,
                "timestamp": datetime.now().isoformat(),
            })

        pred = transform_to_panel_format(
            home, away,
            home_info["code"], away_info["code"],
            home_info["id"], away_info["id"],
            date_val, wpred, rpred,
            feature_engineer=feature_engineer,
            winner_model=winner,
        )
        predictions.append(pred)
        print(f"  {home[:20]:20} vs {away:<20} -> {pred['predictedWinner']} ({pred['confidence']}%)")

    if not predictions:
        return {"success": False, "message": "No se pudieron generar predicciones"}

    api_key = get_api_key()
    if api_key is None:
        return {"success": False, "message": "No se pudo obtener API key"}

    result = send_predictions(predictions, api_key)
    return result
