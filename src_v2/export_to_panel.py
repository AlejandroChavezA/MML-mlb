import os
import json
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


def build_arguments(home_team: str, away_team: str, pred: Dict,
                    runs_pred: Dict) -> Dict:
    for_winner = []
    for_loser = []

    winner_team = pred.get("predicted", "LOCAL")
    confidence = pred.get("confidence", 0)

    for_winner.append(f"Confianza del modelo: {confidence:.0%}")
    for_winner.append(f"Modelo: {pred.get('model', 'desconocido')}")

    probs = pred.get("probabilities", {})
    for_winner.append(f"Probabilidad LOCAL: {probs.get('LOCAL', 0):.0%}")
    for_loser.append(f"Probabilidad VISITANTE: {probs.get('VISITANTE', 0):.0%}")

    if "error" not in (runs_pred or {}):
        ou = (runs_pred or {}).get("markets", {}).get("over_8.5", {})
        if ou:
            for_winner.append(f"O/U 8.5: {ou.get('prediction', '?')} ({ou.get('over_prob', 0):.0%})")

    winner_factors = len(for_winner)
    loser_factors = len(for_loser)

    return {
        "forWinner": for_winner,
        "forLoser": for_loser,
        "summary": {
            "winnerFactors": winner_factors,
            "loserFactors": loser_factors,
            "matchupType": "mlb_regular",
            "betRecommendation": f"{winner_team} with {confidence:.0%} confidence",
        },
    }


def transform_to_panel_format(
    home_team: str, away_team: str, home_code: str, away_code: str,
    home_id: int, away_id: int, game_date: datetime,
    winner_pred: Dict, runs_pred: Dict
) -> Dict:
    confidence_pct = round(winner_pred.get("confidence", 0) * 100)
    risk = confidence_to_risk(winner_pred.get("confidence", 0))

    predicted_winner_code = home_code if winner_pred.get("code") == 1 else away_code

    return {
        "sport": "mlb",
        "homeTeam": home_code,
        "homeTeamFullName": home_team,
        "homeTeamLogo": TEAM_LOGO_URLS.get(home_id, ""),
        "awayTeam": away_code,
        "awayTeamFullName": away_team,
        "awayTeamLogo": TEAM_LOGO_URLS.get(away_id, ""),
        "predictedWinner": predicted_winner_code,
        "confidence": confidence_pct,
        "riskLevel": risk,
        "gameDate": game_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "active",
        "arguments": build_arguments(home_team, away_team, winner_pred, runs_pred),
    }


def send_predictions(predictions: List[Dict], api_key: str = None) -> Dict:
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
        payload = {"predictions": predictions}
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
                "over_under": "OVER" if ou.get("code") == 1 else "UNDER",
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
