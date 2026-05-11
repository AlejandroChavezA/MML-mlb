"""
MLB Data Collector
==================
Recolecta datos desde MLB Stats API (statsapi.mlb.com).

Endpoints:
- /api/v1/schedule?season={year}&sportId=1
- /api/v1/game/{game_pk}/boxscore
- /api/v1/standings?leagueId=103,104&season={year}
- /api/v1/teams?sportId=1
- /api/v1/people/{person_id}/stats?stats=season&group=pitching&season={year}

Dependencias:
- requests

Salida:
- data/games_{year}.csv
- data/standings_{year}.csv
- data/teams.csv
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta


API_BASE = "https://statsapi.mlb.com/api/v1"

LEAGUE_IDS = {"AL": 103, "NL": 104, "MLB": 1}
SPORT_ID = 1


class MLBDataCollector:
    """Recolecta datos desde MLB Stats API"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MML-MLB/1.0",
            "Accept": "application/json",
        })

    def _request(self, endpoint: str, params: Dict = None) -> dict:
        """Hacer request a la API con rate limiting"""
        url = f"{API_BASE}/{endpoint.lstrip('/')}"
        time.sleep(0.15)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_teams(self) -> pd.DataFrame:
        """Obtener lista de equipos MLB"""
        data = self._request("/teams", {"sportId": SPORT_ID})
        teams = []
        for t in data.get("teams", []):
            teams.append({
                "team_id": t["id"],
                "name": t["name"],
                "team_code": t.get("abbreviation", ""),
                "short_name": t.get("shortName", ""),
                "league": t.get("league", {}).get("abbreviation", ""),
                "division": t.get("division", {}).get("name", ""),
                "venue": t.get("venue", {}).get("name", ""),
                "league_id": t.get("league", {}).get("id", ""),
                "division_id": t.get("division", {}).get("id", ""),
            })
        df = pd.DataFrame(teams)
        df.to_csv(self.data_dir / "teams.csv", index=False)
        print(f"  Equipos: {len(df)}")
        return df

    def get_schedule(self, season: int, team_id: Optional[int] = None,
                     hydrate: bool = True) -> pd.DataFrame:
        """Obtener calendario de juegos para una temporada"""
        params = {"sportId": SPORT_ID, "season": season}
        if team_id:
            params["teamId"] = team_id
        if hydrate:
            params["hydrate"] = "probablePitcher"

        data = self._request("/schedule", params)
        games = []

        for date_entry in data.get("dates", []):
            date_str = date_entry.get("date", "")
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("detailedState", "SCHEDULED")
                is_finished = status == "Final"

                home_team_data = game["teams"]["home"]
                away_team_data = game["teams"]["away"]

                home_pitcher = home_team_data.get("probablePitcher", {})
                away_pitcher = away_team_data.get("probablePitcher", {})
                home_pitcher_id = home_pitcher.get("id") if home_pitcher else None
                away_pitcher_id = away_pitcher.get("id") if away_pitcher else None

                game_data = {
                    "game_pk": game["gamePk"],
                    "date": date_str,
                    "status": "FINISHED" if is_finished else "SCHEDULED",
                    "home_team": home_team_data["team"]["name"],
                    "away_team": away_team_data["team"]["name"],
                    "home_team_code": home_team_data["team"].get("abbreviation", ""),
                    "away_team_code": away_team_data["team"].get("abbreviation", ""),
                    "home_team_id": home_team_data["team"]["id"],
                    "away_team_id": away_team_data["team"]["id"],
                    "venue": game.get("venue", {}).get("name", ""),
                    "doubleheader": game.get("doubleHeader", "N") == "Y",
                    "day_night": game.get("dayNight", ""),
                    "home_pitcher_id": home_pitcher_id,
                    "away_pitcher_id": away_pitcher_id,
                }

                if is_finished:
                    game_data["home_runs"] = home_team_data.get("score", 0)
                    game_data["away_runs"] = away_team_data.get("score", 0)
                    game_data["innings"] = game.get("innings", [])
                else:
                    game_data["home_runs"] = np.nan
                    game_data["away_runs"] = np.nan
                    game_data["innings"] = []

                # Boxscore details (only for finished, fetched later if needed)
                game_data["home_hits"] = np.nan
                game_data["away_hits"] = np.nan
                game_data["home_errors"] = np.nan
                game_data["away_errors"] = np.nan

                games.append(game_data)

        df = pd.DataFrame(games)
        if len(df) > 0:
            df["total_runs"] = df["home_runs"].fillna(0) + df["away_runs"].fillna(0)
        return df

    def get_boxscore(self, game_pk: int) -> Dict:
        """Obtener boxscore detallado de un juego"""
        data = self._request(f"/game/{game_pk}/boxscore")
        result = {}

        for team_side in ["home", "away"]:
            team_data = data.get("teams", {}).get(team_side, {})
            team_info = team_data.get("team", {})

            result[f"{team_side}_hits"] = team_data.get("teamStats", {}).get("batting", {}).get("hits", 0)
            result[f"{team_side}_errors"] = team_data.get("teamStats", {}).get("fielding", {}).get("errors", 0)

            pitchers = team_data.get("pitchers", [])
            result[f"{team_side}_pitcher_id"] = pitchers[0] if pitchers else None

        return result

    def enrich_games_with_boxscores(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Enriquecer juegos FINISHED con hits/errors desde boxscore"""
        df = games_df.copy()
        finished = df[df["status"] == "FINISHED"].copy()

        if len(finished) == 0:
            return df

        needs_box = finished["home_hits"].isna() | finished["home_errors"].isna()

        for idx in finished[needs_box].index:
            try:
                box = self.get_boxscore(df.loc[idx, "game_pk"])
                df.loc[idx, "home_hits"] = box.get("home_hits", 0)
                df.loc[idx, "away_hits"] = box.get("away_hits", 0)
                df.loc[idx, "home_errors"] = box.get("home_errors", 0)
                df.loc[idx, "away_errors"] = box.get("away_errors", 0)
            except Exception:
                df.loc[idx, "home_hits"] = 0
                df.loc[idx, "away_hits"] = 0
                df.loc[idx, "home_errors"] = 0
                df.loc[idx, "away_errors"] = 0

        return df

    def get_standings(self, season: int) -> pd.DataFrame:
        """Obtener posiciones para una temporada"""
        data = self._request("/standings", {
            "leagueId": "103,104",
            "season": season,
            "standingsTypes": "regularSeason",
        })

        records = []
        for entry in data.get("records", []):
            league = entry.get("league", {}).get("abbreviation", "")
            division = entry.get("division", {}).get("name", "")
            for team_entry in entry.get("teamRecords", []):
                team = team_entry.get("team", {})
                records.append({
                    "team": team.get("name", ""),
                    "team_code": team.get("abbreviation", ""),
                    "team_id": team.get("id", ""),
                    "league": league,
                    "division": division,
                    "wins": team_entry.get("wins", 0),
                    "losses": team_entry.get("losses", 0),
                    "win_pct": team_entry.get("leagueRecord", {}).get("pct", 0),
                    "gb": team_entry.get("gamesBack", ""),
                    "wild_card_gb": team_entry.get("wildCardGamesBack", ""),
                    "last_10": team_entry.get("records", {}).get("lastTen", ""),
                    "streak": team_entry.get("streak", {}).get("streakCode", ""),
                    "runs_scored": team_entry.get("runsScored", 0),
                    "runs_allowed": team_entry.get("runsAllowed", 0),
                    "run_diff": team_entry.get("runDifferential", 0),
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            output_path = self.data_dir / f"standings_{season}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Standings {season}: {len(df)} equipos")
        return df

    def get_pitcher_season_stats(self, pitcher_id: int, season: int) -> Dict:
        """Obtener stats de temporada para un pitcher"""
        try:
            data = self._request(f"/people/{pitcher_id}/stats", {
                "stats": "season",
                "group": "pitching",
                "season": str(season),
            })
            stats_list = data.get("stats", [])
            if stats_list:
                splits = stats_list[0].get("splits", [])
                if splits:
                    s = splits[0].get("stat", {})
                    return {
                        "pitcher_id": pitcher_id,
                        "season": season,
                        "ERA": float(s.get("era", 0) or 0),
                        "W": int(s.get("wins", 0) or 0),
                        "L": int(s.get("losses", 0) or 0),
                        "G": int(s.get("gamesPlayed", 0) or 0),
                        "GS": int(s.get("gamesStarted", 0) or 0),
                        "IP": float(s.get("inningsPitched", "0").split(".")[0] if "." in str(s.get("inningsPitched", "0")) else float(s.get("inningsPitched", 0) or 0)),
                        "SO": int(s.get("strikeOuts", 0) or 0),
                        "BB": int(s.get("baseOnBalls", 0) or 0),
                        "HR": int(s.get("homeRuns", 0) or 0),
                        "H": int(s.get("hits", 0) or 0),
                        "ER": int(s.get("earnedRuns", 0) or 0),
                        "WHIP": float(s.get("whip", 0) or 0),
                        "K_per_9": float(s.get("strikeoutsPer9Inn", 0) or 0),
                        "BB_per_9": float(s.get("walksPer9Inn", 0) or 0),
                        "HR_per_9": float(s.get("homeRunsPer9", 0) or 0),
                        "BABIP": float(s.get("babip", 0) or 0),
                        "FIP": float(s.get("fip", 0) or 0),
                        "WAR": float(s.get("war", 0) or 0),
                    }
        except Exception:
            pass
        return {}

    def collect_all_pitchers(self, season: int, games_df: pd.DataFrame) -> pd.DataFrame:
        """Recolectar stats de todos los pitchers únicos de una temporada"""
        finished = games_df[games_df["status"] == "FINISHED"]
        pitcher_ids = set()
        for col in ["home_pitcher_id", "away_pitcher_id"]:
            pitcher_ids.update(finished[col].dropna().unique().tolist())

        pitcher_ids = {int(pid) for pid in pitcher_ids if pid and not np.isnan(pid)}
        print(f"  Recolectando stats de {len(pitcher_ids)} pitchers...")

        all_stats = []
        for i, pid in enumerate(pitcher_ids):
            stats = self.get_pitcher_season_stats(pid, season)
            if stats:
                all_stats.append(stats)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(pitcher_ids)}")

        df = pd.DataFrame(all_stats)
        if len(df) > 0:
            df = df.replace([None, ""], np.nan)
            for col in df.columns:
                if col not in ["pitcher_id", "season"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            output_path = self.data_dir / f"pitcher_stats_{season}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Pitchers {season}: {len(df)}")
        return df

    def collect_season(self, season: int, enrich: bool = True) -> Dict[str, pd.DataFrame]:
        """Recolectar todos los datos de una temporada"""
        print(f"\n📡 Recolectando temporada {season}...")

        schedule = self.get_schedule(season)
        if enrich and len(schedule) > 0:
            schedule = self.enrich_games_with_boxscores(schedule)

        games_path = self.data_dir / f"games_{season}.csv"
        schedule.to_csv(games_path, index=False)
        print(f"  Games {season}: {len(schedule)} ({len(schedule[schedule['status']=='FINISHED'])} finished)")

        standings = self.get_standings(season)

        pitchers = self.collect_all_pitchers(season, schedule)

        return {"games": schedule, "standings": standings, "pitchers": pitchers}

    def collect_all(self, seasons: List[int] = None, enrich: bool = True) -> Dict[int, Dict[str, pd.DataFrame]]:
        """Recolectar datos de múltiples temporadas"""
        if seasons is None:
            seasons = [2021, 2022, 2023, 2024, 2025]

        print("📡 RECOLECCIÓN MLB STATS API")
        print("=" * 50)

        teams = self.get_teams()

        results = {}
        for season in seasons:
            results[season] = self.collect_season(season, enrich=enrich)

        print("\n✅ Recolección completa")
        return results


def get_mlb_collector(data_dir: str = "data") -> MLBDataCollector:
    """Factory function"""
    return MLBDataCollector(data_dir)
