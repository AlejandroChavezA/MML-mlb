"""
MLB Feature Engineer
====================
Crea features para modelos de ML desde datos limpios de MLB.

~40 features por juego:
- Pitchers: ERA, FIP, WHIP, K/9, WAR
- Ofensiva equipo: AVG, OPS, wRC+, runs/game
- Pitcheo equipo: ERA, FIP, WHIP, bullpen
- Contextual: descanso, venue, división, día/noche, park factor
- Standings: win%, GB
- Head-to-Head

Dependencias:
- pandas, numpy
- Depende de: data/cleaned/

Salida:
- Features numéricas para modelos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BREF_TEAM_MAP = {
    "Arizona": "AZ", "Atlanta": "ATL", "Baltimore": "BAL", "Boston": "BOS",
    "Cincinnati": "CIN", "Cleveland": "CLE", "Colorado": "COL",
    "Detroit": "DET", "Houston": "HOU", "Kansas City": "KC",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "Oakland": "OAK", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "San Diego": "SD", "San Francisco": "SF", "Seattle": "SEA",
    "St. Louis": "STL", "Tampa Bay": "TB", "Texas": "TEX",
    "Toronto": "TOR", "Washington": "WSH",
    "New York": "NYY", "Los Angeles": "LAD", "Chicago": "CHC",
}

SHORT_TO_CODE = {
    "Diamondbacks": "AZ", "Braves": "ATL", "Orioles": "BAL",
    "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CWS",
    "Reds": "CIN", "Guardians": "CLE", "Rockies": "COL",
    "Tigers": "DET", "Astros": "HOU", "Royals": "KC",
    "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA",
    "Brewers": "MIL", "Twins": "MIN", "Yankees": "NYY",
    "Mets": "NYM", "Athletics": "OAK", "Phillies": "PHI",
    "Pirates": "PIT", "Padres": "SD", "Giants": "SF",
    "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
}


class MLBFeatureEngineer:
    """Ingeniero de features para predicción de juegos MLB"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self.games: Dict[int, pd.DataFrame] = {}
        self.standings: Dict[int, pd.DataFrame] = {}
        self.team_batting: Dict[int, pd.DataFrame] = {}
        self.team_pitching: Dict[int, pd.DataFrame] = {}
        self.pitcher_stats: Dict[int, pd.DataFrame] = {}
        self.teams: pd.DataFrame = None
        self.park_factors: pd.DataFrame = None

    def load_data(self, years: List[int] = None) -> bool:
        """Cargar todos los datos necesarios"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]

        try:
            for year in years:
                self.games[year] = self._load_csv(
                    f"games_{year}_cleaned.csv", self.cleaned_dir
                )
                self.standings[year] = self._load_csv(
                    f"standings_{year}_cleaned.csv", self.cleaned_dir
                )
                self.team_batting[year] = self._load_csv(
                    f"team_batting_{year}.csv", self.data_dir
                )
                self.team_pitching[year] = self._load_csv(
                    f"team_pitching_{year}.csv", self.data_dir
                )
                self.pitcher_stats[year] = self._load_csv(
                    f"pitcher_stats_{year}.csv", self.data_dir
                )
                self.games[year]["date"] = pd.to_datetime(self.games[year]["date"])

            self.teams = self._load_csv("teams_cleaned.csv", self.cleaned_dir)
            self.park_factors = self._load_csv("park_factors.csv", self.data_dir)

            total = sum(len(g) for g in self.games.values())
            print(f"✅ Datos MLB cargados: {total} juegos ({len(years)} temporadas)")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False

    def _load_csv(self, filename: str, directory: Path) -> pd.DataFrame:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")
        return pd.read_csv(path)

    def _get_team_code(self, team_name: str) -> str:
        """Convertir nombre completo de equipo a código 3 letras"""
        parts = team_name.split()
        if parts[-1] in SHORT_TO_CODE:
            return SHORT_TO_CODE[parts[-1]]
        if len(parts) >= 2 and f"{parts[-2]} {parts[-1]}" in SHORT_TO_CODE:
            return SHORT_TO_CODE[f"{parts[-2]} {parts[-1]}"]
        return ""



    def _games_for_season(self, season: int) -> pd.DataFrame:
        """Obtener juegos de una temporada (fallback a la más reciente)"""
        if season in self.games:
            return self.games[season]
        if self.games:
            latest = max(self.games.keys())
            if latest > season:
                return self.games[latest]
            return self.games[latest]
        return pd.DataFrame()

    def get_team_form(self, team_name: str, date: datetime, n_games: int = 10) -> Dict:
        """Calcular forma reciente de un equipo (últimos N juegos)"""
        season = date.year
        games_df = self._games_for_season(season)

        team_games = games_df[
            (
                (games_df["home_team"] == team_name)
                | (games_df["away_team"] == team_name)
            )
            & (games_df["date"] < date)
            & (games_df["status"] == "FINISHED")
        ].sort_values("date", ascending=False).head(n_games)

        if len(team_games) == 0:
            return {"wins": 0, "losses": 0, "win_rate": 0.5, "runs_for": 0,
                    "runs_against": 0, "games_played": 0}

        wins = 0
        runs_for = 0
        runs_against = 0

        for _, g in team_games.iterrows():
            if g["home_team"] == team_name:
                runs_for += g["home_runs"]
                runs_against += g["away_runs"]
                if g["home_runs"] > g["away_runs"]:
                    wins += 1
            else:
                runs_for += g["away_runs"]
                runs_against += g["home_runs"]
                if g["away_runs"] > g["home_runs"]:
                    wins += 1

        return {
            "wins": wins,
            "losses": len(team_games) - wins,
            "win_rate": wins / len(team_games),
            "runs_for": runs_for,
            "runs_against": runs_against,
            "games_played": len(team_games),
        }

    def get_venue_performance(self, team_name: str, venue: str,
                              date: datetime, n_games: int = 20) -> Dict:
        """Rendimiento como local/visitante"""
        season = date.year
        games_df = self._games_for_season(season)

        if venue == "home":
            mask = (
                (games_df["home_team"] == team_name)
                & (games_df["date"] < date)
                & (games_df["status"] == "FINISHED")
            )
            score_col = "home_runs"
            opp_col = "away_runs"
        else:
            mask = (
                (games_df["away_team"] == team_name)
                & (games_df["date"] < date)
                & (games_df["status"] == "FINISHED")
            )
            score_col = "away_runs"
            opp_col = "home_runs"

        games = games_df[mask].sort_values("date", ascending=False).head(n_games)

        if len(games) == 0:
            return {"win_rate": 0.5, "runs_per_game": 0, "runs_allowed_per_game": 0}

        wins = sum(1 for _, g in games.iterrows() if g[score_col] > g[opp_col])
        return {
            "win_rate": wins / len(games),
            "runs_per_game": games[score_col].mean(),
            "runs_allowed_per_game": games[opp_col].mean(),
        }

    def get_rest_days(self, team_name: str, match_date: datetime) -> Dict:
        """Calcular días de descanso"""
        season = match_date.year
        games_df = self._games_for_season(season)

        team_games = games_df[
            ((games_df["home_team"] == team_name) | (games_df["away_team"] == team_name))
            & (games_df["date"] < match_date)
            & (games_df["status"] == "FINISHED")
        ].sort_values("date", ascending=False)

        if len(team_games) == 0:
            return {"rest_days": 7, "games_7_days": 0, "games_14_days": 0}

        last = team_games.iloc[0]
        rest_days = max(0, (match_date - last["date"]).days)

        week_ago = match_date - timedelta(days=7)
        two_weeks_ago = match_date - timedelta(days=14)
        recent = team_games[team_games["date"] >= two_weeks_ago]

        return {
            "rest_days": rest_days,
            "games_7_days": int((recent["date"] >= week_ago).sum()),
            "games_14_days": len(recent),
        }

    def get_head_to_head(self, home: str, away: str, date: datetime,
                         n_games: int = 10) -> Dict:
        """Estadísticas H2H entre dos equipos en la misma temporada"""
        season = date.year
        games_df = self._games_for_season(season)

        h2h = games_df[
            (
                ((games_df["home_team"] == home) & (games_df["away_team"] == away))
                | ((games_df["home_team"] == away) & (games_df["away_team"] == home))
            )
            & (games_df["date"] < date)
            & (games_df["status"] == "FINISHED")
        ].sort_values("date", ascending=False).head(n_games)

        if len(h2h) == 0:
            return {"games": 0, "home_wins": 0, "away_wins": 0, "avg_total_runs": 0}

        home_wins = 0
        away_wins = 0

        for _, g in h2h.iterrows():
            if g["home_team"] == home:
                if g["home_runs"] > g["away_runs"]:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                if g["away_runs"] > g["home_runs"]:
                    home_wins += 1
                else:
                    away_wins += 1

        return {
            "games": len(h2h),
            "home_wins": home_wins,
            "away_wins": away_wins,
            "avg_total_runs": h2h["total_runs"].mean(),
        }

    def get_standings(self, team_name: str, date: datetime) -> Dict:
        """Posición en standings (win%, GB, con fallback)"""
        season = date.year
        if season not in self.standings:
            if self.standings:
                season = max(self.standings.keys())
            else:
                return self._default_standings()

        stand = self.standings[season]
        row = stand[stand["team"] == team_name]

        if len(row) == 0:
            return self._default_standings()

        s = row.iloc[0]
        return {
            "win_pct": float(s.get("win_pct", 0.5)),
            "gb": float(s.get("gb", 0) if s.get("gb") != "-" else 0),
            "wins": int(s.get("wins", 0)),
            "losses": int(s.get("losses", 0)),
        }

    def _get_bref_team_row(self, team_code: str, df: pd.DataFrame) -> pd.Series:
        """Buscar fila BREF que corresponda al team_code.
        
        Nota: BREF combina equipos de la misma ciudad (NY, Chicago, LA).
        Ambos equipos de la misma ciudad reciben los mismos stats combinados.
        """
        if df is None or len(df) == 0:
            return None

        CITY_TO_BREF = {"NYY": "New York", "NYM": "New York",
                        "LAD": "Los Angeles", "LAA": "Los Angeles",
                        "CHC": "Chicago", "CWS": "Chicago"}

        for _, row in df.iterrows():
            bref_name = row["team"]
            if BREF_TEAM_MAP.get(bref_name) == team_code:
                return row
            if bref_name in CITY_TO_BREF.values() and team_code in CITY_TO_BREF:
                if CITY_TO_BREF.get(team_code) == bref_name:
                    return row

        return None

    def get_team_batting_features(self, team_code: str, season: int) -> Dict:
        """Obtener stats ofensivas del equipo para la temporada"""
        batting = self.team_batting.get(season)
        row = self._get_bref_team_row(team_code, batting)
        if row is None:
            return self._default_batting()

        return {
            "AVG": float(row.get("AVG", 0.25)),
            "OPS": float(row.get("OPS", 0.75)),
            "runs_per_game": float(row.get("runs_per_game", 4.5)),
            "HR_per_game": float(row.get("HR_per_game", 1.2)),
            "BB_pct": float(row.get("BB_pct", 0.08)),
            "K_pct": float(row.get("K_pct", 0.22)),
        }

    def get_team_pitching_features(self, team_code: str, season: int) -> Dict:
        """Obtener stats de pitcheo del equipo para la temporada"""
        pitching = self.team_pitching.get(season)
        row = self._get_bref_team_row(team_code, pitching)
        if row is None:
            return self._default_pitching()

        bullpen_era = row.get("bullpen_era")
        return {
            "ERA": float(row.get("ERA", 4.5)),
            "WHIP": float(row.get("WHIP", 1.3)),
            "K_per_9": float(row.get("K_per_9", 8.5)),
            "bullpen_era": float(bullpen_era) if pd.notna(bullpen_era) else 4.5,
            "bullpen_whip": float(row.get("bullpen_whip", 1.3)),
        }

    def get_pitcher_season_stats(self, pitcher_id: int, season: int) -> Dict:
        """Obtener stats de temporada de un pitcher específico"""
        if pd.isna(pitcher_id) or pitcher_id == 0:
            return self._default_pitcher()

        pitcher_id = int(pitcher_id)
        stats = self.pitcher_stats.get(season)
        if stats is None or len(stats) == 0:
            return self._default_pitcher()

        row = stats[stats["pitcher_id"] == pitcher_id]
        if len(row) == 0:
            return self._default_pitcher()

        r = row.iloc[0]
        return {
            "ERA_s": float(r["ERA"]) if pd.notna(r.get("ERA")) else 4.5,
            "WHIP_s": float(r["WHIP"]) if pd.notna(r.get("WHIP")) else 1.3,
            "K9_s": float(r["K_per_9"]) if pd.notna(r.get("K_per_9")) else 8.0,
            "BABIP_s": float(r["BABIP"]) if pd.notna(r.get("BABIP")) else 0.300,
        }

    def get_pitcher_rolling_form(self, pitcher_id: int, date: datetime,
                                 n_starts: int = 3) -> Dict:
        """Calcular forma reciente de un pitcher (últimas N aperturas).
        
        Usa datos del schedule: runs permitidos por el equipo
        como proxy del rendimiento del pitcher.
        """
        if pd.isna(pitcher_id) or pitcher_id == 0:
            return self._default_pitcher_rolling()

        pitcher_id = int(pitcher_id)
        season = date.year
        games_df = self._games_for_season(season)

        if len(games_df) == 0:
            return self._default_pitcher_rolling()

        pitcher_starts = games_df[
            (
                (games_df["home_pitcher_id"] == pitcher_id)
                | (games_df["away_pitcher_id"] == pitcher_id)
            )
            & (games_df["date"] < date)
            & (games_df["status"] == "FINISHED")
        ].sort_values("date", ascending=False).head(n_starts)

        if len(pitcher_starts) == 0:
            return self._default_pitcher_rolling()

        wins = 0
        runs_allowed_total = 0

        for _, g in pitcher_starts.iterrows():
            if g["home_pitcher_id"] == pitcher_id:
                runs_allowed_total += g["away_runs"]
                if g["home_runs"] > g["away_runs"]:
                    wins += 1
            else:
                runs_allowed_total += g["home_runs"]
                if g["away_runs"] > g["home_runs"]:
                    wins += 1

        return {
            "rolling_starts": len(pitcher_starts),
            "rolling_wins": wins,
            "rolling_win_pct": wins / len(pitcher_starts),
            "rolling_avg_runs": runs_allowed_total / len(pitcher_starts),
        }

    def _lookup_game_pitcher_ids(self, home_team: str, away_team: str,
                                  date: datetime) -> tuple:
        """Buscar pitcher IDs para un juego específico"""
        season = date.year
        games_df = self._games_for_season(season)

        if len(games_df) == 0:
            return None, None

        game = games_df[
            (games_df["home_team"] == home_team)
            & (games_df["away_team"] == away_team)
        ]

        if len(game) > 1:
            game = game[game["date"] == date]
        if len(game) > 1:
            game = game.iloc[:1]

        if len(game) == 0:
            return None, None

        g = game.iloc[0]
        return g.get("home_pitcher_id"), g.get("away_pitcher_id")

    def get_park_factor(self, venue: str) -> float:
        """Obtener factor de parque para un venue"""
        if self.park_factors is None or len(self.park_factors) == 0:
            return 1.0
        row = self.park_factors[self.park_factors["venue"] == venue]
        if len(row) == 0:
            return 1.0
        return 1.0

    def _default_pitcher_rolling(self) -> Dict:
        return {"rolling_starts": 0, "rolling_wins": 0,
                "rolling_win_pct": 0.5, "rolling_avg_runs": 4.5}

    def _best_season(self, target_year: int) -> int:
        """Obtener la mejor temporada disponible para un año objetivo"""
        if target_year in self.games:
            return target_year
        if self.games:
            available = sorted(self.games.keys())
            best = [y for y in available if y <= target_year]
            return best[-1] if best else available[-1]
        return target_year

    def create_match_features(self, home_team: str, away_team: str,
                              date: datetime) -> Dict:
        """Crear todas las features para un juego"""
        season = self._best_season(date.year)
        home_code = self._get_team_code(home_team)
        away_code = self._get_team_code(away_team)

        home_form = self.get_team_form(home_team, date)
        away_form = self.get_team_form(away_team, date)

        home_venue = self.get_venue_performance(home_team, "home", date)
        away_venue = self.get_venue_performance(away_team, "away", date)

        home_stand = self.get_standings(home_team, date)
        away_stand = self.get_standings(away_team, date)

        home_rest = self.get_rest_days(home_team, date)
        away_rest = self.get_rest_days(away_team, date)

        h2h = self.get_head_to_head(home_team, away_team, date)

        home_bat = self.get_team_batting_features(home_code, season)
        away_bat = self.get_team_batting_features(away_code, season)

        home_pitch_team = self.get_team_pitching_features(home_code, season)
        away_pitch_team = self.get_team_pitching_features(away_code, season)

        home_pid, away_pid = self._lookup_game_pitcher_ids(home_team, away_team, date)
        home_pitcher = self.get_pitcher_season_stats(home_pid, season)
        away_pitcher = self.get_pitcher_season_stats(away_pid, season)
        home_rolling = self.get_pitcher_rolling_form(home_pid, date)
        away_rolling = self.get_pitcher_rolling_form(away_pid, date)

        return {
            # Form differences
            "form_win_rate_diff": home_form["win_rate"] - away_form["win_rate"],
            "form_runs_diff": (home_form["runs_for"] - home_form["runs_against"])
                              - (away_form["runs_for"] - away_form["runs_against"]),

            # Venue performance
            "home_venue_win_rate": home_venue["win_rate"],
            "away_venue_win_rate": away_venue["win_rate"],
            "venue_advantage": home_venue["win_rate"] - away_venue["win_rate"],

            # Rest
            "home_rest_days": home_rest["rest_days"],
            "away_rest_days": away_rest["rest_days"],
            "rest_diff": home_rest["rest_days"] - away_rest["rest_days"],
            "home_games_14d": home_rest["games_14_days"],
            "away_games_14d": away_rest["games_14_days"],

            # H2H
            "h2h_games": h2h["games"],
            "h2h_home_win_rate": h2h["home_wins"] / max(1, h2h["games"]),
            "h2h_avg_runs": h2h["avg_total_runs"],

            # Standings
            "home_win_pct": home_stand["win_pct"],
            "away_win_pct": away_stand["win_pct"],
            "win_pct_diff": home_stand["win_pct"] - away_stand["win_pct"],
            "home_gb": home_stand["gb"],
            "away_gb": away_stand["gb"],

            # Team batting
            "home_avg": home_bat["AVG"],
            "away_avg": away_bat["AVG"],
            "home_ops": home_bat["OPS"],
            "away_ops": away_bat["OPS"],
            "home_runs_per_game": home_bat["runs_per_game"],
            "away_runs_per_game": away_bat["runs_per_game"],
            "home_home_runs_per_game": home_bat["HR_per_game"],
            "away_home_runs_per_game": away_bat["HR_per_game"],

            # Team pitching
            "home_team_era": home_pitch_team["ERA"],
            "away_team_era": away_pitch_team["ERA"],
            "home_team_whip": home_pitch_team["WHIP"],
            "away_team_whip": away_pitch_team["WHIP"],
            "home_team_k9": home_pitch_team["K_per_9"],
            "away_team_k9": away_pitch_team["K_per_9"],
            "home_bullpen_era": home_pitch_team["bullpen_era"],
            "away_bullpen_era": away_pitch_team["bullpen_era"],

            # Pitcher season stats
            "home_era_s": home_pitcher["ERA_s"],
            "away_era_s": away_pitcher["ERA_s"],
            "era_s_diff": home_pitcher["ERA_s"] - away_pitcher["ERA_s"],
            "home_whip_s": home_pitcher["WHIP_s"],
            "away_whip_s": away_pitcher["WHIP_s"],
            "home_k9_s": home_pitcher["K9_s"],
            "away_k9_s": away_pitcher["K9_s"],

            # Pitcher rolling form (last 3 starts)
            "home_rolling_starts": home_rolling["rolling_starts"],
            "away_rolling_starts": away_rolling["rolling_starts"],
            "home_rolling_win_pct": home_rolling["rolling_win_pct"],
            "away_rolling_win_pct": away_rolling["rolling_win_pct"],
            "rolling_win_pct_diff": home_rolling["rolling_win_pct"] - away_rolling["rolling_win_pct"],
            "home_rolling_avg_runs": home_rolling["rolling_avg_runs"],
            "away_rolling_avg_runs": away_rolling["rolling_avg_runs"],
            "rolling_runs_diff": away_rolling["rolling_avg_runs"] - home_rolling["rolling_avg_runs"],

            # Is home
            "is_home": 1,
        }

    def create_training_dataset(self, years: List[int] = None
                                 ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Crear dataset para entrenamiento desde juegos FINISHED.
        
        Returns: (features_df, winner_targets, runs_targets)
        """
        if years is None:
            years = [2021, 2022, 2023, 2024]

        all_matches = pd.concat(
            [self.games[y][self.games[y]["status"] == "FINISHED"] for y in years],
            ignore_index=True,
        )

        features_list = []
        targets_winner = []
        targets_runs = []
        errors = 0

        for _, m in all_matches.iterrows():
            try:
                date = m["date"]
                if hasattr(date, "tzinfo") and date.tzinfo:
                    date = date.replace(tzinfo=None)

                feats = self.create_match_features(
                    m["home_team"], m["away_team"], date
                )

                target = 1 if m["home_runs"] > m["away_runs"] else 0

                numeric_feats = {k: v for k, v in feats.items()
                                 if k not in ["home_team", "away_team", "date"]}
                features_list.append(numeric_feats)
                targets_winner.append(target)
                targets_runs.append(m["total_runs"])

            except Exception:
                errors += 1
                continue

        if not features_list:
            return pd.DataFrame(), pd.Series([], name="result"), pd.Series([], name="total_runs")

        if errors:
            print(f"  (errores: {errors})")

        features_df = pd.DataFrame(features_list).fillna(0)
        print(f"✅ Dataset MLB: {len(features_df)} samples, {len(features_df.columns)} features")
        return (features_df,
                pd.Series(targets_winner, name="result"),
                pd.Series(targets_runs, name="total_runs"))

    def get_runs_targets(self, features_df: pd.DataFrame,
                         years: List[int] = None) -> pd.Series:
        """Obtener targets de carreras totales alineados con features_df"""
        if years is None:
            years = [2021, 2022, 2023, 2024]

        all_matches = pd.concat(
            [self.games[y][self.games[y]["status"] == "FINISHED"] for y in years],
            ignore_index=True,
        )

        targets = []
        for _, m in all_matches.iterrows():
            try:
                date = m["date"]
                if hasattr(date, "tzinfo") and date.tzinfo:
                    date = date.replace(tzinfo=None)
                feats = self.create_match_features(m["home_team"], m["away_team"], date)
                numeric_feats = {k: v for k, v in feats.items()
                                 if k not in ["home_team", "away_team", "date"]}
                features_list = [numeric_feats]
                targets.append(m["total_runs"])
            except Exception:
                continue

        return pd.Series(targets, name="total_runs")

    def _default_batting(self) -> Dict:
        return {"AVG": 0.250, "OPS": 0.720, "runs_per_game": 4.5,
                "HR_per_game": 1.2, "BB_pct": 0.08, "K_pct": 0.22}

    def _default_pitching(self) -> Dict:
        return {"ERA": 4.5, "WHIP": 1.3, "K_per_9": 8.5,
                "bullpen_era": 4.5, "bullpen_whip": 1.3}

    def _default_pitcher(self) -> Dict:
        return {"ERA_s": 4.5, "WHIP_s": 1.3, "K9_s": 8.0, "BABIP_s": 0.300}

    def _default_standings(self) -> Dict:
        return {"win_pct": 0.5, "gb": 0, "wins": 0, "losses": 0}


def get_mlb_feature_engineer(data_dir: str = "data") -> MLBFeatureEngineer:
    return MLBFeatureEngineer(data_dir)
