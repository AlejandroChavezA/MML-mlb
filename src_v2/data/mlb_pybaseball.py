"""
MLB Pybaseball Wrapper
======================
Wraps pybaseball para obtener estadísticas vía Baseball-Reference.

Funciones:
- get_team_batting_stats(season) → BA, OBP, SLG, OPS, HR, R, BB, SO (por equipo)
- get_team_pitching_stats(season) → ERA, WHIP, BABIP, K/9, HR, BB, SO (por equipo)
- get_pitcher_stats(season) → stats individuales de cada pitcher
- get_park_factors() → factores de parque desde MLB Stats API

Dependencias:
- pybaseball
- pandas

Salida:
- data/team_batting_{year}.csv
- data/team_pitching_{year}.csv
- data/pitcher_stats_{year}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class PybaseballWrapper:
    """Wrapper para pybaseball con manejo de errores"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pybaseball = None

    def _import_pybaseball(self):
        if self._pybaseball is None:
            try:
                import pybaseball as pb
                self._pybaseball = pb
            except ImportError:
                raise ImportError(
                    "pybaseball no está instalado. Ejecuta: pip install pybaseball"
                )
        return self._pybaseball

    def get_team_batting_stats(self, season: int) -> pd.DataFrame:
        """Obtener estadísticas ofensivas por equipo desde Baseball-Reference"""
        pb = self._import_pybaseball()

        try:
            data = pb.batting_stats_bref(season)
        except Exception as e:
            print(f"  ⚠️  Error en batting_stats_bref({season}): {e}")
            return pd.DataFrame()

        if data is None or len(data) == 0:
            return pd.DataFrame()

        team_col = "Tm"
        if team_col not in data.columns:
            return pd.DataFrame()

        single_team = data[~data[team_col].str.contains(",", na=False)].copy()
        if len(single_team) == 0:
            single_team = data.copy()

        sum_cols = ["HR", "R", "H", "BB", "SO", "SB", "G", "PA", "AB"]
        available_sum = {k: "sum" for k in sum_cols if k in data.columns}
        mean_cols = ["BA", "OBP", "SLG", "OPS"]
        available_mean = {k: "mean" for k in mean_cols if k in data.columns}
        available = {**available_sum, **available_mean}
        agg = single_team.groupby(team_col).agg(available).reset_index()
        agg.rename(columns={team_col: "team"}, inplace=True)

        agg["AVG"] = agg["H"] / agg["AB"].replace(0, np.nan)
        agg["runs_per_game"] = agg["R"] / agg["G"].replace(0, np.nan)
        agg["HR_per_game"] = agg["HR"] / agg["G"].replace(0, np.nan)
        agg["BB_pct"] = agg["BB"] / agg["PA"].replace(0, np.nan)
        agg["K_pct"] = agg["SO"] / agg["PA"].replace(0, np.nan)
        agg["season"] = season
        agg = agg.replace([np.inf, -np.inf], np.nan)

        output_path = self.data_dir / f"team_batting_{season}.csv"
        agg.to_csv(output_path, index=False)
        print(f"  Team batting {season}: {len(agg)} equipos")
        return agg

    def get_team_pitching_stats(self, season: int) -> pd.DataFrame:
        """Obtener estadísticas de pitcheo por equipo desde Baseball-Reference"""
        pb = self._import_pybaseball()

        try:
            data = pb.pitching_stats_bref(season)
        except Exception as e:
            print(f"  ⚠️  Error en pitching_stats_bref({season}): {e}")
            return pd.DataFrame()

        if data is None or len(data) == 0:
            return pd.DataFrame()

        team_col = "Tm"
        if team_col not in data.columns:
            return pd.DataFrame()

        single_team = data[~data[team_col].str.contains(",", na=False)].copy()
        if len(single_team) == 0:
            single_team = data.copy()

        sum_cols = ["SO", "BB", "HR", "H", "R", "ER", "IP", "G", "GS"]
        available_sum = {k: "sum" for k in sum_cols if k in data.columns}
        mean_cols = ["BAbip"]
        available_mean = {k: "mean" for k in mean_cols if k in data.columns}
        available = {**available_sum, **available_mean}
        agg = single_team.groupby(team_col).agg(available).reset_index()
        agg.rename(columns={team_col: "team", "BAbip": "BABIP"}, inplace=True)

        agg["ERA"] = agg["ER"] * 9 / agg["IP"].replace(0, np.nan)
        agg["WHIP"] = (agg["BB"] + agg["H"]) / agg["IP"].replace(0, np.nan)
        agg["K_per_9"] = agg["SO"] * 9 / agg["IP"].replace(0, np.nan)
        agg["runs_allowed_per_game"] = agg["R"] / agg["G"].replace(0, np.nan)
        agg["HR_per_9"] = agg["HR"] / (agg["IP"].replace(0, np.nan) / 9)
        agg["BB_per_9"] = agg["BB"] / (agg["IP"].replace(0, np.nan) / 9)

        bullpen = single_team[single_team.get("GS", pd.Series(0)) < single_team.get("G", pd.Series(0)) * 0.5]
        if len(bullpen) > 0:
            bp_sum = bullpen.groupby(team_col)[["ER", "IP", "BB", "H"]].sum().reset_index()
            bp_sum.columns = [team_col, "bp_ER", "bp_IP", "bp_BB", "bp_H"]
            bp_sum["bullpen_era"] = bp_sum["bp_ER"] * 9 / bp_sum["bp_IP"].replace(0, np.nan)
            bp_sum["bullpen_whip"] = (bp_sum["bp_BB"] + bp_sum["bp_H"]) / bp_sum["bp_IP"].replace(0, np.nan)
            agg = agg.merge(
                bp_sum[[team_col, "bullpen_era", "bullpen_whip"]].rename(columns={team_col: "team"}),
                on="team", how="left"
            )

        agg = agg.replace([np.inf, -np.inf], np.nan)

        output_path = self.data_dir / f"team_pitching_{season}.csv"
        agg.to_csv(output_path, index=False)
        print(f"  Team pitching {season}: {len(agg)} equipos")
        return agg

    def get_pitcher_stats(self, season: int) -> pd.DataFrame:
        """Obtener stats individuales de todos los pitchers"""
        pb = self._import_pybaseball()

        try:
            data = pb.pitching_stats_bref(season)
        except Exception as e:
            print(f"  ⚠️  Error en pitcher_stats_bref({season}): {e}")
            return pd.DataFrame()

        if data is None or len(data) == 0:
            return pd.DataFrame()

        cols = ["Name", "Age", "Tm", "G", "GS", "W", "L", "SV", "IP",
                "H", "R", "ER", "BB", "SO", "HR", "ERA", "WHIP", "BAbip",
                "SO9", "mlbID"]
        available = [c for c in cols if c in data.columns]
        result = data[available].copy()
        result = result.rename(columns={
            "Name": "name", "Tm": "team", "BAbip": "BABIP", "SO9": "K_per_9",
            "mlbID": "pitcher_id",
        })
        result["season"] = season
        result = result.replace([np.inf, -np.inf], np.nan)

        output_path = self.data_dir / f"pitcher_stats_{season}.csv"
        result.to_csv(output_path, index=False)
        print(f"  Pitcher stats {season}: {len(result)} pitchers")
        return result

    def get_park_factors(self) -> pd.DataFrame:
        """Obtener factores de parque desde MLB Stats API"""
        import requests
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/teams?sportId=1&fields=teams,id,name,abbreviation,venue",
                timeout=15,
                headers={"User-Agent": "MML-MLB/1.0"},
            )
            data = resp.json()
            rows = []
            for t in data.get("teams", []):
                venue = t.get("venue", {})
                rows.append({
                    "team": t["name"],
                    "team_code": t.get("abbreviation", ""),
                    "venue": venue.get("name", ""),
                    "venue_id": venue.get("id", ""),
                })
            df = pd.DataFrame(rows)
            output_path = self.data_dir / "park_factors.csv"
            df.to_csv(output_path, index=False)
            print(f"  Park factors: {len(df)} estadios (desde MLB API)")
            return df
        except Exception as e:
            print(f"  ⚠️  No se pudieron obtener park factors: {e}")
            return pd.DataFrame()

    def collect_season(self, season: int) -> Dict[str, pd.DataFrame]:
        """Recolectar todas las stats pybaseball para una temporada"""
        print(f"\n📊 Recolectando pybaseball {season}...")
        batting = self.get_team_batting_stats(season)
        pitching = self.get_team_pitching_stats(season)
        pitchers = self.get_pitcher_stats(season)
        return {"batting": batting, "pitching": pitching, "pitchers": pitchers}

    def collect_all(self, seasons: List[int] = None) -> Dict[int, Dict[str, pd.DataFrame]]:
        if seasons is None:
            seasons = [2021, 2022, 2023, 2024, 2025]
        print("📊 RECOLECCIÓN PYBASEBALL (BREF)")
        print("=" * 50)
        park = self.get_park_factors()
        results = {}
        for season in seasons:
            results[season] = self.collect_season(season)
        print("\n✅ Recolección pybaseball completa")
        return results


def get_pybaseball_wrapper(data_dir: str = "data") -> PybaseballWrapper:
    """Factory function"""
    return PybaseballWrapper(data_dir)
