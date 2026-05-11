"""
Data Cleaner (MLB)
=================
Limpieza de datos crudos de MLB -> datos limpios.

Dependencias:
- pandas, numpy

Salida:
- data/cleaned/games_{year}_cleaned.csv
- data/cleaned/standings_{year}_cleaned.csv
- data/cleaned/teams_cleaned.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


BREF_TO_TEAM_CODE = {
    "Arizona": "AZ", "Atlanta": "ATL", "Baltimore": "BAL", "Boston": "BOS",
    "Chi Cubs": "CHC", "Chi White Sox": "CWS", "Cincinnati": "CIN",
    "Cleveland": "CLE", "Colorado": "COL", "Detroit": "DET", "Houston": "HOU",
    "Kansas City": "KC", "Los Angeles": "LAD", "LA Angels": "LAA",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "NY Mets": "NYM", "NY Yankees": "NYY", "Oakland": "OAK",
    "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Diego": "SD",
    "San Francisco": "SF", "Seattle": "SEA", "St. Louis": "STL",
    "Tampa Bay": "TB", "Texas": "TEX", "Toronto": "TOR", "Washington": "WSH",
}

TEAM_CODE_TO_FULL = {
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

BREF_SHORT_TO_TEAM_CODE = {}
for code, full in TEAM_CODE_TO_FULL.items():
    short = full.split()[-1] if full.split()[-1] in [
        "Braves", "Brewers", "Cardinals", "Cubs", "D-backs", "Dodgers",
        "Giants", "Guardians", "Mariners", "Marlins", "Mets", "Orioles",
        "Padres", "Phillies", "Pirates", "Rangers", "Rays", "Red Sox",
        "Reds", "Rockies", "Royals", "Tigers", "Twins", "White Sox",
        "Yankees", "Angels", "Astros", "Athletics", "Blue Jays", "Nationals",
    ] else "Diamondbacks"

# Simple mapping: unique last word -> team_code
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

# Build inverse: team_code -> full_name
# Already have TEAM_CODE_TO_FULL above

# BREF short name -> team_code
BREF_TEAM_MAP = {
    "Arizona": "AZ", "Atlanta": "ATL", "Baltimore": "BAL", "Boston": "BOS",
    "Chicago": None,  # ambiguous (CHC/CWS)
    "Cincinnati": "CIN", "Cleveland": "CLE", "Colorado": "COL",
    "Detroit": "DET", "Houston": "HOU", "Kansas City": "KC",
    "Los Angeles": None,  # ambiguous (LAD/LAA)
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "New York": None,  # ambiguous (NYY/NYM)
    "Oakland": "OAK", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "San Diego": "SD", "San Francisco": "SF", "Seattle": "SEA",
    "St. Louis": "STL", "Tampa Bay": "TB", "Texas": "TEX",
    "Toronto": "TOR", "Washington": "WSH",
}


class DataCleaner:
    """Limpia datos crudos de MLB"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self._ensure_cleaned_dir()

    def _ensure_cleaned_dir(self):
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def clean_games(self, year: int) -> pd.DataFrame:
        """Limpiar datos de juegos para una temporada"""
        input_path = self.data_dir / f"games_{year}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")

        df = pd.read_csv(input_path)
        original_count = len(df)

        df["date"] = pd.to_datetime(df["date"])
        df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
        df["away_runs"] = pd.to_numeric(df["away_runs"], errors="coerce")

        df["total_runs"] = df["home_runs"].fillna(0) + df["away_runs"].fillna(0)
        df["run_difference"] = df["home_runs"].fillna(0) - df["away_runs"].fillna(0)

        def get_result(row):
            if pd.isna(row["home_runs"]):
                return "SCHEDULED"
            elif row["home_runs"] > row["away_runs"]:
                return "LOCAL"
            return "VISITANTE"

        df["result"] = df.apply(get_result, axis=1)

        df["home_team_code"] = df["home_team"].map(
            lambda x: SHORT_TO_CODE.get(x.split()[-1], "")
        )
        df["away_team_code"] = df["away_team"].map(
            lambda x: SHORT_TO_CODE.get(x.split()[-1], "")
        )

        output_path = self.cleaned_dir / f"games_{year}_cleaned.csv"
        df.to_csv(output_path, index=False)

        print(f"  {year}: {original_count} → {len(df)} ({len(df[df['status']=='FINISHED'])} finished)")
        return df

    def clean_standings(self, year: int) -> pd.DataFrame:
        """Limpiar tabla de posiciones (minimal — ya viene limpia de API)"""
        input_path = self.data_dir / f"standings_{year}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")

        df = pd.read_csv(input_path)

        df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce")

        output_path = self.cleaned_dir / f"standings_{year}_cleaned.csv"
        df.to_csv(output_path, index=False)

        print(f"  {year} standings: {len(df)} equipos")
        return df

    def clean_teams(self) -> pd.DataFrame:
        """Dejar teams.csv como está (ya viene limpio de API)"""
        input_path = self.data_dir / "teams.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")

        df = pd.read_csv(input_path)

        output_path = self.cleaned_dir / "teams_cleaned.csv"
        df.to_csv(output_path, index=False)

        print(f"  teams: {len(df)} equipos")
        return df

    def run_cleaning(self, years: Optional[List[int]] = None):
        """Ejecutar limpieza completa"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]

        for year in years:
            print(f"\n[{year}]")
            self.clean_games(year)
            self.clean_standings(year)

        self.clean_teams()

        print(f"\n✅ Limpieza completa → {self.cleaned_dir}")


def get_cleaner(data_dir: str = "data") -> DataCleaner:
    return DataCleaner(data_dir)
