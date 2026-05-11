import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STANDINGS_TO_FULL = {
    "Yankees": "New York Yankees", "Orioles": "Baltimore Orioles",
    "Red Sox": "Boston Red Sox", "Rays": "Tampa Bay Rays",
    "Blue Jays": "Toronto Blue Jays", "Guardians": "Cleveland Guardians",
    "Royals": "Kansas City Royals", "Tigers": "Detroit Tigers",
    "Twins": "Minnesota Twins", "White Sox": "Chicago White Sox",
    "Astros": "Houston Astros", "Mariners": "Seattle Mariners",
    "Rangers": "Texas Rangers", "Athletics": "Athletics",
    "Angels": "Los Angeles Angels", "Phillies": "Philadelphia Phillies",
    "Braves": "Atlanta Braves", "Mets": "New York Mets",
    "Nationals": "Washington Nationals", "Marlins": "Miami Marlins",
    "Brewers": "Milwaukee Brewers", "Cardinals": "St. Louis Cardinals",
    "Cubs": "Chicago Cubs", "Reds": "Cincinnati Reds",
    "Pirates": "Pittsburgh Pirates", "Dodgers": "Los Angeles Dodgers",
    "Padres": "San Diego Padres", "D-backs": "Arizona Diamondbacks",
    "Giants": "San Francisco Giants", "Rockies": "Colorado Rockies",
}


class MLBCompetitiveness:
    """Mide competitividad de MLB por liga/división y detecta tanking"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self.global_score: float = 0.5
        self.global_level: str = "MEDIUM"
        self.division_scores: Dict[str, float] = {}
        self.tanking_teams: List[str] = []
        self.metrics: Dict = {}

    def load_and_calculate(self, years: List[int] = None) -> bool:
        """Cargar standings y calcular competitividad"""
        if years is None:
            years = [2021, 2022, 2023, 2024]

        teams_path = self.cleaned_dir / "teams_cleaned.csv"
        if not teams_path.exists():
            teams_path = self.data_dir / "teams.csv"

        teams_df = None
        if teams_path.exists():
            teams_df = pd.read_csv(teams_path)

        all_standings = []
        for year in years:
            path = self.cleaned_dir / f"standings_{year}_cleaned.csv"
            if not path.exists():
                path = self.data_dir / f"standings_{year}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["season"] = year
                all_standings.append(df)

        if not all_standings:
            print("  No hay datos de standings")
            return False

        combined = pd.concat(all_standings, ignore_index=True)

        if teams_df is not None:
            team_division = {}
            for _, r in teams_df.iterrows():
                team_division[r["name"]] = {
                    "league": r.get("league", ""),
                    "division": r.get("division", ""),
                }
            combined["full_name"] = combined["team"].map(
                lambda x: STANDINGS_TO_FULL.get(x, x)
            )
            combined["league"] = combined["full_name"].map(
                lambda x: team_division.get(x, {}).get("league", "")
            )
            combined["division"] = combined["full_name"].map(
                lambda x: team_division.get(x, {}).get("division", "")
            )
        else:
            combined["full_name"] = combined["team"]

        self._calc_global(combined)
        self._calc_divisions(combined)
        self._detect_tanking(combined)
        return True

    def _calc_global(self, df: pd.DataFrame):
        """Competitividad global basada en win_pct"""
        win_pcts = df["win_pct"].dropna().values
        if len(win_pcts) == 0:
            self.global_score = 0.5
            return

        mean_wp = np.mean(win_pcts)
        std_wp = np.std(win_pcts)
        cv = std_wp / mean_wp if mean_wp > 0 else 0

        # CV tipico en MLB win%: ~0.15-0.30
        # Invertir: bajo CV = alta competitividad
        self.global_score = max(0, min(1, 1 - (cv - 0.12) / 0.18))

        self.metrics["global"] = {
            "mean_win_pct": mean_wp,
            "std_win_pct": std_wp,
            "cv": cv,
            "score": self.global_score,
        }

        self.global_level = self._score_to_level(self.global_score)

    def _calc_divisions(self, df: pd.DataFrame):
        """Competitividad por división"""
        for division in df["division"].dropna().unique():
            div_name = division.strip()
            if not div_name:
                continue
            div_df = df[df["division"] == division]
            wp = div_df["win_pct"].dropna().values
            if len(wp) < 3:
                continue
            mean_wp = np.mean(wp)
            std_wp = np.std(wp)
            cv = std_wp / mean_wp if mean_wp > 0 else 0
            score = max(0, min(1, 1 - (cv - 0.08) / 0.18))
            self.division_scores[div_name] = score

    def _detect_tanking(self, df: pd.DataFrame):
        """Detectar equipos que están tankeando (perdiendo deliberadamente)"""
        win_pcts = df["win_pct"].dropna()
        if len(win_pcts) == 0:
            return

        q25 = win_pcts.quantile(0.25)
        seen = set()
        for _, r in df.iterrows():
            wp = r["win_pct"]
            team = r["team"]
            if pd.notna(wp) and wp < q25 * 0.85 and team not in seen:
                self.tanking_teams.append(team)
                seen.add(team)

    def _score_to_level(self, score: float) -> str:
        if score > 0.6:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        return "LOW"

    def get_global_score(self) -> float:
        return self.global_score

    def get_global_level(self) -> str:
        return self.global_level

    def get_division_score(self, division: str) -> float:
        return self.division_scores.get(division, self.global_score)

    def get_division_level(self, division: str) -> str:
        return self._score_to_level(self.get_division_score(division))

    def is_tanking(self, team_name: str) -> bool:
        return team_name in self.tanking_teams

    def get_tanking_teams(self) -> List[str]:
        return self.tanking_teams

    def get_adjustment_factors(self, division: str = None) -> Dict:
        score = self.get_division_score(division) if division else self.global_score

        return {
            "confidence_factor": 0.7 + score * 0.3,
            "upset_risk_factor": 1.0 + score * 0.5,
            "competitiveness_score": score,
            "competitiveness_level": self._score_to_level(score),
            "tanking_teams": self.tanking_teams,
        }

    def get_upset_risk(self, home_win_pct: float, away_win_pct: float,
                       division: str = None) -> Dict:
        """Calcular riesgo de upset basado en diferencia de win%"""
        wp_diff = abs(home_win_pct - away_win_pct)
        comp_score = self.get_division_score(division) if division else self.global_score

        base_upset = wp_diff * 0.5
        adjusted_upset = base_upset * (1 + comp_score)

        if adjusted_upset > 0.35:
            level = "HIGH"
        elif adjusted_upset > 0.2:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "upset_probability": min(0.5, adjusted_upset),
            "risk_level": level,
            "win_pct_differential": wp_diff,
            "competitiveness_factor": comp_score,
        }

    def print_summary(self):
        print("\n  COMPETITIVIDAD MLB")
        print("  " + "-" * 50)
        print(f"  Global:      {self.global_level} ({self.global_score:.3f})")
        if "global" in self.metrics:
            m = self.metrics["global"]
            print(f"  Mean win%:   {m['mean_win_pct']:.3f}")
            print(f"  Std win%:    {m['std_win_pct']:.3f}")
            print(f"  CV:          {m['cv']:.3f}")

        if self.division_scores:
            print(f"\n  Por division:")
            for div, score in sorted(self.division_scores.items()):
                level = self._score_to_level(score)
                al_nl = div.split()[-1] if div else "?"
                print(f"    {div[:30]:30} {level:6} ({score:.3f})")

        if self.tanking_teams:
            print(f"\n  Tanking: {', '.join(self.tanking_teams)}")
        else:
            print(f"\n  Tanking: Ninguno detectado")

    def adjust_probabilities(self, probs: Dict[str, float], home_team: str = None,
                             away_team: str = None, division: str = None) -> Dict[str, float]:
        """Ajustar probabilidades según competitividad (sin empate)"""
        factors = self.get_adjustment_factors(division)
        adjusted = probs.copy()
        score = factors["competitiveness_score"]

        if score > 0.5:
            local_prob = adjusted.get("LOCAL", 0.5)
            visit_prob = adjusted.get("VISITANTE", 0.5)

            diff = abs(local_prob - visit_prob)
            reduction = diff * (1 - score) * 0.15

            if local_prob > visit_prob:
                adjusted["LOCAL"] = max(0.35, local_prob - reduction)
                adjusted["VISITANTE"] = min(0.65, visit_prob + reduction)
            else:
                adjusted["VISITANTE"] = max(0.35, visit_prob - reduction)
                adjusted["LOCAL"] = min(0.65, local_prob + reduction)

            total = sum(adjusted.values())
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted


def get_competitiveness(data_dir: str = "data") -> MLBCompetitiveness:
    return MLBCompetitiveness(data_dir)
