#!/usr/bin/env python3
"""
Download MLB Historical Data
============================
Script único de setup que construye TODOS los CSVs desde cero
para temporadas 2021-2025.

Uso:
    python download_mlb_historical.py

Requiere:
    - MLB Stats API (statsapi.mlb.com) — sin API key
    - pybaseball (pip install pybaseball)

Salida en data/:
    games_{year}.csv          — Juegos con resultados y pitchers
    standings_{year}.csv      — Posiciones por división
    teams.csv                 — Equipos MLB
    team_batting_{year}.csv   — Stats ofensivas por equipo
    team_pitching_{year}.csv  — Stats de pitcheo por equipo
    pitcher_stats_{year}.csv  — Stats individuales de pitchers
    park_factors.csv          — Factores de parque
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.data.mlb_collector import get_mlb_collector
from src_v2.data.mlb_pybaseball import get_pybaseball_wrapper


SEASONS = [2021, 2022, 2023, 2024, 2025]


def main():
    print("=" * 60)
    print("  MLB HISTORICAL DATA DOWNLOADER 2021-2025")
    print("=" * 60)

    collector = get_mlb_collector("data")
    pybase = get_pybaseball_wrapper("data")

    print("\n[1/3] Recolectando datos desde MLB Stats API...")
    start = time.time()
    collector.collect_all(seasons=SEASONS, enrich=True)
    api_time = time.time() - start
    print(f"  ⏱  {api_time:.0f}s")

    print("\n[2/3] Recolectando estadísticas desde pybaseball...")
    start = time.time()
    pybase.collect_all(seasons=SEASONS)
    pb_time = time.time() - start
    print(f"  ⏱  {pb_time:.0f}s")

    print("\n[3/3] Verificando archivos generados...")
    data_dir = Path("data")
    expected = []
    for season in SEASONS:
        expected.extend([
            f"games_{season}.csv",
            f"standings_{season}.csv",
            f"team_batting_{season}.csv",
            f"team_pitching_{season}.csv",
            f"pitcher_stats_{season}.csv",
        ])
    expected.extend(["teams.csv", "park_factors.csv"])

    total_files = 0
    for f in expected:
        path = data_dir / f
        if path.exists():
            size = path.stat().st_size
            total_files += 1
            print(f"  ✅ {f:45} {size:>8,} bytes")
        else:
            print(f"  ⚠️  {f:45} NO ENCONTRADO")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Descarga completa: {total_files}/{len(expected)} archivos")
    print(f"  ⏱  Tiempo total: {api_time + pb_time:.0f}s ({((api_time + pb_time)/60):.1f}min)")
    print(f"  📁 data/")

    return 0 if total_files == len(expected) else 1


if __name__ == "__main__":
    sys.exit(main())
