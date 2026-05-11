#!/usr/bin/env python3
"""
Update MLB Data
===============
Actualización incremental de datos MLB.

Uso:
    python update_mlb_data.py

Agrega juegos nuevos desde la última actualización.
No re-descarga datos históricos.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.data.mlb_collector import get_mlb_collector
from src_v2.data.mlb_pybaseball import get_pybaseball_wrapper
from src_v2.data.cleaner import get_cleaner


def main():
    print("=" * 50)
    print("  ACTUALIZACIÓN MLB")
    print("=" * 50)

    cache_file = PROJECT_ROOT / ".update_cache.json"

    last_update = None
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
            last_update = datetime.fromisoformat(cache["last_update"])
            print(f"  Última actualización: {last_update}")
        except Exception:
            pass

    current_season = datetime.now().year
    if current_season > 2025:
        current_season = 2025

    print(f"\n  Temporada actual: {current_season}")

    collector = get_mlb_collector("data")
    collector.collect_all(seasons=[current_season], enrich=False)

    pybase = get_pybaseball_wrapper("data")
    pybase.collect_season(current_season)

    print(f"\n Limpiando datos...")
    cleaner = get_cleaner("data")
    cleaner.run_cleaning(years=[current_season])

    with open(cache_file, "w") as f:
        json.dump({"last_update": datetime.now().isoformat()}, f)

    print(f"\n  Datos actualizados!")
    print(f"  Última actualización: {datetime.now()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
