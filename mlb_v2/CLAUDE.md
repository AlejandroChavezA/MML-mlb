# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Sistema MLB v2 de predicción rolling. Arquitectura inspirada en el sistema NBA:
pipeline de datos → rolling features → modelo LogReg (ganador) + Ridge/XGBoost/LightGBM (totales) → CLI interactivo.

Fuente de datos: MLB Stats API (`statsapi.mlb.com/api/v1`) — sin dependencias externas de pago.

## Comandos

```bash
cd mlb_v2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ── Uso diario ──────────────────────────────────────────
# 1. Actualizar datos hasta ayer y regenerar features
python src/ingest/update_data.py

# 2. Reentrenar modelos con datos frescos
python src/models/train.py
python src/models/train_totals_v3.py

# 3. Predecir
python src/predict/predict_cli.py

# ── Setup inicial (solo primera vez) ────────────────────
python src/ingest/build_base_games.py                   # descarga histórico 2015-hoy
python src/ingest/build_team_games_base.py              # box scores por equipo (~4 min)
python src/features/build_team_rolling_features.py      # rolling 5/10/20 con shift(1)
python src/features/build_game_level_features.py        # HOME_ / AWAY_ + DIFF_*

# ── Opciones de update_data.py ───────────────────────────
python src/ingest/update_data.py --dry-run   # ver qué descargaría sin ejecutar
python src/ingest/update_data.py --full      # re-descarga todo desde 2015

# ── Evaluación ───────────────────────────────────────────
python src/evaluation/eval_confidence.py
python tests/verified_values.py
python tests/season_verified.py
```

## Diferencias clave vs src_v2 (modelo original)

| Aspecto | src_v2 (original) | mlb_v2 (este) |
|---|---|---|
| Features | Calculadas en tiempo real por partido | Rolling pre-calculadas con shift(1) |
| Modelos | 3 modelos calibrados + auto-selección por test_acc | LogReg (ganador) + Ridge/XGB/LGB (totales) |
| Almacenamiento | .pkl | .joblib |
| Datos de equipo | pybaseball (Baseball-Reference) | Solo MLB Stats API |
| CLI | Menú con export a safesports-panel | CLI rolling puro, sin export |

## Arquitectura

```
data/raw/            ← opcional: datos descargados manualmente
data/processed/      ← CSVs generados por el pipeline
  mlb_games_base.csv           partidos + HOME_WIN
  mlb_team_games_base.csv      box score por equipo/partido
  mlb_team_games_features.csv  rolling stats pre-partido
  mlb_games_features.csv       dataset final HOME_/AWAY_/DIFF_
models/
  mlb_logreg_rolling.joblib    winner model
  mlb_totals_v3.joblib         totals model
src/
  ingest/    → descarga desde API
  features/  → rolling engineering
  models/    → entrenamiento
  predict/   → CLI + explain_natural
  evaluation/→ eval por umbral
```

## Garantía anti-leakage

`build_team_rolling_features.py` usa `shift(1)` antes de cada `rolling(w)`. Esto asegura que para el partido N, las features rolling usan solo los N-1 partidos anteriores. El split de entrenamiento es temporal 80/20 sin shuffle.

## Formato del CLI

```
BOS @ NYY            # AWAY @ HOME
BOS @ NYY | 8.5      # con línea O/U explícita
DONE                 # terminar sesión manual
```

## Notas de la API

- Rate limit suave: el código duerme 150ms entre requests
- Los boxscores detallados (`/game/{pk}/boxscore`) son lentos en bulk — usar `--no-boxscore` para el primer test
- Temporada 2020 (COVID, 60 juegos) puede sesgar los rolling → excluirla si los resultados son raros
