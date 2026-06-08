# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Sistema de predicción de partidos MLB usando machine learning (sklearn). Predice ganador del juego y over/under (8.5 carreras). Las predicciones se exportan al panel `safesports-panel` vía REST API.

## Comandos Frecuentes

```bash
# Activar entorno virtual (siempre primero)
source mlb-env/bin/activate

# Menú interactivo principal
python3 main.py

# CLI directo
python3 main.py --train              # Entrenar modelos
python3 main.py --update             # Actualizar datos de la temporada actual
python3 main.py --predict NYY BOS    # Predecir juego específico (home away)
python3 main.py --bolo               # Predecir todos los juegos de hoy
python3 main.py --bolo 2025-06-15    # Predecir todos los juegos de una fecha
```

## Arquitectura del Pipeline

El flujo de datos es lineal y unidireccional:

```
Colección → Limpieza → Features → Modelos → Predicción → Export
```

### 1. Colección de Datos (`src_v2/data/`)

Dos fuentes independientes:
- **`mlb_collector.py` (MLBDataCollector):** MLB Stats API oficial. Obtiene schedules, standings, pitcher stats individuales, boxscores.
- **`mlb_pybaseball.py` (PybaseballWrapper):** Baseball-Reference vía pybaseball. Obtiene batting/pitching stats agregados por equipo.

Los datos van a `data/` como CSVs: `games_{year}.csv`, `standings_{year}.csv`, `pitcher_stats_{year}.csv`, `team_batting_{year}.csv`, `team_pitching_{year}.csv`.

### 2. Limpieza (`src_v2/data/cleaner.py`)

`DataCleaner` estandariza nombres de equipos a códigos (e.g. "Yankees" → "NYY"), marca juegos como FINISHED/SCHEDULED, y escribe en `data/cleaned/`.

### 3. Feature Engineering (`src_v2/features/mlb_feature_engineer.py`)

`MLBFeatureEngineer` es el núcleo del sistema. Crea ~50 features por juego combinando datos de todas las fuentes:
- Forma reciente (últimos N juegos), ventaja de sede, días de descanso
- Head-to-head, standings, batting/pitching por equipo
- Stats de pitcher titular (temporada + últimos 3 starts rolling)

Métodos clave:
- `load_data(years)` — carga todos los CSVs para los años dados
- `create_match_features(home, away, date)` — genera features para un partido específico
- `create_training_dataset(years)` — genera el DataFrame completo para entrenamiento

### 4. Modelos (`src_v2/models/`)

Dos predictores independientes, cada uno entrena 3 variantes y selecciona la mejor:

**WinnerPredictor (`winner_predictor.py`):**
- RF, Gradient Boosting, Logistic Regression (todos calibrados con `CalibratedClassifierCV`)
- Target: binario (home=1, away=0)
- Validación cronológica: 80% train / 20% test (no aleatoria)
- Se reentrenan en 100% de datos para producción
- Guardados en `models_mlb/winner_*.pkl`

**RunsPredictor (`runs_predictor.py`):**
- RF Regressor, GB Regressor (convierten a O/U vía Poisson CDF), LR Classifier directo O/U 8.5
- Target: over=1 / under=0 respecto a 8.5 carreras totales
- Guardados en `models_mlb/runs_*.pkl`

Ambos tienen factory functions: `get_winner_predictor()`, `get_runs_predictor()`.

### 5. Competitividad (`src_v2/features/competitiveness.py`)

`MLBCompetitiveness` calcula scores 0-1 por liga/división basados en desviación estándar de win%. Detecta "tanking" (equipos bottom 25%). Se usa para ajustar confianza y calcular upset risk.

### 6. Export a safesports-panel (`src_v2/export_to_panel.py`)

Autenticación: lee `.env.local` → si no hay `SAFESPORTS_USER_API_KEY`, hace POST a `/api/auth/api-key/generate` con email/password.

`transform_to_panel_format()` convierte una predicción al JSON que espera el panel (incluye logos de los 30 equipos hardcodeados como URLs de mlbstatic.com).

`send_predictions()` hace POST a `/api/predictions/import` con `Authorization: Bearer <key>`.

### 7. Tracking (`src_v2/tracking.py`)

Predicciones guardadas en `data/predictions.json`. `get_report(year)` compara contra resultados reales y calcula accuracy por año, banda de confianza y equipo.

## Configuración

**`.env.local`** (no se commitea):
```
SAFESPORTS_PANEL_URL=http://localhost:3000
SAFESPORTS_PANEL_EMAIL=admin@sudo.com
SAFESPORTS_PANEL_PASSWORD=Admin123!
# O usar directamente:
SAFESPORTS_USER_API_KEY=sk_...
```

**Validación temporal:** Train siempre usa años 2021-2023, test usa 2024. No mezclar splits — el split es cronológico, no aleatorio.

**Persistencia de modelos:** Si `models_mlb/winner_best.pkl` existe, se carga sin reentrenar. Borrar el `.pkl` para forzar reentrenamiento.
