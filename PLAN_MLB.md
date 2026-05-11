# Plan de Migración: MML-MLB

**Objetivo:** Transformar el proyecto de predicción de Premier League a un sistema de predicción de MLB, específicamente para:
- **Resultado:** Win/Loss binario (LOCAL / VISITANTE)
- **Over/Under:** Carreras totales vs línea 8.5

**Fuentes de datos:**
- MLB Stats API (`statsapi.mlb.com`) — gratuita, sin API key
- pybaseball — stats agregados con Statcast (barrel rate, exit velo, wRC+, etc.) + park factors

**Validación:** Cronológica — Train 2021-2023, Test 2024, Producción 2025

---

## Archivos a crear — ✅ TODOS COMPLETADOS

### 1. `download_mlb_historical.py` ✅
Script único de setup que construye TODOS los CSVs desde cero para temporadas 2021-2025.
- [x] Recolecta schedule + boxscores + standings + pitchers vía MLB Stats API
- [x] Recolecta stats de equipo/pitchers + park factors vía pybaseball
- [x] Construye: `games_{year}.csv`, `standings_{year}.csv`, `teams.csv`, `team_batting_{year}.csv`, `team_pitching_{year}.csv`, `pitcher_stats_{year}.csv`, `park_factors.csv`

### 2. `update_mlb_data.py` ✅
Actualización incremental diaria. Solo agrega juegos nuevos desde la última actualización.

### 3. `src_v2/data/mlb_collector.py` ✅
Clase `MLBDataCollector` — wraps MLB Stats API:
- [x] `get_schedule(year, team_id=None)` → juegos del calendario
- [x] `get_boxscore(game_pk)` → resultado, pitchers, hits, errors
- [x] `get_standings(year, league_id=103)` → posiciones
- [x] `get_teams()` → lista de equipos
- [x] `get_pitcher_stats(person_id, season)` → stats de pitcher específico

### 4. `src_v2/data/mlb_pybaseball.py` ✅
Clase `PybaseballWrapper` — wraps pybaseball:
- [x] `get_team_batting_stats(season)` → AVG, OBP, SLG, OPS, wRC+, BABIP, barrel%, exit velo, hard hit%, K%, BB%
- [x] `get_team_pitching_stats(season)` → ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, BABIP, barrel% contra
- [x] `get_pitcher_stats(season)` → stats individuales de cada pitcher
- [x] `get_park_factors()` — desde MLB API (pybaseball park_factors no disponible)

### 5. `src_v2/features/mlb_feature_engineer.py` ✅
Clase `MLBFeatureEngineer` — **50 features** por juego (40 planificadas + 15 pitcher, -5 renombradas).

**Pitchers (15 features):**
- `home_era_s`, `away_era_s`, `era_s_diff` — ERA season stats
- `home_whip_s`, `away_whip_s` — WHIP season stats
- `home_k9_s`, `away_k9_s` — K/9 season stats
- `home_rolling_starts`, `away_rolling_starts` — conteo últimos 3 starts
- `home_rolling_win_pct`, `away_rolling_win_pct`, `rolling_win_pct_diff` — win% últimos 3
- `home_rolling_avg_runs`, `away_rolling_avg_runs`, `rolling_runs_diff` — avg runs últimos 3

**Ofensiva equipo (8):**
- `home_avg`, `away_avg`, `home_ops`, `away_ops`
- `home_runs_per_game`, `away_runs_per_game`
- `home_home_runs_per_game`, `away_home_runs_per_game`

**Bullpen/Equipo (6):**
- `home_team_era`, `away_team_era`
- `home_team_whip`, `away_team_whip`
- `home_team_k9`, `away_team_k9`
- `home_bullpen_era`, `away_bullpen_era`

**Contextual (10):**
- `form_win_rate_diff`, `form_runs_diff` — forma reciente
- `home_venue_win_rate`, `away_venue_win_rate`, `venue_advantage`
- `home_rest_days`, `away_rest_days`, `rest_diff`
- `home_games_14d`, `away_games_14d`

**Head-to-Head (3):**
- `h2h_games` (esta temporada)
- `h2h_home_win_rate`
- `h2h_avg_total_runs`

**Standings (5):**
- `home_win_pct`, `away_win_pct`
- `win_pct_diff`
- `home_gb`, `away_gb`

**Extra:**
- `is_home` (1) — sesgo de localía

### 6. `src_v2/models/runs_predictor.py` ✅
Clase `RunsPredictor` — **3 modelos** (RF, GB, LogisticRegression clasificador O/U):
- [x] `random_forest`: `RandomForestRegressor(n_estimators=100, max_depth=10)`
- [x] `gradient_boosting`: `GradientBoostingRegressor(n_estimators=100, max_depth=5, lr=0.05)`
- [x] `logistic_regression`: clasificador directo O/U 8.5 (mejor: 56.9% validación)
- [x] **Auto-selección del mejor** por test O/U accuracy
- [x] Poisson para regresores, clasificación directa para LR

---

## Archivos a modificar — ✅ TODOS COMPLETADOS

### 1. `main.py` ✅
Menú MLB con 5 opciones:
```
1. Predecir juego (interactivo)
2. Entrenar modelos
3. Actualizar datos
4. Exportar predicciones al panel
5. Salir
```
- [x] Menú interactivo con selección de modelo (winner/runs)
- [x] `--train`, `--update`, `--predict` CLI
- [x] Exportación a safesports-panel

### 2. `src_v2/data/cleaner.py` ✅
Adaptado a schema MLB:
- [x] `result` binario: VISITANTE(0) / LOCAL(1) — sin empates
- [x] Derivar: `total_runs`, `run_difference`, `margin_victory`
- [x] Validar rangos de carreras (0-30)
- [x] Separar SCHEDULED vs FINISHED

### 3. `src_v2/models/winner_predictor.py` ✅
**3 modelos** con auto-selección del mejor:
- [x] `random_forest`: `RandomForestClassifier(n_estimators=200, max_depth=10)`
- [x] `gradient_boosting`: `GradientBoostingClassifier(n_estimators=80, max_depth=4, lr=0.05)`
- [x] `logistic_regression`: `LogisticRegression(class_weight='balanced')` ← **mejor: 60.1% val**
- [x] Auto-selección por test accuracy
- [x] Escalado solo para LR (trees usan raw data)
- [x] Comparativa impresa al entrenar

### 4. `src_v2/evaluation/evaluator.py` ✅
- [x] Accuracy en wins + AUC-ROC
- [x] MAE en carreras totales (solo regresores)
- [x] Accuracy O/U 8.5 (todos los modelos)
- [x] Overfitting detection con gap train-test

### 5. `src_v2/predict.py` ✅
- [x] `--game`, `--test`, `--date` flags
- [x] `--w-model` y `--r-model` para elegir modelo
- [x] Mapeo de equipos MLB (3-letter codes)
- [x] Threshold O/U fijo en 8.5
- [x] `sport: "mlb"` (no "baseball" — el panel usa SPORTS.MLB = "mlb")

### 6. `src_v2/train.py` ✅
- [x] Pipeline completo MLB
- [x] Train 2021-2023, Test 2024
- [x] WinnerPredictor (binario) + RunsPredictor
- [x] Evaluación en test set

### 7. `src_v2/features/competitiveness.py` ✅
Adaptado a MLB:
- [x] Sin empates — no hay draw_weight, no hay EMPATE
- [x] Tanking/rebuilding — detecta equipos con win% < 85% del Q25
- [x] `upset_risk` basado en diferencia de win% + competitividad
- [x] Competitiveness por división (6 divisiones) + global
- [x] Ajuste de probabilidades según competitividad (reduce favoritos en liga competitiva)
- [x] Integrado en training pipeline y menú interactivo

### 8. `requirements.txt` ✅
- [x] `pybaseball>=2.2.4`
- [x] `python-dotenv>=1.0.0`

---

## Orden de ejecución por fases — ✅ 7/7 COMPLETADAS

| # | Fase | Archivos | Descripción | Estado |
|---|------|----------|-------------|--------|
| **1** | Setup inicial | `requirements.txt` + `mlb_collector.py` + `mlb_pybaseball.py` | Conexión a fuentes de datos | ✅ |
| **2** | Data pipeline | `download_mlb_historical.py` + `update_mlb_data.py` | Build de CSVs 2021-2025 | ✅ |
| **3** | Clean + Features | `cleaner.py` + `mlb_feature_engineer.py` | Limpieza y feature engineering (50 feats) | ✅ |
| **4** | Modelos | `winner_predictor.py` + `runs_predictor.py` | 3 modelos c/u con auto-selección | ✅ |
| **5** | Train + Eval | `train.py` + `evaluator.py` | Entrenamiento y evaluación cronológica | ✅ |
| **6** | Predictor + UI | `predict.py` + `main.py` + `update_mlb_data.py` | Predicción, menú, export panel | ✅ |
| **7** | Post-MVP | Pitcher rolling stats | Forma reciente de pitchers (últimos 3 starts) | ✅ |

### Rendimiento Actual (2024 test set)
| Modelo | Accuracy |
|--------|----------|
| **Winner Predictor** (LR) | **57.7%** |
| Runs Predictor O/U 8.5 (LR) | **55.1%** |

---

## Schema de CSVs

### `games_{year}.csv`

| Columna | Tipo | Fuente |
|---------|------|--------|
| `game_pk` | int | MLB API Schedule |
| `date` | datetime | MLB API Schedule |
| `home_team` | string | MLB API Schedule |
| `away_team` | string | MLB API Schedule |
| `home_runs` | float64 | Boxscore (NaN si scheduled) |
| `away_runs` | float64 | Boxscore (NaN si scheduled) |
| `total_runs` | float64 | home_runs + away_runs |
| `status` | string | FINISHED / SCHEDULED |
| `venue` | string | MLB API Schedule |
| `home_pitcher_id` | int | Boxscore (abridor) |
| `away_pitcher_id` | int | Boxscore (abridor) |
| `innings` | int | Boxscore |
| `home_hits` | int | Boxscore |
| `away_hits` | int | Boxscore |
| `home_errors` | int | Boxscore |
| `away_errors` | int | Boxscore |
| `doubleheader` | bool | Schedule |
| `day_night` | string | Schedule |
| `home_team_code` | string | Código 3 letras (NYY) |
| `away_team_code` | string | Código 3 letras (BOS) |

### `standings_{year}.csv`

| Columna | Tipo | Notas |
|---------|------|-------|
| `team` | string | Nombre del equipo |
| `team_code` | string | Código 3 letras |
| `league` | string | AL / NL |
| `division` | string | East / Central / West |
| `wins` | int | Victorias |
| `losses` | int | Derrotas |
| `win_pct` | float | % de victorias |
| `gb` | float | Games back en división |
| `wild_card_gb` | float | Games back en wild card |
| `last_10` | string | Forma últimos 10 (W-L) |
| `streak` | string | Racha actual |
| `runs_scored` | int | Carreras anotadas |
| `runs_allowed` | int | Carreras permitidas |
| `run_diff` | int | Diferencia de carreras |

### `teams.csv`

| Columna | Tipo | Notas |
|---------|------|-------|
| `team_id` | int | MLB API team ID |
| `name` | string | Full name (New York Yankees) |
| `team_code` | string | Código (NYY) |
| `league` | string | AL / NL |
| `division` | string | East / Central / West |
| `venue` | string | Estadio |

### `team_batting_{year}.csv`

| Columna | Notas |
|---------|-------|
| `team` | Nombre |
| `AVG` | Average |
| `OBP` | On-base percentage |
| `SLG` | Slugging percentage |
| `OPS` | On-base + slugging |
| `wRC_plus` | Weighted runs created+ |
| `BABIP` | Batting average on balls in play |
| `barrel_rate` | % de barrels |
| `avg_exit_velocity` | Velocidad de salida promedio |
| `launch_angle` | Ángulo de despegue promedio |
| `hard_hit_rate` | % de batazos duros |
| `K_pct` | % de ponches |
| `BB_pct` | % de bases por bolas |
| `runs_per_game` | Carreras por juego |
| `WAR` | Wins above replacement |

### `team_pitching_{year}.csv`

| Columna | Notas |
|---------|-------|
| `team` | Nombre |
| `ERA` | Earned run average |
| `FIP` | Fielding independent pitching |
| `xFIP` | Expected FIP |
| `WHIP` | Walks + hits per inning pitched |
| `K_per_9` | Ponches por 9 entradas |
| `BB_per_9` | Bases por bolas por 9 |
| `HR_per_9` | HR por 9 |
| `BABIP` | BABIP en contra |
| `barrel_rate_against` | Barrel rate en contra |
| `avg_exit_velocity_against` | Exit velo en contra |
| `defensive_efficiency` | Eficiencia defensiva |
| `bullpen_era` | ERA del bullpen |
| `bullpen_whip` | WHIP del bullpen |

### `pitcher_stats_{year}.csv`

| Columna | Notas |
|---------|-------|
| `pitcher_id` | MLB API person ID |
| `name` | Nombre del pitcher |
| `team` | Equipo actual |
| `GS` | Games started |
| `IP` | Innings pitched |
| `W` | Wins |
| `L` | Losses |
| `ERA` | Earned run average |
| `FIP` | FIP |
| `WHIP` | WHIP |
| `K_per_9` | Ponches por 9 |
| `BB_per_9` | Bases por bolas por 9 |
| `HR_per_9` | HR por 9 |
| `BABIP` | BABIP en contra |
| `WAR` | Wins above replacement |

### `park_factors.csv`

| Columna | Notas |
|---------|-------|
| `venue` | Estadio |
| `team` | Equipo dueño |
| `factor_runs` | Factor de carreras (>1.0 = hitter-friendly) |
| `factor_hr` | Factor de HR |
| `factor_single` | Factor de hits sencillos |
| `factor_double` | Factor de dobles |
| `factor_triple` | Factor de triples |

---

## Formato de exportación al panel — ✅ IMPLEMENTADO

El panel usa `SPORTS.MLB = "mlb"` (no "baseball"). Formato real:

```json
{
  "predictions": [{
    "sport": "mlb",
    "homeTeam": "NYY",
    "homeTeamFullName": "New York Yankees",
    "homeTeamLogo": "https://www.mlbstatic.com/team-logos/147.svg",
    "awayTeam": "BOS",
    "awayTeamFullName": "Boston Red Sox",
    "awayTeamLogo": "https://www.mlbstatic.com/team-logos/111.svg",
    "predictedWinner": "NYY",
    "confidence": 62,
    "riskLevel": "medium",
    "gameDate": "2025-06-15T19:05:00Z",
    "status": "active",
    "arguments": {
      "forWinner": [
        "Confianza del modelo: 62%",
        "Modelo: logistic_regression",
        "Probabilidad LOCAL: 62%",
        "O/U 8.5: UNDER (43%)"
      ],
      "forLoser": [
        "Probabilidad VISITANTE: 38%"
      ],
      "summary": {
        "winnerFactors": 4,
        "loserFactors": 1,
        "matchupType": "mlb_regular",
        "betRecommendation": "NYY with 62% confidence"
      }
    }
  }]
}
```

---

## Notas adicionales

- **Premier League se archiva:** ✅ Todo `src/` Premier League eliminado. Solo queda MLB en `src_v2/`.
- **pybaseball vs Statcast crudo:** ✅ Solo stats agregados de pybaseball (`batting_stats_bref`, `pitching_stats_bref`). Park factors vía MLB API.
- **Pitcher form (fase 7):** ✅ Implementado — rolling stats de últimos 3 starts (win%, avg_runs_allowed).
- **Panel:** ✅ Export con `"sport": "mlb"` (SPORTS.MLB en el panel). Envío vía API key con `Authorization: Bearer sk_...`.
- **Competitiveness:** ✅ Adaptado a MLB. Tanking, upset risk, 6 divisiones. Integrado en train + predict.
