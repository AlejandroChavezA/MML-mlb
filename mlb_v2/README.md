# MLB v2 — Sistema de Predicción Rolling

Sistema de machine learning para predecir **ganador** y **total de carreras (O/U 8.5)** de partidos MLB, usando estadísticas rolling pre-partido sin data leakage.

Fuente de datos: MLB Stats API pública (`statsapi.mlb.com`) — sin costo ni API key.

---

## Modelos

| Modelo | Archivo | Objetivo | Algoritmo | Accuracy actual |
|--------|---------|----------|-----------|-----------------|
| **Ganador** | `models/mlb_logreg_rolling.joblib` | Probabilidad de victoria local (`HOME_WIN`) | LogReg + StandardScaler | 53.6% (test) |
| **Totales v3** | `models/mlb_totals_v3.joblib` | Total de carreras y O/U 8.5 | Ridge / XGBoost / LightGBM (auto-selección) | MAE 3.49 / O/U 51.4% |

---

## Estructura

```
mlb_v2/
├── data/
│   └── processed/               # CSVs generados por el pipeline
│       ├── mlb_games_base.csv           partidos históricos + HOME_WIN
│       ├── mlb_team_games_base.csv      box score por equipo/partido
│       ├── mlb_team_games_features.csv  rolling stats pre-partido (shift 1)
│       └── mlb_games_features.csv       dataset final HOME_ / AWAY_ / DIFF_
├── models/                      # modelos entrenados (.joblib)
├── src/
│   ├── ingest/
│   │   ├── build_base_games.py          descarga histórico desde MLB API
│   │   ├── build_team_games_base.py     box scores por equipo (paralelo)
│   │   └── update_data.py               actualización incremental diaria
│   ├── features/
│   │   ├── build_team_rolling_features.py   rolling 5/10/20 con shift(1)
│   │   └── build_game_level_features.py     HOME_ / AWAY_ / DIFF_ + targets
│   ├── models/
│   │   ├── train.py                     entrena modelo ganador
│   │   └── train_totals_v3.py           entrena modelo totales
│   ├── predict/
│   │   ├── predict_cli.py               CLI interactivo
│   │   └── explain_natural.py           señales en lenguaje natural
│   └── evaluation/
│       └── eval_confidence.py           accuracy por umbral de confianza
└── tests/
    ├── verified_values.py
    └── season_verified.py
```

---

## Instalación

```bash
cd mlb_v2
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requisitos:** Python 3.10+, pandas, scikit-learn, xgboost, lightgbm, requests, joblib.

---

## Flujo completo — Primera vez

Ejecutar **en orden** desde la carpeta `mlb_v2/`:

```bash
# 1. Descargar histórico de partidos desde MLB API (2015 → hoy)
python src/ingest/build_base_games.py

# 2. Descargar box scores detallados por equipo
#    Usa 15 threads en paralelo + caché en disco (~4 min, reanudable si se interrumpe)
python src/ingest/build_team_games_base.py

# 3. Calcular rolling features por equipo (ventanas 5, 10, 20 con shift(1))
python src/features/build_team_rolling_features.py

# 4. Construir dataset final por partido (HOME_ / AWAY_ / DIFF_*)
python src/features/build_game_level_features.py

# 5. Entrenar modelo de ganador
python src/models/train.py

# 6. Entrenar modelo de totales
python src/models/train_totals_v3.py

# 7. Predecir
python src/predict/predict_cli.py
```

---

## Flujo diario — Mantener datos al día

```bash
# 1. Descarga solo los partidos nuevos desde el último FINISHED hasta ayer
#    y regenera automáticamente todos los CSVs procesados
python src/ingest/update_data.py

# 2. Reentrenar con datos frescos
python src/models/train.py
python src/models/train_totals_v3.py

# 3. Predecir
python src/predict/predict_cli.py
```

`update_data.py` detecta automáticamente desde qué fecha debe descargar, evita duplicados y solo descarga los boxscores nuevos (el resto está en caché).

---

## Predicciones — CLI

```bash
python src/predict/predict_cli.py
```

### Menú

```
1. Predicción manual (AWAY @ HOME)
2. Predicciones por fecha (busca en API)
3. Salir
```

### Modo manual

Formato `AWAY @ HOME`, línea O/U opcional:

```
BOS @ NYY
LAD @ SF | 8.5
DONE
```

### Modo por fecha

Ingresa una fecha `YYYY-MM-DD` (Enter = hoy). El sistema consulta la API, lista los partidos del día y predice todos.

### Salida por partido

```
  ─────────────────────────────────────────────────────
     SEA @ BAL      (Seattle Mariners @ Baltimore Orioles)
  ─────────────────────────────────────────────────────
  Ganador predicho : SEA  (57% confianza)
  LOCAL win prob   : 43%  |  VISITANTE: 57%
  Total predicho   : 9.1 carreras
  O/U 8.5          : OVER  (over prob: 62%)

  + BAL (LOCAL):
    ✓ Local descansado: 3 días de descanso
    ✗ Ofensiva apagada: solo 2.80 carreras/partido
    ✗ Diferencial negativo: -2.00/partido

  + SEA (VISITANTE):
    ✓ Racha visitante fuerte: 70% en últimos 10 partidos
    ✓ Pitcheo visitante sólido: 2.90 carreras/partido
```

---

## Opciones avanzadas

```bash
# Ver datos que se descargarían sin ejecutar
python src/ingest/update_data.py --dry-run

# Re-descargar todo desde 2015 (lento, ~1 hora)
python src/ingest/update_data.py --full

# Más threads para boxscores
python src/ingest/build_team_games_base.py --workers 20
python src/ingest/update_data.py --workers 20

# Entrenar con menos años de historial
python src/models/train.py --rolling-years 5
python src/models/train_totals_v3.py --rolling-years 3 --no-interactions

# Evaluar accuracy por umbral de confianza
python src/evaluation/eval_confidence.py

# Verificar integridad de los CSVs
python tests/verified_values.py
python tests/season_verified.py
```

---

## Features del modelo

### Rolling por equipo (ventanas 5, 10, 20 partidos)

Calculadas con `shift(1)` — el partido N solo usa datos de los N-1 anteriores.

| Categoría | Métricas |
|-----------|----------|
| Resultado | `WIN_RATE`, `RUN_DIFF` |
| Ofensiva  | `RUNS_FOR`, `HITS`, `HR`, `WALKS`, `SO_BAT` |
| Pitcheo   | `RUNS_AGAINST`, `SO_PIT`, `BB_ALLOWED`, `ERRORS` |
| Contexto  | `DAYS_REST` (cap 5), `BACK_TO_BACK`, `WIN_STREAK` |

### A nivel partido

Cada métrica se duplica con prefijos `HOME_` y `AWAY_`, más diferenciales `DIFF_*`. El modelo de ganador usa 60 features curadas (solo `DIFF_*` + absolutas clave) para evitar colinealidad.

---

## Notas técnicas

- **Anti-leakage:** `shift(1)` en todas las rolling features + split temporal 80/20 sin shuffle.
- **DAYS_REST cappado a 5:** evita que el gap de off-season (sept → marzo = 180+ días) domine el modelo.
- **Boxscores en caché:** `.boxscore_cache.json` persiste entre ejecuciones. Si `update_data.py` se interrumpe, reanuda desde donde quedó.
- **Temporada 2020:** 60 partidos (COVID). Puede sesgar rolling de ventana 20 — excluirla con `--rolling-years 5` si los resultados son ruidosos.
