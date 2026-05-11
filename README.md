<<<<<<< HEAD
# MML-mlb
modelo de predicción de la liga de beisbol MLB. 
=======
# MML-MLB: MLB Match Prediction System

Sistema de predicción de juegos de MLB usando Machine Learning.

**Winner Predictor:** 57.7% accuracy (LogisticRegression)
**Runs O/U 8.5:** 55.1% accuracy (LogisticRegression)

## Instalación

```bash
git clone <repo-url>
cd MML-MLB
python3 -m venv mlb-env
source mlb-env/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
source mlb-env/bin/activate
python3 main.py
```

Menú principal:
1. Predecir juego (interactivo)
2. Entrenar modelos
3. Actualizar datos
4. Exportar predicciones al panel
5. Salir

### Línea de comandos

```bash
python3 main.py --train              # Entrenar modelos
python3 main.py --predict NYY BOS    # Predecir juego
python3 main.py --update             # Actualizar datos
```

## Fuentes de datos

- **MLB Stats API** (`statsapi.mlb.com`) — schedule, boxscores, standings, pitchers
- **pybaseball** (bref endpoints) — batting stats, pitching stats
- **Entrenamiento:** 2021-2023 (8,553 samples)
- **Validación:** 2024 (2,935 juegos)

## Modelos

Cada predictor entrena 3 algoritmos y elige el mejor:

| Predictor | RandomForest | GradientBoosting | LogisticRegression |
|-----------|-------------|-----------------|-------------------|
| Winner | 59.4% | 59.4% | **60.1%** |
| Runs O/U | 56.5% | 55.9% | **56.9%** |

## Features (50)

- Forma reciente equipos, rendimiento local/visitante, descanso
- H2H, standings, stats ofensivas/defensivas por equipo
- Estadísticas de temporada de pitchers (ERA, WHIP, K/9)
- Forma reciente de pitchers (últimos 3 starts)
- Competitividad por división, detección de tanking

## Integración

Exporta predicciones al [safesports-panel](https://github.com/anomalyco/safesports-panel)
vía API REST con formato `sport: "mlb"`.
>>>>>>> 898d1fe (pre merge)
