# Pipeline de Predicción — MLB (MML-MLB)

## Contexto del Proyecto
Sistema automatizado de predicción de juegos de MLB.
Stack: Python (sklearn) + Next.js dashboard (safesports-panel).

## Arquitectura
```
[Modelo sklearn] → [safesports-panel]
```

## Roadmap / Plan de Implementación

### Fase 1 — Setup Inicial ✅
- [x] MLB Stats API collector
- [x] pybaseball wrapper (bref endpoints)
- [x] Park factors desde MLB API

### Fase 2 — Data Pipeline ✅
- [x] Download histórico 2021-2025
- [x] Actualización incremental diaria
- [x] 28 CSVs (games, standings, batting, pitching, pitcher_stats, teams, park_factors)

### Fase 3 — Features ✅
- [x] 50 features por juego
- [x] Pitcher season stats (ERA, WHIP, K/9)
- [x] Pitcher rolling form (últimos 3 starts)
- [x] Team batting/pitching, standings, H2H, rest, venue

### Fase 4 — Modelos ✅
- [x] WinnerPredictor: 3 modelos (RF, GB, LR) con auto-selección
- [x] RunsPredictor: 3 modelos (RF, GB, LR clasificador O/U) con auto-selección
- [x] Mejor winner: LogisticRegression (57.7%)
- [x] Mejor runs: LogisticRegression (55.1% O/U 8.5)

### Fase 5 — Competitividad ✅
- [x] Global, por división (6 divisiones)
- [x] Detección de tanking
- [x] Upset risk basado en win%

### Fase 6 — Integración safesports-panel ✅
- [x] Exportación de predicciones al panel
- [x] API key via `/api/auth/api-key/generate`
- [x] Endpoint `/api/predictions/import`
- [x] Formato con `sport: "mlb"`

## Integración con safesports-panel

### Configuración

**1. En safesports-panel (.env.local):**
```bash
IMPORT_API_SECRET=tu-secret-aqui
```

**2. En MML-MLB (.env.local):**
```bash
SAFESPORTS_PANEL_URL=http://localhost:3000
SAFESPORTS_PANEL_EMAIL=admin@sudo.com
SAFESPORTS_PANEL_PASSWORD=Admin123!
```

O usando API key directa:
```bash
SAFESPORTS_USER_API_KEY=sk_...
```

### Uso desde el Menú

```bash
cd /Users/sas/Documents/Github/MML-MLB
source mlb-env/bin/activate
python3 main.py
```

1. Selecciona opción **4. Exportar predicciones al panel**
2. Las predicciones se envían automáticamente

### Endpoints del Panel

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/auth/api-key/generate` | Genera API key con email/password |
| POST | `/api/predictions/import` | Importa predicciones |

### Formato de Predicciones

```json
{
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
    "forWinner": ["Confianza del modelo: 62%"],
    "forLoser": ["Factor de riesgo: 38%"],
    "summary": {
      "winnerFactors": 4,
      "loserFactors": 1,
      "matchupType": "mlb_regular",
      "betRecommendation": "NYY with 62% confidence"
    }
  }
}
```

---

## Comandos Frecuentes

```bash
# Activar entorno
source mlb-env/bin/activate

# Menú interactivo (recomendado)
python3 main.py

# Entrenar modelos
python3 main.py --train

# Predecir juego específico
python3 main.py --predict NYY BOS

# Ver ayuda
python3 main.py --help
```

### Menú Principal
1. Predecir juego (interactivo)
2. Entrenar modelos
3. Actualizar datos
4. Exportar predicciones al panel
5. Salir

---

## Estructura del Proyecto

```
MML-MLB/
├── main.py                      # Menú principal
├── src_v2/
│   ├── data/
│   │   ├── mlb_collector.py     # MLB Stats API
│   │   ├── mlb_pybaseball.py    # pybaseball wrapper
│   │   └── cleaner.py           # Limpieza de datos
│   ├── features/
│   │   ├── mlb_feature_engineer.py  # 50 features
│   │   └── competitiveness.py       # Competitividad MLB
│   ├── models/
│   │   ├── winner_predictor.py  # 3 modelos winner
│   │   └── runs_predictor.py    # 3 modelos runs
│   └── evaluation/
│       └── evaluator.py         # Métricas
├── data/                        # CSVs
├── models_mlb/                  # Modelos .pkl
├── mlb-env/                     # Entorno virtual
└── .env.local                   # Credenciales panel
```

---

## Variables de Entorno Requeridas

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| SAFESPORTS_PANEL_URL | URL del panel | http://localhost:3000 |
| SAFESPORTS_PANEL_EMAIL | Email del admin | admin@sudo.com |
| SAFESPORTS_PANEL_PASSWORD | Password del admin | Admin123! |

---

## Reglas del Proyecto
- Usar `.env.local` para credenciales (no commitear)
- Responder en español
- 3 modelos por predictor, elegir el mejor automáticamente
- Validación cronológica: train 2021-2023, test 2024
