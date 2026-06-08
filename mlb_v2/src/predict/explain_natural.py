"""
Genera señales en lenguaje natural comparando cada equipo contra la media de liga.
Siempre produce señales para ambos equipos — nunca "sin señales destacadas".
"""

# Medias de liga 2026 (actualizadas desde mlb_team_games_features.csv)
LEAGUE_AVG = {
    "WIN_RATE_roll10":     0.50,
    "RUNS_FOR_roll10":     4.39,
    "RUNS_AGAINST_roll10": 4.38,
    "RUN_DIFF_roll10":     0.01,
    "HR_roll10":           1.09,
    "SO_PIT_roll10":       8.44,
    "WALKS_roll10":        3.46,
    "SO_BAT_roll10":       8.42,
    "RUNS_FOR_roll20":     4.39,
    "RUNS_AGAINST_roll20": 4.38,
    "RUN_DIFF_roll20":     0.01,
    "WIN_RATE_roll20":     0.50,
}


def _vs_avg(val: float, avg: float, pct: bool = False) -> str:
    diff = val - avg
    sign = "+" if diff >= 0 else ""
    if pct:
        return f"{sign}{diff*100:.0f}%"
    return f"{sign}{diff:.1f}"


def _rating(val: float, avg: float, std: float, higher_better: bool = True) -> str:
    """Clasifica un valor vs media de liga."""
    diff = (val - avg) / std if std > 0 else 0
    if higher_better:
        if diff > 1.0:   return "ÉLITE"
        if diff > 0.5:   return "BUENO"
        if diff > -0.5:  return "PROMEDIO"
        if diff > -1.0:  return "BAJO"
        return "MUY BAJO"
    else:  # lower is better (ERA, RUNS_AGAINST)
        if diff < -1.0:  return "ÉLITE"
        if diff < -0.5:  return "BUENO"
        if diff < 0.5:   return "PROMEDIO"
        if diff < 1.0:   return "BAJO"
        return "MUY BAJO"


def build_signals(home_name: str, away_name: str, feat: dict) -> dict:
    """
    Construye señales estructuradas para el formato de salida.

    Retorna:
        favor_winner   → lista de puntos que favorecen al predicho ganador
        favor_loser    → lista de puntos que favorecen al otro equipo
        totals_context → líneas de contexto para la predicción de carreras
        home_stats     → dict con stats clave del local
        away_stats     → dict con stats clave del visitante
    """
    def _h(col):
        return float(feat.get(f"HOME_{col}", 0) or 0)

    def _a(col):
        return float(feat.get(f"AWAY_{col}", 0) or 0)

    def _diff(col):
        return float(feat.get(f"DIFF_{col}", 0) or 0)

    # ── stats clave de cada equipo ──────────────────────────────
    home_stats = {
        "win_rate_10":     _h("WIN_RATE_roll10"),
        "win_rate_20":     _h("WIN_RATE_roll20"),
        "runs_for_10":     _h("RUNS_FOR_roll10"),
        "runs_against_10": _h("RUNS_AGAINST_roll10"),
        "run_diff_10":     _h("RUN_DIFF_roll10"),
        "run_diff_20":     _h("RUN_DIFF_roll20"),
        "hr_10":           _h("HR_roll10"),
        "so_pit_10":       _h("SO_PIT_roll10"),
        "walks_10":        _h("WALKS_roll10"),
        "so_bat_10":       _h("SO_BAT_roll10"),
        "streak":          int(_h("WIN_STREAK")),
        "b2b":             int(_h("BACK_TO_BACK")),
        "rest":            int(_h("DAYS_REST")),
        "hits_10":         _h("HITS_roll10"),
    }
    away_stats = {
        "win_rate_10":     _a("WIN_RATE_roll10"),
        "win_rate_20":     _a("WIN_RATE_roll20"),
        "runs_for_10":     _a("RUNS_FOR_roll10"),
        "runs_against_10": _a("RUNS_AGAINST_roll10"),
        "run_diff_10":     _a("RUN_DIFF_roll10"),
        "run_diff_20":     _a("RUN_DIFF_roll20"),
        "hr_10":           _a("HR_roll10"),
        "so_pit_10":       _a("SO_PIT_roll10"),
        "walks_10":        _a("WALKS_roll10"),
        "so_bat_10":       _a("SO_BAT_roll10"),
        "streak":          int(_a("WIN_STREAK")),
        "b2b":             int(_a("BACK_TO_BACK")),
        "rest":            int(_a("DAYS_REST")),
        "hits_10":         _a("HITS_roll10"),
    }

    # ── Señales por equipo (siempre al menos 4) ─────────────────
    home_lines: list[tuple[float, str]] = []   # (score, texto)
    away_lines: list[tuple[float, str]] = []
    totals_context: list[str] = []

    # helper: agregar con score de relevancia
    def add_h(score: float, text: str):
        home_lines.append((score, text))

    def add_a(score: float, text: str):
        away_lines.append((score, text))

    la = LEAGUE_AVG

    # WIN RATE
    h_wr = home_stats["win_rate_10"]
    a_wr = away_stats["win_rate_10"]
    wins_h = round(h_wr * 10)
    wins_a = round(a_wr * 10)
    add_h(abs(h_wr - 0.5),
          f"{home_name} gana {wins_h}/10 últimos partidos como local "
          f"({_vs_avg(h_wr, la['WIN_RATE_roll10'], pct=True)} vs liga)")
    add_a(abs(a_wr - 0.5),
          f"{away_name} gana {wins_a}/10 últimos partidos como visitante "
          f"({_vs_avg(a_wr, la['WIN_RATE_roll10'], pct=True)} vs liga)")

    # WIN RATE últimos 20
    h_wr20 = home_stats["win_rate_20"]
    a_wr20 = away_stats["win_rate_20"]
    add_h(abs(h_wr20 - 0.5) * 0.6,
          f"Tendencia 20 juegos: {h_wr20:.0%} victorias "
          f"({'por encima' if h_wr20 > 0.5 else 'por debajo'} de la media)")
    add_a(abs(a_wr20 - 0.5) * 0.6,
          f"Tendencia 20 juegos: {a_wr20:.0%} victorias "
          f"({'por encima' if a_wr20 > 0.5 else 'por debajo'} de la media)")

    # RUN DIFFERENTIAL (métrica más predictiva según coeficientes)
    h_rd = home_stats["run_diff_10"]
    a_rd = away_stats["run_diff_10"]
    h_rd20 = home_stats["run_diff_20"]
    a_rd20 = away_stats["run_diff_20"]
    sign_h = "+" if h_rd >= 0 else ""
    sign_a = "+" if a_rd >= 0 else ""
    add_h(abs(h_rd) * 0.5,
          f"Diferencial carreras últimos 10: {sign_h}{h_rd:.1f} por partido "
          f"(últimos 20: {'+' if h_rd20>=0 else ''}{h_rd20:.1f})")
    add_a(abs(a_rd) * 0.5,
          f"Diferencial carreras últimos 10: {sign_a}{a_rd:.1f} por partido "
          f"(últimos 20: {'+' if a_rd20>=0 else ''}{a_rd20:.1f})")

    # OFENSIVA — carreras anotadas
    h_rf = home_stats["runs_for_10"]
    a_rf = away_stats["runs_for_10"]
    add_h(abs(h_rf - la["RUNS_FOR_roll10"]) * 0.4,
          f"Ofensiva: {h_rf:.1f} carreras/partido (liga {la['RUNS_FOR_roll10']:.1f}, "
          f"diferencia {_vs_avg(h_rf, la['RUNS_FOR_roll10'])})")
    add_a(abs(a_rf - la["RUNS_FOR_roll10"]) * 0.4,
          f"Ofensiva: {a_rf:.1f} carreras/partido (liga {la['RUNS_FOR_roll10']:.1f}, "
          f"diferencia {_vs_avg(a_rf, la['RUNS_FOR_roll10'])})")

    # PITCHEO — carreras permitidas
    h_ra = home_stats["runs_against_10"]
    a_ra = away_stats["runs_against_10"]
    add_h(abs(h_ra - la["RUNS_AGAINST_roll10"]) * 0.4,
          f"Pitcheo: concede {h_ra:.1f} carreras/partido (liga {la['RUNS_AGAINST_roll10']:.1f}, "
          f"diferencia {_vs_avg(h_ra, la['RUNS_AGAINST_roll10'])})")
    add_a(abs(a_ra - la["RUNS_AGAINST_roll10"]) * 0.4,
          f"Pitcheo: concede {a_ra:.1f} carreras/partido (liga {la['RUNS_AGAINST_roll10']:.1f}, "
          f"diferencia {_vs_avg(a_ra, la['RUNS_AGAINST_roll10'])})")

    # PONCHES DEL PITCHEO (SO_PIT — segunda métrica más predictiva)
    h_so = home_stats["so_pit_10"]
    a_so = away_stats["so_pit_10"]
    add_h(abs(h_so - la["SO_PIT_roll10"]) * 0.35,
          f"Pitcheo poncha {h_so:.1f}/partido (liga {la['SO_PIT_roll10']:.1f}, "
          f"diferencia {_vs_avg(h_so, la['SO_PIT_roll10'])})")
    add_a(abs(a_so - la["SO_PIT_roll10"]) * 0.35,
          f"Pitcheo poncha {a_so:.1f}/partido (liga {la['SO_PIT_roll10']:.1f}, "
          f"diferencia {_vs_avg(a_so, la['SO_PIT_roll10'])})")

    # HR — poder ofensivo
    h_hr = home_stats["hr_10"]
    a_hr = away_stats["hr_10"]
    add_h(abs(h_hr - la["HR_roll10"]) * 0.3,
          f"Poder: {h_hr:.1f} HR/partido (liga {la['HR_roll10']:.1f})")
    add_a(abs(a_hr - la["HR_roll10"]) * 0.3,
          f"Poder: {a_hr:.1f} HR/partido (liga {la['HR_roll10']:.1f})")

    # DESCANSO / BACK-TO-BACK
    if home_stats["b2b"]:
        add_h(0.8, f"⚠ Juega en back-to-back (segundo partido consecutivo, 0 días descanso)")
    elif home_stats["rest"] >= 3:
        add_h(0.5, f"Descansado: {home_stats['rest']} días desde último partido")
    else:
        add_h(0.2, f"Días de descanso: {home_stats['rest']}")

    if away_stats["b2b"]:
        add_a(0.8, f"⚠ Juega en back-to-back (segundo partido consecutivo, 0 días descanso)")
    elif away_stats["rest"] >= 3:
        add_a(0.5, f"Descansado: {away_stats['rest']} días desde último partido")
    else:
        add_a(0.2, f"Días de descanso: {away_stats['rest']}")

    # RACHA ACTIVA
    h_str = home_stats["streak"]
    a_str = away_stats["streak"]
    if h_str > 0:
        add_h(min(h_str * 0.2, 0.9),
              f"Racha activa: {h_str} {'victoria' if h_str==1 else 'victorias'} consecutiva{'s' if h_str>1 else ''}")
    elif h_str < 0:
        add_h(min(abs(h_str) * 0.2, 0.9),
              f"Racha negativa: {abs(h_str)} {'derrota' if abs(h_str)==1 else 'derrotas'} consecutiva{'s' if abs(h_str)>1 else ''}")
    else:
        add_h(0.1, "Sin racha activa (alternando resultados)")

    if a_str > 0:
        add_a(min(a_str * 0.2, 0.9),
              f"Racha activa: {a_str} {'victoria' if a_str==1 else 'victorias'} consecutiva{'s' if a_str>1 else ''}")
    elif a_str < 0:
        add_a(min(abs(a_str) * 0.2, 0.9),
              f"Racha negativa: {abs(a_str)} {'derrota' if abs(a_str)==1 else 'derrotas'} consecutiva{'s' if abs(a_str)>1 else ''}")
    else:
        add_a(0.1, "Sin racha activa (alternando resultados)")

    # Ordenar por relevancia (score descendente)
    home_lines.sort(key=lambda x: x[0], reverse=True)
    away_lines.sort(key=lambda x: x[0], reverse=True)

    # ── Contexto de totales ─────────────────────────────────────
    total_expected = h_rf + a_ra  # proxy: ofensiva local vs pitcheo visitante
    total_expected2 = a_rf + h_ra  # ofensiva visitante vs pitcheo local

    totals_context.append(
        f"Ofensiva local ({h_rf:.1f}) vs Pitcheo visitante ({a_ra:.1f}) → "
        f"{h_rf + a_ra:.1f} carreras esperadas en esa mitad"
    )
    totals_context.append(
        f"Ofensiva visitante ({a_rf:.1f}) vs Pitcheo local ({h_ra:.1f}) → "
        f"{a_rf + h_ra:.1f} carreras esperadas en esa mitad"
    )

    hr_total = h_hr + a_hr
    if hr_total > 2.5:
        totals_context.append(f"Poder combinado alto: {hr_total:.1f} HR/partido entre ambos equipos → presión OVER")
    elif hr_total < 1.5:
        totals_context.append(f"Poco poder ofensivo combinado: {hr_total:.1f} HR/partido → presión UNDER")
    else:
        totals_context.append(f"Poder combinado moderado: {hr_total:.1f} HR/partido entre ambos equipos")

    if home_stats["b2b"] and away_stats["b2b"]:
        totals_context.append("Ambos equipos en back-to-back → bullpenes cansados, posible más carreras tarde")
    elif home_stats["b2b"] or away_stats["b2b"]:
        team_b2b = home_name if home_stats["b2b"] else away_name
        totals_context.append(f"{team_b2b} en back-to-back → pitcheo puede ser más vulnerable")

    return {
        "home_lines":      [t for _, t in home_lines],
        "away_lines":      [t for _, t in away_lines],
        "totals_context":  totals_context,
        "home_stats":      home_stats,
        "away_stats":      away_stats,
    }
