"""
Поиск уровней S/R, конфлюенс-зоны (Layer 2), HTF-анализ,
генерация сигналов (Layer 1/2/3) и фильтр корреляции.
"""
import logging
import math
from typing import Dict, FrozenSet, List, Optional

import pandas as pd

from config import (
    CORR_MAX, CORR_LOOKBACK,
    ENTRY_PROXIMITY_PERCENT, EMA_PERIOD, ATR_PERIOD,
    SL_ATR_MULT, TP_ATR_MIN_MULT, MIN_RR, ADX_MIN, SHORT_ONLY,
    EXTREMA_WINDOW, TOLERANCE_PERCENT,
    MIN_TOUCHES, TOP_N_LEVELS, MIN_TOUCH_SPACING,
    LEVEL_AGE_MIN_CANDLES, VOLUME_TOUCH_MULTIPLIER, REQUIRE_RETEST,
    EMA_CONFLUENCE_PERIODS, FVG_LOOKBACK, FVG_MIN_GAP_PCT,
    HTF_TIMEFRAME, HTF_EMA_PERIOD, HTF_SR_CANDLES, HTF_STRUCTURE_WINDOW,
    REQUIRE_RSI_DIVERGENCE, REQUIRE_PATTERN, SIGNAL_VOLUME_MULT,
    CRASH_SL_BUFFER, PUMP_SL_BUFFER, HTF_SR_REQUIRE_CONFIRM,
    MIN_CONFLUENCE_SCORE, HTF_RSI_OVERBOUGHT, HTF_RSI_OVERSOLD,
    CHOPPY_ADX_CONFIRM, CHOPPY_ATR_MIN, RR_MAX,
    MAX_SL_PERCENT, LONG_MIN_CONFLUENCE, LONG_HTF_RSI_MIN, HTF_BELOW_EMA_MAX,
    STRUCTURAL_SL, STRUCTURAL_SL_BUFFER, STRUCTURAL_SL_LOOKBACK,
    QUALITY_SIZING, QUALITY_MULT_MIN, QUALITY_MULT_MAX,
    ADAPTIVE_SL, MAX_SL_PERCENT_BEAR, MACRO_EMA_PERIOD, TREND_ALIGN,
)
from indicators import (
    _calc_ema, _calc_atr, _calc_rsi_series, _calc_adx_series,
)

logger = logging.getLogger("signaler")

# ============================================================
# ПОИСК УРОВНЕЙ S/R
# ============================================================

def _find_local_extrema(series: pd.Series, window: int, kind: str) -> List[tuple]:
    values = series.values
    extrema = []
    for i in range(window, len(values) - window):
        segment = values[i - window:i + window + 1]
        center  = values[i]
        if kind == "max" and center == segment.max():
            extrema.append((float(center), i))
        elif kind == "min" and center == segment.min():
            extrema.append((float(center), i))
    return extrema


def _count_touches(
    level: float,
    df: pd.DataFrame,
    tolerance: float,
    min_spacing: int = 0,
    volume_multiplier: float = 0.0,
) -> int:
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    hit = df["high"].between(lower, upper) | df["low"].between(lower, upper)
    if volume_multiplier > 0:
        avg_vol = df["volume"].mean()
        if avg_vol > 0:
            hit = hit & (df["volume"] >= avg_vol * volume_multiplier)
    positions = hit.values.nonzero()[0]
    if len(positions) == 0:
        return 0
    if min_spacing <= 0:
        return len(positions)
    filtered = [positions[0]]
    for pos in positions[1:]:
        if pos - filtered[-1] >= min_spacing:
            filtered.append(pos)
    return len(filtered)


def _has_retest_after_breakout(
    level: float, df: pd.DataFrame, tolerance: float, level_type: str
) -> bool:
    upper  = level * (1 + tolerance)
    lower  = level * (1 - tolerance)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    in_breakout = False
    for i in range(len(closes)):
        if not in_breakout:
            if level_type == "SUPPORT"    and closes[i] < lower: in_breakout = True
            elif level_type == "RESISTANCE" and closes[i] > upper: in_breakout = True
        else:
            if (lower <= highs[i] <= upper) or (lower <= lows[i] <= upper):
                return True
            if level_type == "SUPPORT"    and closes[i] > upper: in_breakout = False
            elif level_type == "RESISTANCE" and closes[i] < lower: in_breakout = False
    return False


def remove_duplicates(levels: List[Dict], tolerance: float) -> List[Dict]:
    if not levels:
        return []
    unique: List[Dict] = []
    for level in sorted(levels, key=lambda x: x["touches"], reverse=True):
        if not level.get("price") or level["price"] <= 0:
            continue
        if not any(
            kept["type"] == level["type"]
            and kept["price"] > 0
            and abs(level["price"] - kept["price"]) / kept["price"] < tolerance
            for kept in unique
        ):
            unique.append(level)
    return unique


# ============================================================
# СЛОЙ 2 — конфлюенс зоны: EMA + FVG + круглые числа
# ============================================================

def _calc_ema_levels(df: pd.DataFrame, periods: List[int]) -> List[Dict]:
    current = float(df["close"].iloc[-1])
    lvls: List[Dict] = []
    for period in periods:
        if len(df) >= period:
            val = float(_calc_ema(df["close"], period).iloc[-1])
            lvls.append({
                "price":   round(val, 8),
                "type":    "SUPPORT" if current > val else "RESISTANCE",
                "subtype": f"EMA{period}",
            })
    return lvls


def _find_fvg_zones(
    df: pd.DataFrame,
    lookback: int = 50,
    min_gap_pct: float = 0.1,
) -> List[Dict]:
    zones: List[Dict] = []
    n     = len(df)
    start = max(2, n - lookback)
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    for i in range(start, n):
        h_prev2, l_prev2 = high[i - 2], low[i - 2]
        h_curr,  l_curr  = high[i],     low[i]
        if l_curr > h_prev2:
            mid = (h_prev2 + l_curr) / 2
            if (l_curr - h_prev2) / mid * 100 >= min_gap_pct:
                zones.append({"type": "SUPPORT", "bottom": float(h_prev2), "top": float(l_curr),
                               "price": float(mid), "age_candles": n - 1 - i})
        elif h_curr < l_prev2:
            mid = (h_curr + l_prev2) / 2
            if (l_prev2 - h_curr) / mid * 100 >= min_gap_pct:
                zones.append({"type": "RESISTANCE", "bottom": float(h_curr), "top": float(l_prev2),
                               "price": float(mid), "age_candles": n - 1 - i})
    return zones


def _is_round_number(price: float, tol_pct: float = TOLERANCE_PERCENT) -> bool:
    if price <= 0:
        return False
    tol = price * tol_pct / 100.0
    mag = 10.0 ** math.floor(math.log10(price))
    for step in (mag, mag / 2.0):
        nearest = round(price / step) * step
        if abs(price - nearest) <= tol:
            return True
    return False


def _add_confluence_scores(
    levels: List[Dict],
    ema_lvls: List[Dict],
    fvg_zones: List[Dict],
    tolerance_pct: float = TOLERANCE_PERCENT,
) -> None:
    tol = tolerance_pct / 100.0
    for lvl in levels:
        tags: List[str] = []
        p = lvl["price"]
        for ema in ema_lvls:
            if ema["type"] == lvl["type"] and p > 0 and abs(ema["price"] - p) / p <= tol:
                tags.append(ema["subtype"])
        for fvg in fvg_zones:
            if fvg["type"] == lvl["type"] and fvg["bottom"] <= p <= fvg["top"]:
                tags.append("FVG")
                break
        if _is_round_number(p, tolerance_pct):
            tags.append("🔢")
        lvl["confluence_score"] = len(tags)
        lvl["confluence_tags"]  = tags


def find_support_resistance(
    df: pd.DataFrame,
    min_touches: int = MIN_TOUCHES,
    tolerance_percent: float = TOLERANCE_PERCENT,
    window: int = EXTREMA_WINDOW,
    top_n: int = TOP_N_LEVELS,
    min_touch_spacing: int = MIN_TOUCH_SPACING,
    level_age_min: int = LEVEL_AGE_MIN_CANDLES,
    volume_multiplier: float = VOLUME_TOUCH_MULTIPLIER,
    require_retest: bool = REQUIRE_RETEST,
) -> List[Dict]:
    tolerance = tolerance_percent / 100.0
    total_candles = len(df)
    resistance_candidates = _find_local_extrema(df["high"], window, "max")
    support_candidates    = _find_local_extrema(df["low"],  window, "min")
    logger.info(
        "Кандидатов: сопр.=%d, подд.=%d",
        len(resistance_candidates), len(support_candidates),
    )
    levels: List[Dict] = []
    for candidates, level_type in [
        (resistance_candidates, "RESISTANCE"),
        (support_candidates,    "SUPPORT"),
    ]:
        for price, idx in candidates:
            if total_candles - 1 - idx < level_age_min:
                continue
            touches = _count_touches(price, df, tolerance,
                                     min_spacing=min_touch_spacing,
                                     volume_multiplier=volume_multiplier)
            if touches < min_touches:
                continue
            has_retest = _has_retest_after_breakout(price, df, tolerance, level_type)
            if require_retest and not has_retest:
                continue
            levels.append({"price": price, "type": level_type,
                           "touches": touches, "has_retest": has_retest})
    levels = remove_duplicates(levels, tolerance)
    levels = sorted(levels, key=lambda x: x["touches"], reverse=True)[:top_n]
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)
    logger.info("Найдено уровней: %d", len(levels))
    return levels


# ============================================================
# HTF — структура рынка и конфлюенс
# ============================================================

def _detect_htf_structure(df: pd.DataFrame, window: int = HTF_STRUCTURE_WINDOW) -> str:
    if df is None or len(df) < window * 6 + 1:
        return "RANGE"
    highs = _find_local_extrema(df["high"], window, "max")
    lows  = _find_local_extrema(df["low"],  window, "min")
    if len(highs) < 3 or len(lows) < 3:
        return "RANGE"
    h2, h1, h0 = highs[-3][0], highs[-2][0], highs[-1][0]
    l2, l1, l0 = lows[-3][0],  lows[-2][0],  lows[-1][0]
    if h0 > h1 > h2 and l0 > l1 > l2:
        return "BULLISH"
    if h0 < h1 < h2 and l0 < l1 < l2:
        return "BEARISH"
    return "RANGE"


def _fetch_htf_confluence(client, pair: str):
    """
    Один запрос HTF_TIMEFRAME + дополнительный запрос 1d для макро-тренда.
    Возвращает (trend, ema, sr_levels, structure, htf_rsi, htf_below_ema, macro_trend).
    """
    _default = (None, None, [], "RANGE", None, None, None)
    try:
        needed = max(HTF_SR_CANDLES, HTF_EMA_PERIOD + 30)
        raw = client.fetch_ohlcv(pair, HTF_TIMEFRAME, limit=needed)
        if not raw or len(raw) < HTF_EMA_PERIOD:
            return _default
        df_htf = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_htf["timestamp"] = pd.to_datetime(df_htf["timestamp"], unit="ms")
        closes    = df_htf["close"].astype(float)
        ema_s     = closes.ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
        ema       = float(ema_s.iloc[-1])
        trend     = "UP" if float(closes.iloc[-1]) > ema else "DOWN"
        structure = _detect_htf_structure(df_htf)
        rsi_raw   = float(_calc_rsi_series(df_htf).iloc[-1])
        htf_rsi   = rsi_raw if not pd.isna(rsi_raw) else None
        htf_below_ema = int((closes.iloc[-5:].values < ema_s.iloc[-5:].values).sum())
        htf_sr: List[Dict] = []
        if HTF_SR_CANDLES > 0 and len(df_htf) >= EXTREMA_WINDOW * 2 + 1:
            htf_sr = find_support_resistance(df_htf)

        # Macro trend (daily EMA200) — used as an input for adaptive SL sizing.
        macro_trend: Optional[str] = None
        try:
            raw_d = client.fetch_ohlcv(pair, "1d", limit=MACRO_EMA_PERIOD)
            if raw_d and len(raw_d) >= MACRO_EMA_PERIOD:
                daily_closes = pd.Series(
                    [c[4] for c in raw_d],
                    index=pd.to_datetime([c[0] for c in raw_d], unit="ms", utc=True),
                    dtype=float,
                )
                ema_d = daily_closes.ewm(span=MACRO_EMA_PERIOD, adjust=False).mean()
                macro_trend = "UP" if float(daily_closes.iloc[-1]) > float(ema_d.iloc[-1]) else "DOWN"
        except Exception:
            pass

        return trend, round(ema, 8), htf_sr, structure, htf_rsi, htf_below_ema, macro_trend
    except Exception as exc:
        logger.warning("Ошибка HTF анализа %s: %s", pair, exc)
        return _default


def _mark_htf_confirmed(
    levels: List[Dict], htf_levels: List[Dict], tolerance_pct: float
) -> None:
    tol = tolerance_pct / 100.0
    for lvl in levels:
        lvl["htf_confirmed"] = any(
            h["type"] == lvl["type"]
            and abs(h["price"] - lvl["price"]) / lvl["price"] <= tol
            for h in htf_levels
        )


# ============================================================
# ПАТТЕРНЫ СВЕЧЕЙ + ДИВЕРГЕНЦИЯ RSI
# ============================================================

def _detect_candle_pattern(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 3:
        return None
    c    = df.iloc[-2]
    prev = df.iloc[-3]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    total_range = h - l
    if total_range == 0:
        return None
    body        = abs(cl - o)
    upper_wick  = h - max(o, cl)
    lower_wick  = min(o, cl) - l
    body_ratio  = body / total_range
    if body_ratio < 0.3 and lower_wick / total_range > 0.6 and lower_wick > 2 * upper_wick:
        return "молот 🔨"
    if body_ratio < 0.3 and upper_wick / total_range > 0.6 and upper_wick > 2 * lower_wick:
        return "падающая звезда ⭐"
    p_o, p_cl = float(prev["open"]), float(prev["close"])
    if p_cl < p_o and cl > o and o <= p_cl and cl >= p_o:
        return "бычье поглощение 🕯"
    if p_cl > p_o and cl < o and o >= p_cl and cl <= p_o:
        return "медвежье поглощение 🕯"
    return None


# Должны совпадать со строками _detect_candle_pattern
_BULLISH_PATTERNS: FrozenSet[str] = frozenset({"молот 🔨", "бычье поглощение 🕯"})
_BEARISH_PATTERNS: FrozenSet[str] = frozenset({"падающая звезда ⭐", "медвежье поглощение 🕯"})


def _detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    direction: str,
    level_price: float,
    window: int = EXTREMA_WINDOW,
    proximity_pct: float = TOLERANCE_PERCENT,
) -> bool:
    if level_price is None or level_price <= 0:
        return False
    tol    = proximity_pct / 100.0 * 4
    n      = len(rsi_series)
    long   = direction == "LONG"
    series = df["low"] if long else df["high"]
    kind   = "min" if long else "max"
    swings = _find_local_extrema(series, window, kind)
    near   = [(p, i) for p, i in swings if abs(p - level_price) / level_price <= tol]
    if len(near) < 2:
        return False
    (p1, i1), (p2, i2) = near[-2], near[-1]
    price_extended = p2 < p1 if long else p2 > p1
    if not price_extended:
        return False
    r1 = float(rsi_series.iloc[i1]) if i1 < n else 50.0
    r2 = float(rsi_series.iloc[i2]) if i2 < n else 50.0
    return (r2 > r1) if long else (r2 < r1)


# ============================================================
# СТРУКТУРНЫЙ SL + КАЧЕСТВО СИГНАЛА
# ============================================================

def _find_structural_sl(
    df: pd.DataFrame,
    entry: float,
    direction: str,
) -> Optional[float]:
    """SL за ближайшим swing low/high вместо ATR. Возвращает цену SL или None."""
    window = max(3, EXTREMA_WINDOW // 2)
    needed = STRUCTURAL_SL_LOOKBACK + window * 2
    if len(df) < needed:
        return None
    recent = df.iloc[-needed:]
    if direction == "LONG":
        candidates = [p for p, _ in _find_local_extrema(recent["low"], window, "min") if p < entry]
        if not candidates:
            return None
        return max(candidates) * (1.0 - STRUCTURAL_SL_BUFFER / 100.0)
    else:
        candidates = [p for p, _ in _find_local_extrema(recent["high"], window, "max") if p > entry]
        if not candidates:
            return None
        return min(candidates) * (1.0 + STRUCTURAL_SL_BUFFER / 100.0)


def _calc_quality_mult(lvl: Dict, rr: float, htf_confirmed: bool, divergence: bool = False) -> float:
    """Множитель размера позиции по качеству сигнала [QUALITY_MULT_MIN, QUALITY_MULT_MAX]."""
    score = lvl.get("confluence_score", 0)
    if score == 0:
        mult = 0.7
    elif score == 1:
        mult = 0.85
    elif score == 2:
        mult = 1.0
    else:
        mult = 1.2
    if rr is not None:
        if rr >= 3.5:
            mult *= 1.2
        elif rr >= 2.5:
            mult *= 1.1
    if htf_confirmed:
        mult *= 1.1
    if REQUIRE_RSI_DIVERGENCE and divergence:
        mult *= 1.05
    return round(min(max(mult, QUALITY_MULT_MIN), QUALITY_MULT_MAX), 2)


# ============================================================
# ФИЛЬТР КОРРЕЛЯЦИИ
# ============================================================

_returns_cache: Dict[str, pd.Series] = {}


def _check_correlation(pair: str, open_pairs: List[str]) -> bool:
    if not open_pairs or CORR_MAX <= 0:
        return True
    r1 = _returns_cache.get(pair)
    if r1 is None:
        return True
    for op in open_pairs:
        r2 = _returns_cache.get(op)
        if r2 is None:
            continue
        combined = pd.concat([r1, r2], axis=1, sort=False).dropna()
        if len(combined) < 10:
            continue
        if abs(combined.iloc[:, 0].corr(combined.iloc[:, 1])) > CORR_MAX:
            return False
    return True


# ============================================================
# ГЕНЕРАЦИЯ СИГНАЛОВ (Слои 1/2/3)
# ============================================================

def _layer1_passes(
    direction: str,
    htf_structure: Optional[str],
    special: bool,
) -> bool:
    """HTF-structure gate (Layer 1).

    None is treated the same as "RANGE" — no signals when 4h data is
    unavailable; trading without HTF confirmation is not allowed.

    RSI-divergence is no longer a hard requirement here — it is rewarded as a
    position-sizing bonus in `_calc_quality_mult` instead (a hard gate measurably
    hurt PnL in both bull and bear backtests without changing trade direction).
    """
    if not special:
        if htf_structure is None or htf_structure == "RANGE":
            return False
        if htf_structure == "BULLISH" and direction != "LONG":
            return False
        if htf_structure == "BEARISH" and direction != "SHORT":
            return False
    return True


def _layer3_passes(
    pattern_confirmed: bool,
    vol_spike: bool,
    special: bool,
) -> bool:
    """Candle-pattern + volume gate (Layer 3)."""
    if not special and REQUIRE_PATTERN and not pattern_confirmed:
        return False
    if not special and SIGNAL_VOLUME_MULT > 0 and not vol_spike:
        return False
    return True


def find_entry_signals(
    df: pd.DataFrame,
    levels: List[Dict],
    current_price: float,
    htf_trend: Optional[str] = None,
    adx_val: Optional[float] = None,
    proximity_percent: float = ENTRY_PROXIMITY_PERCENT,
    ema_period: int = EMA_PERIOD,
    atr_period: int = ATR_PERIOD,
    sl_atr_mult: float = SL_ATR_MULT,
    tp_atr_min_mult: float = TP_ATR_MIN_MULT,
    min_rr: float = MIN_RR,
    adx_min: float = ADX_MIN,
    regime: str = "NORMAL",
    crash_low: Optional[float] = None,
    pump_high: Optional[float] = None,
    htf_structure: Optional[str] = None,
    rsi_series: Optional[pd.Series] = None,
    htf_rsi: Optional[float] = None,
    atr_ratio: float = 1.0,
    adx_series: Optional[pd.Series] = None,
    htf_below_ema: Optional[int] = None,
    macro_trend: Optional[str] = None,
) -> List[Dict]:
    if regime in ("CRASH", "PUMP"):
        return []

    if ema_period >= len(df):
        logger.debug("Недостаточно свечей для EMA%d (%d)", ema_period, len(df))
        return []

    proximity = proximity_percent / 100.0
    ema_val   = float(_calc_ema(df["close"], ema_period).iloc[-1])
    ema_trend = "UP" if current_price > ema_val else "DOWN"
    pattern   = _detect_candle_pattern(df)
    atr       = _calc_atr(df, atr_period)
    if pd.isna(atr) or atr <= 0:
        logger.warning("ATR некорректен (%.6f) для %s — сигналы пропущены", atr, regime)
        return []

    if SIGNAL_VOLUME_MULT > 0:
        sig_vol   = float(df["volume"].iloc[-2])
        avg_vol   = float(df["volume"].iloc[-51:-1].mean())
        vol_ratio = round(sig_vol / avg_vol, 1) if avg_vol > 0 else None
        vol_spike = vol_ratio is not None and vol_ratio >= SIGNAL_VOLUME_MULT
    else:
        vol_ratio = None
        vol_spike = False

    recovery   = (regime == "RECOVERY")
    correction = (regime == "CORRECTION")
    special    = recovery or correction

    if not special:
        # ATR choppy filter: reject if current volatility is far below its average
        if CHOPPY_ATR_MIN > 0 and atr_ratio < CHOPPY_ATR_MIN:
            logger.info("ATR ratio %.2f < %.2f — слишком тихо, сигналы пропущены", atr_ratio, CHOPPY_ATR_MIN)
            return []
        # ADX persistence filter: require ADX ≥ adx_min for last N consecutive bars
        if adx_min > 0 and CHOPPY_ADX_CONFIRM > 0:
            adx_s = adx_series if adx_series is not None else _calc_adx_series(df, atr_period)[0]
            if len(adx_s) >= CHOPPY_ADX_CONFIRM:
                if float(adx_s.iloc[-CHOPPY_ADX_CONFIRM:].min()) < adx_min:
                    logger.info("ADX choppy — min %.1f < %.1f за последние %d баров",
                                float(adx_s.iloc[-CHOPPY_ADX_CONFIRM:].min()), adx_min, CHOPPY_ADX_CONFIRM)
                    return []
        elif adx_val is not None and adx_min > 0 and adx_val < adx_min:
            logger.info("ADX %.1f < %.1f — боковик, сигналы пропущены", adx_val, adx_min)
            return []

    signals = []
    for lvl in levels:
        if not lvl["price"]:
            continue
        distance = abs(current_price - lvl["price"]) / lvl["price"]
        if distance > proximity:
            continue

        if not special and lvl.get("confluence_score", 0) < MIN_CONFLUENCE_SCORE:
            continue

        if recovery:
            if lvl["type"] != "SUPPORT":
                continue
            direction = "LONG"
        elif correction:
            if lvl["type"] != "RESISTANCE":
                continue
            direction = "SHORT"
        else:
            if lvl["type"] == "SUPPORT" and ema_trend == "UP":
                direction = "LONG"
            elif lvl["type"] == "RESISTANCE" and ema_trend == "DOWN":
                direction = "SHORT"
            else:
                continue

        entry = lvl["price"]

        if SHORT_ONLY and direction == "LONG":
            continue

        if TREND_ALIGN:
            if not special and macro_trend is not None:
                if macro_trend == "DOWN" and direction == "LONG":
                    continue
                if macro_trend == "UP" and direction == "SHORT":
                    continue
            if special and macro_trend is None:
                # Skip crash/pump bounces on tokens with no macro history (<200 trading days)
                continue

        if not special and direction == "LONG" and LONG_MIN_CONFLUENCE > 0:
            if lvl.get("confluence_score", 0) < LONG_MIN_CONFLUENCE:
                logger.debug("LONG пропущен: confluence=%d < LONG_MIN_CONFLUENCE=%d",
                             lvl.get("confluence_score", 0), LONG_MIN_CONFLUENCE)
                continue

        if not special and htf_rsi is not None:
            if direction == "LONG"  and htf_rsi > HTF_RSI_OVERBOUGHT:
                continue
            if direction == "SHORT" and htf_rsi < HTF_RSI_OVERSOLD:
                continue
            if direction == "LONG" and LONG_HTF_RSI_MIN > 0 and htf_rsi < LONG_HTF_RSI_MIN:
                continue

        if (not special and direction == "LONG"
                and htf_below_ema is not None and HTF_BELOW_EMA_MAX >= 0
                and htf_below_ema > HTF_BELOW_EMA_MAX):
            logger.debug("LONG пропущен: %d из 5 HTF свечей ниже EMA > HTF_BELOW_EMA_MAX=%d",
                         htf_below_ema, HTF_BELOW_EMA_MAX)
            continue

        divergence = (
            _detect_rsi_divergence(df, rsi_series, direction, entry)
            if rsi_series is not None else False
        )
        if not _layer1_passes(direction, htf_structure, special):
            continue

        pattern_confirmed = pattern in (_BULLISH_PATTERNS if direction == "LONG" else _BEARISH_PATTERNS)
        if not _layer3_passes(pattern_confirmed, vol_spike, special):
            continue

        if recovery and crash_low is not None:
            sl      = crash_low * (1.0 - CRASH_SL_BUFFER / 100.0)
            sl_dist = entry - sl
            if sl_dist <= 0:
                continue
            tp_min = sl_dist * min_rr
            above  = sorted(
                [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > entry],
                key=lambda x: x["price"],
            )
            tp = above[0]["price"] if above and above[0]["price"] >= entry + tp_min else entry + tp_min
        elif correction and pump_high is not None:
            sl      = pump_high * (1.0 + PUMP_SL_BUFFER / 100.0)
            sl_dist = sl - entry
            if sl_dist <= 0:
                continue
            tp_min = sl_dist * min_rr
            below  = sorted(
                [l for l in levels if l["type"] == "SUPPORT" and l["price"] < entry],
                key=lambda x: x["price"], reverse=True,
            )
            tp = below[0]["price"] if below and below[0]["price"] <= entry - tp_min else entry - tp_min
        elif special:
            logger.warning("режим %s без crash_low/pump_high — SL по ATR", regime)
            sl_dist = atr * sl_atr_mult
            tp_min  = sl_dist * min_rr
            sl = entry - sl_dist if recovery else entry + sl_dist
            tp = entry + tp_min  if recovery else entry - tp_min
        else:
            # Structural SL (swing low/high) with ATR fallback
            struct_sl = _find_structural_sl(df, entry, direction) if STRUCTURAL_SL else None
            if struct_sl is not None:
                sl      = struct_sl
                sl_dist = abs(entry - sl)
            else:
                sl_dist = atr * sl_atr_mult
                sl      = entry - sl_dist if direction == "LONG" else entry + sl_dist
            tp_min = sl_dist * min_rr

            if direction == "LONG":
                above = sorted(
                    [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > entry],
                    key=lambda x: x["price"],
                )
                if above:
                    nearest    = above[0]["price"]
                    nearest_rr = (nearest - entry) / sl_dist
                    if nearest_rr >= min_rr:
                        tp = nearest
                    else:
                        continue  # nearest S/R doesn't meet RR — skip
                else:
                    tp = entry + tp_min
            else:
                below = sorted(
                    [l for l in levels if l["type"] == "SUPPORT" and l["price"] < entry],
                    key=lambda x: x["price"], reverse=True,
                )
                if below:
                    nearest    = below[0]["price"]
                    nearest_rr = (entry - nearest) / sl_dist
                    if nearest_rr >= min_rr:
                        tp = nearest
                    else:
                        continue  # nearest S/R doesn't meet RR — skip
                else:
                    tp = entry - tp_min

        # Reject trades where SL is too wide regardless of regime.
        # Adaptive cap: tighten the limit during a macro downtrend (daily close
        # below EMA200) — backtests show a tighter cap helps in bear regimes but
        # costs PnL in bull regimes, so the cap follows the macro trend.
        max_sl_percent = MAX_SL_PERCENT
        if ADAPTIVE_SL and macro_trend == "DOWN":
            max_sl_percent = MAX_SL_PERCENT_BEAR
        if max_sl_percent > 0 and sl_dist > 0:
            if sl_dist / entry * 100 > max_sl_percent:
                continue

        # Cap TP at RR_MAX to avoid unreachable targets (e.g. RR=31, 74)
        if RR_MAX > 0 and sl_dist > 0:
            if direction == "LONG":
                tp = min(tp, entry + sl_dist * RR_MAX)
            else:
                tp = max(tp, entry - sl_dist * RR_MAX)

        tp_dist    = abs(tp - entry)
        rr         = round(tp_dist / sl_dist, 1) if sl_dist > 0 else None
        if rr is None or rr < min_rr:
            continue

        risk_pct    = round(sl_dist / entry * 100, 2)
        reward_pct  = round(tp_dist / entry * 100, 2)
        quality_mult = (
            _calc_quality_mult(lvl, rr, lvl.get("htf_confirmed", False), divergence)
            if QUALITY_SIZING else 1.0
        )

        signals.append({
            "direction":         direction,
            "level":             lvl,
            "pattern":           pattern,
            "ema":               ema_val,
            "ema_trend":         ema_trend,
            "htf_trend":         htf_trend,
            "htf_structure":     htf_structure,
            "adx":               round(adx_val, 1) if adx_val is not None else None,
            "atr":               round(atr, 8),
            "distance_percent":  round(distance * 100, 2),
            "sl":                sl,
            "tp":                tp,
            "rr":                rr,
            "risk_pct":          risk_pct,
            "reward_pct":        reward_pct,
            "regime":            regime,
            "divergence":        divergence,
            "pattern_confirmed": pattern_confirmed,
            "vol_ratio":         vol_ratio,
            "volume_spike":      vol_spike,
            "quality_mult":      quality_mult,
        })
    return signals
