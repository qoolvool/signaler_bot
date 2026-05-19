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
    SL_ATR_MULT, TP_ATR_MIN_MULT, MIN_RR, ADX_MIN,
    EXTREMA_WINDOW, TOLERANCE_PERCENT,
    MIN_TOUCHES, TOP_N_LEVELS, MIN_TOUCH_SPACING,
    LEVEL_AGE_MIN_CANDLES, VOLUME_TOUCH_MULTIPLIER, REQUIRE_RETEST,
    EMA_CONFLUENCE_PERIODS, FVG_LOOKBACK, FVG_MIN_GAP_PCT,
    HTF_TIMEFRAME, HTF_EMA_PERIOD, HTF_SR_CANDLES, HTF_STRUCTURE_WINDOW,
    REQUIRE_RSI_DIVERGENCE, REQUIRE_PATTERN, SIGNAL_VOLUME_MULT,
    CRASH_SL_BUFFER, PUMP_SL_BUFFER, HTF_SR_REQUIRE_CONFIRM,
)
from indicators import _calc_ema, _calc_atr

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
    avg_volume   = df["volume"].mean()
    touch_indices = []
    for i in range(len(df)):
        high, low = df["high"].iat[i], df["low"].iat[i]
        if not ((lower <= high <= upper) or (lower <= low <= upper)):
            continue
        if volume_multiplier > 0 and df["volume"].iat[i] < avg_volume * volume_multiplier:
            continue
        touch_indices.append(i)
    if not touch_indices:
        return 0
    if min_spacing <= 0:
        return len(touch_indices)
    filtered = [touch_indices[0]]
    for idx in touch_indices[1:]:
        if idx - filtered[-1] >= min_spacing:
            filtered.append(idx)
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
        if not any(
            kept["type"] == level["type"]
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
            if ema["type"] == lvl["type"] and abs(ema["price"] - p) / p <= tol:
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
    if df is None or len(df) < window * 4 + 1:
        return "RANGE"
    highs = _find_local_extrema(df["high"], window, "max")
    lows  = _find_local_extrema(df["low"],  window, "min")
    if len(highs) < 2 or len(lows) < 2:
        return "RANGE"
    h_prev, h_last = highs[-2][0], highs[-1][0]
    l_prev, l_last = lows[-2][0],  lows[-1][0]
    if h_last > h_prev and l_last > l_prev:
        return "BULLISH"
    if h_last < h_prev and l_last < l_prev:
        return "BEARISH"
    return "RANGE"


def _fetch_htf_confluence(client, pair: str):
    """Один запрос HTF_TIMEFRAME → (trend, ema, sr_levels, structure)."""
    try:
        needed = max(HTF_SR_CANDLES, HTF_EMA_PERIOD + 30)
        raw = client.fetch_ohlcv(pair, HTF_TIMEFRAME, limit=needed)
        if not raw or len(raw) < HTF_EMA_PERIOD:
            return None, None, [], "RANGE"
        df_htf = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_htf["timestamp"] = pd.to_datetime(df_htf["timestamp"], unit="ms")
        closes    = df_htf["close"].astype(float)
        ema       = float(closes.ewm(span=HTF_EMA_PERIOD, adjust=False).mean().iloc[-1])
        trend     = "UP" if float(closes.iloc[-1]) > ema else "DOWN"
        structure = _detect_htf_structure(df_htf)
        htf_sr: List[Dict] = []
        if HTF_SR_CANDLES > 0 and len(df_htf) >= EXTREMA_WINDOW * 2 + 1:
            htf_sr = find_support_resistance(df_htf)
        return trend, round(ema, 8), htf_sr, structure
    except Exception as exc:
        logger.warning("Ошибка HTF анализа %s: %s", pair, exc)
        return None, None, [], "RANGE"


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
        combined = pd.concat([r1, r2], axis=1).dropna()
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
    divergence: bool,
    special: bool,
) -> bool:
    """HTF-structure gate + RSI-divergence requirement (Layer 1)."""
    if not special and htf_structure is not None:
        if htf_structure == "RANGE":
            return False
        if htf_structure != ("BULLISH" if direction == "LONG" else "BEARISH"):
            return False
    if not special and REQUIRE_RSI_DIVERGENCE and not divergence:
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
) -> List[Dict]:
    if regime in ("CRASH", "PUMP"):
        return []

    if ema_period >= len(df):
        logger.warning("Недостаточно свечей для EMA%d (%d)", ema_period, len(df))
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

    if not special and adx_val is not None and adx_min > 0 and adx_val < adx_min:
        logger.info("ADX %.1f < %.1f — боковик, сигналы пропущены", adx_val, adx_min)
        return []

    signals = []
    for lvl in levels:
        distance = abs(current_price - lvl["price"]) / lvl["price"]
        if distance > proximity:
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

        divergence = (
            _detect_rsi_divergence(df, rsi_series, direction, entry)
            if rsi_series is not None else False
        )
        if not _layer1_passes(direction, htf_structure, divergence, special):
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
            sl_dist = atr * sl_atr_mult
            tp_min  = atr * tp_atr_min_mult
            if direction == "LONG":
                sl    = entry - sl_dist
                above = sorted(
                    [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > entry],
                    key=lambda x: x["price"],
                )
                if above:
                    nearest = above[0]["price"]
                    if nearest < entry + tp_min:
                        continue
                    tp = nearest
                else:
                    tp = entry + tp_min
            else:
                sl    = entry + sl_dist
                below = sorted(
                    [l for l in levels if l["type"] == "SUPPORT" and l["price"] < entry],
                    key=lambda x: x["price"], reverse=True,
                )
                if below:
                    nearest = below[0]["price"]
                    if nearest > entry - tp_min:
                        continue
                    tp = nearest
                else:
                    tp = entry - tp_min

        tp_dist    = abs(tp - entry)
        rr         = round(tp_dist / sl_dist, 1) if sl_dist > 0 else None
        if rr is None or rr < min_rr:
            continue

        risk_pct   = round(sl_dist / entry * 100, 2)
        reward_pct = round(tp_dist / entry * 100, 2)

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
        })
    return signals
