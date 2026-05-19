"""Технические индикаторы: чистые pandas-функции без I/O."""
import logging

import pandas as pd

from config import (
    CRASH_PAUSE_ATR_RATIO, CRASH_RESUME_ATR_RATIO,
    RSI_OVERSOLD, RSI_OVERSOLD_LOOKBACK, RSI_RECOVERY_MAX,
    RSI_OVERBOUGHT, RSI_OVERBOUGHT_LOOKBACK, RSI_CORRECTION_MIN,
)

logger = logging.getLogger("signaler")


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _calc_adx(df: pd.DataFrame, period: int = 14):
    """Wilder's ADX. Возвращает (adx, +DI, -DI). При нехватке данных — (None, None, None)."""
    if len(df) < period * 2:
        return None, None, None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    tr  = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up, dn = high - ph, pl - low
    pdm = ((up > dn) & (up > 0)).astype(float) * up
    ndm = ((dn > up) & (dn > 0)).astype(float) * dn
    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    pdm_s = pdm.ewm(alpha=alpha, adjust=False).mean()
    ndm_s = ndm.ewm(alpha=alpha, adjust=False).mean()
    pdi   = 100.0 * pdm_s / atr_s.replace(0, float("nan"))
    ndi   = 100.0 * ndm_s / atr_s.replace(0, float("nan"))
    denom = (pdi + ndi).replace(0, float("nan"))
    dx    = 100.0 * (pdi - ndi).abs() / denom
    adx   = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])


def _calc_rsi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_g = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_atr_ratio(df: pd.DataFrame, atr_period: int = 14, avg_period: int = 20) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(atr_period).mean().dropna()
    if len(atr_series) < avg_period + 1:
        return 1.0
    current  = float(atr_series.iloc[-1])
    hist_avg = float(atr_series.iloc[-(avg_period + 1):-1].mean())
    return round(current / hist_avg, 2) if hist_avg > 0 else 1.0


def _detect_regime(rsi_series: pd.Series, atr_ratio: float) -> str:
    """Возвращает 'CRASH'/'RECOVERY'/'PUMP'/'CORRECTION'/'NORMAL'."""
    curr_rsi = float(rsi_series.iloc[-1])
    if atr_ratio >= CRASH_PAUSE_ATR_RATIO:
        return "CRASH" if curr_rsi < 50 else "PUMP"
    if atr_ratio < CRASH_RESUME_ATR_RATIO:
        was_panic = bool((rsi_series.iloc[-RSI_OVERSOLD_LOOKBACK:]  < RSI_OVERSOLD).any())
        was_pump  = bool((rsi_series.iloc[-RSI_OVERBOUGHT_LOOKBACK:] > RSI_OVERBOUGHT).any())
        if was_panic and RSI_OVERSOLD < curr_rsi < RSI_RECOVERY_MAX:
            return "RECOVERY"
        if was_pump and curr_rsi > RSI_CORRECTION_MIN:
            return "CORRECTION"
    return "NORMAL"


def _find_crash_low(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["low"].iloc[-lookback:].min())


def _find_pump_high(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["high"].iloc[-lookback:].max())
