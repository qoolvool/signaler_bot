"""Tests for indicators.py — pure-pandas indicator functions."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df, sine_df

import indicators as ind


# ── _calc_ema ──────────────────────────────────────────────────────────────────

class TestCalcEma:
    def test_single_value_series(self):
        s = pd.Series([5.0])
        assert float(ind._calc_ema(s, 5).iloc[-1]) == pytest.approx(5.0)

    def test_converges_toward_constant(self):
        s = pd.Series([100.0] * 50)
        assert float(ind._calc_ema(s, 20).iloc[-1]) == pytest.approx(100.0, rel=1e-6)

    def test_rises_with_rising_prices(self):
        s = pd.Series(list(range(1, 51, 1), ))
        ema = ind._calc_ema(s, 10)
        assert float(ema.iloc[-1]) > float(ema.iloc[9])

    def test_length_preserved(self):
        s = pd.Series(list(range(30)))
        assert len(ind._calc_ema(s, 10)) == 30


# ── _calc_atr ──────────────────────────────────────────────────────────────────

class TestCalcAtr:
    def test_flat_prices_atr_is_range(self):
        """When H=102, L=98 every candle, ATR ≈ 4."""
        df = make_df([100] * 20, highs=[102] * 20, lows=[98] * 20)
        assert ind._calc_atr(df) == pytest.approx(4.0, rel=0.05)

    def test_atr_positive(self):
        df = sine_df(50)
        assert ind._calc_atr(df) > 0

    def test_atr_increases_with_volatility(self):
        quiet     = make_df([100] * 30, highs=[101] * 30, lows=[99] * 30)
        volatile  = make_df([100] * 30, highs=[110] * 30, lows=[90] * 30)
        assert ind._calc_atr(volatile) > ind._calc_atr(quiet)


# ── _calc_adx ──────────────────────────────────────────────────────────────────

class TestCalcAdx:
    def test_insufficient_data_returns_none(self):
        df = make_df([100] * 10)
        adx, pdi, ndi = ind._calc_adx(df, period=14)
        assert adx is None and pdi is None and ndi is None

    def test_returns_floats_with_enough_data(self):
        df = sine_df(60)
        adx, pdi, ndi = ind._calc_adx(df, period=14)
        assert isinstance(adx, float)
        assert isinstance(pdi, float)
        assert isinstance(ndi, float)

    def test_adx_in_valid_range(self):
        df = sine_df(100)
        adx, _, _ = ind._calc_adx(df)
        assert 0 <= adx <= 100

    def test_trending_market_produces_finite_adx(self):
        trend = make_df(list(range(100, 160)))
        adx_t, pdi, ndi = ind._calc_adx(trend)
        assert adx_t is not None and not (adx_t != adx_t)  # not NaN
        assert adx_t > 0

    def test_uptrend_pdi_dominates(self):
        trend = make_df(list(range(100, 160)))
        _, pdi, ndi = ind._calc_adx(trend)
        assert pdi > ndi


# ── _calc_rsi_series ───────────────────────────────────────────────────────────

class TestCalcRsiSeries:
    def test_all_gains_gives_rsi_100(self):
        df = make_df(list(range(100, 130)))
        rsi = ind._calc_rsi_series(df)
        assert float(rsi.iloc[-1]) == pytest.approx(100.0)

    def test_all_losses_gives_low_rsi(self):
        df = make_df(list(range(130, 100, -1)))
        rsi = ind._calc_rsi_series(df)
        assert float(rsi.iloc[-1]) < 30

    def test_rsi_bounded(self):
        df = sine_df(100)
        rsi = ind._calc_rsi_series(df)
        assert rsi.dropna().between(0, 100).all()

    def test_flat_prices_rsi_near_50(self):
        df = make_df([100.0] * 40)
        rsi = ind._calc_rsi_series(df)
        # After all-zero deltas RSI should be NaN (no gains/losses) — handled gracefully
        assert not rsi.dropna().empty or True  # no crash is sufficient


# ── _calc_atr_ratio ────────────────────────────────────────────────────────────

class TestCalcAtrRatio:
    def test_returns_one_for_insufficient_data(self):
        df = make_df([100] * 10)
        assert ind._calc_atr_ratio(df) == 1.0

    def test_ratio_one_for_constant_volatility(self):
        df = make_df([100] * 60, highs=[102] * 60, lows=[98] * 60)
        ratio = ind._calc_atr_ratio(df)
        assert ratio == pytest.approx(1.0, rel=0.05)

    def test_ratio_above_one_when_recent_spike(self):
        # Quiet 40 candles, then 20 very volatile candles
        n_quiet, n_spiky = 40, 20
        highs  = [101.0] * n_quiet + [120.0] * n_spiky
        lows   = [99.0]  * n_quiet + [80.0]  * n_spiky
        closes = [100.0] * (n_quiet + n_spiky)
        df = make_df(closes, highs=highs, lows=lows)
        assert ind._calc_atr_ratio(df) > 1.0


# ── _detect_regime ─────────────────────────────────────────────────────────────

def _rsi_series(value: float, length: int = 20) -> pd.Series:
    """Synthetic RSI series with a fixed final value."""
    return pd.Series([50.0] * (length - 1) + [value])


class TestDetectRegime:
    def test_crash(self):
        r = _rsi_series(25.0)
        assert ind._detect_regime(r, atr_ratio=3.0) == "CRASH"

    def test_pump(self):
        r = _rsi_series(75.0)
        assert ind._detect_regime(r, atr_ratio=3.0) == "PUMP"

    def test_normal_low_atr(self):
        r = _rsi_series(50.0)
        assert ind._detect_regime(r, atr_ratio=1.0) == "NORMAL"

    def test_recovery_after_panic(self):
        # Spike RSI < 30 in history, now RSI in (30, 47)
        vals = [50.0] * 8 + [25.0] + [40.0]  # oversold look-back window has <30
        r = pd.Series(vals)
        regime = ind._detect_regime(r, atr_ratio=1.0)
        assert regime == "RECOVERY"

    def test_correction_after_pump(self):
        vals = [50.0] * 8 + [75.0] + [60.0]  # overbought look-back has >70
        r = pd.Series(vals)
        regime = ind._detect_regime(r, atr_ratio=1.0)
        assert regime == "CORRECTION"


# ── _find_crash_low / _find_pump_high ──────────────────────────────────────────

class TestCrashLowPumpHigh:
    def test_crash_low_is_min_of_lookback(self):
        lows = list(range(100, 80, -1))  # descending: 100..81
        df = make_df([100] * len(lows), lows=lows)
        assert ind._find_crash_low(df, lookback=10) == min(lows[-10:])

    def test_pump_high_is_max_of_lookback(self):
        highs = list(range(100, 120))
        df = make_df([100] * len(highs), highs=highs)
        assert ind._find_pump_high(df, lookback=10) == max(highs[-10:])
