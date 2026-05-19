"""Extended coverage for indicators.py — gaps after first pass."""
import pandas as pd
import pytest
from conftest import make_df

import indicators as ind


# ── _calc_atr_ratio: hist_avg == 0 ────────────────────────────────────────────

class TestAtrRatioEdgeCases:
    def test_returns_one_when_hist_avg_is_zero(self):
        # Flat prices → TR = 0 on every candle → hist_avg = 0 → fallback 1.0
        n = 50
        df = make_df([100.0] * n, highs=[100.0] * n, lows=[100.0] * n)
        ratio = ind._calc_atr_ratio(df, atr_period=14, avg_period=20)
        assert ratio == 1.0

    def test_insufficient_data_returns_one(self):
        # Fewer candles than avg_period+1 → early return 1.0
        df = make_df([100.0] * 10)
        assert ind._calc_atr_ratio(df, atr_period=5, avg_period=20) == 1.0


# ── _find_crash_low / _find_pump_high: lookback > series ──────────────────────

class TestCrashLowPumpHighOversizedLookback:
    def test_crash_low_lookback_exceeds_length(self):
        lows = [90.0, 85.0, 80.0]
        df = make_df([100.0] * 3, lows=lows)
        # lookback=50 > len(df)=3 → iloc[-50:] returns entire series safely
        result = ind._find_crash_low(df, lookback=50)
        assert result == 80.0

    def test_pump_high_lookback_exceeds_length(self):
        highs = [110.0, 115.0, 120.0]
        df = make_df([100.0] * 3, highs=highs)
        result = ind._find_pump_high(df, lookback=50)
        assert result == 120.0


# ── _calc_rsi_series: all-loss series ─────────────────────────────────────────

class TestRsiAllLoss:
    def test_all_losses_rsi_is_zero(self):
        # Strictly falling prices → avg_g = 0, avg_l > 0 → rs = 0 → RSI = 0
        closes = list(range(100, 80, -1))
        df = make_df(closes)
        rsi = ind._calc_rsi_series(df)
        # RSI should approach 0 (may not be exactly 0 due to EWM warm-up)
        assert float(rsi.iloc[-1]) < 10.0


# ── _detect_regime: boundary values ───────────────────────────────────────────

class TestDetectRegimeBoundary:
    def _rsi(self, values):
        return pd.Series(values, dtype=float)

    def test_rsi_exactly_50_with_high_atr_is_pump(self):
        from config import CRASH_PAUSE_ATR_RATIO
        rsi = self._rsi([50.0] * 20)
        regime = ind._detect_regime(rsi, atr_ratio=CRASH_PAUSE_ATR_RATIO)
        assert regime == "PUMP"

    def test_rsi_below_50_with_high_atr_is_crash(self):
        from config import CRASH_PAUSE_ATR_RATIO
        rsi = self._rsi([49.9] * 20)
        regime = ind._detect_regime(rsi, atr_ratio=CRASH_PAUSE_ATR_RATIO)
        assert regime == "CRASH"

    def test_no_special_history_returns_normal(self):
        from config import CRASH_RESUME_ATR_RATIO
        # atr below resume threshold, no previous oversold/overbought
        rsi = self._rsi([55.0] * 20)
        regime = ind._detect_regime(rsi, atr_ratio=CRASH_RESUME_ATR_RATIO - 0.1)
        assert regime == "NORMAL"
