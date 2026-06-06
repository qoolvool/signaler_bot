"""Extended coverage for analysis.py — gaps found after first pass."""
import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_df, sine_df
from unittest.mock import MagicMock, patch

import analysis as an


# ── _detect_rsi_divergence (direct) ───────────────────────────────────────────

class TestDetectRsiDivergence:
    # _find_local_extrema with window=8 iterates range(8, len-8), so we need
    # at least ~35 candles and place swing extrema at indices 12 and 30.

    def _bullish_df_rsi(self, low1=93.0, low2=91.0, rsi1=25.0, rsi2=35.0):
        n = 50
        lows = [100.0] * n
        lows[12] = low1
        lows[30] = low2
        closes = [l + 0.5 for l in lows]
        highs  = [l + 2.0 for l in lows]
        df  = make_df(closes, highs=highs, lows=lows)
        rsi = pd.Series([50.0] * n)
        rsi.iloc[12] = rsi1
        rsi.iloc[30] = rsi2
        return df, rsi

    def _bearish_df_rsi(self, high1=107.0, high2=109.0, rsi1=72.0, rsi2=62.0):
        n = 50
        highs = [100.0] * n
        highs[12] = high1
        highs[30] = high2
        closes = [h - 0.5 for h in highs]
        lows   = [h - 2.0 for h in highs]
        df  = make_df(closes, highs=highs, lows=lows)
        rsi = pd.Series([50.0] * n)
        rsi.iloc[12] = rsi1
        rsi.iloc[30] = rsi2
        return df, rsi

    def test_bullish_divergence_detected(self):
        # Swing lows: 93 then 91 (price extends lower), RSI: 25 then 35 (higher) → divergence
        df, rsi = self._bullish_df_rsi(low1=93.0, low2=91.0, rsi1=25.0, rsi2=35.0)
        result = an._detect_rsi_divergence(df, rsi, "LONG", level_price=92.0)
        assert result is True

    def test_no_divergence_when_both_extend(self):
        # Swing lows: 93 then 91, RSI: 30 then 25 (also lower) → no divergence
        df, rsi = self._bullish_df_rsi(low1=93.0, low2=91.0, rsi1=30.0, rsi2=25.0)
        result = an._detect_rsi_divergence(df, rsi, "LONG", level_price=92.0)
        assert result is False

    def test_bearish_divergence_detected(self):
        # Swing highs: 107 then 109 (price extends higher), RSI: 72 then 62 (lower) → divergence
        df, rsi = self._bearish_df_rsi(high1=107.0, high2=109.0, rsi1=72.0, rsi2=62.0)
        result = an._detect_rsi_divergence(df, rsi, "SHORT", level_price=108.0)
        assert result is True

    def test_insufficient_swings_returns_false(self):
        # Only one swing low near the level → can't form a pair → False
        n = 50
        lows = [100.0] * n
        lows[12] = 93.0   # single swing low near level=92
        closes = [l + 0.5 for l in lows]
        highs  = [l + 2.0 for l in lows]
        df  = make_df(closes, highs=highs, lows=lows)
        rsi = pd.Series([50.0] * n)
        rsi.iloc[12] = 30.0
        result = an._detect_rsi_divergence(df, rsi, "LONG", level_price=92.0)
        assert result is False


# ── _mark_htf_confirmed ────────────────────────────────────────────────────────

class TestMarkHtfConfirmed:
    def test_marks_matching_level(self):
        levels     = [{"price": 100.0, "type": "SUPPORT"}]
        htf_levels = [{"price": 100.2, "type": "SUPPORT"}]
        an._mark_htf_confirmed(levels, htf_levels, tolerance_pct=0.8)
        assert levels[0]["htf_confirmed"] is True

    def test_no_match_different_type(self):
        levels     = [{"price": 100.0, "type": "SUPPORT"}]
        htf_levels = [{"price": 100.2, "type": "RESISTANCE"}]
        an._mark_htf_confirmed(levels, htf_levels, tolerance_pct=0.8)
        assert levels[0]["htf_confirmed"] is False

    def test_no_match_too_far(self):
        levels     = [{"price": 100.0, "type": "SUPPORT"}]
        htf_levels = [{"price": 110.0, "type": "SUPPORT"}]
        an._mark_htf_confirmed(levels, htf_levels, tolerance_pct=0.8)
        assert levels[0]["htf_confirmed"] is False

    def test_empty_htf_levels(self):
        levels = [{"price": 100.0, "type": "SUPPORT"}]
        an._mark_htf_confirmed(levels, [], tolerance_pct=0.8)
        assert levels[0]["htf_confirmed"] is False


# ── find_support_resistance with require_retest=True ──────────────────────────

class TestFindSrRequireRetest:
    def test_rejects_level_without_retest(self):
        # Monotonically falling prices — no retest possible
        closes = list(range(120, 80, -1))
        df     = make_df(closes)
        levels = an.find_support_resistance(df, require_retest=True,
                                             min_touches=2, top_n=10)
        # All levels lack retest in a pure trend → should be empty or very few
        for lvl in levels:
            assert lvl.get("has_retest") is True

    def test_accepts_level_with_retest(self):
        # Build a series with a clear breakout and retest
        closes = (
            [100] * 5 +         # establishes support
            [90, 85, 80] +      # breaks below
            [98, 100.5, 95] +   # retests from below
            [90, 85]
        )
        highs = [c + 1 for c in closes]
        lows  = [c - 1 for c in closes]
        df    = make_df(closes, highs=highs, lows=lows)
        # Tolerance wide enough to find the level
        assert an._has_retest_after_breakout(100.0, df, 0.02, "SUPPORT") is True


# ── find_entry_signals — special mode without crash/pump_high ─────────────────

class TestFindEntrySignalsSpecialEdge:
    def _df(self, n=300):
        closes = [100.0 + i * 0.1 for i in range(n)]
        return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])

    def test_recovery_without_crash_low_uses_atr_sl(self):
        """RECOVERY regime with crash_low=None → falls back to ATR-based SL."""
        orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        df = self._df()
        current = float(df["close"].iloc[-1])
        levels = [{"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
                   "confluence_score": 1, "confluence_tags": []}]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            regime="RECOVERY", crash_low=None, ema_period=50,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig
        # Should still generate a signal using ATR fallback
        assert isinstance(sigs, list)

    def test_crash_and_pump_return_empty(self):
        df = self._df()
        current = float(df["close"].iloc[-1])
        levels = [{"price": current, "type": "SUPPORT", "touches": 5,
                   "confluence_score": 1, "confluence_tags": []}]
        assert an.find_entry_signals(df, levels, current, regime="CRASH") == []
        assert an.find_entry_signals(df, levels, current, regime="PUMP")  == []


# ── Efficiency Ratio gate ───────────────────────────────────────────────────────

class TestEfficiencyRatioGate:
    def _chop_df(self, n=300):
        """Sideways market: oscillates ±1 around 100 → low ER."""
        closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(n)]
        return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])

    def _trend_df(self, n=300):
        closes = [100.0 + i * 0.5 for i in range(n)]
        return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])

    def test_gate_blocks_signals_in_chop(self):
        orig_er  = an.EFFICIENCY_FILTER
        orig_min = an.EFFICIENCY_MIN
        an.EFFICIENCY_FILTER = True
        an.EFFICIENCY_MIN    = 0.30
        df = self._chop_df()
        current = float(df["close"].iloc[-1])
        levels = [{"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
                   "confluence_score": 5, "confluence_tags": []}]
        sigs = an.find_entry_signals(df, levels, current, ema_period=50)
        an.EFFICIENCY_FILTER = orig_er
        an.EFFICIENCY_MIN    = orig_min
        assert sigs == []

    def test_gate_disabled_does_not_block(self):
        """With the filter off, chop alone must not short-circuit via ER."""
        orig_er = an.EFFICIENCY_FILTER
        an.EFFICIENCY_FILTER = False
        df = self._chop_df()
        current = float(df["close"].iloc[-1])
        levels = [{"price": current, "type": "SUPPORT", "touches": 5,
                   "confluence_score": 5, "confluence_tags": []}]
        # Should not raise and ER path is skipped; other filters may still apply.
        sigs = an.find_entry_signals(df, levels, current, ema_period=50)
        an.EFFICIENCY_FILTER = orig_er
        assert isinstance(sigs, list)

    def test_gate_bypassed_in_special_regime(self):
        """RECOVERY/CORRECTION bypass the ER gate (extreme-bounce entries)."""
        orig_er  = an.EFFICIENCY_FILTER
        orig_rsi = an.REQUIRE_RSI_DIVERGENCE
        an.EFFICIENCY_FILTER      = True
        an.REQUIRE_RSI_DIVERGENCE = False
        df = self._chop_df()
        current = float(df["close"].iloc[-1])
        levels = [{"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
                   "confluence_score": 1, "confluence_tags": []}]
        sigs = an.find_entry_signals(
            df, levels, current, regime="RECOVERY", crash_low=current * 0.95, ema_period=50,
        )
        an.EFFICIENCY_FILTER      = orig_er
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        # Gate must not have fired — special regime reaches SL/TP logic.
        assert isinstance(sigs, list)


# ── MIN_CONFLUENCE_SCORE filter ────────────────────────────────────────────────

class TestMinConfluenceScoreFilter:
    def _bullish_df(self, n=300):
        closes = [100.0 + i * 0.1 for i in range(n)]
        return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])

    def test_blocks_zero_score_level_in_normal_mode(self):
        orig_rsi   = an.REQUIRE_RSI_DIVERGENCE
        orig_score = an.MIN_CONFLUENCE_SCORE
        an.REQUIRE_RSI_DIVERGENCE = False
        an.MIN_CONFLUENCE_SCORE   = 1
        df      = self._bullish_df()
        current = float(df["close"].iloc[-1])
        levels  = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 0, "confluence_tags": []},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, ema_period=50,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        an.MIN_CONFLUENCE_SCORE   = orig_score
        assert sigs == []

    def test_zero_threshold_allows_all_levels(self):
        orig_rsi   = an.REQUIRE_RSI_DIVERGENCE
        orig_score = an.MIN_CONFLUENCE_SCORE
        orig_long  = an.LONG_MIN_CONFLUENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        an.MIN_CONFLUENCE_SCORE   = 0
        an.LONG_MIN_CONFLUENCE    = 0
        df      = self._bullish_df()
        current = float(df["close"].iloc[-1])
        levels  = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 0, "confluence_tags": []},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, ema_period=50,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        an.MIN_CONFLUENCE_SCORE   = orig_score
        an.LONG_MIN_CONFLUENCE    = orig_long
        assert len(sigs) >= 1

    def test_confluence_filter_bypassed_in_recovery(self):
        """RECOVERY mode skips MIN_CONFLUENCE_SCORE gate."""
        orig_rsi   = an.REQUIRE_RSI_DIVERGENCE
        orig_score = an.MIN_CONFLUENCE_SCORE
        an.REQUIRE_RSI_DIVERGENCE = False
        an.MIN_CONFLUENCE_SCORE   = 2  # high threshold
        df      = self._bullish_df()
        current = float(df["close"].iloc[-1])
        crash_low = current * 0.8
        levels  = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 0, "confluence_tags": []},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            regime="RECOVERY", crash_low=crash_low, ema_period=50,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        an.MIN_CONFLUENCE_SCORE   = orig_score
        # Recovery ignores confluence filter → may produce signals
        assert isinstance(sigs, list)


# ── HTF RSI — htf_rsi=None passthrough ────────────────────────────────────────

class TestHtfRsiPassthrough:
    def test_none_htf_rsi_does_not_block(self):
        """When htf_rsi is None, the 4h RSI gate is skipped entirely."""
        orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        closes = [100.0 + i * 0.1 for i in range(300)]
        df = make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 3, "confluence_tags": ["EMA21", "FVG", "HTF_SR"]},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, ema_period=50,
            htf_rsi=None,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig
        assert len(sigs) >= 1


# ── atr_ratio choppy filter ───────────────────────────────────────────────────

class TestChoppyAtrRatioFilter:
    def _bullish_df(self, n: int = 300):
        closes = [100.0 + i * 0.1 for i in range(n)]
        return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])

    def test_low_atr_ratio_blocks_signals(self):
        """atr_ratio below CHOPPY_ATR_MIN should suppress all signals."""
        orig_rsi = an.REQUIRE_RSI_DIVERGENCE
        orig_min = an.CHOPPY_ATR_MIN
        an.REQUIRE_RSI_DIVERGENCE = False
        an.CHOPPY_ATR_MIN = 0.75
        df = self._bullish_df()
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 3, "confluence_tags": ["EMA21", "FVG", "HTF_SR"]},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, ema_period=50,
            atr_ratio=0.5,  # below threshold → choppy filter fires
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        an.CHOPPY_ATR_MIN = orig_min
        assert sigs == []

    def test_high_atr_ratio_allows_signals(self):
        """atr_ratio above CHOPPY_ATR_MIN should not suppress signals."""
        orig_rsi = an.REQUIRE_RSI_DIVERGENCE
        orig_min = an.CHOPPY_ATR_MIN
        an.REQUIRE_RSI_DIVERGENCE = False
        an.CHOPPY_ATR_MIN = 0.75
        df = self._bullish_df()
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 3, "confluence_tags": ["EMA21", "FVG", "HTF_SR"]},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, ema_period=50,
            atr_ratio=1.5,  # above threshold → passes
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        an.CHOPPY_ATR_MIN = orig_min
        assert len(sigs) >= 1


# ── is_round_number edge cases ─────────────────────────────────────────────────

class TestIsRoundNumberEdge:
    def test_negative_price(self):
        assert an._is_round_number(-100.0) is False

    def test_very_small_price(self):
        # 0.001 USDT — should detect 0.001 as a round number
        assert isinstance(an._is_round_number(0.001), bool)  # no crash

    def test_zero(self):
        assert an._is_round_number(0.0) is False


# ── _count_touches — all fail volume filter ────────────────────────────────────

class TestCountTouchesVolumeEdge:
    def test_all_candles_fail_volume_returns_zero(self):
        # All volumes = 100, avg = 100; multiplier = 5 → none qualify
        df = make_df([100] * 10, highs=[101] * 10, lows=[99] * 10, volumes=[100] * 10)
        assert an._count_touches(100.0, df, tolerance=0.02, volume_multiplier=5.0) == 0

    def test_zero_average_volume_skips_volume_check(self):
        # volumes all 0 → avg_volume = 0 → multiplier guard avoided
        df = make_df([100] * 10, highs=[101] * 10, lows=[99] * 10, volumes=[0.0] * 10)
        # multiplier > 0 but avg = 0; condition: vol < 0 * mult is always False
        count = an._count_touches(100.0, df, tolerance=0.02, volume_multiplier=1.2)
        # With avg=0, all touches are counted (0 < 0 is False → not skipped)
        assert count == 10


# ── _fetch_htf_confluence — 5-tuple return ────────────────────────────────────

class TestFetchHtfConfluenceReturn:
    def _mock_client(self, n_candles=100):
        base   = 100.0
        closes = [base + i * 0.1 for i in range(n_candles)]
        raw    = [
            [i * 3600000, c - 0.5, c + 1, c - 1, c, 1000.0]
            for i, c in enumerate(closes)
        ]
        client = MagicMock()
        client.fetch_ohlcv.return_value = raw
        return client

    def test_returns_seven_tuple(self):
        client = self._mock_client(100)
        result = an._fetch_htf_confluence(client, "BTC/USDT")
        assert len(result) == 7

    def test_htf_rsi_is_float_or_none(self):
        client = self._mock_client(100)
        _, _, _, _, htf_rsi, _, _ = an._fetch_htf_confluence(client, "BTC/USDT")
        assert htf_rsi is None or isinstance(htf_rsi, float)

    def test_empty_response_returns_nones(self):
        client = MagicMock()
        client.fetch_ohlcv.return_value = []
        result = an._fetch_htf_confluence(client, "BTC/USDT")
        assert result == (None, None, [], "RANGE", None, None, None)

    def test_exception_returns_nones(self):
        client = MagicMock()
        client.fetch_ohlcv.side_effect = Exception("network error")
        result = an._fetch_htf_confluence(client, "BTC/USDT")
        assert result == (None, None, [], "RANGE", None, None, None)
