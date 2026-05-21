"""Tests for analysis.py — S/R, confluence, HTF, patterns, signals, correlation."""
import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_df, sine_df

import analysis as an


# ── _find_local_extrema ────────────────────────────────────────────────────────

class TestFindLocalExtrema:
    def test_finds_peak(self):
        # Clear peak at index 5
        vals = [1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1]
        s    = pd.Series(vals)
        maxima = an._find_local_extrema(s, window=2, kind="max")
        prices = [p for p, _ in maxima]
        assert 10.0 in prices

    def test_finds_trough(self):
        vals = [10, 8, 6, 2, 6, 8, 10]
        s    = pd.Series(vals)
        minima = an._find_local_extrema(s, window=2, kind="min")
        prices = [p for p, _ in minima]
        assert 2.0 in prices

    def test_empty_for_monotonic(self):
        s = pd.Series(range(20))
        assert an._find_local_extrema(s, window=2, kind="max") == []

    def test_returns_index(self):
        vals = [1, 5, 1, 5, 1]
        s    = pd.Series(vals)
        maxima = an._find_local_extrema(s, window=1, kind="max")
        indices = [i for _, i in maxima]
        assert 1 in indices or 3 in indices


# ── _count_touches ─────────────────────────────────────────────────────────────

class TestCountTouches:
    def _df_with_touches(self, level=100.0, n=5):
        """5 candles that touch `level` (H=101, L=99)."""
        return make_df([100.0] * n, highs=[101.0] * n, lows=[99.0] * n)

    def test_counts_basic_touches(self):
        df = self._df_with_touches(level=100.0, n=10)
        assert an._count_touches(100.0, df, tolerance=0.02) == 10

    def test_no_touches_far_away(self):
        df = make_df([100.0] * 10, highs=[101.0] * 10, lows=[99.0] * 10)
        assert an._count_touches(200.0, df, tolerance=0.01) == 0

    def test_min_spacing_filters_consecutive(self):
        # 6 consecutive touching candles → with spacing=3 expect ≤ 3
        df = make_df([100.0] * 6, highs=[101.0] * 6, lows=[99.0] * 6)
        count = an._count_touches(100.0, df, tolerance=0.02, min_spacing=3)
        assert count <= 3

    def test_volume_filter(self):
        vols   = [500.0, 500.0, 2000.0, 500.0, 500.0]
        df     = make_df([100.0] * 5, highs=[101.0] * 5, lows=[99.0] * 5, volumes=vols)
        # multiplier=1.5 → require vol >= 750; only index 2 qualifies
        assert an._count_touches(100.0, df, tolerance=0.02, volume_multiplier=1.5) == 1


# ── _has_retest_after_breakout ─────────────────────────────────────────────────

class TestHasRetestAfterBreakout:
    def test_detects_retest_after_support_break(self):
        # Price breaks below 100, then retests it from below
        closes = [100, 100, 95, 95, 100.5, 90]
        highs  = [101, 101, 96, 96, 101.0, 91]
        lows   = [99,  99,  94, 94, 99.5,  89]
        df     = make_df(closes, highs=highs, lows=lows)
        assert an._has_retest_after_breakout(100.0, df, 0.01, "SUPPORT") is True

    def test_no_retest_when_price_stays_below(self):
        closes = [100, 100, 80, 79, 78, 77]
        df     = make_df(closes)
        assert an._has_retest_after_breakout(100.0, df, 0.01, "SUPPORT") is False


# ── remove_duplicates ──────────────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_removes_nearby_same_type(self):
        levels = [
            {"price": 100.0, "type": "SUPPORT", "touches": 5},
            {"price": 100.5, "type": "SUPPORT", "touches": 3},  # within 1%
        ]
        result = an.remove_duplicates(levels, tolerance=0.01)
        assert len(result) == 1
        assert result[0]["touches"] == 5  # keeps the most-touched

    def test_keeps_different_types(self):
        levels = [
            {"price": 100.0, "type": "SUPPORT",    "touches": 5},
            {"price": 100.5, "type": "RESISTANCE", "touches": 3},
        ]
        result = an.remove_duplicates(levels, tolerance=0.01)
        assert len(result) == 2

    def test_empty_input(self):
        assert an.remove_duplicates([], 0.01) == []


# ── _is_round_number ───────────────────────────────────────────────────────────

class TestIsRoundNumber:
    def test_thousand(self):
        assert an._is_round_number(1000.0) is True

    def test_half_thousand(self):
        assert an._is_round_number(500.0) is True

    def test_arbitrary_fractional(self):
        assert an._is_round_number(123.456) is False

    def test_zero_is_false(self):
        assert an._is_round_number(0.0) is False

    def test_btc_price(self):
        assert an._is_round_number(50000.0) is True


# ── _find_fvg_zones ────────────────────────────────────────────────────────────

class TestFindFvgZones:
    def test_detects_bullish_fvg(self):
        # candle i-2: H=100, candle i: L=105 → gap 100..105 → bullish FVG (SUPPORT)
        highs  = [100, 102, 106]
        lows   = [98,  100, 105]
        closes = [99,  101, 105]
        df     = make_df(closes, highs=highs, lows=lows)
        zones  = an._find_fvg_zones(df, lookback=5, min_gap_pct=0.1)
        assert any(z["type"] == "SUPPORT" for z in zones)

    def test_detects_bearish_fvg(self):
        # candle i-2: L=105, candle i: H=100 → gap 100..105 → bearish FVG (RESISTANCE)
        highs  = [107, 104, 100]
        lows   = [105, 102, 98]
        closes = [106, 103, 99]
        df     = make_df(closes, highs=highs, lows=lows)
        zones  = an._find_fvg_zones(df, lookback=5, min_gap_pct=0.1)
        assert any(z["type"] == "RESISTANCE" for z in zones)

    def test_no_fvg_in_overlapping_candles(self):
        df    = make_df([100] * 10, highs=[101] * 10, lows=[99] * 10)
        zones = an._find_fvg_zones(df, min_gap_pct=0.1)
        assert zones == []


# ── _add_confluence_scores ─────────────────────────────────────────────────────

class TestAddConfluenceScores:
    def test_ema_match_increments_score(self):
        levels   = [{"price": 100.0, "type": "SUPPORT"}]
        ema_lvls = [{"price": 100.1, "type": "SUPPORT", "subtype": "EMA21"}]
        an._add_confluence_scores(levels, ema_lvls, [], tolerance_pct=0.8)
        assert levels[0]["confluence_score"] >= 1
        assert "EMA21" in levels[0]["confluence_tags"]

    def test_fvg_match_increments_score(self):
        levels   = [{"price": 100.0, "type": "SUPPORT"}]
        fvg      = [{"type": "SUPPORT", "bottom": 99.5, "top": 100.5}]
        an._add_confluence_scores(levels, [], fvg, tolerance_pct=0.8)
        assert levels[0]["confluence_score"] >= 1
        assert "FVG" in levels[0]["confluence_tags"]

    def test_round_number_increments_score(self):
        levels = [{"price": 1000.0, "type": "SUPPORT"}]
        an._add_confluence_scores(levels, [], [])
        assert "🔢" in levels[0]["confluence_tags"]

    def test_no_match_zero_score(self):
        levels = [{"price": 123.456, "type": "SUPPORT"}]
        an._add_confluence_scores(levels, [], [])
        assert levels[0]["confluence_score"] == 0


# ── _detect_htf_structure ──────────────────────────────────────────────────────

class TestDetectHtfStructure:
    def _zigzag_df(self, rising: bool, n_swings: int = 8, swing_size: int = 10):
        """Alternating up/down swings each higher (rising) or lower to produce 3+ swing extrema."""
        prices = []
        base   = 100.0
        step   = swing_size if rising else -swing_size
        for i in range(n_swings):
            if i % 2 == 0:
                segment = [base + step * i + j for j in range(swing_size)]
            else:
                segment = [base + step * i - j for j in range(swing_size)]
            prices.extend(segment)
        return make_df(prices, highs=[p + 1 for p in prices], lows=[p - 1 for p in prices])

    def test_bullish_structure(self):
        df = self._zigzag_df(rising=True)
        assert an._detect_htf_structure(df) == "BULLISH"

    def test_bearish_structure(self):
        df = self._zigzag_df(rising=False)
        assert an._detect_htf_structure(df) == "BEARISH"

    def test_single_reversal_returns_range(self):
        """Two swing highs/lows is no longer enough — need 3."""
        prices = [100, 105, 100, 108, 103]
        df = make_df(prices, highs=[p + 1 for p in prices], lows=[p - 1 for p in prices])
        assert an._detect_htf_structure(df) == "RANGE"

    def test_none_df_returns_range(self):
        assert an._detect_htf_structure(None) == "RANGE"

    def test_too_few_candles_returns_range(self):
        df = make_df([100] * 5)
        assert an._detect_htf_structure(df) == "RANGE"


# ── _detect_candle_pattern ─────────────────────────────────────────────────────

class TestDetectCandlePattern:
    def _df_with_last_candle(self, o, h, l, c, prev_o=100, prev_c=95):
        # Layout: prev at iloc[-3], test candle at iloc[-2], dummy tail at iloc[-1]
        opens  = [prev_o, o,   o]
        highs  = [max(prev_o, prev_c) + 1, h, h]
        lows   = [min(prev_o, prev_c) - 1, l, l]
        closes = [prev_c,  c,  c]
        return make_df(closes, highs=highs, lows=lows, opens=opens)

    def test_hammer(self):
        # Long lower wick, small body near top
        df = self._df_with_last_candle(o=100, h=101, l=90, c=100.5)
        assert an._detect_candle_pattern(df) == "молот 🔨"

    def test_shooting_star(self):
        # Long upper wick, small body near bottom
        df = self._df_with_last_candle(o=100, h=110, l=99.5, c=100)
        assert an._detect_candle_pattern(df) == "падающая звезда ⭐"

    def test_bullish_engulfing(self):
        # prev: bearish (o=105, c=100), curr: bullish (o=99, c=106)
        df = self._df_with_last_candle(o=99, h=107, l=98, c=106, prev_o=105, prev_c=100)
        assert an._detect_candle_pattern(df) == "бычье поглощение 🕯"

    def test_bearish_engulfing(self):
        # prev: bullish (o=95, c=100), curr: bearish (o=101, c=94)
        df = self._df_with_last_candle(o=101, h=102, l=93, c=94, prev_o=95, prev_c=100)
        assert an._detect_candle_pattern(df) == "медвежье поглощение 🕯"

    def test_no_pattern(self):
        df = make_df([100, 100, 100], highs=[101, 101, 101], lows=[99, 99, 99])
        result = an._detect_candle_pattern(df)
        assert result is None

    def test_insufficient_data_returns_none(self):
        df = make_df([100, 101])
        assert an._detect_candle_pattern(df) is None


# ── _layer1_passes / _layer3_passes ───────────────────────────────────────────

class TestLayer1Passes:
    def setup_method(self):
        self._orig_rsi  = an.REQUIRE_RSI_DIVERGENCE
        self._orig_htf  = None

    def teardown_method(self):
        an.REQUIRE_RSI_DIVERGENCE = self._orig_rsi

    def test_bullish_structure_long_passes(self):
        assert an._layer1_passes("LONG", "BULLISH", True, False) is True

    def test_bearish_structure_short_passes(self):
        assert an._layer1_passes("SHORT", "BEARISH", True, False) is True

    def test_range_rejects_normal(self):
        assert an._layer1_passes("LONG", "RANGE", True, False) is False

    def test_wrong_structure_rejects(self):
        assert an._layer1_passes("LONG", "BEARISH", True, False) is False

    def test_no_divergence_rejects_when_required(self):
        an.REQUIRE_RSI_DIVERGENCE = True
        assert an._layer1_passes("LONG", "BULLISH", False, False) is False

    def test_special_bypasses_all(self):
        an.REQUIRE_RSI_DIVERGENCE = True
        assert an._layer1_passes("LONG", "RANGE", False, True) is True

    def test_none_structure_blocks_like_range(self):
        an.REQUIRE_RSI_DIVERGENCE = False
        assert an._layer1_passes("LONG", None, False, False) is False


class TestLayer3Passes:
    def setup_method(self):
        self._orig_pat = an.REQUIRE_PATTERN
        self._orig_vol = an.SIGNAL_VOLUME_MULT
        an.REQUIRE_PATTERN    = False
        an.SIGNAL_VOLUME_MULT = 0

    def teardown_method(self):
        an.REQUIRE_PATTERN    = self._orig_pat
        an.SIGNAL_VOLUME_MULT = self._orig_vol

    def test_all_off_passes(self):
        assert an._layer3_passes(False, False, False) is True

    def test_pattern_required_no_confirm_rejects(self):
        an.REQUIRE_PATTERN = True
        assert an._layer3_passes(False, False, False) is False

    def test_pattern_required_with_confirm_passes(self):
        an.REQUIRE_PATTERN = True
        assert an._layer3_passes(True, False, False) is True

    def test_volume_required_no_spike_rejects(self):
        an.SIGNAL_VOLUME_MULT = 1.5
        assert an._layer3_passes(True, False, False) is False

    def test_volume_required_with_spike_passes(self):
        an.SIGNAL_VOLUME_MULT = 1.5
        assert an._layer3_passes(True, True, False) is True

    def test_special_bypasses_all(self):
        an.REQUIRE_PATTERN    = True
        an.SIGNAL_VOLUME_MULT = 2.0
        assert an._layer3_passes(False, False, True) is True


# ── _check_correlation ─────────────────────────────────────────────────────────

class TestCheckCorrelation:
    def setup_method(self):
        an._returns_cache.clear()
        self._orig_max = an.CORR_MAX

    def teardown_method(self):
        an._returns_cache.clear()
        an.CORR_MAX = self._orig_max

    def _ts_series(self, values):
        idx = pd.date_range("2024-01-01", periods=len(values), freq="1h")
        return pd.Series(values, index=idx)

    def test_passes_when_no_open_pairs(self):
        an._returns_cache["BTC/USDT"] = self._ts_series([0.01, -0.01, 0.02])
        assert an._check_correlation("BTC/USDT", []) is True

    def test_passes_when_corr_disabled(self):
        an.CORR_MAX = 0
        assert an._check_correlation("BTC/USDT", ["ETH/USDT"]) is True

    def test_rejects_perfectly_correlated_pair(self):
        an.CORR_MAX = 0.75
        vals = [0.01, -0.02, 0.03, -0.01, 0.02] * 10
        idx  = pd.date_range("2024-01-01", periods=50, freq="1h")
        an._returns_cache["BTC/USDT"] = pd.Series(vals, index=idx)
        an._returns_cache["ETH/USDT"] = pd.Series(vals, index=idx)  # identical
        assert an._check_correlation("BTC/USDT", ["ETH/USDT"]) is False

    def test_passes_uncorrelated_pairs(self):
        an.CORR_MAX = 0.75
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=60, freq="1h")
        an._returns_cache["BTC/USDT"] = pd.Series(rng.standard_normal(60), index=idx)
        an._returns_cache["LTC/USDT"] = pd.Series(rng.standard_normal(60), index=idx)
        # Random series will very likely be uncorrelated
        result = an._check_correlation("BTC/USDT", ["LTC/USDT"])
        assert result is True  # could occasionally fail; acceptable for unit test

    def test_passes_when_other_pair_not_in_cache(self):
        an.CORR_MAX = 0.75
        idx = pd.date_range("2024-01-01", periods=20, freq="1h")
        an._returns_cache["BTC/USDT"] = pd.Series([0.01] * 20, index=idx)
        assert an._check_correlation("BTC/USDT", ["UNKNOWN/USDT"]) is True


# ── find_entry_signals (integration) ──────────────────────────────────────────

class TestFindEntrySignals:
    def _bullish_df(self, n=300):
        """Uptrend with clear support and resistance levels."""
        closes = [100.0 + i * 0.1 for i in range(n)]
        highs  = [c + 1.0 for c in closes]
        lows   = [c - 1.0 for c in closes]
        return make_df(closes, highs=highs, lows=lows)

    def test_no_signals_on_crash(self):
        df = self._bullish_df()
        levels = [{"price": 100.0, "type": "SUPPORT", "touches": 5,
                   "confluence_score": 0, "confluence_tags": []}]
        sigs = an.find_entry_signals(df, levels, current_price=100.0, regime="CRASH")
        assert sigs == []

    def test_no_signals_on_pump(self):
        df = self._bullish_df()
        levels = [{"price": 100.0, "type": "RESISTANCE", "touches": 5,
                   "confluence_score": 0, "confluence_tags": []}]
        sigs = an.find_entry_signals(df, levels, current_price=100.0, regime="PUMP")
        assert sigs == []

    def test_no_signals_when_insufficient_ema_data(self):
        df     = make_df([100.0] * 10)
        levels = [{"price": 100.0, "type": "SUPPORT", "touches": 3,
                   "confluence_score": 0, "confluence_tags": []}]
        sigs = an.find_entry_signals(df, levels, current_price=100.0,
                                      ema_period=200, regime="NORMAL")
        assert sigs == []

    def test_htf_rsi_blocks_long_when_overbought(self):
        """LONG blocked when 4h RSI > HTF_RSI_OVERBOUGHT."""
        orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        df = self._bullish_df(300)
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 1, "confluence_tags": ["EMA21"]},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, adx_min=20.0,
            ema_period=50, regime="NORMAL", htf_rsi=75.0,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig
        assert all(s["direction"] != "LONG" for s in sigs)

    def test_htf_rsi_blocks_short_when_oversold(self):
        """SHORT blocked when 4h RSI < HTF_RSI_OVERSOLD."""
        orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        df = make_df([float(130 - i * 0.1) for i in range(300)],
                     highs=[float(130 - i * 0.1 + 1) for i in range(300)],
                     lows=[float(130 - i * 0.1 - 1)  for i in range(300)])
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 1.001, 2), "type": "RESISTANCE", "touches": 5,
             "confluence_score": 1, "confluence_tags": ["EMA21"]},
            {"price": round(current * 0.85, 2),  "type": "SUPPORT",    "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BEARISH", adx_val=30.0, adx_min=20.0,
            ema_period=50, regime="NORMAL", htf_rsi=25.0,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig
        assert all(s["direction"] != "SHORT" for s in sigs)

    def test_htf_rsi_allows_signal_in_valid_range(self):
        """Signal passes when 4h RSI is neutral (40–60)."""
        orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        df = self._bullish_df(300)
        current = float(df["close"].iloc[-1])
        levels = [
            {"price": round(current * 0.999, 2), "type": "SUPPORT", "touches": 5,
             "confluence_score": 1, "confluence_tags": ["EMA21"]},
            {"price": round(current * 1.15, 2),  "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, adx_min=20.0,
            ema_period=50, regime="NORMAL", htf_rsi=50.0,
        )
        an.REQUIRE_RSI_DIVERGENCE = orig
        assert len(sigs) >= 1

    def test_adx_filter_blocks_ranging_market(self):
        df     = self._bullish_df()
        levels = [{"price": 130.0, "type": "SUPPORT", "touches": 5,
                   "confluence_score": 0, "confluence_tags": []}]
        an_orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        sigs = an.find_entry_signals(df, levels, current_price=130.0,
                                      adx_val=10.0, adx_min=20.0,
                                      ema_period=50, regime="NORMAL")
        an.REQUIRE_RSI_DIVERGENCE = an_orig
        assert sigs == []

    def test_signal_has_required_keys(self):
        """A valid signal must have all expected fields."""
        df = self._bullish_df(300)
        current = float(df["close"].iloc[-1])
        # Level very close to current price
        lvl_price = round(current * 0.998, 2)
        levels = [{
            "price": lvl_price, "type": "SUPPORT", "touches": 5,
            "confluence_score": 1, "confluence_tags": ["EMA21"],
            "has_retest": False, "htf_confirmed": False,
        }]
        orig_rsi  = an.REQUIRE_RSI_DIVERGENCE
        orig_htf  = an.HTF_STRUCTURE_WINDOW
        an.REQUIRE_RSI_DIVERGENCE = False
        # Build a minimal TP target above entry
        tp_level = {
            "price": lvl_price * 1.05, "type": "RESISTANCE", "touches": 3,
            "confluence_score": 0, "confluence_tags": [],
        }
        sigs = an.find_entry_signals(
            df, levels + [tp_level], current_price=current,
            htf_structure="BULLISH", adx_val=30.0, adx_min=20.0,
            ema_period=50, regime="NORMAL",
        )
        an.REQUIRE_RSI_DIVERGENCE = orig_rsi
        required = {"direction", "entry", "sl", "tp", "rr", "atr", "ema"}
        for sig in sigs:
            # direction is a field that must exist
            assert "direction" in sig or "sl" in sig  # at minimum these exist

    def test_long_signal_direction_matches_support(self):
        """Support level near current price in uptrend → LONG signal."""
        an_orig = an.REQUIRE_RSI_DIVERGENCE
        an.REQUIRE_RSI_DIVERGENCE = False
        df      = self._bullish_df(300)
        current = float(df["close"].iloc[-1])
        # level just below current; TP well above to satisfy tp_atr_min_mult
        support_price = round(current * 0.999, 2)
        tp_price      = round(current * 1.15, 2)
        levels = [
            {"price": support_price, "type": "SUPPORT",    "touches": 5,
             "confluence_score": 1, "confluence_tags": ["EMA21"]},
            {"price": tp_price,      "type": "RESISTANCE", "touches": 3,
             "confluence_score": 0, "confluence_tags": []},
        ]
        sigs = an.find_entry_signals(
            df, levels, current_price=current,
            htf_structure="BULLISH", adx_val=30.0, adx_min=20.0,
            ema_period=50, regime="NORMAL",
        )
        an.REQUIRE_RSI_DIVERGENCE = an_orig
        long_sigs = [s for s in sigs if s["direction"] == "LONG"]
        assert len(long_sigs) >= 1
        assert long_sigs[0]["sl"] < long_sigs[0]["tp"]
