"""Tests for paper_trader.py — PaperPortfolio lifecycle."""
import pytest
from unittest.mock import patch

from paper_trader import PaperPortfolio


def _portfolio(**kwargs):
    """Create an in-memory portfolio with file I/O stubbed out."""
    defaults = dict(
        initial_balance=1000.0,
        trade_size_percent=2.0,
        max_open_trades=0,
        pending_expiry_checks=10,
        leverage=1,
        commission_maker=0.0,
        commission_taker=0.0,
        breakeven_threshold=0.5,
        fixed_risk_mode=False,
        trailing_stop=False,
    )
    defaults.update(kwargs)
    with patch.object(PaperPortfolio, "_connect_gsheets", lambda self: None), \
         patch.object(PaperPortfolio, "_load",  lambda self: None), \
         patch.object(PaperPortfolio, "_save",  lambda self: None):
        return PaperPortfolio(**defaults)


# ── create_pending_order ───────────────────────────────────────────────────────

class TestCreatePendingOrder:
    def test_creates_order(self):
        p = _portfolio()
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert o is not None
        assert o["pair"] == "BTC/USDT"
        assert o["direction"] == "LONG"

    def test_rejects_duplicate(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        dup = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert dup is None
        assert len(p.pending_orders) == 1

    def test_allows_opposite_direction(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG",  100.0, sl=95.0, tp=120.0)
        o = p.create_pending_order("BTC/USDT", "SHORT", 100.0, sl=105.0, tp=80.0)
        assert o is not None
        assert len(p.pending_orders) == 2

    def test_respects_max_open_trades(self):
        p = _portfolio(max_open_trades=1)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        o2 = p.create_pending_order("ETH/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert o2 is None

    def test_zero_balance_blocks_order(self):
        p = _portfolio()
        p.balance = 0.0
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert o is None

    def test_increments_orders_created(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert p.orders_created == 1

    def test_size_usd_does_not_exceed_balance(self):
        p = _portfolio(initial_balance=100.0, trade_size_percent=200.0)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=95.0, tp=120.0)
        assert o["size_usd"] <= 100.0

    def test_fixed_risk_mode_sizing(self):
        # 1% risk, entry=100, sl=90 (10% away) → size = 1%*1000 / 10% = 100
        p = _portfolio(fixed_risk_mode=True, risk_per_trade_percent=1.0,
                       max_trade_size_percent=100.0)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=200.0)
        assert o["size_usd"] == pytest.approx(100.0, rel=0.01)

    def test_rr_calculated(self):
        p = _portfolio()
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        # risk=10, reward=20 → RR=2.0
        assert o["rr"] == pytest.approx(2.0, abs=0.1)

    def test_trailing_dist_set_when_enabled(self):
        p = _portfolio(trailing_stop=True, trailing_mult=1.5)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=130.0)
        assert o["trail_dist"] == pytest.approx(15.0, rel=0.01)  # (100-90)*1.5

    def test_trailing_dist_zero_when_disabled(self):
        p = _portfolio(trailing_stop=False)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=130.0)
        assert o["trail_dist"] == 0.0


# ── check_pending_orders ───────────────────────────────────────────────────────

class TestCheckPendingOrders:
    def test_triggers_long_on_low(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        triggered, _ = p.check_pending_orders("BTC/USDT", candle_high=105.0, candle_low=99.0)
        assert len(triggered) == 1
        assert triggered[0]["status"] == "OPEN"

    def test_triggers_short_on_high(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "SHORT", 100.0, sl=110.0, tp=80.0)
        triggered, _ = p.check_pending_orders("BTC/USDT", candle_high=100.5, candle_low=95.0)
        assert len(triggered) == 1

    def test_no_trigger_when_price_far(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        triggered, _ = p.check_pending_orders("BTC/USDT", candle_high=110.0, candle_low=102.0)
        assert triggered == []

    def test_cancels_expired_order(self):
        p = _portfolio(pending_expiry_checks=1)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        _, cancelled = p.check_pending_orders("BTC/USDT", candle_high=115.0, candle_low=108.0)
        assert len(cancelled) == 1
        assert p.orders_cancelled == 1
        assert p.pending_orders == []

    def test_bounce_confirm_blocks_close_below_entry(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        triggered, _ = p.check_pending_orders(
            "BTC/USDT", candle_high=101.0, candle_low=99.5,
            candle_close=99.0, require_bounce=True,
        )
        assert triggered == []

    def test_bounce_confirm_triggers_when_close_above_entry(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        triggered, _ = p.check_pending_orders(
            "BTC/USDT", candle_high=101.0, candle_low=99.5,
            candle_close=101.0, require_bounce=True,
        )
        assert len(triggered) == 1

    def test_orders_for_other_pairs_not_touched(self):
        p = _portfolio()
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p.create_pending_order("ETH/USDT", "LONG", 50.0,  sl=45.0, tp=60.0)
        triggered, _ = p.check_pending_orders("ETH/USDT", candle_high=55.0, candle_low=49.0)
        assert len(triggered) == 1
        assert triggered[0]["pair"] == "ETH/USDT"
        assert len(p.pending_orders) == 1  # BTC still pending


# ── check_breakeven ────────────────────────────────────────────────────────────

class TestCheckBreakeven:
    def _open_trade(self, p, direction, entry, sl, tp):
        p.create_pending_order("BTC/USDT", direction, entry, sl=sl, tp=tp)
        p.check_pending_orders("BTC/USDT",
                                candle_high=entry + 1,
                                candle_low=entry - 1)

    def test_long_moves_sl_to_entry(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        moved = p.check_breakeven("BTC/USDT", candle_high=111.0, candle_low=105.0)
        assert len(moved) == 1
        assert p.trades[0]["sl"] == pytest.approx(100.0)
        assert p.trades[0]["sl_at_breakeven"] is True

    def test_long_no_move_before_threshold(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        moved = p.check_breakeven("BTC/USDT", candle_high=108.0, candle_low=102.0)
        assert moved == []

    def test_short_moves_sl_to_entry(self):
        p = _portfolio()
        self._open_trade(p, "SHORT", entry=100.0, sl=110.0, tp=80.0)
        # 50% of TP distance = 90.0
        moved = p.check_breakeven("BTC/USDT", candle_high=95.0, candle_low=89.0)
        assert len(moved) == 1
        assert p.trades[0]["sl"] == pytest.approx(100.0)

    def test_no_double_move(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        p.check_breakeven("BTC/USDT", candle_high=112.0, candle_low=105.0)
        moved2 = p.check_breakeven("BTC/USDT", candle_high=112.0, candle_low=108.0)
        assert moved2 == []

    def test_long_phase2_triggers_at_75pct(self):
        p = _portfolio(trail_lock_trigger=0.75, trail_lock_target=0.50)
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        moved = p.check_breakeven("BTC/USDT", candle_high=116.0, candle_low=110.0)
        assert any(t.get("sl_at_half_tp") for t in p.trades)
        assert p.trades[0]["sl"] == pytest.approx(110.0)

    def test_long_phase2_no_trigger_before_threshold(self):
        p = _portfolio(trail_lock_trigger=0.75, trail_lock_target=0.50)
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        p.check_breakeven("BTC/USDT", candle_high=114.0, candle_low=108.0)
        assert not p.trades[0].get("sl_at_half_tp")


# ── check_trailing_stop ────────────────────────────────────────────────────────

class TestCheckTrailingStop:
    def _open_trailing_trade(self, p, direction, entry, sl, tp):
        p.create_pending_order("BTC/USDT", direction, entry, sl=sl, tp=tp)
        p.check_pending_orders("BTC/USDT",
                                candle_high=entry + 0.5,
                                candle_low=entry - 0.5)

    def test_long_sl_follows_price_up(self):
        p = _portfolio(trailing_stop=True, trailing_mult=1.0)
        self._open_trailing_trade(p, "LONG", entry=100.0, sl=90.0, tp=130.0)
        # trail_dist = (100-90)*1 = 10; new peak = 115 → new SL = 105
        moved = p.check_trailing_stop("BTC/USDT", candle_high=115.0, candle_low=110.0)
        assert len(moved) == 1
        assert p.trades[0]["sl"] == pytest.approx(105.0)

    def test_short_sl_follows_price_down(self):
        p = _portfolio(trailing_stop=True, trailing_mult=1.0)
        self._open_trailing_trade(p, "SHORT", entry=100.0, sl=110.0, tp=70.0)
        # trail_dist = (110-100)*1 = 10; new peak = 80 → new SL = 90
        moved = p.check_trailing_stop("BTC/USDT", candle_high=95.0, candle_low=80.0)
        assert len(moved) == 1
        assert p.trades[0]["sl"] == pytest.approx(90.0)

    def test_sl_never_moves_against_trade(self):
        p = _portfolio(trailing_stop=True, trailing_mult=1.0)
        self._open_trailing_trade(p, "LONG", entry=100.0, sl=90.0, tp=130.0)
        moved = p.check_trailing_stop("BTC/USDT", candle_high=98.0, candle_low=95.0)
        assert moved == []

    def test_no_trail_when_trailing_disabled(self):
        p = _portfolio(trailing_stop=False)
        self._open_trailing_trade(p, "LONG", entry=100.0, sl=90.0, tp=130.0)
        moved = p.check_trailing_stop("BTC/USDT", candle_high=120.0, candle_low=115.0)
        assert moved == []


# ── check_sl_tp ────────────────────────────────────────────────────────────────

class TestCheckSlTp:
    def _open_trade(self, p, direction, entry, sl, tp):
        p.create_pending_order("BTC/USDT", direction, entry, sl=sl, tp=tp)
        p.check_pending_orders("BTC/USDT",
                                candle_high=entry + 0.5,
                                candle_low=entry - 0.5)

    def test_long_tp_hit(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=121.0, candle_low=105.0)
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "TP"
        assert closed[0]["pnl_usd"] > 0

    def test_long_sl_hit(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=95.0, candle_low=88.0)
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "SL"
        assert closed[0]["pnl_usd"] < 0

    def test_short_tp_hit(self):
        p = _portfolio()
        self._open_trade(p, "SHORT", entry=100.0, sl=110.0, tp=80.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=95.0, candle_low=78.0)
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "TP"
        assert closed[0]["pnl_usd"] > 0

    def test_short_sl_hit(self):
        p = _portfolio()
        self._open_trade(p, "SHORT", entry=100.0, sl=110.0, tp=80.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=112.0, candle_low=105.0)
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "SL"
        assert closed[0]["pnl_usd"] < 0

    def test_no_close_when_price_mid(self):
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=110.0, candle_low=95.0)
        assert closed == []

    def test_tp_wins_over_sl_on_same_candle(self):
        """If both TP and SL touched, TP is checked first."""
        p = _portfolio()
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        closed = p.check_sl_tp("BTC/USDT", candle_high=121.0, candle_low=89.0)
        assert closed[0]["close_reason"] == "TP"

    def test_balance_updated_on_close(self):
        p = _portfolio(initial_balance=1000.0, trade_size_percent=10.0,
                       commission_maker=0.0, commission_taker=0.0)
        self._open_trade(p, "LONG", entry=100.0, sl=90.0, tp=120.0)
        initial_bal = p.balance
        p.check_sl_tp("BTC/USDT", candle_high=121.0, candle_low=105.0)
        assert p.balance != initial_bal


# ── PnL calculation ────────────────────────────────────────────────────────────

class TestPnlCalculation:
    def test_long_pnl_correct(self):
        """$100 notional LONG from 100 to 110 → +10% → +$10."""
        p = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                       trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.0)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        trade = p.open_trades[0]
        notional = trade["notional"]
        result = p._close_trade(trade, 110.0, "TP")
        assert result["pnl_usd"] == pytest.approx(notional * 0.1, rel=0.01)

    def test_short_pnl_correct(self):
        """SHORT from 100 to 90 → +10% of notional."""
        p = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                       trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.0)
        p.create_pending_order("BTC/USDT", "SHORT", 100.0, sl=110.0, tp=80.0)
        p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        trade    = p.open_trades[0]
        notional = trade["notional"]
        result   = p._close_trade(trade, 90.0, "TP")
        assert result["pnl_usd"] == pytest.approx(notional * 0.1, rel=0.01)

    def test_commission_deducted(self):
        # SL exit → taker rate charged; TP exit → maker rate (0.0) charged
        p = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                       trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.001)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        trade    = p.open_trades[0]
        notional = trade["notional"]
        # SL close: gross = notional * (90-100)/100, commission = taker only
        result   = p._close_trade(trade, 90.0, "SL")
        expected = notional * (-0.1) - notional * 0.001
        assert result["pnl_usd"] == pytest.approx(expected, rel=0.01)
        # TP close: maker rate = 0.0, so no commission
        p2 = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                        trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.001)
        p2.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p2.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        trade2   = p2.open_trades[0]
        notional2 = trade2["notional"]
        result2  = p2._close_trade(trade2, 110.0, "TP")
        assert result2["pnl_usd"] == pytest.approx(notional2 * 0.1, rel=0.01)

    def test_leverage_amplifies_pnl(self):
        pnls = []
        for lev in (1, 10):
            p = _portfolio(leverage=lev, fixed_risk_mode=False,
                           trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.0)
            p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
            p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
            t = p.open_trades[0]
            p._close_trade(t, 110.0, "TP")
            pnls.append(p.closed_trades[0]["pnl_usd"])
        assert pnls[1] == pytest.approx(pnls[0] * 10, rel=0.01)

    def test_sl_pnl_is_negative(self):
        p = _portfolio(fixed_risk_mode=False, trade_size_percent=10.0,
                       commission_maker=0.0, commission_taker=0.0)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        trade = p.open_trades[0]
        result = p._close_trade(trade, 90.0, "SL")
        assert result["pnl_usd"] < 0


# ── get_stats / get_equity ─────────────────────────────────────────────────────

class TestGetStats:
    def test_empty_stats(self):
        p     = _portfolio()
        stats = p.get_stats()
        assert stats["total_closed"] == 0
        assert stats["winrate"]      == 0.0
        assert stats["balance"]      == 1000.0

    def test_winrate_calculation(self):
        p = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                       trade_size_percent=5.0, commission_maker=0.0, commission_taker=0.0)

        def _trade(pair, direction, entry, sl, tp, h_close, l_close):
            p.create_pending_order(pair, direction, entry, sl=sl, tp=tp)
            p.check_pending_orders(pair, candle_high=entry+1, candle_low=entry-1)
            p.check_sl_tp(pair, candle_high=h_close, candle_low=l_close)

        _trade("BTC/USDT", "LONG", 100.0, 90.0, 120.0, 121.0, 105.0)  # win
        _trade("ETH/USDT", "LONG", 100.0, 90.0, 120.0, 121.0, 105.0)  # win
        _trade("SOL/USDT", "LONG", 100.0, 90.0, 120.0,  95.0,  88.0)  # loss

        stats = p.get_stats()
        assert stats["total_closed"] == 3
        assert stats["wins"]         == 2
        assert stats["losses"]       == 1
        assert stats["winrate"]      == pytest.approx(66.7, abs=0.1)

    def test_get_equity_includes_unrealized(self):
        p = _portfolio(initial_balance=1000.0, fixed_risk_mode=False,
                       trade_size_percent=10.0, commission_maker=0.0, commission_taker=0.0)
        p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=90.0, tp=120.0)
        p.check_pending_orders("BTC/USDT", candle_high=101, candle_low=99)
        notional = p.open_trades[0]["notional"]
        equity   = p.get_equity({"BTC/USDT": 110.0})
        assert equity == pytest.approx(1000.0 + notional * 0.1, rel=0.01)

    def test_equity_equals_balance_when_no_open_trades(self):
        p = _portfolio(initial_balance=500.0)
        assert p.get_equity({}) == 500.0
