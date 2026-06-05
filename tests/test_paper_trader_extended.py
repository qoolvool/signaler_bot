"""Extended coverage for paper_trader.py — gaps after first pass."""
import pytest
from unittest.mock import patch

from paper_trader import PaperPortfolio


def _portfolio(**kwargs):
    defaults = dict(
        initial_balance=1000.0,
        trade_size_percent=2.0,
        max_open_trades=0,
        pending_expiry_checks=10,
        leverage=1,
        commission_maker=0.0, commission_taker=0.0,
        breakeven_threshold=0.5,
        fixed_risk_mode=False,
        trailing_stop=False,
    )
    defaults.update(kwargs)
    with patch.object(PaperPortfolio, "_connect_gsheets", lambda self: None), \
         patch.object(PaperPortfolio, "_load",  lambda self: None), \
         patch.object(PaperPortfolio, "_save",  lambda self: None):
        return PaperPortfolio(**defaults)


def _open_trade(p, pair="BTC/USDT", direction="LONG",
                entry=100.0, sl=90.0, tp=120.0):
    """Helper: place a pending order and trigger it immediately."""
    p.create_pending_order(pair, direction, entry, sl=sl, tp=tp)
    triggered, _ = p.check_pending_orders(
        pair, candle_high=entry + 1, candle_low=entry - 1
    )
    return triggered[0]


# ── cancel_order ───────────────────────────────────────────────────────────────

class TestCancelOrder:
    def test_cancel_removes_order(self):
        p = _portfolio()
        o = p.create_pending_order("ETH/USDT", "LONG", 100.0, sl=95.0, tp=115.0)
        assert len(p.pending_orders) == 1
        p.cancel_order(o)
        assert len(p.pending_orders) == 0

    def test_cancel_increments_orders_cancelled(self):
        p = _portfolio()
        o = p.create_pending_order("ETH/USDT", "LONG", 100.0, sl=95.0, tp=115.0)
        before = p.orders_cancelled
        p.cancel_order(o)
        assert p.orders_cancelled == before + 1

    def test_cancel_unknown_id_is_noop(self):
        p = _portfolio()
        p.create_pending_order("ETH/USDT", "LONG", 100.0, sl=95.0, tp=115.0)
        fake_order = {"id": "XXXXXXXX", "direction": "LONG", "pair": "ETH/USDT"}
        p.cancel_order(fake_order)
        assert len(p.pending_orders) == 1  # original order still there


# ── save_report / get_report ───────────────────────────────────────────────────

class TestReports:
    def test_save_and_get_report(self):
        p = _portfolio()
        p.save_report("BTC/USDT", "some analysis text")
        r = p.get_report("BTC/USDT")
        assert r is not None
        assert r["pair"] == "BTC/USDT"
        assert r["text"] == "some analysis text"

    def test_get_report_missing_returns_none(self):
        p = _portfolio()
        assert p.get_report("XYZ/USDT") is None

    def test_overwrite_report(self):
        p = _portfolio()
        p.save_report("BTC/USDT", "first")
        p.save_report("BTC/USDT", "second")
        assert p.get_report("BTC/USDT")["text"] == "second"


# ── recent_trades sort order ───────────────────────────────────────────────────

class TestRecentTrades:
    def test_returns_most_recent_first(self):
        p = _portfolio(trade_size_percent=5.0)
        # Open two trades then close them with different timestamps
        _open_trade(p, "BTC/USDT", entry=100.0, sl=90.0, tp=120.0)
        _open_trade(p, "ETH/USDT", entry=50.0,  sl=45.0, tp=60.0)

        p.check_sl_tp("BTC/USDT", candle_high=125.0, candle_low=95.0)
        p.check_sl_tp("ETH/USDT", candle_high=62.0,  candle_low=48.0)

        recent = p.recent_trades(n=10)
        assert len(recent) == 2
        # closed_at timestamps sort lexicographically (ISO format), descending
        ts = [t["closed_at"] for t in recent]
        assert ts == sorted(ts, reverse=True)

    def test_respects_n_limit(self):
        pairs = [f"COIN{i}/USDT" for i in range(5)]
        p = _portfolio(trade_size_percent=2.0)
        for pair in pairs:
            _open_trade(p, pair, entry=100.0, sl=90.0, tp=120.0)
            p.check_sl_tp(pair, candle_high=125.0, candle_low=99.0)
        assert len(p.recent_trades(n=3)) == 3


# ── create_pending_order: sl == entry ─────────────────────────────────────────

class TestPendingOrderSlEqualsEntry:
    def test_sl_equals_entry_creates_order(self):
        # sl_pct=0 → fixed_risk_mode guard skips → falls back to trade_size_percent
        p = _portfolio(fixed_risk_mode=True, risk_per_trade_percent=1.0, leverage=1)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=100.0, tp=120.0)
        assert o is not None

    def test_sl_equals_entry_rr_is_none(self):
        p = _portfolio(fixed_risk_mode=False)
        o = p.create_pending_order("BTC/USDT", "LONG", 100.0, sl=100.0, tp=120.0)
        assert o["rr"] is None


# ── _close_trade with missing/zero entry ──────────────────────────────────────

class TestCloseTradeCorruptEntry:
    def test_zero_entry_closes_with_zero_pnl(self):
        p = _portfolio()
        trade = {
            "id": "CORRUPT1", "pair": "BTC/USDT", "direction": "LONG",
            "entry_price": 0.0, "sl": 85.0, "tp": 120.0,
            "size_usd": 50.0, "notional": 50.0, "leverage": 1,
            "risk_pct": 5.0, "reward_pct": 20.0, "rr": 4.0,
            "opened_at": "2024-01-01 00:00:00 UTC", "closed_at": None,
            "close_price": None, "close_reason": None,
            "pnl_usd": None, "pnl_percent": None, "commission": None,
            "sl_at_breakeven": False, "sl_at_half_tp": False, "trail_dist": 0.0, "peak_price": 0.0,
            "status": "OPEN",
        }
        p.trades.append(trade)
        closed = p._close_trade(trade, close_price=110.0, reason="TP")
        assert closed["status"] == "CLOSED"
        assert closed["pnl_usd"] == 0.0

    def test_none_entry_closes_with_zero_pnl(self):
        p = _portfolio()
        trade = {
            "id": "CORRUPT2", "pair": "ETH/USDT", "direction": "SHORT",
            "entry_price": None, "sl": 115.0, "tp": 80.0,
            "size_usd": 50.0, "notional": 50.0, "leverage": 1,
            "risk_pct": 5.0, "reward_pct": 20.0, "rr": 4.0,
            "opened_at": "2024-01-01 00:00:00 UTC", "closed_at": None,
            "close_price": None, "close_reason": None,
            "pnl_usd": None, "pnl_percent": None, "commission": None,
            "sl_at_breakeven": False, "sl_at_half_tp": False, "trail_dist": 0.0, "peak_price": None,
            "status": "OPEN",
        }
        p.trades.append(trade)
        closed = p._close_trade(trade, close_price=80.0, reason="TP")
        assert closed["pnl_usd"] == 0.0


# ── get_equity notional fallback ───────────────────────────────────────────────

class TestEquityNotionalFallback:
    def test_uses_size_usd_when_notional_absent(self):
        p = _portfolio(initial_balance=1000.0, commission_maker=0.0, commission_taker=0.0)
        # Manually inject an open trade without "notional" key
        trade = {
            "id": "T1", "pair": "BTC/USDT", "direction": "LONG",
            "entry_price": 100.0, "sl": 90.0, "tp": 120.0,
            "size_usd": 50.0, "leverage": 1,
            "status": "OPEN",
        }
        p.trades.append(trade)
        # Price +10% → unrealized = 50 * 0.1 = 5.0
        equity = p.get_equity({"BTC/USDT": 110.0})
        assert equity == pytest.approx(1005.0, rel=0.001)


# ── get_stats with None pnl_usd ───────────────────────────────────────────────

class TestStatsNonePnl:
    def test_none_pnl_counted_as_zero(self):
        p = _portfolio()
        # Inject a closed trade where pnl_usd is None (e.g. data gap)
        p.trades.append({
            "id": "T1", "pair": "BTC/USDT", "direction": "LONG",
            "entry_price": 100.0, "sl": 90.0, "tp": 120.0,
            "size_usd": 20.0, "notional": 20.0, "leverage": 1,
            "status": "CLOSED", "pnl_usd": None, "pnl_percent": None,
            "closed_at": "2024-01-01 00:00:00 UTC",
        })
        stats = p.get_stats({})
        assert stats["total_closed"] == 1
        assert stats["total_pnl"] == 0.0  # None treated as 0


# ── SHORT breakeven equality edge ─────────────────────────────────────────────

class TestShortBreakevenEquality:
    def test_sl_already_at_entry_marks_breakeven_but_not_in_moved(self):
        # entry=sl initially; breakeven threshold reached; sl_at_breakeven flips
        # but new_sl = min(entry, entry) == trade["sl"] → not added to moved
        p = _portfolio(breakeven_threshold=0.5, commission_maker=0.0, commission_taker=0.0)
        _open_trade(p, "BTC/USDT", "SHORT", entry=100.0, sl=100.0, tp=80.0)
        trade = p.open_trades[0]

        # target = entry - (entry - tp) * 0.5 = 100 - 10 = 90; candle_low = 88 <= 90
        moved = p.check_breakeven("BTC/USDT", candle_high=101.0, candle_low=88.0)
        assert trade["sl_at_breakeven"] is True
        assert trade not in moved  # sl value didn't change, so not "moved"


# ── LONG trailing equality edge ───────────────────────────────────────────────

class TestLongTrailingEquality:
    def test_sl_not_in_moved_when_new_sl_equals_current(self):
        # trail_dist is large enough that new_sl < current_sl → no update
        p = _portfolio(trailing_stop=True, trailing_mult=100.0, commission_maker=0.0, commission_taker=0.0)
        _open_trade(p, "BTC/USDT", "LONG", entry=100.0, sl=90.0, tp=130.0)
        trade = p.open_trades[0]
        # trail_dist = |100 - 90| * 100 = 1000 — so new_sl = peak - 1000 << current sl
        moved = p.check_trailing_stop("BTC/USDT", candle_high=101.0, candle_low=99.0)
        assert trade not in moved  # new_sl not > trade["sl"]
