"""
Paper trading simulation engine.
Pending (limit) orders wait for price to touch the entry level before opening.
State persists in trades.json and portfolio.json.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("signaler.paper")

_BASE = Path(os.getenv("DATA_DIR", str(Path(__file__).parent)))
TRADES_FILE    = _BASE / "trades.json"
PORTFOLIO_FILE = _BASE / "portfolio.json"


class PaperPortfolio:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        trade_size_percent: float = 2.0,
        max_open_trades: int = 5,
        pending_expiry_checks: int = 8,
        leverage: int = 1,
    ):
        self.initial_balance       = initial_balance
        self.trade_size_percent    = trade_size_percent
        self.max_open_trades       = max_open_trades
        self.pending_expiry_checks = pending_expiry_checks
        self.leverage              = leverage
        self.balance: float        = initial_balance
        self.trades: List[Dict]    = []
        self.pending_orders: List[Dict] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if PORTFOLIO_FILE.exists():
            try:
                data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
                self.balance        = data.get("balance",         self.initial_balance)
                self.initial_balance = data.get("initial_balance", self.initial_balance)
                self.pending_orders  = data.get("pending_orders",  [])
            except Exception as exc:
                logger.error("Ошибка чтения portfolio.json: %s", exc)

        if TRADES_FILE.exists():
            try:
                self.trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Ошибка чтения trades.json: %s", exc)

    def _save(self) -> None:
        try:
            PORTFOLIO_FILE.write_text(
                json.dumps(
                    {
                        "balance":          self.balance,
                        "initial_balance":  self.initial_balance,
                        "pending_orders":   self.pending_orders,
                        "updated_at":       _utcnow(),
                    },
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            TRADES_FILE.write_text(
                json.dumps(self.trades, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Ошибка сохранения данных: %s", exc)

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> List[Dict]:
        return [t for t in self.trades if t["status"] == "OPEN"]

    @property
    def closed_trades(self) -> List[Dict]:
        return [t for t in self.trades if t["status"] == "CLOSED"]

    def has_open_trade(self, pair: str, direction: str) -> bool:
        return any(
            t["pair"] == pair and t["direction"] == direction
            for t in self.open_trades
        )

    def has_pending_order(self, pair: str, direction: str) -> bool:
        return any(
            o["pair"] == pair and o["direction"] == direction
            for o in self.pending_orders
        )

    # ── pending (limit) orders ────────────────────────────────────────────────

    def create_pending_order(
        self,
        pair: str,
        direction: str,
        entry_price: float,  # exact level price
        sl: float,
        tp: float,
    ) -> Optional[Dict]:
        """Place a limit order at entry_price. Activates when price touches the level."""
        if self.has_pending_order(pair, direction):
            logger.info("Ордер %s %s уже ожидает.", direction, pair)
            return None
        if self.has_open_trade(pair, direction):
            logger.info("Позиция %s %s уже открыта.", direction, pair)
            return None
        active = len(self.open_trades) + len(self.pending_orders)
        if active >= self.max_open_trades:
            logger.info("Лимит позиций (%d) достигнут.", self.max_open_trades)
            return None

        size_usd   = round(self.balance * self.trade_size_percent / 100.0, 2)
        notional   = round(size_usd * self.leverage, 2)
        risk_pct   = round(abs(entry_price - sl) / entry_price * 100, 2)
        reward_pct = round(abs(tp - entry_price) / entry_price * 100, 2)
        rr         = round(reward_pct / risk_pct, 1) if risk_pct > 0 else None

        order: Dict = {
            "id":               str(uuid.uuid4())[:8].upper(),
            "pair":             pair,
            "direction":        direction,
            "entry_price":      entry_price,
            "sl":               sl,
            "tp":               tp,
            "size_usd":         size_usd,
            "notional":         notional,
            "leverage":         self.leverage,
            "risk_pct":         risk_pct,
            "reward_pct":       reward_pct,
            "rr":               rr,
            "created_at":       _utcnow(),
            "checks_remaining": self.pending_expiry_checks,
        }
        self.pending_orders.append(order)
        self._save()
        logger.info(
            "Ордер #%s %s %s @ %.6f | SL %.6f | TP %.6f | $%.2f",
            order["id"], direction, pair, entry_price, sl, tp, size_usd,
        )
        return order

    def check_pending_orders(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Check if any pending order for this pair was triggered by the candle.
        LONG triggered when candle_low  <= entry_price  (limit buy filled)
        SHORT triggered when candle_high >= entry_price  (limit sell filled)
        Returns (triggered_trades, cancelled_orders).
        """
        triggered: List[Dict] = []
        cancelled: List[Dict] = []
        remaining: List[Dict] = []

        for order in self.pending_orders:
            if order["pair"] != pair:
                remaining.append(order)
                continue

            order["checks_remaining"] -= 1

            hit = (
                order["direction"] == "LONG"  and candle_low  <= order["entry_price"] or
                order["direction"] == "SHORT" and candle_high >= order["entry_price"]
            )

            if hit:
                trade = self._make_trade(order)
                self.trades.append(trade)
                triggered.append(trade)
                logger.info(
                    "Ордер #%s исполнен: %s %s @ %.6f",
                    trade["id"], trade["direction"], trade["pair"], trade["entry_price"],
                )
            elif order["checks_remaining"] <= 0:
                cancelled.append(order)
                logger.info(
                    "Ордер #%s отменён (истёк): %s %s @ %.6f",
                    order["id"], order["direction"], order["pair"], order["entry_price"],
                )
            else:
                remaining.append(order)

        self.pending_orders = remaining
        if triggered or cancelled:
            self._save()

        return triggered, cancelled

    def _make_trade(self, order: Dict) -> Dict:
        return {
            "id":           order["id"],
            "pair":         order["pair"],
            "direction":    order["direction"],
            "entry_price":  order["entry_price"],
            "sl":           order["sl"],
            "tp":           order["tp"],
            "size_usd":     order["size_usd"],
            "notional":     order["notional"],
            "leverage":     order["leverage"],
            "risk_pct":     order["risk_pct"],
            "reward_pct":   order["reward_pct"],
            "rr":           order["rr"],
            "opened_at":    _utcnow(),
            "closed_at":    None,
            "close_price":  None,
            "close_reason": None,
            "pnl_usd":      None,
            "pnl_percent":  None,
            "status":       "OPEN",
        }

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def check_sl_tp(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> List[Dict]:
        """Check open trades for this pair against candle high/low. Returns closed trades."""
        closed = []
        for trade in list(self.open_trades):
            if trade["pair"] != pair:
                continue
            close_price, reason = None, None
            if trade["direction"] == "LONG":
                if candle_low  <= trade["sl"]: close_price, reason = trade["sl"], "SL"
                elif candle_high >= trade["tp"]: close_price, reason = trade["tp"], "TP"
            else:
                if candle_high >= trade["sl"]: close_price, reason = trade["sl"], "SL"
                elif candle_low  <= trade["tp"]: close_price, reason = trade["tp"], "TP"
            if close_price is not None:
                closed.append(self._close_trade(trade, close_price, reason))
        return closed

    def _close_trade(self, trade: Dict, close_price: float, reason: str) -> Dict:
        leverage = trade.get("leverage", 1)
        notional = trade.get("notional", trade["size_usd"])
        if trade["direction"] == "LONG":
            pnl = (close_price - trade["entry_price"]) / trade["entry_price"] * notional
        else:
            pnl = (trade["entry_price"] - close_price) / trade["entry_price"] * notional
        pnl_pct = pnl / trade["size_usd"] * 100  # % от маржи
        trade.update(
            status="CLOSED", closed_at=_utcnow(), close_price=close_price,
            close_reason=reason, pnl_usd=round(pnl, 2), pnl_percent=round(pnl_pct, 2),
        )
        self.balance = round(self.balance + pnl, 2)
        self._save()
        logger.info(
            "Закрыта #%s %s %s: %s @ %.6f | PnL $%.2f (%.2f%%)",
            trade["id"], trade["direction"], trade["pair"],
            reason, close_price, pnl, pnl_pct,
        )
        return trade

    # ── statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        closed   = self.closed_trades
        wins     = [t for t in closed if (t["pnl_usd"] or 0) > 0]
        losses   = [t for t in closed if (t["pnl_usd"] or 0) <= 0]
        total_pnl = sum(t["pnl_usd"] or 0 for t in closed)
        winrate  = len(wins) / len(closed) * 100 if closed else 0.0
        bal_chg  = (self.balance - self.initial_balance) / self.initial_balance * 100
        best  = max(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        worst = min(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        return {
            "balance":           self.balance,
            "initial_balance":   self.initial_balance,
            "balance_change_pct": round(bal_chg, 2),
            "open_count":        len(self.open_trades),
            "pending_count":     len(self.pending_orders),
            "total_closed":      len(closed),
            "wins":              len(wins),
            "losses":            len(losses),
            "winrate":           round(winrate, 1),
            "total_pnl":         round(total_pnl, 2),
            "best":              best,
            "worst":             worst,
        }

    def recent_trades(self, n: int = 10) -> List[Dict]:
        return sorted(
            self.closed_trades,
            key=lambda x: x["closed_at"] or "",
            reverse=True,
        )[:n]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
