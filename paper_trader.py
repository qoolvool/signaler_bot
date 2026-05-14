"""
Paper trading simulation engine.
Tracks a virtual portfolio without placing real exchange orders.
Persists state to trades.json and portfolio.json.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("signaler.paper")

_BASE = Path(__file__).parent
TRADES_FILE = _BASE / "trades.json"
PORTFOLIO_FILE = _BASE / "portfolio.json"


class PaperPortfolio:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        trade_size_percent: float = 2.0,
        max_open_trades: int = 5,
    ):
        self.initial_balance = initial_balance
        self.trade_size_percent = trade_size_percent
        self.max_open_trades = max_open_trades
        self.balance: float = initial_balance
        self.trades: List[Dict] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if PORTFOLIO_FILE.exists():
            try:
                data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
                self.balance = data.get("balance", self.initial_balance)
                self.initial_balance = data.get("initial_balance", self.initial_balance)
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
                        "balance": self.balance,
                        "initial_balance": self.initial_balance,
                        "updated_at": _utcnow(),
                    },
                    indent=2,
                    ensure_ascii=False,
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

    # ── trade lifecycle ───────────────────────────────────────────────────────

    def open_trade(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> Optional[Dict]:
        """Open a paper trade. Returns the trade dict or None if rejected."""
        if self.has_open_trade(pair, direction):
            logger.info("Позиция %s %s уже открыта — пропускаем.", direction, pair)
            return None
        if len(self.open_trades) >= self.max_open_trades:
            logger.info("Лимит открытых сделок (%d) достигнут.", self.max_open_trades)
            return None

        size_usd = round(self.balance * self.trade_size_percent / 100.0, 2)
        if size_usd <= 0:
            return None

        risk_pct = round(abs(entry_price - sl) / entry_price * 100, 2)
        reward_pct = round(abs(tp - entry_price) / entry_price * 100, 2)
        rr = round(reward_pct / risk_pct, 1) if risk_pct > 0 else None

        trade: Dict = {
            "id": str(uuid.uuid4())[:8].upper(),
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "size_usd": size_usd,
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "rr": rr,
            "opened_at": _utcnow(),
            "closed_at": None,
            "close_price": None,
            "close_reason": None,
            "pnl_usd": None,
            "pnl_percent": None,
            "status": "OPEN",
        }
        self.trades.append(trade)
        self._save()
        logger.info(
            "Открыта #%s %s %s @ %.6f | SL %.6f | TP %.6f | $%.2f",
            trade["id"], direction, pair, entry_price, sl, tp, size_usd,
        )
        return trade

    def check_sl_tp(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> List[Dict]:
        """Check open trades for this pair against a candle high/low.
        Returns list of trades that were closed."""
        closed = []
        for trade in list(self.open_trades):
            if trade["pair"] != pair:
                continue
            close_price, reason = None, None
            if trade["direction"] == "LONG":
                if candle_low <= trade["sl"]:
                    close_price, reason = trade["sl"], "SL"
                elif candle_high >= trade["tp"]:
                    close_price, reason = trade["tp"], "TP"
            else:  # SHORT
                if candle_high >= trade["sl"]:
                    close_price, reason = trade["sl"], "SL"
                elif candle_low <= trade["tp"]:
                    close_price, reason = trade["tp"], "TP"
            if close_price is not None:
                closed.append(self._close_trade(trade, close_price, reason))
        return closed

    def _close_trade(self, trade: Dict, close_price: float, reason: str) -> Dict:
        if trade["direction"] == "LONG":
            pnl = (close_price - trade["entry_price"]) / trade["entry_price"] * trade["size_usd"]
        else:
            pnl = (trade["entry_price"] - close_price) / trade["entry_price"] * trade["size_usd"]
        pnl_pct = pnl / trade["size_usd"] * 100

        trade.update(
            status="CLOSED",
            closed_at=_utcnow(),
            close_price=close_price,
            close_reason=reason,
            pnl_usd=round(pnl, 2),
            pnl_percent=round(pnl_pct, 2),
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
        closed = self.closed_trades
        wins = [t for t in closed if (t["pnl_usd"] or 0) > 0]
        losses = [t for t in closed if (t["pnl_usd"] or 0) <= 0]
        total_pnl = sum(t["pnl_usd"] or 0 for t in closed)
        winrate = len(wins) / len(closed) * 100 if closed else 0.0
        bal_change = (self.balance - self.initial_balance) / self.initial_balance * 100
        best = max(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        worst = min(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "balance_change_pct": round(bal_change, 2),
            "open_count": len(self.open_trades),
            "total_closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(winrate, 1),
            "total_pnl": round(total_pnl, 2),
            "best": best,
            "worst": worst,
        }

    def recent_trades(self, n: int = 10) -> List[Dict]:
        return sorted(
            self.closed_trades,
            key=lambda x: x["closed_at"] or "",
            reverse=True,
        )[:n]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
