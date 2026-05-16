"""
Paper trading simulation engine.
Storage: Google Sheets (primary) or JSON files (fallback).

Google Sheets setup:
  1. Google Cloud Console → Service Accounts → create → download JSON key
  2. Share the spreadsheet with the service account e-mail (Editor)
  3. Set env vars:
       GOOGLE_SPREADSHEET_ID=<id from spreadsheet URL>
       GOOGLE_CREDENTIALS_JSON=<full content of JSON key>   (recommended)
       OR
       GOOGLE_CREDENTIALS_FILE=<path to JSON key file>
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("signaler.paper")

_BASE          = Path(os.getenv("DATA_DIR", str(Path(__file__).parent)))
TRADES_FILE    = _BASE / "trades.json"
PORTFOLIO_FILE = _BASE / "portfolio.json"
REPORTS_FILE   = _BASE / "reports.json"

_TRADE_HEADERS = [
    "id", "pair", "direction", "entry_price", "sl", "tp",
    "size_usd", "notional", "leverage", "risk_pct", "reward_pct", "rr",
    "opened_at", "closed_at", "close_price", "close_reason",
    "pnl_usd", "pnl_percent", "commission", "sl_at_breakeven",
    "trail_dist", "peak_price", "status",
]

try:
    import gspread
    from google.oauth2.service_account import Credentials as _SACredentials
    _GSPREAD_OK = True
except ImportError:
    _GSPREAD_OK = False


def _cell(v) -> str:
    return "" if v is None else str(v)


class PaperPortfolio:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        trade_size_percent: float = 2.0,
        max_open_trades: int = 5,
        pending_expiry_checks: int = 8,
        leverage: int = 1,
        commission_rate: float = 0.001,
        breakeven_threshold: float = 0.5,
        fixed_risk_mode: bool = True,
        risk_per_trade_percent: float = 1.0,
        max_trade_size_percent: float = 20.0,
        trailing_stop: bool = False,
        trailing_mult: float = 1.0,
    ):
        self.initial_balance        = initial_balance
        self.trade_size_percent     = trade_size_percent
        self.max_open_trades        = max_open_trades
        self.pending_expiry_checks  = pending_expiry_checks
        self.leverage               = leverage
        self.commission_rate        = commission_rate
        self.breakeven_threshold    = breakeven_threshold
        self.fixed_risk_mode        = fixed_risk_mode
        self.risk_per_trade_percent = risk_per_trade_percent
        self.max_trade_size_percent = max_trade_size_percent
        self.trailing_stop          = trailing_stop
        self.trailing_mult          = trailing_mult

        self.balance: float             = initial_balance
        self.trades: List[Dict]         = []
        self.pending_orders: List[Dict] = []
        self.orders_created: int        = 0
        self.orders_cancelled: int      = 0

        self._ws_portfolio = None
        self._ws_trades    = None
        self._ws_reports   = None
        self._trades_dirty = False  # True = trades sheet needs rewrite

        self.reports: Dict[str, Dict]   = {}
        self._live_prices: Dict[str, float] = {}

        self._connect_gsheets()
        self._load()

    # ── Google Sheets connection ──────────────────────────────────────────────

    def _connect_gsheets(self) -> None:
        spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID", "")
        if not spreadsheet_id:
            return
        if not _GSPREAD_OK:
            logger.warning(
                "gspread не установлен — используются файлы. "
                "Запусти: pip install gspread google-auth"
            )
            return
        creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
        creds_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "")
        if not creds_json and not creds_file:
            logger.warning(
                "GOOGLE_SPREADSHEET_ID задан, но ключи не найдены. "
                "Укажи GOOGLE_CREDENTIALS_JSON или GOOGLE_CREDENTIALS_FILE."
            )
            return
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = (
                _SACredentials.from_service_account_info(
                    json.loads(creds_json), scopes=scopes
                )
                if creds_json
                else _SACredentials.from_service_account_file(
                    creds_file, scopes=scopes
                )
            )
            gc = gspread.authorize(creds)
            ss = gc.open_by_key(spreadsheet_id)
            self._ws_portfolio = self._init_sheet(
                ss, "Portfolio",
                ["balance", "initial_balance", "orders_created",
                 "orders_cancelled", "pending_orders_json", "updated_at"],
            )
            self._ws_trades  = self._init_sheet(ss, "Trades",  _TRADE_HEADERS)
            self._ws_reports = self._init_sheet(ss, "Reports", ["pair", "text", "saved_at"])
            logger.info("Google Sheets подключён ✓  id=%s", spreadsheet_id)
        except Exception as exc:
            logger.error("Google Sheets недоступен, использую файлы: %s", exc)

    def _init_sheet(self, ss, title: str, headers: List[str]):
        try:
            ws = ss.worksheet(title)
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(title=title, rows="1000", cols=str(len(headers) + 2))
        if not ws.row_values(1):
            ws.update("A1", [headers])
        return ws

    @property
    def _use_gsheets(self) -> bool:
        return self._ws_portfolio is not None

    # ── load / save ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._use_gsheets:
            self._load_gsheets()
        else:
            self._load_files()

    def _save(self) -> None:
        if self._use_gsheets:
            self._save_gsheets()
        else:
            self._save_files()

    # ── Google Sheets ─────────────────────────────────────────────────────────

    def _load_gsheets(self) -> None:
        try:
            rows = self._ws_portfolio.get_all_records()
            if rows:
                r = rows[0]
                self.balance          = float(r.get("balance",          self.initial_balance))
                self.initial_balance  = float(r.get("initial_balance",  self.initial_balance))
                self.orders_created   = int(float(r.get("orders_created",  0)))
                self.orders_cancelled = int(float(r.get("orders_cancelled", 0)))
                raw = r.get("pending_orders_json") or "[]"
                self.pending_orders   = json.loads(raw)

            trade_rows = self._ws_trades.get_all_records()
            # If trades sheet is empty but a JSON backup exists, recover from it
            if not trade_rows and TRADES_FILE.exists():
                try:
                    self.trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
                    logger.warning(
                        "Trades sheet пуст — восстановлено %d сделок из JSON-резерва.",
                        len(self.trades),
                    )
                    self._trades_dirty = True  # schedule rewrite to GSheets
                    trade_rows = []  # skip the parsing loop below
                except Exception as exc:
                    logger.error("Ошибка чтения резервного trades.json: %s", exc)
                    trade_rows = []
            self.trades = [] if trade_rows else self.trades
            for r in trade_rows:
                t = {k: (None if v == "" else v) for k, v in r.items()}
                for f in ("entry_price", "sl", "tp", "size_usd", "notional",
                          "risk_pct", "reward_pct", "rr", "pnl_usd", "pnl_percent",
                          "trail_dist", "peak_price"):
                    if t.get(f) is not None:
                        try:
                            t[f] = float(t[f])
                        except (ValueError, TypeError):
                            t[f] = None
                if t.get("leverage") is not None:
                    try:
                        t["leverage"] = int(float(t["leverage"]))
                    except (ValueError, TypeError):
                        pass
                # Boolean field: GSheets хранит как строку "True"/"False"
                raw_be = t.get("sl_at_breakeven")
                t["sl_at_breakeven"] = str(raw_be).lower() in ("true", "1") if raw_be else False
                self.trades.append(t)

            logger.info(
                "Загружено из Google Sheets: баланс $%.2f, сделок %d, ордеров %d",
                self.balance, len(self.trades), len(self.pending_orders),
            )
        except Exception as exc:
            logger.error("Ошибка загрузки из Google Sheets: %s", exc)

    def _save_gsheets(self) -> None:
        try:
            # Portfolio: update single data row — 1 API call
            self._ws_portfolio.update("A2", [[
                self.balance,
                self.initial_balance,
                self.orders_created,
                self.orders_cancelled,
                json.dumps(self.pending_orders, ensure_ascii=False),
                _utcnow(),
            ]])
            # Trades: full rewrite only when something changed
            if self._trades_dirty:
                # JSON backup first — protects data if GSheets write is interrupted
                try:
                    TRADES_FILE.write_text(
                        json.dumps(self.trades, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                rows = [_TRADE_HEADERS] + [
                    [_cell(t.get(h)) for h in _TRADE_HEADERS] for t in self.trades
                ]
                n = len(rows)
                # Write data first (no empty-sheet window), then trim stale trailing rows
                self._ws_trades.update("A1", rows)
                if self._ws_trades.row_count > n:
                    self._ws_trades.resize(rows=n)
                self._trades_dirty = False
        except Exception as exc:
            logger.error("Ошибка сохранения в Google Sheets: %s", exc)

    # ── JSON files ────────────────────────────────────────────────────────────

    def _load_files(self) -> None:
        if PORTFOLIO_FILE.exists():
            try:
                data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
                self.balance          = data.get("balance",          self.initial_balance)
                self.initial_balance  = data.get("initial_balance",  self.initial_balance)
                self.pending_orders   = data.get("pending_orders",   [])
                self.orders_created   = data.get("orders_created",   0)
                self.orders_cancelled = data.get("orders_cancelled", 0)
            except Exception as exc:
                logger.error("Ошибка чтения portfolio.json: %s", exc)
        if TRADES_FILE.exists():
            try:
                self.trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Ошибка чтения trades.json: %s", exc)
        if REPORTS_FILE.exists():
            try:
                self.reports = json.loads(REPORTS_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Ошибка чтения reports.json: %s", exc)

    def _save_files(self) -> None:
        try:
            PORTFOLIO_FILE.write_text(
                json.dumps({
                    "balance":          self.balance,
                    "initial_balance":  self.initial_balance,
                    "pending_orders":   self.pending_orders,
                    "orders_created":   self.orders_created,
                    "orders_cancelled": self.orders_cancelled,
                    "updated_at":       _utcnow(),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            TRADES_FILE.write_text(
                json.dumps(self.trades, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if self.reports:
                REPORTS_FILE.write_text(
                    json.dumps(self.reports, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.error("Ошибка сохранения данных: %s", exc)

    # ── live prices / equity ──────────────────────────────────────────────────

    def update_price(self, pair: str, price: float) -> None:
        self._live_prices[pair] = price

    def get_equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        p = prices if prices is not None else self._live_prices
        unrealized = 0.0
        for trade in self.open_trades:
            cur = p.get(trade["pair"])
            if cur is None:
                continue
            notional = trade.get("notional", trade["size_usd"])
            if trade["direction"] == "LONG":
                upnl = (cur - trade["entry_price"]) / trade["entry_price"] * notional
            else:
                upnl = (trade["entry_price"] - cur) / trade["entry_price"] * notional
            unrealized += upnl
        return round(self.balance + unrealized, 2)

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
        entry_price: float,
        sl: float,
        tp: float,
    ) -> Optional[Dict]:
        if self.balance <= 0:
            logger.warning("Баланс <= 0 ($%.2f) — создание ордеров остановлено.", self.balance)
            return None
        if self.has_pending_order(pair, direction):
            logger.info("Ордер %s %s уже ожидает.", direction, pair)
            return None
        if self.has_open_trade(pair, direction):
            logger.info("Позиция %s %s уже открыта.", direction, pair)
            return None
        if self.max_open_trades > 0 and len(self.open_trades) + len(self.pending_orders) >= self.max_open_trades:
            logger.info("Лимит позиций (%d) достигнут.", self.max_open_trades)
            return None

        sl_pct = abs(entry_price - sl) / entry_price
        if self.fixed_risk_mode and sl_pct > 0 and self.leverage > 0:
            risk_amount = self.balance * self.risk_per_trade_percent / 100.0
            size_usd = risk_amount / (sl_pct * self.leverage)
            size_usd = min(size_usd, self.balance * self.max_trade_size_percent / 100.0)
        else:
            size_usd = self.balance * self.trade_size_percent / 100.0
        size_usd = round(min(size_usd, self.balance), 2)
        notional   = round(size_usd * self.leverage, 2)
        risk_pct   = round(sl_pct * 100, 2)
        reward_pct = round(abs(tp - entry_price) / entry_price * 100, 2)
        rr         = round(reward_pct / risk_pct, 1) if risk_pct > 0 else None

        trail_dist = (
            round(abs(entry_price - sl) * self.trailing_mult, 8)
            if self.trailing_stop else 0.0
        )
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
            "trail_dist":       trail_dist,
            "created_at":       _utcnow(),
            "checks_remaining": self.pending_expiry_checks,
        }
        self.pending_orders.append(order)
        self.orders_created += 1
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
        candle_close: Optional[float] = None,
        require_bounce: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        triggered: List[Dict] = []
        cancelled: List[Dict] = []
        remaining: List[Dict] = []

        for order in self.pending_orders:
            if order["pair"] != pair:
                remaining.append(order)
                continue

            order["checks_remaining"] -= 1
            hit = (
                (order["direction"] == "LONG"  and candle_low  <= order["entry_price"]) or
                (order["direction"] == "SHORT" and candle_high >= order["entry_price"])
            )

            if hit and require_bounce and candle_close is not None:
                if order["direction"] == "LONG" and candle_close <= order["entry_price"]:
                    hit = False  # коснулась уровня, но закрытие ниже — отскок не подтверждён
                elif order["direction"] == "SHORT" and candle_close >= order["entry_price"]:
                    hit = False  # коснулась уровня, но закрытие выше — отскок не подтверждён

            if hit:
                trade = self._make_trade(order)
                self.trades.append(trade)
                self._trades_dirty = True
                triggered.append(trade)
                logger.info(
                    "Ордер #%s исполнен: %s %s @ %.6f | баланс $%.2f",
                    trade["id"], trade["direction"], trade["pair"],
                    trade["entry_price"], self.balance,
                )
            elif order["checks_remaining"] <= 0:
                cancelled.append(order)
                self.orders_cancelled += 1
                logger.info(
                    "Ордер #%s отменён (истёк): %s %s",
                    order["id"], order["direction"], order["pair"],
                )
            else:
                remaining.append(order)

        self.pending_orders = remaining
        if triggered or cancelled:
            self._save()
        return triggered, cancelled

    def _make_trade(self, order: Dict) -> Dict:
        return {
            "id":              order["id"],
            "pair":            order["pair"],
            "direction":       order["direction"],
            "entry_price":     order["entry_price"],
            "sl":              order["sl"],
            "tp":              order["tp"],
            "size_usd":        order["size_usd"],
            "notional":        order["notional"],
            "leverage":        order["leverage"],
            "risk_pct":        order["risk_pct"],
            "reward_pct":      order["reward_pct"],
            "rr":              order["rr"],
            "opened_at":       _utcnow(),
            "closed_at":       None,
            "close_price":     None,
            "close_reason":    None,
            "pnl_usd":         None,
            "pnl_percent":     None,
            "commission":      None,
            "sl_at_breakeven": False,
            "trail_dist":      order.get("trail_dist", 0.0),
            "peak_price":      order["entry_price"],
            "status":          "OPEN",
        }

    def cancel_order(self, order: Dict) -> None:
        self.pending_orders = [o for o in self.pending_orders if o["id"] != order["id"]]
        self.orders_cancelled += 1
        self._save()
        logger.info("Ордер #%s отменён: %s %s", order["id"], order["direction"], order["pair"])

    # ── Breakeven ────────────────────────────────────────────────────────────

    def check_breakeven(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> List[Dict]:
        """Переносит SL в безубыток когда цена прошла breakeven_threshold пути к TP."""
        moved = []
        for trade in self.open_trades:
            if trade["pair"] != pair or trade.get("sl_at_breakeven"):
                continue
            entry = trade["entry_price"]
            tp    = trade["tp"]
            if trade["direction"] == "LONG":
                target = entry + (tp - entry) * self.breakeven_threshold
                if candle_high >= target:
                    trade["sl"]              = entry
                    trade["sl_at_breakeven"] = True
                    self._trades_dirty       = True
                    moved.append(trade)
                    logger.info(
                        "BE #%s %s %s: SL перенесён → %.6f",
                        trade["id"], trade["direction"], trade["pair"], entry,
                    )
            else:
                target = entry - (entry - tp) * self.breakeven_threshold
                if candle_low <= target:
                    trade["sl"]              = entry
                    trade["sl_at_breakeven"] = True
                    self._trades_dirty       = True
                    moved.append(trade)
                    logger.info(
                        "BE #%s %s %s: SL перенесён → %.6f",
                        trade["id"], trade["direction"], trade["pair"], entry,
                    )
        if moved:
            self._save()
        return moved

    # ── Trailing stop ─────────────────────────────────────────────────────────

    def check_trailing_stop(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> List[Dict]:
        """Поджимает SL вслед за пиковой ценой. SL никогда не опускается ниже текущего."""
        moved = []
        for trade in self.open_trades:
            if trade["pair"] != pair:
                continue
            trail = trade.get("trail_dist") or 0.0
            if not trail:
                continue
            entry    = trade["entry_price"]
            peak     = trade.get("peak_price") or entry
            if trade["direction"] == "LONG":
                new_peak = max(peak, candle_high)
                new_sl   = new_peak - trail
                if new_sl > trade["sl"]:
                    trade["peak_price"] = new_peak
                    trade["sl"]         = round(new_sl, 8)
                    self._trades_dirty  = True
                    moved.append(trade)
                    logger.info(
                        "Trail #%s %s %s: peak=%.6f → SL=%.6f",
                        trade["id"], trade["direction"], trade["pair"],
                        new_peak, trade["sl"],
                    )
            else:
                new_peak = min(peak, candle_low)
                new_sl   = new_peak + trail
                if new_sl < trade["sl"]:
                    trade["peak_price"] = new_peak
                    trade["sl"]         = round(new_sl, 8)
                    self._trades_dirty  = True
                    moved.append(trade)
                    logger.info(
                        "Trail #%s %s %s: peak=%.6f → SL=%.6f",
                        trade["id"], trade["direction"], trade["pair"],
                        new_peak, trade["sl"],
                    )
        if moved:
            self._save()
        return moved

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def check_sl_tp(
        self,
        pair: str,
        candle_high: float,
        candle_low: float,
    ) -> List[Dict]:
        closed = []
        for trade in list(self.open_trades):
            if trade["pair"] != pair:
                continue
            close_price, reason = None, None
            if trade["direction"] == "LONG":
                # БАГ 12 fix: TP проверяем первым — если свеча пробила оба уровня,
                # нельзя определить порядок; TP-first даёт нейтральную оценку.
                if candle_high >= trade["tp"]:   close_price, reason = trade["tp"], "TP"
                elif candle_low <= trade["sl"]:  close_price, reason = trade["sl"], "SL"
            else:
                if candle_low  <= trade["tp"]:   close_price, reason = trade["tp"], "TP"
                elif candle_high >= trade["sl"]: close_price, reason = trade["sl"], "SL"
            if close_price is not None:
                closed.append(self._close_trade(trade, close_price, reason))
        return closed

    def _close_trade(self, trade: Dict, close_price: float, reason: str) -> Dict:
        notional = trade.get("notional", trade["size_usd"])
        entry    = trade["entry_price"]
        size_usd = trade["size_usd"]
        if not entry or not size_usd:
            logger.error(
                "Сделка #%s имеет entry_price=%s size_usd=%s — принудительное закрытие",
                trade["id"], entry, size_usd,
            )
            trade.update(
                status="CLOSED", closed_at=_utcnow(), close_price=close_price,
                close_reason=reason, pnl_usd=0.0, pnl_percent=0.0,
            )
            self._trades_dirty = True
            self._save()
            return trade
        if trade["direction"] == "LONG":
            pnl = (close_price - entry) / entry * notional
        else:
            pnl = (entry - close_price) / entry * notional
        # Комиссия: 0.1% round-trip от notional (открытие + закрытие одной ставкой)
        commission = round(notional * self.commission_rate, 4)
        pnl -= commission
        pnl_pct = pnl / size_usd * 100
        trade.update(
            status="CLOSED", closed_at=_utcnow(), close_price=close_price,
            close_reason=reason, pnl_usd=round(pnl, 2), pnl_percent=round(pnl_pct, 2),
            commission=commission,
        )
        self.balance = round(self.balance + pnl, 2)
        self._trades_dirty = True
        self._save()
        logger.info(
            "Закрыта #%s %s %s: %s @ %.6f | PnL $%.2f (%.2f%%) | баланс $%.2f",
            trade["id"], trade["direction"], trade["pair"],
            reason, close_price, pnl, pnl_pct, self.balance,
        )
        return trade

    # ── reports ───────────────────────────────────────────────────────────────

    def save_report(self, pair: str, text: str) -> None:
        if self._use_gsheets:
            try:
                cell = None
                try:
                    cell = self._ws_reports.find(pair, in_column=1)
                except Exception:
                    pass
                row = [pair, text, _utcnow()]
                if cell:
                    self._ws_reports.update(f"A{cell.row}", [row])
                else:
                    self._ws_reports.append_row(row)
            except Exception as exc:
                logger.error("Ошибка сохранения отчёта %s: %s", pair, exc)
        else:
            self.reports[pair] = {"pair": pair, "text": text, "saved_at": _utcnow()}
            try:
                REPORTS_FILE.write_text(
                    json.dumps(self.reports, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.error("Ошибка записи reports.json: %s", exc)

    def get_report(self, pair: str) -> Optional[Dict]:
        if self._use_gsheets:
            try:
                cell = self._ws_reports.find(pair, in_column=1)
                if cell:
                    row = self._ws_reports.row_values(cell.row)
                    # БАГ 16 fix: защита от неполных строк
                    if len(row) < 2:
                        return None
                    return {"pair": row[0], "text": row[1], "saved_at": row[2] if len(row) > 2 else ""}
            except Exception as exc:
                logger.error("Ошибка загрузки отчёта %s: %s", pair, exc)
            return None
        return self.reports.get(pair)

    # ── statistics ────────────────────────────────────────────────────────────

    def get_stats(self, prices: Optional[Dict[str, float]] = None) -> Dict:
        closed    = self.closed_trades
        wins      = [t for t in closed if (t["pnl_usd"] or 0) > 0]
        losses    = [t for t in closed if (t["pnl_usd"] or 0) < 0]
        total_pnl = sum(t["pnl_usd"] or 0 for t in closed)
        winrate   = len(wins) / len(closed) * 100 if closed else 0.0
        avg_pnl   = total_pnl / len(closed) if closed else 0.0
        equity    = self.get_equity(prices)
        bal_chg   = (equity - self.initial_balance) / self.initial_balance * 100
        best  = max(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        worst = min(closed, key=lambda x: x["pnl_usd"] or 0, default=None)
        return {
            "balance":            self.balance,
            "equity":             equity,
            "initial_balance":    self.initial_balance,
            "balance_change_pct": round(bal_chg, 2),
            "open_count":         len(self.open_trades),
            "pending_count":      len(self.pending_orders),
            "total_closed":       len(closed),
            "wins":               len(wins),
            "losses":             len(losses),
            "winrate":            round(winrate, 1),
            "total_pnl":          round(total_pnl, 2),
            "avg_pnl":            round(avg_pnl, 2),
            "best":               best,
            "worst":              worst,
            "orders_created":     self.orders_created,
            "orders_cancelled":   self.orders_cancelled,
            "orders_triggered":   len(self.trades),
        }

    def recent_trades(self, n: int = 10) -> List[Dict]:
        return sorted(
            self.closed_trades,
            key=lambda x: x["closed_at"] or "",
            reverse=True,
        )[:n]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
