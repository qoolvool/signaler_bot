"""
MEXC Support & Resistance Signaler + Paper Trader
==================================================
Поиск уровней S/R, генерация точек входа и симуляция сделок
на виртуальном балансе $1 000.

Установка:
    pip3 install -r requirements.txt

Настройка:
    cp .env.example .env  # заполни ключи

Запуск:
    python3 signaler.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from paper_trader import PaperPortfolio

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MEXC_API_KEY      = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY   = os.getenv("MEXC_SECRET_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

TRADING_PAIRS = [
    p.strip() for p in os.getenv("TRADING_PAIRS", (
        "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,BNB/USDT,"
        "ADA/USDT,AVAX/USDT,DOT/USDT,ATOM/USDT,LINK/USDT,"
        "LTC/USDT,NEAR/USDT,UNI/USDT,FIL/USDT,INJ/USDT,"
        "RNDR/USDT,TON/USDT,SUI/USDT,ARB/USDT,OP/USDT"
    )).split(",")
    if p.strip()
]

TIMEFRAME             = os.getenv("TIMEFRAME", "1h")
CANDLES_LIMIT         = int(os.getenv("CANDLES_LIMIT", "250"))
MIN_TOUCHES           = int(os.getenv("MIN_TOUCHES", "3"))
TOLERANCE_PERCENT     = float(os.getenv("TOLERANCE_PERCENT", "0.8"))
EXTREMA_WINDOW        = int(os.getenv("EXTREMA_WINDOW", "8"))
TOP_N_LEVELS          = int(os.getenv("TOP_N_LEVELS", "5"))
MIN_TOUCH_SPACING     = int(os.getenv("MIN_TOUCH_SPACING", "3"))
LEVEL_AGE_MIN_CANDLES = int(os.getenv("LEVEL_AGE_MIN_CANDLES", "10"))
VOLUME_TOUCH_MULTIPLIER = float(os.getenv("VOLUME_TOUCH_MULTIPLIER", "1.2"))
REQUIRE_RETEST        = os.getenv("REQUIRE_RETEST", "false").lower() == "true"

ENTRY_PROXIMITY_PERCENT = float(os.getenv("ENTRY_PROXIMITY_PERCENT", "0.5"))
EMA_PERIOD              = int(os.getenv("EMA_PERIOD", "200"))
AUTO_TOP_PAIRS          = int(os.getenv("AUTO_TOP_PAIRS", "0"))
SL_PERCENT              = float(os.getenv("SL_PERCENT", "1.5"))  # стоп-лосс от цены входа
TP_PERCENT              = float(os.getenv("TP_PERCENT", "3.0"))  # тейк-профит от цены входа

DELAY_BETWEEN_PAIRS  = int(os.getenv("DELAY_BETWEEN_PAIRS", "2"))
RUN_INTERVAL_HOURS   = float(os.getenv("RUN_INTERVAL_HOURS", "3"))

# --- Paper trading ---
INITIAL_BALANCE        = float(os.getenv("INITIAL_BALANCE", "1000"))
TRADE_SIZE_PERCENT     = float(os.getenv("TRADE_SIZE_PERCENT", "2"))
MAX_OPEN_TRADES        = int(os.getenv("MAX_OPEN_TRADES", "5"))
PENDING_EXPIRY_CHECKS  = int(os.getenv("PENDING_EXPIRY_CHECKS", "8"))
LEVERAGE               = int(os.getenv("LEVERAGE", "10"))


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signaler")


# ============================================================
# ГЛОБАЛЬНЫЙ ПОРТФЕЛЬ
# ============================================================
portfolio = PaperPortfolio(
    initial_balance=INITIAL_BALANCE,
    trade_size_percent=TRADE_SIZE_PERCENT,
    max_open_trades=MAX_OPEN_TRADES,
    pending_expiry_checks=PENDING_EXPIRY_CHECKS,
    leverage=LEVERAGE,
)


# ============================================================
# ПРОВЕРКА КОНФИГА
# ============================================================
def validate_config() -> None:
    missing = []
    if not MEXC_API_KEY:       missing.append("MEXC_API_KEY")
    if not MEXC_SECRET_KEY:    missing.append("MEXC_SECRET_KEY")
    if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:   missing.append("TELEGRAM_CHAT_ID")
    if not TRADING_PAIRS and AUTO_TOP_PAIRS <= 0:
        missing.append("TRADING_PAIRS (или установи AUTO_TOP_PAIRS > 0)")
    if missing:
        logger.error("Не заполнены обязательные поля: %s", ", ".join(missing))
        sys.exit(1)
    logger.info("Конфиг проверен.")


# ============================================================
# MEXC CLIENT
# ============================================================
def get_mexc_client() -> ccxt.mexc:
    logger.info("Подключение к MEXC...")
    try:
        client = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_SECRET_KEY,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        client.load_markets()
        logger.info("MEXC подключён. Рынков: %d", len(client.markets))
        return client
    except ccxt.AuthenticationError as exc:
        logger.error("Ошибка авторизации MEXC: %s", exc)
        raise
    except Exception as exc:
        logger.error("Ошибка подключения к MEXC: %s", exc)
        raise


def fetch_top_pairs(client: ccxt.mexc, n: int = 10) -> List[str]:
    """Получить топ-N USDT пар по объёму за 24ч."""
    logger.info("Получение топ-%d пар по объёму...", n)
    try:
        tickers = client.fetch_tickers()
        usdt = {
            sym: t for sym, t in tickers.items()
            if sym.endswith("/USDT") and t.get("quoteVolume")
        }
        top = sorted(usdt.items(), key=lambda x: x[1]["quoteVolume"] or 0, reverse=True)[:n]
        pairs = [sym for sym, _ in top]
        logger.info("Топ-%d: %s", n, ", ".join(pairs))
        return pairs
    except Exception as exc:
        logger.error("Ошибка при получении топ пар: %s", exc)
        return TRADING_PAIRS


def fetch_ohlcv(
    client: ccxt.mexc,
    symbol: str,
    timeframe: str = TIMEFRAME,
    limit: int = CANDLES_LIMIT,
) -> Optional[pd.DataFrame]:
    logger.info("Загрузка %d свечей %s для %s...", limit, timeframe, symbol)
    try:
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            logger.warning("Пустой ответ для %s", symbol)
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if len(df) < EXTREMA_WINDOW * 2 + 1:
            logger.warning("Слишком мало свечей для %s: %d", symbol, len(df))
            return None
        logger.info("Получено %d свечей для %s", len(df), symbol)
        return df
    except ccxt.BadSymbol:
        logger.error("Пара %s не найдена на MEXC", symbol)
        return None
    except Exception as exc:
        logger.error("Ошибка загрузки OHLCV для %s: %s", symbol, exc)
        return None


# ============================================================
# ПОИСК УРОВНЕЙ
# ============================================================
def _find_local_extrema(series: pd.Series, window: int, kind: str) -> List[tuple]:
    values = series.values
    extrema = []
    for i in range(window, len(values) - window):
        segment = values[i - window:i + window + 1]
        center = values[i]
        if kind == "max" and center == segment.max():
            extrema.append((float(center), i))
        elif kind == "min" and center == segment.min():
            extrema.append((float(center), i))
    return extrema


def _count_touches(
    level: float,
    df: pd.DataFrame,
    tolerance: float,
    min_spacing: int = 0,
    volume_multiplier: float = 0.0,
) -> int:
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    avg_volume = df["volume"].mean()
    touch_indices = []
    for i in range(len(df)):
        high, low = df["high"].iat[i], df["low"].iat[i]
        if not ((lower <= high <= upper) or (lower <= low <= upper)):
            continue
        if volume_multiplier > 0 and df["volume"].iat[i] < avg_volume * volume_multiplier:
            continue
        touch_indices.append(i)
    if not touch_indices:
        return 0
    if min_spacing <= 0:
        return len(touch_indices)
    filtered = [touch_indices[0]]
    for idx in touch_indices[1:]:
        if idx - filtered[-1] >= min_spacing:
            filtered.append(idx)
    return len(filtered)


def _has_retest_after_breakout(
    level: float, df: pd.DataFrame, tolerance: float, level_type: str
) -> bool:
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
    in_breakout = False
    for i in range(len(closes)):
        if not in_breakout:
            if level_type == "SUPPORT" and closes[i] < lower:
                in_breakout = True
            elif level_type == "RESISTANCE" and closes[i] > upper:
                in_breakout = True
        else:
            if (lower <= highs[i] <= upper) or (lower <= lows[i] <= upper):
                return True
            if level_type == "SUPPORT" and closes[i] > upper:
                in_breakout = False
            elif level_type == "RESISTANCE" and closes[i] < lower:
                in_breakout = False
    return False


def remove_duplicates(levels: List[Dict], tolerance: float) -> List[Dict]:
    if not levels:
        return []
    sorted_levels = sorted(levels, key=lambda x: x["touches"], reverse=True)
    unique: List[Dict] = []
    for level in sorted_levels:
        if not any(
            kept["type"] == level["type"]
            and abs(level["price"] - kept["price"]) / kept["price"] < tolerance
            for kept in unique
        ):
            unique.append(level)
    return unique


def find_support_resistance(
    df: pd.DataFrame,
    min_touches: int = MIN_TOUCHES,
    tolerance_percent: float = TOLERANCE_PERCENT,
    window: int = EXTREMA_WINDOW,
    top_n: int = TOP_N_LEVELS,
    min_touch_spacing: int = MIN_TOUCH_SPACING,
    level_age_min: int = LEVEL_AGE_MIN_CANDLES,
    volume_multiplier: float = VOLUME_TOUCH_MULTIPLIER,
    require_retest: bool = REQUIRE_RETEST,
) -> List[Dict]:
    tolerance = tolerance_percent / 100.0
    total_candles = len(df)
    resistance_candidates = _find_local_extrema(df["high"], window, "max")
    support_candidates    = _find_local_extrema(df["low"],  window, "min")
    logger.info(
        "Кандидатов: сопр.=%d, подд.=%d",
        len(resistance_candidates), len(support_candidates),
    )
    levels: List[Dict] = []
    for candidates, level_type in [
        (resistance_candidates, "RESISTANCE"),
        (support_candidates,    "SUPPORT"),
    ]:
        for price, idx in candidates:
            if total_candles - 1 - idx < level_age_min:
                continue
            touches = _count_touches(price, df, tolerance,
                                     min_spacing=min_touch_spacing,
                                     volume_multiplier=volume_multiplier)
            if touches < min_touches:
                continue
            has_retest = _has_retest_after_breakout(price, df, tolerance, level_type)
            if require_retest and not has_retest:
                continue
            levels.append({"price": price, "type": level_type,
                           "touches": touches, "has_retest": has_retest})
    levels = remove_duplicates(levels, tolerance)
    levels = sorted(levels, key=lambda x: x["touches"], reverse=True)[:top_n]
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)
    logger.info("Найдено уровней: %d", len(levels))
    return levels


# ============================================================
# ТОЧКИ ВХОДА
# ============================================================
def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _detect_candle_pattern(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 3:
        return None
    c = df.iloc[-2]
    prev = df.iloc[-3]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    total_range = h - l
    if total_range == 0:
        return None
    body = abs(cl - o)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_ratio = body / total_range
    if body_ratio < 0.3 and lower_wick / total_range > 0.6 and lower_wick > 2 * upper_wick:
        return "молот 🔨"
    if body_ratio < 0.3 and upper_wick / total_range > 0.6 and upper_wick > 2 * lower_wick:
        return "падающая звезда ⭐"
    p_o, p_cl = float(prev["open"]), float(prev["close"])
    if p_cl < p_o and cl > o and o <= p_cl and cl >= p_o:
        return "бычье поглощение 🕯"
    if p_cl > p_o and cl < o and o >= p_cl and cl <= p_o:
        return "медвежье поглощение 🕯"
    return None


def find_entry_signals(
    df: pd.DataFrame,
    levels: List[Dict],
    current_price: float,
    proximity_percent: float = ENTRY_PROXIMITY_PERCENT,
    ema_period: int = EMA_PERIOD,
    sl_percent: float = SL_PERCENT,
    tp_percent: float = TP_PERCENT,
) -> List[Dict]:
    if ema_period >= len(df):
        logger.warning("Недостаточно свечей для EMA%d (%d)", ema_period, len(df))
        return []
    proximity = proximity_percent / 100.0
    ema_val   = float(_calc_ema(df["close"], ema_period).iloc[-1])
    ema_trend = "UP" if current_price > ema_val else "DOWN"
    pattern   = _detect_candle_pattern(df)
    signals = []
    for lvl in levels:
        distance = abs(current_price - lvl["price"]) / lvl["price"]
        if distance > proximity:
            continue
        if lvl["type"] == "SUPPORT" and ema_trend == "UP":
            direction = "LONG"
        elif lvl["type"] == "RESISTANCE" and ema_trend == "DOWN":
            direction = "SHORT"
        else:
            continue
        entry = lvl["price"]
        if direction == "LONG":
            sl = entry * (1 - sl_percent / 100)
            tp = entry * (1 + tp_percent / 100)
        else:
            sl = entry * (1 + sl_percent / 100)
            tp = entry * (1 - tp_percent / 100)
        rr = round(tp_percent / sl_percent, 1)
        signals.append({
            "direction": direction,
            "level": lvl,
            "pattern": pattern,
            "ema": ema_val,
            "ema_trend": ema_trend,
            "distance_percent": round(distance * 100, 2),
            "sl": sl,
            "tp": tp,
            "rr": rr,
        })
    return signals


# ============================================================
# TELEGRAM — отправка
# ============================================================
async def send_msg(bot: Bot, text: str) -> bool:
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return True
    except TelegramError as exc:
        logger.error("Ошибка Telegram: %s", exc)
        return False


# ============================================================
# ФОРМАТИРОВАНИЕ СООБЩЕНИЙ
# ============================================================
def _fp(price: float) -> str:
    """Адаптивное форматирование цены."""
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.8f}"


REPLY_KB = ReplyKeyboardMarkup(
    [
        ["📊 Статистика", "📋 Лог сделок", "📂 Позиции"],
        ["📈 Монеты"],
    ],
    resize_keyboard=True,
    is_persistent=True,
)


def _build_pairs_keyboard() -> InlineKeyboardMarkup:
    buttons: list = []
    row: list = []
    for pair in TRADING_PAIRS:
        base = pair.split("/")[0]
        row.append(InlineKeyboardButton(base, callback_data=f"report:{pair}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


def fmt_analysis(
    pair: str,
    timeframe: str,
    current_price: float,
    levels: List[Dict],
    signals: List[Dict],
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📊 <b>{pair}</b>  •  <code>{timeframe}</code>",
        f"💰 Цена: <b>{_fp(current_price)}</b>",
        f"🕒 {now}", "",
    ]
    if signals:
        lines.append("🎯 <b>СИГНАЛЫ ВХОДА:</b>")
        for sig in signals:
            lvl = sig["level"]
            de  = "📈" if sig["direction"] == "LONG" else "📉"
            ta  = "↑" if sig["ema_trend"] == "UP" else "↓"
            lines.append(
                f"{de} <b>{sig['direction']}</b>  вблизи {_fp(lvl['price'])}"
                f"  <i>({sig['distance_percent']}%)</i>"
            )
            lines.append(f"   Тренд: {ta} EMA{EMA_PERIOD} = {_fp(sig['ema'])}")
            if sig["pattern"]:
                lines.append(f"   Паттерн: {sig['pattern']}")
            tp_str = _fp(sig["tp"]) if sig["tp"] else "—"
            rr_str = f"  •  R:R 1:{sig['rr']}" if sig["rr"] else ""
            lines.append(f"   SL: <b>{_fp(sig['sl'])}</b>  •  TP: <b>{tp_str}</b>{rr_str}")
        lines.append("")
    else:
        lines += ["⏳ <i>Сигналов нет — цена далеко от уровней.</i>", ""]

    if not levels:
        lines.append("⚠️ <i>Уровни не найдены.</i>")
        return "\n".join(lines)

    lines.append("<b>Уровни:</b>")
    for lvl in levels:
        emoji   = "🔴" if lvl["type"] == "RESISTANCE" else "🟢"
        ru_type = "Сопр." if lvl["type"] == "RESISTANCE" else "Подд."
        dist    = (lvl["price"] - current_price) / current_price * 100
        sign    = "+" if dist >= 0 else ""
        retest  = " ✅" if lvl.get("has_retest") else ""
        lines.append(
            f"{emoji} <b>{_fp(lvl['price'])}</b> — {ru_type} "
            f"(касаний: <b>{lvl['touches']}</b>, {sign}{dist:.2f}%{retest})"
        )
    return "\n".join(lines)


def fmt_trade_opened(trade: Dict, balance: float) -> str:
    de  = "📈" if trade["direction"] == "LONG" else "📉"
    rr  = f"  •  R:R 1:{trade['rr']}" if trade["rr"] else ""
    return (
        f"{de} <b>СДЕЛКА ОТКРЫТА #{trade['id']}</b>\n"
        f"<b>{trade['pair']}</b>  •  {trade['direction']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Вход:  <b>{_fp(trade['entry_price'])}</b>\n"
        f"SL:    <b>{_fp(trade['sl'])}</b>  (-{trade['risk_pct']}%)\n"
        f"TP:    <b>{_fp(trade['tp'])}</b>  (+{trade['reward_pct']}%)\n"
        f"Размер: <b>${trade['size_usd']}</b>{rr}\n"
        f"Баланс: ${balance:,.2f}"
    )


def fmt_trade_closed(trade: Dict, balance: float) -> str:
    won    = (trade["pnl_usd"] or 0) > 0
    emoji  = "✅" if won else "❌"
    reason = "ТЕЙК-ПРОФИТ 🎯" if trade["close_reason"] == "TP" else "СТОП-ЛОСС 🛑"
    sign   = "+" if (trade["pnl_usd"] or 0) >= 0 else ""
    return (
        f"{emoji} <b>{reason} #{trade['id']}</b>\n"
        f"<b>{trade['pair']}</b>  •  {trade['direction']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Вход: {_fp(trade['entry_price'])} → Закрыт: <b>{_fp(trade['close_price'])}</b>\n"
        f"P&L: <b>{sign}${trade['pnl_usd']}  ({sign}{trade['pnl_percent']}%)</b>\n"
        f"Баланс: <b>${balance:,.2f}</b>"
    )


def fmt_pending_created(order: Dict) -> str:
    de  = "📈" if order["direction"] == "LONG" else "📉"
    rr  = f"  •  R:R 1:{order['rr']}" if order["rr"] else ""
    chk = order["checks_remaining"]
    lev = order.get("leverage", 1)
    ntl = order.get("notional", order["size_usd"])
    return (
        f"⏳ <b>ЛИМИТНЫЙ ОРДЕР #{order['id']}</b>\n"
        f"{de} <b>{order['pair']}</b>  •  {order['direction']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Вход (лимит): <b>{_fp(order['entry_price'])}</b>\n"
        f"SL: <b>{_fp(order['sl'])}</b>  (-{order['risk_pct']}%)\n"
        f"TP: <b>{_fp(order['tp'])}</b>  (+{order['reward_pct']}%){rr}\n"
        f"Маржа: <b>${order['size_usd']}</b>  •  Плечо: <b>{lev}×</b>  •  Позиция: <b>${ntl}</b>\n"
        f"<i>Ждём касания уровня... (до {chk} проверок)</i>"
    )


def fmt_pending_triggered(trade: Dict, balance: float) -> str:
    de = "📈" if trade["direction"] == "LONG" else "📉"
    return (
        f"✅ <b>ОРДЕР ИСПОЛНЕН #{trade['id']}</b>\n"
        f"{de} <b>{trade['pair']}</b>  •  {trade['direction']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Цена коснулась уровня: <b>{_fp(trade['entry_price'])}</b>\n"
        f"SL: <b>{_fp(trade['sl'])}</b>  •  TP: <b>{_fp(trade['tp'])}</b>\n"
        f"Баланс: ${balance:,.2f}"
    )


def fmt_pending_cancelled(order: Dict) -> str:
    de = "📈" if order["direction"] == "LONG" else "📉"
    return (
        f"🗑 <b>ОРДЕР ОТМЕНЁН #{order['id']}</b>\n"
        f"{de} {order['pair']}  •  {order['direction']}\n"
        f"<i>Цена не коснулась {_fp(order['entry_price'])} за отведённое время</i>"
    )


def fmt_stats(ptf: PaperPortfolio) -> str:
    s    = ptf.get_stats()
    sign = "+" if s["balance_change_pct"] >= 0 else ""
    ps   = "+" if s["total_pnl"] >= 0 else ""
    lines = [
        "📊 <b>СТАТИСТИКА ПОРТФЕЛЯ</b>", "",
        f"💰 Баланс: <b>${s['balance']:,.2f}</b>  ({sign}{s['balance_change_pct']}%)",
        f"🏦 Начальный: ${s['initial_balance']:,.2f}",
        f"📂 Открытых: <b>{s['open_count']}</b>  •  ⏳ Ожидающих: <b>{s['pending_count']}</b>", "",
        f"📈 Всего закрыто: <b>{s['total_closed']}</b>",
        f"  ✅ Прибыльных: <b>{s['wins']}</b>  ({s['winrate']}% winrate)",
        f"  ❌ Убыточных:  <b>{s['losses']}</b>", "",
        f"💵 Итоговый P&L: <b>{ps}${s['total_pnl']:,.2f}</b>",
    ]
    if s["best"]:
        b = s["best"]
        lines.append(f"🏆 Лучшая:  <b>+${b['pnl_usd']}</b>  ({b['pair']} {b['direction']})")
    if s["worst"]:
        w = s["worst"]
        lines.append(f"💔 Худшая:  <b>${w['pnl_usd']}</b>  ({w['pair']} {w['direction']})")
    return "\n".join(lines)


def fmt_log(ptf: PaperPortfolio, n: int = 10) -> str:
    trades = ptf.recent_trades(n)
    if not trades:
        return "📋 <b>Лог сделок</b>\n\n<i>Закрытых сделок пока нет.</i>"
    lines = [f"📋 <b>Последние {len(trades)} сделок:</b>", ""]
    for i, t in enumerate(trades, 1):
        pnl  = t["pnl_usd"] or 0
        em   = "✅" if pnl > 0 else "❌"
        sign = "+" if pnl >= 0 else ""
        rsn  = "TP" if t["close_reason"] == "TP" else "SL"
        lines.append(
            f"{i}. {em} <b>{t['pair']}</b> {t['direction']} [{rsn}]  "
            f"<b>{sign}${t['pnl_usd']}</b> ({sign}{t['pnl_percent']}%)"
        )
        lines.append(
            f"   {_fp(t['entry_price'])} → {_fp(t['close_price'])}"
            f"  •  {(t['closed_at'] or '')[:16]}"
        )
    return "\n".join(lines)


def fmt_open_trades(ptf: PaperPortfolio, prices: Dict[str, float]) -> str:
    open_t   = ptf.open_trades
    pending  = ptf.pending_orders
    if not open_t and not pending:
        return "📂 <b>Активных позиций нет</b>\n\n<i>Жду сигналов...</i>"

    lines = ["📂 <b>ТЕКУЩИЕ ПОЗИЦИИ</b>", ""]

    for t in open_t:
        de  = "📈" if t["direction"] == "LONG" else "📉"
        cur = prices.get(t["pair"])
        if cur:
            lev  = t.get("leverage", 1)
            ntl  = t.get("notional", t["size_usd"])
            upnl = ((cur - t["entry_price"]) / t["entry_price"] * ntl
                    if t["direction"] == "LONG"
                    else (t["entry_price"] - cur) / t["entry_price"] * ntl)
            upnl_pct = upnl / t["size_usd"] * 100
            sign = "+" if upnl >= 0 else ""
            pnl_line = f"\n   PnL: <b>{sign}${upnl:.2f}  ({sign}{upnl_pct:.1f}%)</b>  •  Цена: {_fp(cur)}"
        else:
            pnl_line = ""
        lines.append(
            f"{de} <b>#{t['id']}</b> {t['pair']}  •  {t['direction']}\n"
            f"   Вход: {_fp(t['entry_price'])}  SL: {_fp(t['sl'])}  TP: {_fp(t['tp'])}"
            f"{pnl_line}"
        )

    if pending:
        lines += ["", "⏳ <b>Ожидающие ордера:</b>"]
        for o in pending:
            de = "📈" if o["direction"] == "LONG" else "📉"
            lines.append(
                f"{de} <b>#{o['id']}</b> {o['pair']}  •  {o['direction']}\n"
                f"   Лимит: {_fp(o['entry_price'])}  •  Осталось проверок: {o['checks_remaining']}"
            )

    return "\n".join(lines)


def _fetch_current_prices(client, trades: List[Dict]) -> Dict[str, float]:
    if not client or not trades:
        return {}
    prices: Dict[str, float] = {}
    for pair in {t["pair"] for t in trades}:
        try:
            prices[pair] = float(client.fetch_ticker(pair)["last"])
        except Exception:
            pass
    return prices


# ============================================================
# АНАЛИЗ ПАРЫ (с бумажной торговлей)
# ============================================================
async def analyze_pair(
    client: ccxt.mexc,
    bot: Bot,
    pair: str,
) -> None:
    logger.info("Анализ: %s", pair)

    df = fetch_ohlcv(client, pair, TIMEFRAME, CANDLES_LIMIT)
    if df is None or df.empty:
        logger.warning("Пропускаем %s (нет данных)", pair)
        return

    current_price = float(df["close"].iloc[-1])

    last = df.iloc[-2]  # последняя ЗАВЕРШЁННАЯ свеча
    h, l = float(last["high"]), float(last["low"])

    # 1. Закрытие открытых сделок по SL/TP
    for trade in portfolio.check_sl_tp(pair, h, l):
        await send_msg(bot, fmt_trade_closed(trade, portfolio.balance))

    # 2. Проверка ожидающих ордеров — коснулась ли цена уровня
    triggered, cancelled = portfolio.check_pending_orders(pair, h, l)
    for trade in triggered:
        await send_msg(bot, fmt_pending_triggered(trade, portfolio.balance))
    for order in cancelled:
        await send_msg(bot, fmt_pending_cancelled(order))

    # 3. Уровни и сигналы
    levels  = find_support_resistance(df)
    signals = find_entry_signals(df, levels, current_price)

    # 4. Сохраняем отчёт в БД; в чат отправляем только если есть сигнал
    report_text = fmt_analysis(pair, TIMEFRAME, current_price, levels, signals)
    portfolio.save_report(pair, report_text)
    if signals:
        await send_msg(bot, report_text)

    # 5. Размещаем лимитные ордера на точной цене уровня
    for sig in signals:
        if sig["tp"] is None:
            continue
        order = portfolio.create_pending_order(
            pair=pair,
            direction=sig["direction"],
            entry_price=sig["level"]["price"],  # точная цена уровня, не текущая
            sl=sig["sl"],
            tp=sig["tp"],
        )
        if order:
            await send_msg(bot, fmt_pending_created(order))


# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Бот запущен. Используй кнопки внизу.",
        reply_markup=REPLY_KB,
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        fmt_stats(portfolio), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
    )


async def cmd_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        fmt_log(portfolio), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    txt = update.message.text or ""
    if "Статистика" in txt:
        msg = fmt_stats(portfolio)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=REPLY_KB)
    elif "Лог" in txt:
        msg = fmt_log(portfolio)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=REPLY_KB)
    elif "Позиции" in txt:
        prices = _fetch_current_prices(
            context.bot_data.get("client"), portfolio.open_trades
        )
        msg = fmt_open_trades(portfolio, prices)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=REPLY_KB)
    elif "Монеты" in txt:
        await update.message.reply_text(
            "📈 <b>Выбери монету для просмотра отчёта:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=_build_pairs_keyboard(),
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "pairs_list":
        await query.edit_message_text(
            "📈 <b>Выбери монету для просмотра отчёта:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=_build_pairs_keyboard(),
        )
        return

    if data.startswith("report:"):
        pair = data[7:]
        report = portfolio.get_report(pair)
        if report:
            msg = f"{report['text']}\n\n<i>🕒 Обновлено: {report.get('saved_at', '')}</i>"
        else:
            msg = f"⚠️ <i>Отчёт по {pair} ещё не готов — подождите следующего цикла анализа.</i>"
        back_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ К монетам", callback_data="pairs_list")
        ]])
        await query.edit_message_text(msg, parse_mode=ParseMode.HTML, reply_markup=back_kb)


# ============================================================
# JOB: ПЕРИОДИЧЕСКИЙ АНАЛИЗ
# ============================================================
async def analysis_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data["client"]
    bot    = context.bot

    pairs = fetch_top_pairs(client, AUTO_TOP_PAIRS) if AUTO_TOP_PAIRS > 0 else TRADING_PAIRS
    logger.info("=== Запуск анализа (%d пар) ===", len(pairs))

    for idx, pair in enumerate(pairs):
        try:
            await analyze_pair(client, bot, pair)
        except Exception as exc:
            logger.error("Ошибка %s: %s", pair, exc)
        if idx < len(pairs) - 1:
            await asyncio.sleep(DELAY_BETWEEN_PAIRS)


# ============================================================
# СТАРТОВОЕ СООБЩЕНИЕ (post_init)
# ============================================================
async def post_init(app: Application) -> None:
    pairs_mode = (f"топ-{AUTO_TOP_PAIRS} по объёму" if AUTO_TOP_PAIRS > 0
                  else f"{len(TRADING_PAIRS)} пар из конфига")
    mode = f"каждые {RUN_INTERVAL_HOURS}ч" if RUN_INTERVAL_HOURS > 0 else "одноразовый"

    text = (
        f"🚀 <b>Signaler + Paper Trading запущен</b>\n"
        f"Пары: {pairs_mode}  •  TF: <code>{TIMEFRAME}</code>\n"
        f"Режим: <b>{mode}</b>\n\n"
        f"💰 Баланс: <b>${portfolio.balance:,.2f}</b>  "
        f"(из ${portfolio.initial_balance:,.2f})\n"
        f"Маржа на сделку: {TRADE_SIZE_PERCENT}%  •  Плечо: <b>{LEVERAGE}×</b>  •  Max: {MAX_OPEN_TRADES}\n"
        f"SL: {SL_PERCENT}%  •  TP: {TP_PERCENT}%  •  R:R 1:{round(TP_PERCENT/SL_PERCENT,1)}\n"
        f"EMA: {EMA_PERIOD}  •  Entry proximity: {ENTRY_PROXIMITY_PERCENT}%\n\n"
        f"Первый анализ через ~15 сек."
    )
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=REPLY_KB,
        )
    except TelegramError as exc:
        logger.error("Ошибка стартового сообщения: %s", exc)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    validate_config()

    try:
        client = get_mexc_client()
    except Exception:
        logger.error("Невозможно подключиться к MEXC.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.bot_data["client"] = client

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("log",   cmd_log))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    interval_sec = int(RUN_INTERVAL_HOURS * 3600)
    if interval_sec > 0:
        app.job_queue.run_repeating(analysis_job, interval=interval_sec, first=15)
    else:
        app.job_queue.run_once(analysis_job, when=15)

    logger.info("Бот запущен, анализ каждые %.1fч.", RUN_INTERVAL_HOURS)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    except Exception as exc:
        logger.error("Фатальная ошибка: %s", exc)
        sys.exit(1)
