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
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

# load_dotenv MUST come before local imports so DATA_DIR is applied to file paths
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Local modules — config must be imported after dotenv
from config import *  # noqa: F401,F403
from analysis import (
    _add_confluence_scores,
    _calc_ema_levels,
    _check_correlation,
    _fetch_htf_confluence,
    _find_fvg_zones,
    _mark_htf_confirmed,
    _returns_cache,
    find_entry_signals,
    find_support_resistance,
)
from formatters import (
    REPLY_KB,
    _fp,
    _pairs_inline_kb,
    fmt_analysis,
    fmt_log,
    fmt_open_trades,
    fmt_pending_orders,
    fmt_sl_to_breakeven,
    fmt_stats,
    fmt_trade_closed,
    fmt_trade_opened,
)
from indicators import (
    _calc_adx,
    _calc_atr_ratio,
    _calc_rsi_series,
    _detect_regime,
    _find_crash_low,
    _find_pump_high,
)

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
    commission_maker=COMMISSION_MAKER,
    commission_taker=COMMISSION_TAKER,
    breakeven_threshold=BREAKEVEN_THRESHOLD,
    fixed_risk_mode=FIXED_RISK_MODE,
    risk_per_trade_percent=RISK_PER_TRADE_PERCENT,
    max_trade_size_percent=MAX_TRADE_SIZE_PERCENT,
    trailing_stop=TRAILING_STOP,
    trailing_mult=TRAILING_MULT,
)

# ============================================================
# ДАННЫЕ С БИРЖИ
# ============================================================

def fetch_top_pairs(client: ccxt.mexc, n: int = 10) -> List[str]:
    """Топ-N USDT пар по объёму за 24ч."""
    logger.info("Получение топ-%d пар по объёму...", n)
    try:
        tickers = client.fetch_tickers()
        usdt = [
            (sym, t.get("quoteVolume") or 0)
            for sym, t in tickers.items()
            if sym.endswith("/USDT") and t.get("quoteVolume")
        ]
        top = sorted(usdt, key=lambda x: x[1], reverse=True)[:n]
        pairs = [sym for sym, _ in top]
        logger.info("Топ-%d пар: %s", n, pairs)
        return pairs
    except Exception as exc:
        logger.error("Ошибка получения топ-пар: %s", exc)
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


def _fetch_hl(
    client: ccxt.mexc,
    pair: str,
    current_price: float,
    fallback_h: Optional[float] = None,
    fallback_l: Optional[float] = None,
) -> Tuple[float, float]:
    """Fetch recent high/low from CHECK_TIMEFRAME; falls back to supplied values or current_price."""
    try:
        raw = client.fetch_ohlcv(pair, CHECK_TIMEFRAME, limit=5)
        if raw and len(raw) >= 2:
            h = max(r[2] for r in raw[-2:])
            l = min(r[3] for r in raw[-2:])
        else:
            raise ValueError("мало свечей")
    except Exception:
        h = fallback_h if fallback_h is not None else current_price
        l = fallback_l if fallback_l is not None else current_price
    return max(h, current_price), min(l, current_price)


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
# TELEGRAM — отправка + проверки позиций
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
    except Exception as exc:
        logger.error("Непредвиденная ошибка при отправке сообщения: %s", exc)
        return False


async def _run_position_checks(
    bot: Bot,
    pair: str,
    h: float,
    l: float,
    current_price: float,
) -> None:
    """BE → trailing → SL/TP → pending fill → BE → trailing → SL/TP (post-fill)."""
    for trade in portfolio.check_breakeven(pair, h, l):
        await send_msg(bot, fmt_sl_to_breakeven(trade))
    portfolio.check_trailing_stop(pair, h, l)
    for closed in portfolio.check_sl_tp(pair, h, l):
        await send_msg(bot, fmt_trade_closed(closed, portfolio.get_equity()))

    triggered, _ = portfolio.check_pending_orders(
        pair, h, l, candle_close=current_price, require_bounce=BOUNCE_CONFIRM,
    )
    if triggered:
        for trade in triggered:
            await send_msg(bot, fmt_trade_opened(trade))
        for trade in portfolio.check_breakeven(pair, h, l):
            await send_msg(bot, fmt_sl_to_breakeven(trade))
        portfolio.check_trailing_stop(pair, h, l)
        for closed in portfolio.check_sl_tp(pair, h, l):
            await send_msg(bot, fmt_trade_closed(closed, portfolio.get_equity()))


# ============================================================
# АНАЛИЗ ПАРЫ (с бумажной торговлей)
# ============================================================

class _PairAnalysis(NamedTuple):
    levels:        List[Dict]
    signals:       List[Dict]
    htf_trend:     Optional[str]
    htf_ema:       Optional[float]
    adx_val:       Optional[float]
    rsi_val:       float
    atr_ratio:     float
    regime:        str
    htf_structure: Optional[str]
    htf_rsi:       Optional[float]
    macro_trend:   Optional[str] = None


async def _compute_analysis(
    df: pd.DataFrame,
    pair: str,
    client: ccxt.mexc,
    current_price: float,
) -> _PairAnalysis:
    levels    = find_support_resistance(df)
    ema_lvls  = _calc_ema_levels(df, EMA_CONFLUENCE_PERIODS)
    fvg_zones = _find_fvg_zones(df, FVG_LOOKBACK, FVG_MIN_GAP_PCT)
    _add_confluence_scores(levels, ema_lvls, fvg_zones, TOLERANCE_PERCENT)

    htf_trend, htf_ema, htf_sr, htf_structure, htf_rsi, htf_below_ema, macro_trend = (
        await asyncio.to_thread(_fetch_htf_confluence, client, pair)
    )
    if htf_sr:
        _mark_htf_confirmed(levels, htf_sr, TOLERANCE_PERCENT)
    adx_val, pdi_val, ndi_val = _calc_adx(df, ADX_PERIOD)

    rsi_series = _calc_rsi_series(df, RSI_PERIOD)
    rsi_raw    = float(rsi_series.iloc[-1])
    rsi_val    = rsi_raw if not pd.isna(rsi_raw) else 50.0
    atr_ratio  = _calc_atr_ratio(df, ATR_PERIOD)
    regime     = _detect_regime(rsi_series, atr_ratio)
    crash_low  = _find_crash_low(df, CRASH_LOW_LOOKBACK) if regime == "RECOVERY"   else None
    pump_high  = _find_pump_high(df, PUMP_HIGH_LOOKBACK) if regime == "CORRECTION" else None

    logger.info(
        "%s | HTF(%s)=%s  структура=%s  RSI4h=%s  ADX=%s  RSI=%.1f  ATR×=%.1f  режим=%s  макро=%s",
        pair, HTF_TIMEFRAME, htf_trend or "N/A",
        htf_structure or "N/A",
        f"{htf_rsi:.1f}" if htf_rsi is not None else "N/A",
        f"{adx_val:.1f}" if adx_val is not None else "N/A",
        rsi_val, atr_ratio, regime,
        macro_trend or "N/A",
    )

    signal_levels = (
        [lvl for lvl in levels if lvl.get("htf_confirmed")]
        if HTF_SR_REQUIRE_CONFIRM and htf_sr
        else levels
    )
    signals = find_entry_signals(
        df, signal_levels, current_price,
        htf_trend=htf_trend,
        adx_val=adx_val,
        regime=regime,
        crash_low=crash_low,
        pump_high=pump_high,
        htf_structure=htf_structure,
        rsi_series=rsi_series,
        htf_rsi=htf_rsi,
        atr_ratio=atr_ratio,
        htf_below_ema=htf_below_ema,
        macro_trend=macro_trend,
        pdi_val=pdi_val,
        ndi_val=ndi_val,
    )
    return _PairAnalysis(
        levels=levels, signals=signals,
        htf_trend=htf_trend, htf_ema=htf_ema,
        adx_val=adx_val, rsi_val=rsi_val,
        atr_ratio=atr_ratio, regime=regime,
        htf_structure=htf_structure,
        htf_rsi=htf_rsi,
        macro_trend=macro_trend,
    )


def _apply_orders(pair: str, signals: List[Dict]) -> None:
    best: Dict[str, Optional[Dict]] = {"LONG": None, "SHORT": None}
    for sig in sorted(signals, key=lambda s: (
        not s["level"].get("htf_confirmed", False),
        -s["level"].get("confluence_score", 0),
        s["distance_percent"],
    )):
        if best[sig["direction"]] is None:
            best[sig["direction"]] = sig

    open_pairs = [t["pair"] for t in portfolio.open_trades if t["pair"] != pair]
    if not _check_correlation(pair, open_pairs):
        logger.info(
            "Пропуск ордеров %s: высокая корреляция с открытыми позициями (>%.2f)",
            pair, CORR_MAX,
        )
        best = {"LONG": None, "SHORT": None}

    tolerance = TOLERANCE_PERCENT / 100
    for order in list(portfolio.pending_orders):
        if order["pair"] != pair:
            continue
        b = best.get(order["direction"])
        same_level = (
            b is not None
            and abs(b["level"]["price"] - order["entry_price"]) / order["entry_price"] <= tolerance
        )
        if not same_level:
            portfolio.cancel_order(order)

    for sig in best.values():
        if sig is None or sig["tp"] is None:
            continue
        portfolio.create_pending_order(
            pair=pair,
            direction=sig["direction"],
            entry_price=sig["level"]["price"],
            sl=sig["sl"],
            tp=sig["tp"],
            risk_mult=sig.get("quality_mult", 1.0),
        )


async def analyze_pair(
    client: ccxt.mexc,
    bot: Bot,
    pair: str,
) -> None:
    logger.info("Анализ: %s", pair)

    df = await asyncio.to_thread(fetch_ohlcv, client, pair, TIMEFRAME, CANDLES_LIMIT)
    if df is None or df.empty:
        logger.warning("Пропускаем %s (нет данных)", pair)
        return

    _returns_cache[pair] = (
        df.set_index("timestamp")["close"].pct_change().dropna().tail(CORR_LOOKBACK)
    )

    try:
        ticker = await asyncio.to_thread(client.fetch_ticker, pair)
        current_price = float(ticker["last"])
    except Exception:
        current_price = float(df["close"].iloc[-1])
    portfolio.update_price(pair, current_price)

    h, l = await asyncio.to_thread(
        _fetch_hl, client, pair, current_price,
        max(float(df["high"].iloc[-2]), float(df["high"].iloc[-1])),
        min(float(df["low"].iloc[-2]),  float(df["low"].iloc[-1])),
    )
    await _run_position_checks(bot, pair, h, l, current_price)

    analysis = await _compute_analysis(df, pair, client, current_price)

    portfolio.save_report(pair, fmt_analysis(
        pair, TIMEFRAME, current_price, analysis.levels, analysis.signals,
        htf_trend=analysis.htf_trend, htf_ema=analysis.htf_ema,
        adx=round(analysis.adx_val, 1) if analysis.adx_val is not None else None,
        rsi=round(analysis.rsi_val, 1),
        atr_ratio=analysis.atr_ratio,
        regime=analysis.regime,
        htf_structure=analysis.htf_structure,
    ))

    _apply_orders(pair, analysis.signals)


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
        await update.message.reply_text(
            fmt_stats(portfolio), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
        )
    elif "Лог" in txt:
        await update.message.reply_text(
            fmt_log(portfolio), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
        )
    elif "Позиции" in txt:
        prices = _fetch_current_prices(
            context.bot_data.get("client"), portfolio.open_trades,
        )
        await update.message.reply_text(
            fmt_open_trades(portfolio, prices), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
        )
    elif "Ордера" in txt:
        await update.message.reply_text(
            fmt_pending_orders(portfolio), parse_mode=ParseMode.HTML, reply_markup=REPLY_KB,
        )
    elif "Монеты" in txt:
        pairs = context.bot_data.get("pairs", TRADING_PAIRS)
        await update.message.reply_text(
            "🪙 Выбери монету для просмотра последнего отчёта:",
            reply_markup=_pairs_inline_kb(pairs),
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("report_"):
        pair   = data[len("report_"):]
        report = portfolio.get_report(pair)
        if report:
            text  = report["text"]
            saved = (report.get("saved_at") or "")[:16]
            await query.message.reply_text(
                f"{text}\n\n<i>Сохранён: {saved}</i>",
                parse_mode=ParseMode.HTML,
            )
        else:
            await query.message.reply_text(
                f"⚠️ Отчёт по <b>{pair}</b> пока не готов — подожди первого цикла анализа.",
                parse_mode=ParseMode.HTML,
            )


# ============================================================
# JOB: БЫСТРАЯ ПРОВЕРКА SL/TP (каждые N минут)
# ============================================================

async def fast_check_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data.get("client")
    bot    = context.bot
    if not client:
        return

    pairs = (
        {t["pair"] for t in portfolio.open_trades} |
        {o["pair"] for o in portfolio.pending_orders}
    )
    if not pairs:
        return

    logger.info("=== Быстрая проверка SL/TP (%d пар) ===", len(pairs))
    for pair in pairs:
        try:
            try:
                ticker = await asyncio.to_thread(client.fetch_ticker, pair)
                current_price = float(ticker["last"])
            except Exception:
                continue
            portfolio.update_price(pair, current_price)

            h, l = await asyncio.to_thread(_fetch_hl, client, pair, current_price)
            await _run_position_checks(bot, pair, h, l, current_price)

        except Exception as exc:
            logger.error("fast_check %s: %s", pair, exc)


# ============================================================
# JOB: ПЕРИОДИЧЕСКИЙ АНАЛИЗ
# ============================================================

async def analysis_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data["client"]
    bot    = context.bot

    pairs = (
        await asyncio.to_thread(fetch_top_pairs, client, AUTO_TOP_PAIRS)
        if AUTO_TOP_PAIRS > 0 else TRADING_PAIRS
    )
    context.bot_data["pairs"] = pairs
    for stale in list(_returns_cache.keys() - set(pairs)):
        del _returns_cache[stale]
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
        f"Режим: анализ каждые <b>{mode}</b>  •  SL/TP каждые <b>{SL_TP_CHECK_INTERVAL_MIN} мин</b>\n\n"
        f"💰 Equity: <b>${portfolio.get_equity():,.2f}</b>  "
        f"(из ${portfolio.initial_balance:,.2f})\n"
        f"{'Риск' if FIXED_RISK_MODE else 'Маржа'}/сделку: "
        f"<b>{RISK_PER_TRADE_PERCENT if FIXED_RISK_MODE else TRADE_SIZE_PERCENT}%</b>  •  "
        f"Плечо: <b>{LEVERAGE}×</b>  •  Max: {MAX_OPEN_TRADES}\n"
        f"Комиссия: maker <b>{COMMISSION_MAKER*100:.2f}%</b> / taker <b>{COMMISSION_TAKER*100:.3f}%</b>  •  "
        f"Безубыток при <b>{int(BREAKEVEN_THRESHOLD*100)}%</b> пути к TP\n"
        f"Трейлинг: {'✓ ×' + str(TRAILING_MULT) if TRAILING_STOP else '✗'}  •  "
        f"Отскок: {'✓' if BOUNCE_CONFIRM else '✗'}\n"
        f"SL: ATR×{SL_ATR_MULT}  •  TP: ближ. уровень (мин ATR×{TP_ATR_MIN_MULT})  •  Min R:R 1:{MIN_RR}\n"
        f"ATR period: {ATR_PERIOD}  •  EMA{EMA_PERIOD}  •  Entry proximity: {ENTRY_PROXIMITY_PERCENT}%\n"
        f"Тренд-фильтры: {HTF_TIMEFRAME} EMA{HTF_EMA_PERIOD}  •  ADX≥{ADX_MIN}  •  "
        f"HTF S/R: {'✓' if HTF_SR_CANDLES > 0 else '✗'}"
        f"{'  (только HTF)' if HTF_SR_REQUIRE_CONFIRM else ''}\n\n"
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

    check_sec = SL_TP_CHECK_INTERVAL_MIN * 60
    app.job_queue.run_repeating(fast_check_job, interval=check_sec, first=30)

    interval_sec = int(RUN_INTERVAL_HOURS * 3600)
    if interval_sec > 0:
        app.job_queue.run_repeating(analysis_job, interval=interval_sec, first=15)
    else:
        app.job_queue.run_once(analysis_job, when=15)

    logger.info(
        "Бот запущен: анализ каждые %.1fч, SL/TP проверка каждые %d мин.",
        RUN_INTERVAL_HOURS, SL_TP_CHECK_INTERVAL_MIN,
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    except Exception as exc:
        logger.error("Фатальная ошибка: %s", exc)
        sys.exit(1)
