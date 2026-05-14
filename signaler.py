"""
MEXC Support & Resistance Signaler
====================================
Скрипт для поиска уровней поддержки и сопротивления на бирже MEXC
и отправки сигналов в Telegram.

Script for finding support and resistance levels on MEXC exchange
and sending signals to Telegram.

Установка зависимостей / Install dependencies:
    pip3 install -r requirements.txt

Настройка / Setup:
    cp .env.example .env  и заполни ключи в .env

Запуск / Run:
    python3 signaler.py
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import ccxt
import numpy as np
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Опциональная загрузка .env (если установлен python-dotenv) /
# Optional .env loading (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Если python-dotenv не установлен, читаем напрямую из окружения
    pass


# ============================================================
# КОНФИГУРАЦИЯ / CONFIGURATION
# ============================================================
# Секреты загружаются из переменных окружения (см. .env.example) /
# Secrets are loaded from environment variables (see .env.example)

# --- Секретные данные (из .env) / Secrets (from .env) ---
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Несекретные параметры / Non-sensitive parameters ---
# Можно переопределить через env или менять прямо здесь
# Can be overridden via env or edited here directly

# Торговые пары для анализа (через запятую в .env) /
# Trading pairs to analyze (comma-separated in .env)
TRADING_PAIRS = [
    p.strip() for p in os.getenv("TRADING_PAIRS", "BTC/USDT,ETH/USDT").split(",")
    if p.strip()
]

# Таймфрейм / Timeframe (1m, 5m, 15m, 1h, 4h, 1d, ...)
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

# Количество свечей для анализа / Number of candles to analyze
CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", "250"))

# Минимальное количество касаний уровня / Minimum touches for a valid level
MIN_TOUCHES = int(os.getenv("MIN_TOUCHES", "5"))

# Толерантность при поиске касаний (в процентах) / Touch tolerance (percent)
TOLERANCE_PERCENT = float(os.getenv("TOLERANCE_PERCENT", "0.8"))

# Размер окна для поиска локальных экстремумов / Local extrema window size
EXTREMA_WINDOW = int(os.getenv("EXTREMA_WINDOW", "8"))

# Топ N самых сильных уровней / Top N strongest levels
TOP_N_LEVELS = int(os.getenv("TOP_N_LEVELS", "5"))

# Минимальный промежуток между касаниями (в свечах) / Min candles between consecutive touches
MIN_TOUCH_SPACING = int(os.getenv("MIN_TOUCH_SPACING", "3"))

# Минимальный возраст уровня (свечей от конца данных) / Min level age (candles from data end)
LEVEL_AGE_MIN_CANDLES = int(os.getenv("LEVEL_AGE_MIN_CANDLES", "10"))

# Множитель среднего объёма для засчитывания касания (0 = фильтр отключён) /
# Avg-volume multiplier required at touch (0 = disabled)
VOLUME_TOUCH_MULTIPLIER = float(os.getenv("VOLUME_TOUCH_MULTIPLIER", "1.2"))

# Требовать ретест после пробоя / Require retest after breakout
REQUIRE_RETEST = os.getenv("REQUIRE_RETEST", "false").lower() == "true"

# --- Точки входа / Entry signals ---
# Порог близости к уровню для генерации сигнала (%) / Proximity to level for entry signal (%)
ENTRY_PROXIMITY_PERCENT = float(os.getenv("ENTRY_PROXIMITY_PERCENT", "0.5"))

# Период EMA для определения тренда / EMA period for trend filter
EMA_PERIOD = int(os.getenv("EMA_PERIOD", "200"))

# Автоматически брать топ N пар по объёму (0 = использовать TRADING_PAIRS) /
# Auto-fetch top N pairs by 24h volume (0 = use TRADING_PAIRS)
AUTO_TOP_PAIRS = int(os.getenv("AUTO_TOP_PAIRS", "10"))

# Задержка между анализом пар (сек) / Delay between pairs (sec)
DELAY_BETWEEN_PAIRS = int(os.getenv("DELAY_BETWEEN_PAIRS", "2"))

# Интервал между полными прогонами (часы) / Interval between full runs (hours)
# 0 = одноразовый запуск / 0 = run once and exit
RUN_INTERVAL_HOURS = float(os.getenv("RUN_INTERVAL_HOURS", "1"))

# ============================================================
# ЛОГИРОВАНИЕ / LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signaler")


# ============================================================
# ПРОВЕРКА КОНФИГА / CONFIG VALIDATION
# ============================================================
def validate_config() -> None:
    """Проверка, что все обязательные поля заполнены."""
    missing = []
    if not MEXC_API_KEY:
        missing.append("MEXC_API_KEY")
    if not MEXC_SECRET_KEY:
        missing.append("MEXC_SECRET_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if not TRADING_PAIRS:
        missing.append("TRADING_PAIRS")

    if missing:
        logger.error(
            "Не заполнены обязательные поля конфига: %s",
            ", ".join(missing),
        )
        sys.exit(1)

    logger.info("Конфиг проверен: все поля заполнены.")


# ============================================================
# MEXC CLIENT
# ============================================================
def get_mexc_client() -> ccxt.mexc:
    """
    Создать и проверить подключение к MEXC.
    Create and verify MEXC connection.
    """
    logger.info("Подключение к MEXC...")
    try:
        client = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_SECRET_KEY,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # Загружаем рынки чтобы убедиться в работоспособности
        client.load_markets()
        logger.info("Подключение к MEXC установлено. Рынков: %d", len(client.markets))
        return client
    except ccxt.AuthenticationError as exc:
        logger.error("Ошибка авторизации MEXC: %s", exc)
        raise
    except Exception as exc:
        logger.error("Ошибка подключения к MEXC: %s", exc)
        raise


def fetch_top_pairs(client: ccxt.mexc, n: int = 10) -> List[str]:
    """Получить топ-N USDT пар по объёму торгов за 24ч."""
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
    """
    Скачать исторические свечи (OHLCV).
    Fetch historical OHLCV candles.

    Returns DataFrame с колонками: timestamp, open, high, low, close, volume.
    """
    logger.info("Загрузка %d свечей %s для %s...", limit, timeframe, symbol)
    try:
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            logger.warning("Пустой ответ для %s", symbol)
            return None

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # Валидация: нужно достаточно данных для анализа
        if len(df) < EXTREMA_WINDOW * 2 + 1:
            logger.warning("Слишком мало свечей для %s: %d", symbol, len(df))
            return None

        logger.info("Получено %d свечей для %s", len(df), symbol)
        return df
    except ccxt.BadSymbol:
        logger.error("Пара %s не найдена на MEXC", symbol)
        return None
    except Exception as exc:
        logger.error("Ошибка при загрузке OHLCV для %s: %s", symbol, exc)
        return None


# ============================================================
# ПОИСК УРОВНЕЙ / LEVEL DETECTION
# ============================================================
def _find_local_extrema(
    series: pd.Series,
    window: int,
    kind: str,
) -> List[tuple]:
    """
    Найти локальные максимумы/минимумы в серии.
    Find local maxima/minima within a rolling window.

    Returns list of (price, index) tuples.
    kind: 'max' -> локальные максимумы (сопротивление)
          'min' -> локальные минимумы (поддержка)
    """
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
    """
    Посчитать количество касаний цены к уровню.
    Count how many times price touched a level (within tolerance).

    Критерии качества касания:
    - min_spacing: минимальный промежуток между касаниями (в свечах)
    - volume_multiplier: засчитывается только при объёме >= avg * multiplier
    """
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    avg_volume = df["volume"].mean()

    touch_indices = []
    for i in range(len(df)):
        high = df["high"].iat[i]
        low = df["low"].iat[i]
        if not ((lower <= high <= upper) or (lower <= low <= upper)):
            continue
        if volume_multiplier > 0 and df["volume"].iat[i] < avg_volume * volume_multiplier:
            continue
        touch_indices.append(i)

    if not touch_indices:
        return 0

    if min_spacing <= 0:
        return len(touch_indices)

    # Оставляем только касания, разделённые минимальным промежутком
    filtered = [touch_indices[0]]
    for idx in touch_indices[1:]:
        if idx - filtered[-1] >= min_spacing:
            filtered.append(idx)

    return len(filtered)


def _has_retest_after_breakout(
    level: float,
    df: pd.DataFrame,
    tolerance: float,
    level_type: str,
) -> bool:
    """
    Проверить, был ли ретест уровня после его пробоя.
    Check if there was a retest of the level after a breakout.

    Пробой: свеча закрывается за пределами зоны уровня.
    Ретест: после пробоя цена возвращается к зоне уровня.
    """
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

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
            # Ложный пробой — цена вернулась на исходную сторону
            if level_type == "SUPPORT" and closes[i] > upper:
                in_breakout = False
            elif level_type == "RESISTANCE" and closes[i] < lower:
                in_breakout = False

    return False


def remove_duplicates(
    levels: List[Dict],
    tolerance: float,
) -> List[Dict]:
    """
    Удалить близкие дубликаты уровней.
    Remove near-duplicate levels (within tolerance).

    Сортируем по силе (touches), оставляем самые сильные.
    """
    if not levels:
        return []

    # Сортируем по силе (по убыванию)
    sorted_levels = sorted(levels, key=lambda x: x["touches"], reverse=True)
    unique: List[Dict] = []

    for level in sorted_levels:
        is_duplicate = False
        for kept in unique:
            # Если уровень слишком близко к уже сохранённому того же типа — пропускаем
            if kept["type"] != level["type"]:
                continue
            diff = abs(level["price"] - kept["price"]) / kept["price"]
            if diff < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
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
    """
    Найти уровни поддержки и сопротивления.
    Find support and resistance levels.

    Алгоритм:
    1. Находим локальные максимумы/минимумы (кандидаты в уровни).
    2. Фильтруем слишком молодые уровни (level_age_min).
    3. Считаем касания с учётом промежутка во времени и объёма.
    4. Отфильтровываем уровни с touches < min_touches.
    5. Опционально: требуем ретест после пробоя (require_retest).
    6. Удаляем дубликаты, берём топ-N.
    """
    tolerance = tolerance_percent / 100.0
    total_candles = len(df)

    resistance_candidates = _find_local_extrema(df["high"], window, "max")
    support_candidates = _find_local_extrema(df["low"], window, "min")

    logger.info(
        "Кандидатов: сопротивление=%d, поддержка=%d",
        len(resistance_candidates),
        len(support_candidates),
    )

    levels: List[Dict] = []

    for candidates, level_type in [
        (resistance_candidates, "RESISTANCE"),
        (support_candidates, "SUPPORT"),
    ]:
        for price, idx in candidates:
            # Критерий возраста: уровень должен быть сформирован достаточно давно
            age = total_candles - 1 - idx
            if age < level_age_min:
                continue

            touches = _count_touches(
                price, df, tolerance,
                min_spacing=min_touch_spacing,
                volume_multiplier=volume_multiplier,
            )
            if touches < min_touches:
                continue

            has_retest = _has_retest_after_breakout(price, df, tolerance, level_type)
            if require_retest and not has_retest:
                continue

            levels.append({
                "price": price,
                "type": level_type,
                "touches": touches,
                "has_retest": has_retest,
            })

    levels = remove_duplicates(levels, tolerance)
    levels = sorted(levels, key=lambda x: x["touches"], reverse=True)[:top_n]
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)

    logger.info("Найдено уровней: %d", len(levels))
    return levels


# ============================================================
# ТОЧКИ ВХОДА / ENTRY SIGNALS
# ============================================================
def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _detect_candle_pattern(df: pd.DataFrame) -> Optional[str]:
    """Определить паттерн на последней завершённой свече."""
    if len(df) < 3:
        return None

    c = df.iloc[-2]     # последняя завершённая (текущая ещё формируется)
    prev = df.iloc[-3]

    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    total_range = h - l
    if total_range == 0:
        return None

    body = abs(cl - o)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_ratio = body / total_range

    # Молот (бычий пин-бар)
    if body_ratio < 0.3 and lower_wick / total_range > 0.6 and lower_wick > 2 * upper_wick:
        return "молот 🔨"

    # Падающая звезда (медвежий пин-бар)
    if body_ratio < 0.3 and upper_wick / total_range > 0.6 and upper_wick > 2 * lower_wick:
        return "падающая звезда ⭐"

    p_o, p_cl = float(prev["open"]), float(prev["close"])

    # Бычье поглощение
    if p_cl < p_o and cl > o and o <= p_cl and cl >= p_o:
        return "бычье поглощение 🕯"

    # Медвежье поглощение
    if p_cl > p_o and cl < o and o >= p_cl and cl <= p_o:
        return "медвежье поглощение 🕯"

    return None


def find_entry_signals(
    df: pd.DataFrame,
    levels: List[Dict],
    current_price: float,
    proximity_percent: float = ENTRY_PROXIMITY_PERCENT,
    ema_period: int = EMA_PERIOD,
    tolerance_percent: float = TOLERANCE_PERCENT,
) -> List[Dict]:
    """
    Найти точки входа вблизи уровней.

    Критерии (все три должны совпасть):
    1. Цена в пределах proximity_percent% от уровня.
    2. Тренд по EMA: LONG только выше EMA, SHORT только ниже.
    3. Паттерн свечи (необязателен, но отображается если есть).

    Каждый сигнал содержит SL, TP и соотношение R:R.
    """
    if ema_period >= len(df):
        logger.warning("Недостаточно свечей для EMA%d (%d свечей)", ema_period, len(df))
        return []

    proximity = proximity_percent / 100.0
    tolerance = tolerance_percent / 100.0

    ema_val = float(_calc_ema(df["close"], ema_period).iloc[-1])
    ema_trend = "UP" if current_price > ema_val else "DOWN"
    pattern = _detect_candle_pattern(df)

    signals = []
    for lvl in levels:
        level_price = lvl["price"]
        distance = abs(current_price - level_price) / level_price
        if distance > proximity:
            continue

        if lvl["type"] == "SUPPORT" and ema_trend == "UP":
            direction = "LONG"
        elif lvl["type"] == "RESISTANCE" and ema_trend == "DOWN":
            direction = "SHORT"
        else:
            continue

        sl = (level_price * (1 - tolerance * 2) if direction == "LONG"
              else level_price * (1 + tolerance * 2))

        if direction == "LONG":
            targets = [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > current_price]
            tp = min(targets, key=lambda x: x["price"])["price"] if targets else None
        else:
            targets = [l for l in levels if l["type"] == "SUPPORT" and l["price"] < current_price]
            tp = max(targets, key=lambda x: x["price"])["price"] if targets else None

        risk = abs(current_price - sl)
        reward = abs(tp - current_price) if tp is not None else None
        rr = round(reward / risk, 1) if reward and risk > 0 else None

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
# TELEGRAM
# ============================================================
async def send_telegram_message(bot: Bot, text: str) -> bool:
    """
    Отправить HTML-сообщение в Telegram.
    Send an HTML message to Telegram chat.
    """
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Сообщение отправлено в Telegram")
        return True
    except TelegramError as exc:
        logger.error("Ошибка Telegram: %s", exc)
        return False
    except Exception as exc:
        logger.error("Неожиданная ошибка Telegram: %s", exc)
        return False


# ============================================================
# ФОРМАТИРОВАНИЕ СООБЩЕНИЯ / MESSAGE FORMATTING
# ============================================================
def _format_price(price: float) -> str:
    """Форматирование цены с адаптивной точностью."""
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.8f}"


def format_message(
    pair: str,
    timeframe: str,
    current_price: float,
    levels: List[Dict],
    signals: List[Dict],
) -> str:
    """Форматирует HTML-сообщение для Telegram."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"📊 <b>{pair}</b>  •  <code>{timeframe}</code>",
        f"💰 Текущая цена: <b>{_format_price(current_price)}</b>",
        f"🕒 {now}",
        "",
    ]

    # --- Блок сигналов входа ---
    if signals:
        lines.append("🎯 <b>СИГНАЛЫ ВХОДА:</b>")
        for sig in signals:
            lvl = sig["level"]
            dir_emoji = "📈" if sig["direction"] == "LONG" else "📉"
            trend_arrow = "↑" if sig["ema_trend"] == "UP" else "↓"
            lines.append(
                f"{dir_emoji} <b>{sig['direction']}</b>  вблизи {_format_price(lvl['price'])}"
                f"  <i>({sig['distance_percent']}% от цены)</i>"
            )
            lines.append(
                f"   Тренд: {trend_arrow} EMA{EMA_PERIOD} = {_format_price(sig['ema'])}"
            )
            if sig["pattern"]:
                lines.append(f"   Паттерн: {sig['pattern']}")
            tp_str = _format_price(sig["tp"]) if sig["tp"] else "—"
            rr_str = f"  •  R:R 1:{sig['rr']}" if sig["rr"] else ""
            lines.append(
                f"   SL: <b>{_format_price(sig['sl'])}</b>"
                f"  •  TP: <b>{tp_str}</b>{rr_str}"
            )
        lines.append("")
    else:
        lines.append("⏳ <i>Сигналов нет — цена далеко от уровней.</i>")
        lines.append("")

    # --- Блок уровней ---
    if not levels:
        lines.append("⚠️ <i>Уровни не найдены (недостаточно касаний).</i>")
        return "\n".join(lines)

    lines.append("<b>Уровни поддержки и сопротивления:</b>")
    for lvl in levels:
        emoji = "🔴" if lvl["type"] == "RESISTANCE" else "🟢"
        ru_type = "Сопр." if lvl["type"] == "RESISTANCE" else "Подд."
        distance = (lvl["price"] - current_price) / current_price * 100
        sign = "+" if distance >= 0 else ""
        retest_mark = " ✅" if lvl.get("has_retest") else ""
        lines.append(
            f"{emoji} <b>{_format_price(lvl['price'])}</b> — {ru_type} "
            f"(касаний: <b>{lvl['touches']}</b>, {sign}{distance:.2f}%{retest_mark})"
        )

    return "\n".join(lines)


# ============================================================
# АНАЛИЗ ПАРЫ / PAIR ANALYSIS
# ============================================================
async def analyze_pair(
    client: ccxt.mexc,
    bot: Bot,
    pair: str,
) -> None:
    """
    Полный цикл анализа одной торговой пары.
    Full analysis pipeline for one trading pair.
    """
    logger.info("=" * 50)
    logger.info("Анализ пары: %s", pair)

    df = fetch_ohlcv(client, pair, TIMEFRAME, CANDLES_LIMIT)
    if df is None or df.empty:
        logger.warning("Пропускаем %s (нет данных)", pair)
        await send_telegram_message(
            bot,
            f"⚠️ <b>{pair}</b>: не удалось загрузить данные.",
        )
        return

    current_price = float(df["close"].iloc[-1])
    logger.info("Текущая цена %s: %s", pair, _format_price(current_price))

    levels = find_support_resistance(df)
    signals = find_entry_signals(df, levels, current_price)

    for lvl in levels:
        logger.info("  %s @ %s (touches=%d)", lvl["type"], _format_price(lvl["price"]), lvl["touches"])
    for sig in signals:
        logger.info(
            "  СИГНАЛ %s @ %s → SL %s, TP %s, R:R %s",
            sig["direction"], _format_price(current_price),
            _format_price(sig["sl"]),
            _format_price(sig["tp"]) if sig["tp"] else "—",
            sig["rr"],
        )
    if not levels:
        logger.info("Для %s уровни не найдены.", pair)

    message = format_message(pair, TIMEFRAME, current_price, levels, signals)
    # Дублируем в консоль для удобства
    print("\n" + "-" * 50)
    print(message.replace("<b>", "").replace("</b>", "")
                 .replace("<code>", "").replace("</code>", "")
                 .replace("<i>", "").replace("</i>", ""))
    print("-" * 50 + "\n")

    await send_telegram_message(bot, message)


# ============================================================
# MAIN
# ============================================================
async def run_once(client: ccxt.mexc, bot: Bot, pairs: List[str]) -> None:
    """Один полный прогон по списку пар."""
    for idx, pair in enumerate(pairs):
        try:
            await analyze_pair(client, bot, pair)
        except Exception as exc:
            logger.error("Ошибка при анализе %s: %s", pair, exc)
            await send_telegram_message(
                bot,
                f"❌ <b>{pair}</b>: ошибка анализа — <code>{exc}</code>",
            )
        if idx < len(pairs) - 1:
            await asyncio.sleep(DELAY_BETWEEN_PAIRS)


async def run() -> None:
    """Главная асинхронная функция."""
    validate_config()

    try:
        client = get_mexc_client()
    except Exception:
        logger.error("Невозможно продолжить без подключения к MEXC.")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    # Стартовое сообщение (отправляется один раз при старте)
    try:
        me = await bot.get_me()
        logger.info("Telegram бот: @%s", me.username)
    except TelegramError as exc:
        logger.error("Не удалось подключиться к Telegram: %s", exc)
        return

    interval_seconds = int(RUN_INTERVAL_HOURS * 3600)
    mode = "одноразовый" if interval_seconds <= 0 else f"каждые {RUN_INTERVAL_HOURS}ч"
    pairs_mode = (f"топ-{AUTO_TOP_PAIRS} по объёму" if AUTO_TOP_PAIRS > 0
                  else f"{len(TRADING_PAIRS)} пар из конфига")

    start_msg = (
        f"🚀 <b>Signaler запущен</b>\n"
        f"Пары: {pairs_mode}  •  TF: <code>{TIMEFRAME}</code>\n"
        f"Режим: <b>{mode}</b>\n"
        f"Min touches: {MIN_TOUCHES}  •  Tolerance: {TOLERANCE_PERCENT}%\n"
        f"Touch spacing: {MIN_TOUCH_SPACING}св  •  Level age: {LEVEL_AGE_MIN_CANDLES}св\n"
        f"Volume ×{VOLUME_TOUCH_MULTIPLIER}  •  Retest: {'вкл' if REQUIRE_RETEST else 'выкл'}\n"
        f"Entry proximity: {ENTRY_PROXIMITY_PERCENT}%  •  EMA: {EMA_PERIOD}"
    )
    await send_telegram_message(bot, start_msg)

    iteration = 0
    while True:
        iteration += 1
        logger.info("========== Итерация #%d ==========", iteration)

        pairs = fetch_top_pairs(client, AUTO_TOP_PAIRS) if AUTO_TOP_PAIRS > 0 else TRADING_PAIRS
        await run_once(client, bot, pairs)

        if interval_seconds <= 0:
            logger.info("Одноразовый режим — завершаюсь.")
            break

        next_run = datetime.utcnow() + timedelta(seconds=interval_seconds)
        logger.info(
            "Следующая итерация через %.1fч (в %s UTC)",
            RUN_INTERVAL_HOURS,
            next_run.strftime("%Y-%m-%d %H:%M"),
        )
        await asyncio.sleep(interval_seconds)


def main() -> None:
    """Точка входа."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    except Exception as exc:
        logger.error("Фатальная ошибка: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
