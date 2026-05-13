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
CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", "150"))

# Минимальное количество касаний уровня / Minimum touches for a valid level
MIN_TOUCHES = int(os.getenv("MIN_TOUCHES", "4"))

# Толерантность при поиске касаний (в процентах) / Touch tolerance (percent)
TOLERANCE_PERCENT = float(os.getenv("TOLERANCE_PERCENT", "0.5"))

# Размер окна для поиска локальных экстремумов / Local extrema window size
EXTREMA_WINDOW = int(os.getenv("EXTREMA_WINDOW", "8"))

# Топ N самых сильных уровней / Top N strongest levels
TOP_N_LEVELS = int(os.getenv("TOP_N_LEVELS", "5"))

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
) -> List[float]:
    """
    Найти локальные максимумы/минимумы в серии.
    Find local maxima/minima within a rolling window.

    kind: 'max' -> локальные максимумы (сопротивление)
          'min' -> локальные минимумы (поддержка)
    """
    values = series.values
    extrema = []
    for i in range(window, len(values) - window):
        segment = values[i - window:i + window + 1]
        center = values[i]
        if kind == "max" and center == segment.max():
            extrema.append(float(center))
        elif kind == "min" and center == segment.min():
            extrema.append(float(center))
    return extrema


def _count_touches(
    level: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tolerance: float,
) -> int:
    """
    Посчитать количество касаний цены к уровню.
    Count how many times price touched a level (within tolerance).
    """
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    # Касание = цена high или low оказалась внутри коридора уровня
    touched_high = np.sum((highs <= upper) & (highs >= lower))
    touched_low = np.sum((lows <= upper) & (lows >= lower))
    return int(touched_high + touched_low)


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
) -> List[Dict]:
    """
    Найти уровни поддержки и сопротивления.
    Find support and resistance levels.

    Алгоритм:
    1. Находим локальные максимумы (потенциальные сопротивления)
       и локальные минимумы (потенциальные поддержки).
    2. Для каждого кандидата считаем количество касаний.
    3. Отфильтровываем уровни с touches < MIN_TOUCHES.
    4. Удаляем дубликаты (близкие уровни).
    5. Возвращаем топ-N самых сильных.
    """
    tolerance = tolerance_percent / 100.0

    highs = df["high"].values
    lows = df["low"].values

    # 1. Локальные экстремумы
    resistance_candidates = _find_local_extrema(df["high"], window, "max")
    support_candidates = _find_local_extrema(df["low"], window, "min")

    logger.info(
        "Кандидатов: сопротивление=%d, поддержка=%d",
        len(resistance_candidates),
        len(support_candidates),
    )

    levels: List[Dict] = []

    # 2. Считаем касания для каждого кандидата сопротивления
    for price in resistance_candidates:
        touches = _count_touches(price, highs, lows, tolerance)
        if touches >= min_touches:
            levels.append({
                "price": price,
                "type": "RESISTANCE",
                "touches": touches,
            })

    # 2b. Считаем касания для кандидатов поддержки
    for price in support_candidates:
        touches = _count_touches(price, highs, lows, tolerance)
        if touches >= min_touches:
            levels.append({
                "price": price,
                "type": "SUPPORT",
                "touches": touches,
            })

    # 3. Удаляем дубликаты
    levels = remove_duplicates(levels, tolerance)

    # 4. Берём топ-N
    levels = sorted(levels, key=lambda x: x["touches"], reverse=True)[:top_n]

    # 5. Сортируем для красивого вывода: по цене (по убыванию)
    levels = sorted(levels, key=lambda x: x["price"], reverse=True)

    logger.info("Найдено уровней: %d", len(levels))
    return levels


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
) -> str:
    """
    Красиво форматирует HTML-сообщение для Telegram.
    Format a nice HTML message for Telegram.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"📊 <b>{pair}</b>  •  <code>{timeframe}</code>",
        f"💰 Текущая цена: <b>{_format_price(current_price)}</b>",
        f"🕒 {now}",
        "",
    ]

    if not levels:
        lines.append("⚠️ <i>Уровни не найдены (недостаточно касаний).</i>")
        return "\n".join(lines)

    lines.append("<b>Найденные уровни:</b>")
    for lvl in levels:
        emoji = "🔴" if lvl["type"] == "RESISTANCE" else "🟢"
        ru_type = "Сопротивление" if lvl["type"] == "RESISTANCE" else "Поддержка"
        distance = (lvl["price"] - current_price) / current_price * 100
        sign = "+" if distance >= 0 else ""
        lines.append(
            f"{emoji} <b>{_format_price(lvl['price'])}</b> "
            f"— {ru_type} "
            f"(касаний: <b>{lvl['touches']}</b>, "
            f"расстояние: <b>{sign}{distance:.2f}%</b>)"
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

    # Логируем результат
    if levels:
        for lvl in levels:
            logger.info(
                "  %s @ %s (touches=%d)",
                lvl["type"],
                _format_price(lvl["price"]),
                lvl["touches"],
            )
    else:
        logger.info("Для %s уровни не найдены.", pair)

    message = format_message(pair, TIMEFRAME, current_price, levels)
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
async def run_once(client: ccxt.mexc, bot: Bot) -> None:
    """
    Один полный прогон по всем парам.
    One full pass through all trading pairs.
    """
    for idx, pair in enumerate(TRADING_PAIRS):
        try:
            await analyze_pair(client, bot, pair)
        except Exception as exc:
            logger.error("Ошибка при анализе %s: %s", pair, exc)
            await send_telegram_message(
                bot,
                f"❌ <b>{pair}</b>: ошибка анализа — <code>{exc}</code>",
            )
        # Небольшая пауза между парами, чтобы не упереться в rate limit
        if idx < len(TRADING_PAIRS) - 1:
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

    start_msg = (
        f"🚀 <b>Signaler запущен</b>\n"
        f"Пар: {len(TRADING_PAIRS)}  •  TF: <code>{TIMEFRAME}</code>\n"
        f"Режим: <b>{mode}</b>\n"
        f"Min touches: {MIN_TOUCHES}  •  Tolerance: {TOLERANCE_PERCENT}%"
    )
    await send_telegram_message(bot, start_msg)

    # Основной цикл / Main loop
    iteration = 0
    while True:
        iteration += 1
        logger.info("========== Итерация #%d ==========", iteration)

        await run_once(client, bot)

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
