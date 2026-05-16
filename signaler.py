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
from datetime import datetime
from typing import Dict, List, Optional

# load_dotenv MUST come before paper_trader import so DATA_DIR is applied to file paths
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


# ============================================================
# ВЕРСИЯ
# ============================================================
BOT_VERSION = "1.5.0"

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
CHECK_TIMEFRAME       = os.getenv("CHECK_TIMEFRAME", "5m")   # для проверки ордеров и SL/TP
CANDLES_LIMIT         = int(os.getenv("CANDLES_LIMIT", "500"))
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
ATR_PERIOD              = int(os.getenv("ATR_PERIOD",   "14"))   # период ATR
SL_ATR_MULT             = float(os.getenv("SL_ATR_MULT",   "1.5"))  # SL = entry ± ATR × mult
# TP_ATR_MIN_MULT должен быть >= SL_ATR_MULT × MIN_RR, иначе фоллбэк не пройдёт MIN_RR фильтр
# По умолчанию: 1.5 × 1.5 = 2.25 → ставим 2.5 для небольшого запаса
TP_ATR_MIN_MULT         = float(os.getenv("TP_ATR_MIN_MULT", "2.5"))  # TP не ближе ATR × mult
MIN_RR                  = float(os.getenv("MIN_RR",         "1.5"))  # пропустить если R:R < этого

# --- Многотаймфреймовый тренд (HTF) + ADX ---
HTF_TIMEFRAME  = os.getenv("HTF_TIMEFRAME",  "4h")   # старший TF для определения тренда
HTF_EMA_PERIOD = int(os.getenv("HTF_EMA_PERIOD", "50"))   # EMA на HTF
ADX_PERIOD     = int(os.getenv("ADX_PERIOD", "14"))    # период ADX
ADX_MIN        = float(os.getenv("ADX_MIN",  "20"))    # ниже = боковик, не торговать

# --- Краш-детектор и режим восстановления / Памп-детектор и режим коррекции ---
RSI_PERIOD            = int(os.getenv("RSI_PERIOD",            "14"))
# Падение
RSI_OVERSOLD          = float(os.getenv("RSI_OVERSOLD",        "30"))   # ниже = паника продаж
RSI_RECOVERY_MAX      = float(os.getenv("RSI_RECOVERY_MAX",    "47"))   # выше = отскок завершён
RSI_OVERSOLD_LOOKBACK = int(os.getenv("RSI_OVERSOLD_LOOKBACK", "12"))   # свечей назад
CRASH_LOW_LOOKBACK    = int(os.getenv("CRASH_LOW_LOOKBACK",    "24"))   # свечей для поиска лоя
CRASH_SL_BUFFER       = float(os.getenv("CRASH_SL_BUFFER",    "0.35"))  # % ниже лоя краша для SL
# Рост
RSI_OVERBOUGHT        = float(os.getenv("RSI_OVERBOUGHT",      "70"))   # выше = памп
RSI_CORRECTION_MIN    = float(os.getenv("RSI_CORRECTION_MIN",  "55"))   # ниже = коррекция завершена
RSI_OVERBOUGHT_LOOKBACK = int(os.getenv("RSI_OVERBOUGHT_LOOKBACK","12"))
PUMP_HIGH_LOOKBACK    = int(os.getenv("PUMP_HIGH_LOOKBACK",    "24"))   # свечей для поиска хая
PUMP_SL_BUFFER        = float(os.getenv("PUMP_SL_BUFFER",     "0.35"))  # % выше хая памп для SL
# Общий ATR-ratio порог (один для обоих направлений)
CRASH_PAUSE_ATR_RATIO = float(os.getenv("CRASH_PAUSE_ATR_RATIO", "2.5"))
CRASH_RESUME_ATR_RATIO= float(os.getenv("CRASH_RESUME_ATR_RATIO","2.0"))

DELAY_BETWEEN_PAIRS       = int(os.getenv("DELAY_BETWEEN_PAIRS", "2"))
# Полный анализ (уровни + сигналы) — рекомендуется 1h при TIMEFRAME=1h
RUN_INTERVAL_HOURS        = float(os.getenv("RUN_INTERVAL_HOURS", "1.0"))
# Быстрая проверка SL/TP и ожидающих ордеров
SL_TP_CHECK_INTERVAL_MIN  = int(os.getenv("SL_TP_CHECK_INTERVAL_MIN", "3"))
# Комиссия round-trip (открытие + закрытие), MEXC maker ≈ 0.1%
COMMISSION_RATE           = float(os.getenv("COMMISSION_RATE", "0.001"))
# Доля пути к TP для переноса SL в безубыток (0.5 = 50%)
BREAKEVEN_THRESHOLD       = float(os.getenv("BREAKEVEN_THRESHOLD", "0.5"))

# --- Трейлинг-стоп ---
# true = SL поджимается за ценой; trail_dist = оригинальный SL-дистанс × TRAILING_MULT
TRAILING_STOP  = os.getenv("TRAILING_STOP",  "true").lower() == "true"
TRAILING_MULT  = float(os.getenv("TRAILING_MULT",  "1.0"))

# --- Подтверждение отскока ---
# true = вход только если цена закрылась с нужной стороны уровня (закрытие > entry для LONG и т.д.)
BOUNCE_CONFIRM            = os.getenv("BOUNCE_CONFIRM", "true").lower() == "true"

# --- Фиксированный риск на сделку ---
# true = позиция рассчитывается так, чтобы потеря при SL = RISK_PER_TRADE_PERCENT% баланса
FIXED_RISK_MODE           = os.getenv("FIXED_RISK_MODE", "true").lower() == "true"
RISK_PER_TRADE_PERCENT    = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
MAX_TRADE_SIZE_PERCENT    = float(os.getenv("MAX_TRADE_SIZE_PERCENT", "20.0"))

# --- HTF S/R подтверждение ---
# Кол-во свечей HTF_TIMEFRAME для поиска уровней S/R (0 = отключить)
HTF_SR_CANDLES            = int(os.getenv("HTF_SR_CANDLES", "200"))
# true = сигналы только с HTF-подтверждением; false = предпочитать HTF, но не блокировать остальные
HTF_SR_REQUIRE_CONFIRM    = os.getenv("HTF_SR_REQUIRE_CONFIRM", "false").lower() == "true"

# --- Paper trading ---
INITIAL_BALANCE        = float(os.getenv("INITIAL_BALANCE", "1000"))
TRADE_SIZE_PERCENT     = float(os.getenv("TRADE_SIZE_PERCENT", "2"))
MAX_OPEN_TRADES        = int(os.getenv("MAX_OPEN_TRADES", "0"))  # 0 = без лимита
PENDING_EXPIRY_CHECKS  = int(os.getenv("PENDING_EXPIRY_CHECKS", "99999"))  # отмена по близости, не по счётчику
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
    commission_rate=COMMISSION_RATE,
    breakeven_threshold=BREAKEVEN_THRESHOLD,
    fixed_risk_mode=FIXED_RISK_MODE,
    risk_per_trade_percent=RISK_PER_TRADE_PERCENT,
    max_trade_size_percent=MAX_TRADE_SIZE_PERCENT,
    trailing_stop=TRAILING_STOP,
    trailing_mult=TRAILING_MULT,
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


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _calc_adx(df: pd.DataFrame, period: int = 14):
    """Wilder's ADX. Возвращает (adx, +DI, -DI). При нехватке данных — (None, None, None)."""
    if len(df) < period * 2:
        return None, None, None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    tr  = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up, dn = high - ph, pl - low
    pdm = ((up > dn) & (up > 0)).astype(float) * up
    ndm = ((dn > up) & (dn > 0)).astype(float) * dn
    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    pdm_s = pdm.ewm(alpha=alpha, adjust=False).mean()
    ndm_s = ndm.ewm(alpha=alpha, adjust=False).mean()
    pdi   = 100.0 * pdm_s / atr_s.replace(0, float("nan"))
    ndi   = 100.0 * ndm_s / atr_s.replace(0, float("nan"))
    denom = (pdi + ndi).replace(0, float("nan"))
    dx    = 100.0 * (pdi - ndi).abs() / denom
    adx   = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])


def _calc_rsi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_g = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_atr_ratio(df: pd.DataFrame, atr_period: int = 14, avg_period: int = 20) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(atr_period).mean().dropna()
    if len(atr_series) < avg_period + 1:
        return 1.0
    current      = float(atr_series.iloc[-1])
    hist_avg     = float(atr_series.iloc[-(avg_period + 1):-1].mean())
    return round(current / hist_avg, 2) if hist_avg > 0 else 1.0


def _detect_regime(rsi_series: pd.Series, atr_ratio: float) -> str:
    """Возвращает 'CRASH'/'RECOVERY'/'PUMP'/'CORRECTION'/'NORMAL'."""
    curr_rsi = float(rsi_series.iloc[-1])
    if atr_ratio >= CRASH_PAUSE_ATR_RATIO:
        return "CRASH" if curr_rsi < 50 else "PUMP"
    if atr_ratio < CRASH_RESUME_ATR_RATIO:
        # БАГ 1 fix: берём срезы напрямую из исходной серии, без двойного среза
        was_panic  = bool((rsi_series.iloc[-RSI_OVERSOLD_LOOKBACK:]  < RSI_OVERSOLD).any())
        was_pump   = bool((rsi_series.iloc[-RSI_OVERBOUGHT_LOOKBACK:] > RSI_OVERBOUGHT).any())
        # БАГ 2 fix: CORRECTION активна пока RSI > RSI_CORRECTION_MIN (55),
        # т.е. цена ещё высоко — коррекция только началась. Аналог RECOVERY: RSI < RSI_RECOVERY_MAX.
        if was_panic and RSI_OVERSOLD < curr_rsi < RSI_RECOVERY_MAX:
            return "RECOVERY"
        if was_pump and curr_rsi > RSI_CORRECTION_MIN:
            return "CORRECTION"
    return "NORMAL"


def _find_crash_low(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["low"].iloc[-lookback:].min())


def _find_pump_high(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["high"].iloc[-lookback:].max())


def _fetch_htf_confluence(client, pair: str):
    """Один запрос 4h → (trend, ema, sr_levels). Тренд + S/R без двойного API-вызова."""
    try:
        needed = max(HTF_SR_CANDLES, HTF_EMA_PERIOD + 30)
        raw = client.fetch_ohlcv(pair, HTF_TIMEFRAME, limit=needed)
        if not raw or len(raw) < HTF_EMA_PERIOD:
            return None, None, []
        closes = pd.Series([float(r[4]) for r in raw])
        ema    = float(closes.ewm(span=HTF_EMA_PERIOD, adjust=False).mean().iloc[-1])
        trend  = "UP" if float(raw[-1][4]) > ema else "DOWN"
        htf_sr: List[Dict] = []
        if HTF_SR_CANDLES > 0 and len(raw) >= EXTREMA_WINDOW * 2 + 1:
            df_htf = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_htf["timestamp"] = pd.to_datetime(df_htf["timestamp"], unit="ms")
            htf_sr = find_support_resistance(df_htf)
        return trend, round(ema, 8), htf_sr
    except Exception as exc:
        logger.warning("Ошибка HTF анализа %s: %s", pair, exc)
        return None, None, []


def _mark_htf_confirmed(levels: List[Dict], htf_levels: List[Dict], tolerance_pct: float) -> None:
    """Помечает htf_confirmed=True на уровнях, совпадающих с уровнем того же типа на HTF."""
    tol = tolerance_pct / 100.0
    for lvl in levels:
        lvl["htf_confirmed"] = any(
            h["type"] == lvl["type"]
            and abs(h["price"] - lvl["price"]) / lvl["price"] <= tol
            for h in htf_levels
        )


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
    htf_trend: Optional[str] = None,
    adx_val: Optional[float] = None,
    proximity_percent: float = ENTRY_PROXIMITY_PERCENT,
    ema_period: int = EMA_PERIOD,
    atr_period: int = ATR_PERIOD,
    sl_atr_mult: float = SL_ATR_MULT,
    tp_atr_min_mult: float = TP_ATR_MIN_MULT,
    min_rr: float = MIN_RR,
    adx_min: float = ADX_MIN,
    regime: str = "NORMAL",
    crash_low: Optional[float] = None,
    pump_high: Optional[float] = None,
) -> List[Dict]:
    # Во время паники (краш или памп) — не создавать новых ордеров
    if regime in ("CRASH", "PUMP"):
        return []

    if ema_period >= len(df):
        logger.warning("Недостаточно свечей для EMA%d (%d)", ema_period, len(df))
        return []

    proximity   = proximity_percent / 100.0
    ema_val     = float(_calc_ema(df["close"], ema_period).iloc[-1])
    ema_trend   = "UP" if current_price > ema_val else "DOWN"
    pattern     = _detect_candle_pattern(df)
    atr         = _calc_atr(df, atr_period)
    # БАГ 7 fix: ATR может быть NaN при недостатке данных — всё что дальше использует его сломается
    if pd.isna(atr) or atr <= 0:
        logger.warning("ATR некорректен (%.6f) для %s — сигналы пропущены", atr, regime)
        return []
    recovery    = (regime == "RECOVERY")
    correction  = (regime == "CORRECTION")
    special     = recovery or correction  # оба режима отключают стандартные фильтры

    # ADX фильтр пропускаем в спецрежимах — рынок только вышел из аномального движения
    if not special and adx_val is not None and adx_min > 0 and adx_val < adx_min:
        logger.info("ADX %.1f < %.1f — боковик, сигналы пропущены", adx_val, adx_min)
        return []

    signals = []
    for lvl in levels:
        distance = abs(current_price - lvl["price"]) / lvl["price"]
        if distance > proximity:
            continue

        if recovery:
            if lvl["type"] != "SUPPORT":
                continue
            direction = "LONG"
        elif correction:
            if lvl["type"] != "RESISTANCE":
                continue
            direction = "SHORT"
        else:
            if lvl["type"] == "SUPPORT" and ema_trend == "UP":
                direction = "LONG"
            elif lvl["type"] == "RESISTANCE" and ema_trend == "DOWN":
                direction = "SHORT"
            else:
                continue
            if htf_trend is not None:
                if htf_trend != ("UP" if direction == "LONG" else "DOWN"):
                    continue

        entry = lvl["price"]

        if recovery and crash_low is not None:
            sl      = crash_low * (1.0 - CRASH_SL_BUFFER / 100.0)
            sl_dist = entry - sl
            if sl_dist <= 0:
                continue
            tp_min = sl_dist * min_rr
            above  = sorted(
                [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > entry],
                key=lambda x: x["price"],
            )
            tp = above[0]["price"] if above and above[0]["price"] >= entry + tp_min else entry + tp_min
        elif correction and pump_high is not None:
            sl      = pump_high * (1.0 + PUMP_SL_BUFFER / 100.0)
            sl_dist = sl - entry
            if sl_dist <= 0:
                continue
            tp_min = sl_dist * min_rr
            below  = sorted(
                [l for l in levels if l["type"] == "SUPPORT" and l["price"] < entry],
                key=lambda x: x["price"], reverse=True,
            )
            tp = below[0]["price"] if below and below[0]["price"] <= entry - tp_min else entry - tp_min
        elif special:
            # БАГ 3/4 fix: спецрежим без опорной цены — ATR-fallback, предупреждаем
            logger.warning("режим %s без crash_low/pump_high — SL по ATR", regime)
            sl_dist = atr * sl_atr_mult
            tp_min  = sl_dist * min_rr
            sl = entry - sl_dist if recovery else entry + sl_dist
            tp = entry + tp_min  if recovery else entry - tp_min
        else:
            sl_dist = atr * sl_atr_mult
            tp_min  = atr * tp_atr_min_mult
            if direction == "LONG":
                sl    = entry - sl_dist
                above = sorted(
                    [l for l in levels if l["type"] == "RESISTANCE" and l["price"] > entry],
                    key=lambda x: x["price"],
                )
                if above:
                    nearest = above[0]["price"]
                    if nearest < entry + tp_min:
                        continue
                    tp = nearest
                else:
                    tp = entry + tp_min
            else:
                sl    = entry + sl_dist
                below = sorted(
                    [l for l in levels if l["type"] == "SUPPORT" and l["price"] < entry],
                    key=lambda x: x["price"], reverse=True,
                )
                if below:
                    nearest = below[0]["price"]
                    if nearest > entry - tp_min:
                        continue
                    tp = nearest
                else:
                    tp = entry - tp_min

        tp_dist    = abs(tp - entry)
        rr         = round(tp_dist / sl_dist, 1) if sl_dist > 0 else None
        if rr is None or rr < min_rr:
            continue

        risk_pct   = round(sl_dist / entry * 100, 2)
        reward_pct = round(tp_dist / entry * 100, 2)

        signals.append({
            "direction":        direction,
            "level":            lvl,
            "pattern":          pattern,
            "ema":              ema_val,
            "ema_trend":        ema_trend,
            "htf_trend":        htf_trend,
            "adx":              round(adx_val, 1) if adx_val is not None else None,
            "atr":              round(atr, 8),
            "distance_percent": round(distance * 100, 2),
            "sl":               sl,
            "tp":               tp,
            "rr":               rr,
            "risk_pct":         risk_pct,
            "reward_pct":       reward_pct,
            "regime":           regime,
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
        ["⏳ Ордера", "🪙 Монеты"],
    ],
    resize_keyboard=True,
    is_persistent=True,
)


def _pairs_inline_kb(pairs: List[str]) -> InlineKeyboardMarkup:
    """Инлайн-клавиатура со списком монет для просмотра отчётов."""
    buttons = []
    row: List[InlineKeyboardButton] = []
    for pair in pairs:
        label = pair.replace("/USDT", "")
        row.append(InlineKeyboardButton(label, callback_data=f"report_{pair}"))
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
    htf_trend: Optional[str] = None,
    htf_ema: Optional[float] = None,
    adx: Optional[float] = None,
    rsi: Optional[float] = None,
    atr_ratio: Optional[float] = None,
    regime: str = "NORMAL",
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📊 <b>{pair}</b>  •  <code>{timeframe}</code>",
        f"💰 Цена: <b>{_fp(current_price)}</b>",
        f"🕒 {now}",
    ]
    # Шапка: HTF тренд + ADX
    htf_str = ""
    if htf_trend:
        ht = "↑ UP" if htf_trend == "UP" else "↓ DOWN"
        htf_str = f"  •  {HTF_TIMEFRAME} EMA{HTF_EMA_PERIOD}: <b>{ht}</b>"
        if htf_ema:
            htf_str += f" ({_fp(htf_ema)})"
    adx_str = ""
    if adx is not None:
        label = "тренд" if adx >= ADX_MIN else "боковик"
        adx_str = f"  •  ADX: <b>{adx:.1f}</b> ({label})"
    if htf_str or adx_str:
        lines.append(f"📡{htf_str}{adx_str}")
    # RSI + ATR-ratio + режим рынка
    rsi_str = ""
    if rsi is not None:
        if rsi < RSI_OVERSOLD:
            rsi_label = "🔴 перепродан"
        elif rsi < RSI_RECOVERY_MAX:
            rsi_label = "🟡 восст."
        elif rsi > RSI_OVERBOUGHT:
            rsi_label = "🔴 перекуплен"
        elif rsi > RSI_CORRECTION_MIN:
            rsi_label = "🟡 корр."
        else:
            rsi_label = "🟢 норма"
        rsi_str = f"RSI: <b>{rsi:.1f}</b> ({rsi_label})"
    ratio_str = ""
    if atr_ratio is not None:
        ratio_str = f"ATR×: <b>{atr_ratio:.1f}</b>"
    regime_map = {
        "CRASH":      "⚠️ КРАШ — пауза",
        "RECOVERY":   "🔄 ВОССТАНОВЛЕНИЕ",
        "PUMP":       "🚀 ПАМП — пауза",
        "CORRECTION": "📉 КОРРЕКЦИЯ",
        "NORMAL":     "",
    }
    regime_str = regime_map.get(regime, "")
    extra = "  •  ".join(filter(None, [rsi_str, ratio_str, regime_str]))
    if extra:
        lines.append(f"⚡ {extra}")
    lines.append("")

    if signals:
        lines.append("🎯 <b>СИГНАЛЫ ВХОДА:</b>")
        for sig in signals:
            lvl    = sig["level"]
            de     = "📈" if sig["direction"] == "LONG" else "📉"
            ta_30  = "↑" if sig["ema_trend"] == "UP" else "↓"
            ta_htf = ""
            if sig.get("htf_trend"):
                ta_htf = "↑" if sig["htf_trend"] == "UP" else "↓"
                ta_htf = f"  •  {HTF_TIMEFRAME}:{ta_htf}"
            adx_s = f"  •  ADX:{sig['adx']}" if sig.get("adx") is not None else ""
            htf_mark = " ✨" if lvl.get("htf_confirmed") else ""
            lines.append(
                f"{de} <b>{sig['direction']}</b>  вблизи {_fp(lvl['price'])}{htf_mark}"
                f"  <i>({sig['distance_percent']}%)</i>"
            )
            lines.append(
                f"   {ta_30} EMA{EMA_PERIOD}={_fp(sig['ema'])}{ta_htf}{adx_s}"
                f"  •  ATR={_fp(sig['atr'])}"
            )
            if sig["pattern"]:
                lines.append(f"   Паттерн: {sig['pattern']}")
            rr_str = f"  •  R:R 1:{sig['rr']}" if sig["rr"] else ""
            lines.append(
                f"   SL: <b>{_fp(sig['sl'])}</b> (-{sig['risk_pct']}%)"
                f"  •  TP: <b>{_fp(sig['tp'])}</b> (+{sig['reward_pct']}%){rr_str}"
            )
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
        retest   = " ✅" if lvl.get("has_retest") else ""
        htf_mark = " ✨" if lvl.get("htf_confirmed") else ""
        lines.append(
            f"{emoji} <b>{_fp(lvl['price'])}</b> — {ru_type} "
            f"(касаний: <b>{lvl['touches']}</b>, {sign}{dist:.2f}%{retest}{htf_mark})"
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
        f"Equity: ${balance:,.2f}"
    )


def fmt_sl_to_breakeven(trade: Dict) -> str:
    de = "📈" if trade["direction"] == "LONG" else "📉"
    return (
        f"🔒 <b>SL → безубыток #{trade['id']}</b>\n"
        f"{de} <b>{trade['pair']}</b>  •  {trade['direction']}\n"
        f"Новый SL: <b>{_fp(trade['entry_price'])}</b>  (= цена входа)"
    )


def fmt_trade_closed(trade: Dict, balance: float) -> str:
    won    = (trade["pnl_usd"] or 0) > 0
    emoji  = "✅" if won else "❌"
    reason = "ТЕЙК-ПРОФИТ 🎯" if trade["close_reason"] == "TP" else "СТОП-ЛОСС 🛑"
    sign   = "+" if (trade["pnl_usd"] or 0) >= 0 else ""
    comm   = trade.get("commission")
    comm_line = f"\nКомиссия: <b>-${comm}</b>" if comm else ""
    be_line = "\n<i>SL был перенесён в безубыток</i>" if trade.get("sl_at_breakeven") else ""
    return (
        f"{emoji} <b>{reason} #{trade['id']}</b>\n"
        f"<b>{trade['pair']}</b>  •  {trade['direction']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Вход: {_fp(trade['entry_price'])} → Закрыт: <b>{_fp(trade['close_price'])}</b>\n"
        f"P&L: <b>{sign}${trade['pnl_usd']}  ({sign}{trade['pnl_percent']}%)</b>"
        f"{comm_line}{be_line}\n"
        f"Equity: <b>${balance:,.2f}</b>"
    )



def fmt_stats(ptf: PaperPortfolio) -> str:
    s    = ptf.get_stats()
    sign = "+" if s["balance_change_pct"] >= 0 else ""
    ps   = "+" if s["total_pnl"] >= 0 else ""
    ap   = "+" if s["avg_pnl"] >= 0 else ""
    lines = [
        "📊 <b>СТАТИСТИКА ПОРТФЕЛЯ</b>", "",
        f"💰 Equity: <b>${s['equity']:,.2f}</b>  ({sign}{s['balance_change_pct']}%)",
        f"🏦 Начальный: ${s['initial_balance']:,.2f}  •  Реализован: ${s['balance']:,.2f}",
        f"📂 Открытых: <b>{s['open_count']}</b>  •  ⏳ Ожидающих: <b>{s['pending_count']}</b>", "",
        f"📋 Ордеров создано: <b>{s['orders_created']}</b>  •  "
        f"Срабатывало: <b>{s['orders_triggered']}</b>  •  "
        f"Отменено: <b>{s['orders_cancelled']}</b>", "",
        f"📈 Всего закрыто: <b>{s['total_closed']}</b>",
        f"  ✅ Прибыльных: <b>{s['wins']}</b>  ({s['winrate']}% winrate)",
        f"  ❌ Убыточных:  <b>{s['losses']}</b>", "",
        f"💵 Итоговый P&L: <b>{ps}${s['total_pnl']:,.2f}</b>  •  "
        f"Средняя сделка: <b>{ap}${s['avg_pnl']:,.2f}</b>",
    ]
    if s["best"]:
        b = s["best"]
        lines.append(f"🏆 Лучшая:  <b>+${b['pnl_usd'] or 0}</b>  ({b['pair']} {b['direction']})")
    if s["worst"]:
        w = s["worst"]
        lines.append(f"💔 Худшая:  <b>${w['pnl_usd'] or 0}</b>  ({w['pair']} {w['direction']})")
    lines.append(f"\n<i>v{BOT_VERSION}</i>")
    return "\n".join(lines)


def fmt_log(ptf: PaperPortfolio, n: int = 20) -> str:
    closed = ptf.recent_trades(n)
    total  = len(ptf.closed_trades)
    if not closed:
        return "📋 <b>Лог сделок</b>\n\n<i>Закрытых сделок пока нет.</i>"
    lines = [f"📋 <b>Лог сделок</b>  (всего закрытых: {total})", ""]

    for i, t in enumerate(closed, 1):
        pnl  = t["pnl_usd"] or 0
        em   = "✅" if pnl > 0 else "❌"
        sign = "+" if pnl >= 0 else ""
        rsn  = "TP" if t["close_reason"] == "TP" else "SL"
        lines.append(
            f"{i}. {em} <b>{t['pair']}</b> {t['direction']} [{rsn}]  "
            f"<b>{sign}${t['pnl_usd'] or 0}</b> ({sign}{t['pnl_percent'] or 0}%)"
        )
        lines.append(
            f"   {_fp(t['entry_price'])} → {_fp(t['close_price'])}"
            f"  •  {(t['closed_at'] or '')[:16]}"
        )
    return "\n".join(lines)


def fmt_open_trades(ptf: PaperPortfolio, prices: Dict[str, float]) -> str:
    open_t = ptf.open_trades
    if not open_t:
        return "📂 <b>Открытых позиций нет</b>\n\n<i>Для просмотра ожидающих ордеров нажми ⏳ Ордера.</i>"

    lines = [f"📂 <b>ОТКРЫТЫЕ ПОЗИЦИИ</b>  ({len(open_t)} шт.)", ""]

    for t in open_t:
        de  = "📈" if t["direction"] == "LONG" else "📉"
        cur = prices.get(t["pair"])
        if cur:
            ntl  = t.get("notional", t["size_usd"])
            upnl = ((cur - t["entry_price"]) / t["entry_price"] * ntl
                    if t["direction"] == "LONG"
                    else (t["entry_price"] - cur) / t["entry_price"] * ntl)
            upnl_pct = upnl / t["size_usd"] * 100
            sign = "+" if upnl >= 0 else ""
            pnl_line = f"\n   PnL: <b>{sign}${upnl:.2f}  ({sign}{upnl_pct:.1f}%)</b>  •  Цена: {_fp(cur)}"
        else:
            pnl_line = ""
        ep = t["entry_price"]
        sl_pct = round(abs(ep - t["sl"]) / ep * 100, 2) if ep else "?"
        tp_pct = round(abs(t["tp"] - ep) / ep * 100, 2) if ep else "?"
        be_mark    = " 🔒" if t.get("sl_at_breakeven") else ""
        trail_mark = " 🔄" if t.get("trail_dist") else ""
        lines.append(
            f"{de} <b>#{t['id']}</b> {t['pair']}  •  {t['direction']}{be_mark}{trail_mark}\n"
            f"   Вход: {_fp(ep)}\n"
            f"   SL: {_fp(t['sl'])}  (-{sl_pct}%)  •  TP: {_fp(t['tp'])}  (+{tp_pct}%)"
            f"{pnl_line}"
        )

    return "\n".join(lines)


def fmt_pending_orders(ptf: PaperPortfolio) -> str:
    orders = ptf.pending_orders
    if not orders:
        return "⏳ <b>Ожидающих ордеров нет</b>\n\n<i>Ордера появятся когда цена подойдёт к уровню.</i>"
    lines = [f"⏳ <b>ОЖИДАЮЩИЕ ОРДЕРА</b>  ({len(orders)} шт.)", ""]
    for o in orders:
        de     = "📈" if o["direction"] == "LONG" else "📉"
        ep     = o["entry_price"]
        sl_pct = o.get("risk_pct")   or (round(abs(ep - o["sl"]) / ep * 100, 2) if ep else "?")
        tp_pct = o.get("reward_pct") or (round(abs(o["tp"] - ep) / ep * 100, 2) if ep else "?")
        rr     = o.get("rr")
        rr_str = f"  •  R:R 1:{rr}" if rr else ""
        lines.append(
            f"{de} <b>#{o['id']}</b>  {o['pair']}  •  {o['direction']}\n"
            f"   Лимит: <b>{_fp(ep)}</b>\n"
            f"   SL: {_fp(o['sl'])}  (-{sl_pct}%)  •  TP: {_fp(o['tp'])}  (+{tp_pct}%){rr_str}"
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

    # Основной TF для анализа уровней и сигналов
    df = fetch_ohlcv(client, pair, TIMEFRAME, CANDLES_LIMIT)
    if df is None or df.empty:
        logger.warning("Пропускаем %s (нет данных)", pair)
        return

    try:
        current_price = float(client.fetch_ticker(pair)["last"])
    except Exception:
        current_price = float(df["close"].iloc[-1])
    portfolio.update_price(pair, current_price)

    # Короткий TF для SL/TP — прямой вызов CCXT (без проверки min_candles из fetch_ohlcv)
    try:
        raw_check = client.fetch_ohlcv(pair, CHECK_TIMEFRAME, limit=5)
        if raw_check and len(raw_check) >= 2:
            h = max(r[2] for r in raw_check[-2:])  # high
            l = min(r[3] for r in raw_check[-2:])  # low
        else:
            raise ValueError("мало свечей")
    except Exception:
        h = max(float(df["high"].iloc[-2]), float(df["high"].iloc[-1]))
        l = min(float(df["low"].iloc[-2]),  float(df["low"].iloc[-1]))

    # Всегда включаем текущую живую цену, чтобы SL/TP срабатывал без задержки
    h = max(h, current_price)
    l = min(l, current_price)

    # 1. Безубыток → трейлинг → SL/TP для уже открытых сделок
    for trade in portfolio.check_breakeven(pair, h, l):
        await send_msg(bot, fmt_sl_to_breakeven(trade))
    portfolio.check_trailing_stop(pair, h, l)
    for trade in portfolio.check_sl_tp(pair, h, l):
        await send_msg(bot, fmt_trade_closed(trade, portfolio.get_equity()))

    # 2. Проверка ожидающих ордеров — коснулась ли цена уровня
    triggered, _ = portfolio.check_pending_orders(pair, h, l,
                                                   candle_close=current_price,
                                                   require_bounce=BOUNCE_CONFIRM)
    # Та же свеча могла сразу достичь BE или SL/TP у только что открытых сделок
    for trade in portfolio.check_breakeven(pair, h, l):
        await send_msg(bot, fmt_sl_to_breakeven(trade))
    portfolio.check_trailing_stop(pair, h, l)
    for closed in portfolio.check_sl_tp(pair, h, l):
        await send_msg(bot, fmt_trade_closed(closed, portfolio.get_equity()))

    # 3. Уровни и сигналы
    levels = find_support_resistance(df)

    # HTF: тренд + S/R — один API-запрос на пару
    htf_trend, htf_ema, htf_sr = _fetch_htf_confluence(client, pair)
    if htf_sr:
        _mark_htf_confirmed(levels, htf_sr, TOLERANCE_PERCENT)
    adx_val, _, _ = _calc_adx(df, ADX_PERIOD)

    # RSI + ATR-ratio → режим рынка
    rsi_series = _calc_rsi_series(df, RSI_PERIOD)
    rsi_raw    = float(rsi_series.iloc[-1])
    rsi_val    = rsi_raw if not pd.isna(rsi_raw) else 50.0  # NaN → нейтральное значение
    atr_ratio  = _calc_atr_ratio(df, ATR_PERIOD)
    regime     = _detect_regime(rsi_series, atr_ratio)
    crash_low  = _find_crash_low(df, CRASH_LOW_LOOKBACK)  if regime == "RECOVERY"   else None
    pump_high  = _find_pump_high(df, PUMP_HIGH_LOOKBACK)  if regime == "CORRECTION" else None

    logger.info(
        "%s | HTF(%s)=%s  ADX=%s  RSI=%.1f  ATR×=%.1f  режим=%s",
        pair, HTF_TIMEFRAME, htf_trend or "N/A",
        f"{adx_val:.1f}" if adx_val is not None else "N/A",
        rsi_val, atr_ratio, regime,
    )

    signal_levels = (
        [l for l in levels if l.get("htf_confirmed")]
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
    )

    # 4. Сохраняем отчёт в БД (без авто-отправки — доступен по кнопке монеты)
    portfolio.save_report(pair, fmt_analysis(
        pair, TIMEFRAME, current_price, levels, signals,
        htf_trend=htf_trend, htf_ema=htf_ema,
        adx=round(adx_val, 1) if adx_val is not None else None,
        rsi=round(rsi_val, 1),
        atr_ratio=atr_ratio,
        regime=regime,
    ))

    # 5. Ордера: всегда только для ближайшего уровня по направлению
    best: Dict[str, Optional[Dict]] = {"LONG": None, "SHORT": None}
    for sig in sorted(signals, key=lambda s: (not s["level"].get("htf_confirmed", False), s["distance_percent"])):
        if best[sig["direction"]] is None:
            best[sig["direction"]] = sig

    # Отменяем ордера, у которых теперь не самый близкий уровень (молча)
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

    # Создаём ордер только для ближайшего уровня (молча)
    for sig in best.values():
        if sig is None or sig["tp"] is None:
            continue
        portfolio.create_pending_order(
            pair=pair,
            direction=sig["direction"],
            entry_price=sig["level"]["price"],
            sl=sig["sl"],
            tp=sig["tp"],
        )


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
    elif "Ордера" in txt:
        msg = fmt_pending_orders(portfolio)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=REPLY_KB)
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
        pair = data[len("report_"):]
        report = portfolio.get_report(pair)
        if report:
            text = report["text"]
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
                current_price = float(client.fetch_ticker(pair)["last"])
            except Exception:
                continue
            portfolio.update_price(pair, current_price)

            try:
                raw = client.fetch_ohlcv(pair, CHECK_TIMEFRAME, limit=5)
                if raw and len(raw) >= 2:
                    h = max(r[2] for r in raw[-2:])
                    l = min(r[3] for r in raw[-2:])
                else:
                    raise ValueError("мало свечей")
            except Exception:
                h = l = current_price
            h = max(h, current_price)
            l = min(l, current_price)

            for trade in portfolio.check_breakeven(pair, h, l):
                await send_msg(bot, fmt_sl_to_breakeven(trade))
            portfolio.check_trailing_stop(pair, h, l)

            for closed in portfolio.check_sl_tp(pair, h, l):
                await send_msg(bot, fmt_trade_closed(closed, portfolio.get_equity()))

            triggered, _ = portfolio.check_pending_orders(pair, h, l,
                                                           candle_close=current_price,
                                                           require_bounce=BOUNCE_CONFIRM)
            if triggered:
                for trade in portfolio.check_breakeven(pair, h, l):
                    await send_msg(bot, fmt_sl_to_breakeven(trade))
                portfolio.check_trailing_stop(pair, h, l)
                for closed in portfolio.check_sl_tp(pair, h, l):
                    await send_msg(bot, fmt_trade_closed(closed, portfolio.get_equity()))

        except Exception as exc:
            logger.error("fast_check %s: %s", pair, exc)


# ============================================================
# JOB: ПЕРИОДИЧЕСКИЙ АНАЛИЗ
# ============================================================
async def analysis_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data["client"]
    bot    = context.bot

    pairs = fetch_top_pairs(client, AUTO_TOP_PAIRS) if AUTO_TOP_PAIRS > 0 else TRADING_PAIRS
    context.bot_data["pairs"] = pairs
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
        f"Комиссия: <b>{COMMISSION_RATE*100:.2f}%</b> round-trip  •  "
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

    # Быстрая проверка SL/TP + безубыток (стартует через 30 сек, затем каждые N мин)
    check_sec = SL_TP_CHECK_INTERVAL_MIN * 60
    app.job_queue.run_repeating(fast_check_job, interval=check_sec, first=30)

    # Полный анализ уровней и сигналов (стартует через 15 сек, затем каждые RUN_INTERVAL_HOURS)
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
