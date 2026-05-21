"""
Все конфигурационные константы, читаемые из переменных окружения.
Остальные модули делают: from config import *
"""
import logging
import os
import sys

import ccxt

BOT_VERSION = "1.8.0"

MEXC_API_KEY       = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY    = os.getenv("MEXC_SECRET_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

TRADING_PAIRS = [
    p.strip() for p in os.getenv("TRADING_PAIRS", (
        # 12 пар из разных секторов — минимальная межсекторная корреляция
        "BTC/USDT,"   # цифровое золото / benchmark
        "ETH/USDT,"   # L1 / DeFi-база
        "SOL/USDT,"   # high-perf L1
        "XRP/USDT,"   # payments / регуляторный сектор
        "BNB/USDT,"   # exchange token
        "LINK/USDT,"  # oracle / инфраструктура
        "ARB/USDT,"   # L2 экосистема Ethereum
        "RNDR/USDT,"  # AI / GPU compute
        "ATOM/USDT,"  # межсетевое взаимодействие
        "TON/USDT,"   # мессенджер-экосистема
        "SUI/USDT,"   # новый L1
        "LTC/USDT"    # Bitcoin-proxy с отдельной динамикой
    )).split(",")
    if p.strip()
]

TIMEFRAME             = os.getenv("TIMEFRAME", "1h")
CHECK_TIMEFRAME       = os.getenv("CHECK_TIMEFRAME", "5m")
CANDLES_LIMIT         = int(os.getenv("CANDLES_LIMIT", "500"))
MIN_TOUCHES           = int(os.getenv("MIN_TOUCHES", "3"))
TOLERANCE_PERCENT     = float(os.getenv("TOLERANCE_PERCENT", "0.8"))
EXTREMA_WINDOW        = int(os.getenv("EXTREMA_WINDOW", "8"))
TOP_N_LEVELS          = int(os.getenv("TOP_N_LEVELS", "5"))
MIN_TOUCH_SPACING     = int(os.getenv("MIN_TOUCH_SPACING", "3"))
LEVEL_AGE_MIN_CANDLES = int(os.getenv("LEVEL_AGE_MIN_CANDLES", "10"))
VOLUME_TOUCH_MULTIPLIER = float(os.getenv("VOLUME_TOUCH_MULTIPLIER", "1.2"))
REQUIRE_RETEST        = os.getenv("REQUIRE_RETEST", "false").lower() == "true"
HTF_STRUCTURE_WINDOW  = int(os.getenv("HTF_STRUCTURE_WINDOW", "5"))
REQUIRE_RSI_DIVERGENCE = os.getenv("REQUIRE_RSI_DIVERGENCE", "true").lower() == "true"

ENTRY_PROXIMITY_PERCENT = float(os.getenv("ENTRY_PROXIMITY_PERCENT", "0.5"))
EMA_PERIOD              = int(os.getenv("EMA_PERIOD", "200"))
AUTO_TOP_PAIRS          = int(os.getenv("AUTO_TOP_PAIRS", "0"))
ATR_PERIOD              = int(os.getenv("ATR_PERIOD", "14"))
SL_ATR_MULT             = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MIN_MULT         = float(os.getenv("TP_ATR_MIN_MULT", "4.0"))
MIN_RR                  = float(os.getenv("MIN_RR", "2.0"))

HTF_TIMEFRAME  = os.getenv("HTF_TIMEFRAME", "4h")
HTF_EMA_PERIOD = int(os.getenv("HTF_EMA_PERIOD", "50"))
ADX_PERIOD     = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN        = float(os.getenv("ADX_MIN", "20"))

RSI_PERIOD            = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD          = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_RECOVERY_MAX      = float(os.getenv("RSI_RECOVERY_MAX", "47"))
RSI_OVERSOLD_LOOKBACK = int(os.getenv("RSI_OVERSOLD_LOOKBACK", "12"))
CRASH_LOW_LOOKBACK    = int(os.getenv("CRASH_LOW_LOOKBACK", "24"))
CRASH_SL_BUFFER       = float(os.getenv("CRASH_SL_BUFFER", "0.35"))
RSI_OVERBOUGHT        = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_CORRECTION_MIN    = float(os.getenv("RSI_CORRECTION_MIN", "55"))
RSI_OVERBOUGHT_LOOKBACK = int(os.getenv("RSI_OVERBOUGHT_LOOKBACK", "12"))
PUMP_HIGH_LOOKBACK    = int(os.getenv("PUMP_HIGH_LOOKBACK", "24"))
PUMP_SL_BUFFER        = float(os.getenv("PUMP_SL_BUFFER", "0.35"))
CRASH_PAUSE_ATR_RATIO = float(os.getenv("CRASH_PAUSE_ATR_RATIO", "2.5"))
CRASH_RESUME_ATR_RATIO = float(os.getenv("CRASH_RESUME_ATR_RATIO", "2.0"))

DELAY_BETWEEN_PAIRS      = int(os.getenv("DELAY_BETWEEN_PAIRS", "2"))
RUN_INTERVAL_HOURS       = float(os.getenv("RUN_INTERVAL_HOURS", "1.0"))
SL_TP_CHECK_INTERVAL_MIN = int(os.getenv("SL_TP_CHECK_INTERVAL_MIN", "3"))
COMMISSION_MAKER         = float(os.getenv("COMMISSION_MAKER", "0.0000"))
COMMISSION_TAKER         = float(os.getenv("COMMISSION_TAKER", "0.0005"))
BREAKEVEN_THRESHOLD      = float(os.getenv("BREAKEVEN_THRESHOLD", "0.65"))

TRAILING_STOP = os.getenv("TRAILING_STOP", "true").lower() == "true"
TRAILING_MULT = float(os.getenv("TRAILING_MULT", "1.2"))

BOUNCE_CONFIRM         = os.getenv("BOUNCE_CONFIRM", "true").lower() == "true"
FIXED_RISK_MODE        = os.getenv("FIXED_RISK_MODE", "true").lower() == "true"
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
MAX_TRADE_SIZE_PERCENT = float(os.getenv("MAX_TRADE_SIZE_PERCENT", "10.0"))

HTF_SR_CANDLES       = int(os.getenv("HTF_SR_CANDLES", "200"))
HTF_SR_REQUIRE_CONFIRM = os.getenv("HTF_SR_REQUIRE_CONFIRM", "false").lower() == "true"

EMA_CONFLUENCE_PERIODS = [
    int(x.strip()) for x in os.getenv("EMA_CONFLUENCE_PERIODS", "21,50").split(",")
    if x.strip()
]
FVG_LOOKBACK   = int(os.getenv("FVG_LOOKBACK", "50"))
FVG_MIN_GAP_PCT = float(os.getenv("FVG_MIN_GAP_PCT", "0.1"))

REQUIRE_PATTERN      = os.getenv("REQUIRE_PATTERN", "false").lower() == "true"
SIGNAL_VOLUME_MULT   = float(os.getenv("SIGNAL_VOLUME_MULT", "0"))
MIN_CONFLUENCE_SCORE   = int(os.getenv("MIN_CONFLUENCE_SCORE", "1"))
HTF_RSI_OVERBOUGHT     = float(os.getenv("HTF_RSI_OVERBOUGHT", "65"))
HTF_RSI_OVERSOLD       = float(os.getenv("HTF_RSI_OVERSOLD",   "35"))

CORR_LOOKBACK = int(os.getenv("CORR_LOOKBACK", "48"))
CORR_MAX      = float(os.getenv("CORR_MAX", "0.75"))

INITIAL_BALANCE       = float(os.getenv("INITIAL_BALANCE", "1000"))
TRADE_SIZE_PERCENT    = float(os.getenv("TRADE_SIZE_PERCENT", "2"))
MAX_OPEN_TRADES       = int(os.getenv("MAX_OPEN_TRADES", "0"))
PENDING_EXPIRY_CHECKS = int(os.getenv("PENDING_EXPIRY_CHECKS", "96"))
LEVERAGE              = int(os.getenv("LEVERAGE", "10"))

logger = logging.getLogger("signaler")


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
