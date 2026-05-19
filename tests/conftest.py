"""Shared stubs so telegram/ccxt don't need real credentials."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _make_telegram_stubs():
    names = [
        "telegram", "telegram.ext", "telegram.ext._application",
        "telegram.constants", "telegram.error", "telegram.ext.filters",
    ]
    for name in names:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m

    tg  = sys.modules["telegram"]
    ext = sys.modules["telegram.ext"]

    for cls in ["Update", "Bot", "Message", "Chat", "InlineKeyboardButton",
                "InlineKeyboardMarkup", "CallbackQuery",
                "ReplyKeyboardMarkup", "KeyboardButton"]:
        setattr(tg, cls, type(cls, (), {"__init__": lambda self, *a, **kw: None}))

    for cls in ["Application", "ApplicationBuilder", "CommandHandler",
                "MessageHandler", "CallbackQueryHandler", "ConversationHandler"]:
        setattr(ext, cls, type(cls, (), {"__init__": lambda self, *a, **kw: None,
                                         "DEFAULT_TYPE": None}))

    class ContextTypes:
        DEFAULT_TYPE = None
        def __init__(self, *a, **kw): pass
    ext.ContextTypes = ContextTypes

    const = sys.modules["telegram.constants"]
    const.ParseMode = type("ParseMode", (), {"MARKDOWN_V2": "MarkdownV2", "HTML": "HTML"})()

    class TelegramError(Exception): pass
    sys.modules["telegram.error"].TelegramError = TelegramError
    sys.modules["telegram.ext.filters"].Filters = type("Filters", (), {"text": None})()


_make_telegram_stubs()

for k, v in [("MEXC_API_KEY", "t"), ("MEXC_SECRET_KEY", "t"),
             ("TELEGRAM_BOT_TOKEN", "1:a"), ("TELEGRAM_CHAT_ID", "-1")]:
    os.environ.setdefault(k, v)


# ── helpers ────────────────────────────────────────────────────────────────────

def make_df(closes, highs=None, lows=None, opens=None, volumes=None, n=None):
    """Build a minimal OHLCV DataFrame."""
    if n and not closes:
        closes = list(range(100, 100 + n))
    closes  = list(closes)
    size    = len(closes)
    highs   = list(highs)   if highs   else [c * 1.01 for c in closes]
    lows    = list(lows)    if lows    else [c * 0.99 for c in closes]
    opens   = list(opens)   if opens   else closes[:]
    volumes = list(volumes) if volumes else [1000.0] * size
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


def sine_df(periods=100, amplitude=10, base=100.0):
    """Sinusoidal price series — useful for ADX / RSI / ATR stability tests."""
    x = np.linspace(0, 4 * np.pi, periods)
    closes = base + amplitude * np.sin(x)
    return make_df(closes)
