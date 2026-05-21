#!/usr/bin/env python3
"""
backtest.py — Walk-forward backtesting for signaler_bot
========================================================
Phases:
  1. DATA   — download & cache 4 years of 1h + 4h OHLCV from Binance
  2. SIM    — candle-by-candle walk-forward simulation (exact bot logic)
  3. REPORT — trades.csv, equity.csv, stats.json, report.html

Usage:
    python backtest.py                      # full run
    python backtest.py --skip-download      # reuse cached data
    python backtest.py --download-only      # only refresh data cache
    python backtest.py --interval 1         # analysis every 1h (max fidelity, ~4x slower)
    python backtest.py --years 2            # only 2 years
    python backtest.py --pairs BTC/USDT,ETH/USDT  # subset of pairs

Results:
    results/trades.csv   — full trade log
    results/equity.csv   — equity snapshot every hour
    results/stats.json   — summary metrics
    results/report.html  — charts (equity, drawdown, monthly PnL, per-pair)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

# ── dotenv (optional) ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Bot modules ──────────────────────────────────────────────────────────────
from config import (
    TRADING_PAIRS, CANDLES_LIMIT, TOLERANCE_PERCENT,
    EXTREMA_WINDOW, TOP_N_LEVELS, HTF_STRUCTURE_WINDOW,
    ENTRY_PROXIMITY_PERCENT, EMA_PERIOD,
    ATR_PERIOD, SL_ATR_MULT, TP_ATR_MIN_MULT, MIN_RR,
    HTF_TIMEFRAME, HTF_EMA_PERIOD, ADX_PERIOD, ADX_MIN,
    RSI_PERIOD, CRASH_LOW_LOOKBACK, PUMP_HIGH_LOOKBACK,
    COMMISSION_RATE, BREAKEVEN_THRESHOLD, TRAILING_STOP, TRAILING_MULT,
    BOUNCE_CONFIRM, FIXED_RISK_MODE, RISK_PER_TRADE_PERCENT,
    MAX_TRADE_SIZE_PERCENT, HTF_SR_CANDLES, HTF_SR_REQUIRE_CONFIRM,
    EMA_CONFLUENCE_PERIODS, FVG_LOOKBACK, FVG_MIN_GAP_PCT,
    MIN_CONFLUENCE_SCORE, HTF_RSI_OVERBOUGHT, HTF_RSI_OVERSOLD,
    INITIAL_BALANCE, LEVERAGE, PENDING_EXPIRY_CHECKS, MAX_OPEN_TRADES,
    CORR_MAX, CORR_LOOKBACK, TRADE_SIZE_PERCENT,
)
from analysis import (
    find_support_resistance, find_entry_signals,
    _add_confluence_scores, _calc_ema_levels, _find_fvg_zones,
    _mark_htf_confirmed, _detect_htf_structure, _returns_cache,
)
from indicators import (
    _calc_adx, _calc_rsi_series, _calc_atr_ratio,
    _detect_regime, _find_crash_low, _find_pump_high,
)
from paper_trader import PaperPortfolio

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data" / "ohlcv"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Binance (public endpoint, no API key required) ───────────────────────────
def _binance() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True})


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

def _csv_path(pair: str, tf: str) -> Path:
    return DATA_DIR / pair.replace("/", "_") / f"{tf}.csv"


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index().astype(float)


def _download_pair(
    client: ccxt.binance,
    pair: str,
    tf: str,
    since_ms: int,
) -> Optional[pd.DataFrame]:
    """Paginated download of OHLCV. Returns DataFrame or None on failure."""
    tf_ms     = client.parse_timeframe(tf) * 1000
    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []
    since     = since_ms
    retries   = 0

    while True:
        try:
            batch = client.fetch_ohlcv(pair, tf, since=since, limit=1000)
            retries = 0
        except ccxt.RateLimitExceeded:
            time.sleep(15)
            continue
        except ccxt.BadSymbol:
            logger.warning("  %s not on Binance — skipped", pair)
            return None
        except Exception as exc:
            retries += 1
            if retries > 5:
                logger.error("  %s %s download failed: %s", pair, tf, exc)
                return None
            wait = 2 ** retries
            logger.warning("  retry %d/%d for %s %s in %ds", retries, 5, pair, tf, wait)
            time.sleep(wait)
            continue

        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= now_ms - tf_ms:
            break
        since = last_ts + tf_ms
        time.sleep(0.3)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates("timestamp")
          .set_index("timestamp")
          .sort_index()
          .astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    )
    logger.info("  %-12s %-3s  %d candles  %s → %s",
                pair, tf, len(df), df.index[0].date(), df.index[-1].date())
    return df


def phase1_download(pairs: List[str], years: int = 4, force: bool = False) -> None:
    since_ms  = int(
        (datetime.now(timezone.utc) - timedelta(days=years * 365 + 7)).timestamp() * 1000
    )
    since_str = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    client    = _binance()

    logger.info("=== PHASE 1: Downloading %dy history for %d pairs (since %s) ===",
                years, len(pairs), since_str)

    for pair in pairs:
        for tf in ("1h", "4h"):
            path = _csv_path(pair, tf)
            if path.exists() and not force:
                df = _load_df(path)
                logger.info("  [CACHED] %-12s %-3s  %d candles", pair, tf, len(df))
                continue
            df = _download_pair(client, pair, tf, since_ms)
            if df is not None:
                _save_df(df, path)

    logger.info("=== PHASE 1 complete ===\n")


# ============================================================
# BACKTEST PORTFOLIO — PaperPortfolio without file / GSheets I/O
# ============================================================

class BacktestPortfolio(PaperPortfolio):
    _sim_time: str = ""

    def _utcnow(self) -> str:
        return self._sim_time or super()._utcnow()

    def _connect_gsheets(self) -> None: pass
    def _load(self)          -> None: pass
    def _save(self)          -> None: pass


# ============================================================
# HTF CONFLUENCE FROM DATAFRAME (mirrors _fetch_htf_confluence)
# ============================================================

def _htf_from_df(df_4h: Optional[pd.DataFrame]) -> Tuple:
    """Return (trend, ema, htf_sr, structure, htf_rsi) from pre-loaded 4h data."""
    try:
        if df_4h is None or len(df_4h) < HTF_EMA_PERIOD:
            return None, None, [], "RANGE", None
        closes    = df_4h["close"].astype(float)
        ema_val   = float(closes.ewm(span=HTF_EMA_PERIOD, adjust=False).mean().iloc[-1])
        trend     = "UP" if float(closes.iloc[-1]) > ema_val else "DOWN"
        structure = _detect_htf_structure(df_4h)
        rsi_raw   = float(_calc_rsi_series(df_4h).iloc[-1])
        htf_rsi   = rsi_raw if not np.isnan(rsi_raw) else None
        htf_sr: List[Dict] = []
        if HTF_SR_CANDLES > 0 and len(df_4h) >= EXTREMA_WINDOW * 2 + 1:
            htf_sr = find_support_resistance(df_4h.tail(HTF_SR_CANDLES), min_touches=2, top_n=10)
        return trend, round(ema_val, 8), htf_sr, structure, htf_rsi
    except Exception as exc:
        logger.debug("HTF error: %s", exc)
        return None, None, [], "RANGE", None


# ============================================================
# POSITION CHECKS (mirrors signaler._run_position_checks)
# ============================================================

def _position_checks(
    pf: BacktestPortfolio,
    pair: str,
    high: float,
    low: float,
    close: float,
) -> List[Dict]:
    """Run breakeven → trailing → SL/TP → pending fill → repeat. Returns closed trades."""
    closed: List[Dict] = []

    pf.check_breakeven(pair, high, low)
    pf.check_trailing_stop(pair, high, low)
    closed.extend(pf.check_sl_tp(pair, high, low))

    triggered, _ = pf.check_pending_orders(
        pair, high, low, candle_close=close, require_bounce=BOUNCE_CONFIRM,
    )
    if triggered:
        pf.check_breakeven(pair, high, low)
        pf.check_trailing_stop(pair, high, low)

    return closed


# ============================================================
# ORDER PLACEMENT (mirrors signaler._apply_orders)
# ============================================================

def _apply_orders(pf: BacktestPortfolio, pair: str, signals: List[Dict]) -> None:
    best: Dict[str, Optional[Dict]] = {"LONG": None, "SHORT": None}
    for sig in sorted(signals, key=lambda s: (
        not s["level"].get("htf_confirmed", False),
        -s["level"].get("confluence_score", 0),
        s["distance_percent"],
    )):
        if best[sig["direction"]] is None:
            best[sig["direction"]] = sig

    # Correlation filter (uses _returns_cache populated in sim loop)
    if CORR_MAX > 0:
        open_pairs = [t["pair"] for t in pf.open_trades if t["pair"] != pair]
        r1 = _returns_cache.get(pair)
        if r1 is not None:
            for op in open_pairs:
                r2 = _returns_cache.get(op)
                if r2 is None:
                    continue
                combined = pd.concat([r1, r2], axis=1).dropna()
                if len(combined) >= 10 and abs(combined.iloc[:, 0].corr(combined.iloc[:, 1])) > CORR_MAX:
                    best = {"LONG": None, "SHORT": None}
                    break

    # Cancel stale pending orders that no longer match best signal
    tol = TOLERANCE_PERCENT / 100
    for order in list(pf.pending_orders):
        if order["pair"] != pair:
            continue
        b = best.get(order["direction"])
        if b is None or abs(b["level"]["price"] - order["entry_price"]) / order["entry_price"] > tol:
            pf.cancel_order(order)

    for sig in best.values():
        if sig is None or sig["tp"] is None:
            continue
        pf.create_pending_order(
            pair=pair,
            direction=sig["direction"],
            entry_price=sig["level"]["price"],
            sl=sig["sl"],
            tp=sig["tp"],
        )


# ============================================================
# PHASE 2 — WALK-FORWARD SIMULATION
# ============================================================

def phase2_simulate(
    pairs: List[str],
    analysis_interval_h: int = 4,
) -> Tuple[BacktestPortfolio, List[Dict]]:
    """
    Walk-forward simulation.
    Returns (portfolio, equity_log).
    """
    # ── Load cached data ──────────────────────────────────────
    logger.info("=== PHASE 2: Loading data ===")
    data_1h: Dict[str, pd.DataFrame] = {}
    data_4h: Dict[str, Optional[pd.DataFrame]] = {}
    active: List[str] = []

    for pair in pairs:
        p1 = _csv_path(pair, "1h")
        p4 = _csv_path(pair, "4h")
        if not p1.exists():
            logger.warning("  No 1h data for %s — skipped (run without --skip-download)", pair)
            continue
        data_1h[pair] = _load_df(p1)
        data_4h[pair] = _load_df(p4) if p4.exists() else None
        active.append(pair)
        logger.info("  %-12s  1h:%d  4h:%s", pair, len(data_1h[pair]),
                    str(len(data_4h[pair])) if data_4h[pair] is not None else "N/A")

    if not active:
        logger.error("No data found. Run without --skip-download first.")
        sys.exit(1)

    # ── Build unified 1h timeline ─────────────────────────────
    all_ts = sorted(set().union(*[set(data_1h[p].index) for p in active]))
    logger.info("Timeline: %d hourly steps  %s → %s\n",
                len(all_ts), all_ts[0].date(), all_ts[-1].date())

    # ── Portfolio ──────────────────────────────────────────────
    pf = BacktestPortfolio(
        initial_balance        = INITIAL_BALANCE,
        trade_size_percent     = TRADE_SIZE_PERCENT,
        max_open_trades        = MAX_OPEN_TRADES,
        pending_expiry_checks  = PENDING_EXPIRY_CHECKS,
        leverage               = LEVERAGE,
        commission_rate        = COMMISSION_RATE,
        breakeven_threshold    = BREAKEVEN_THRESHOLD,
        fixed_risk_mode        = FIXED_RISK_MODE,
        risk_per_trade_percent = RISK_PER_TRADE_PERCENT,
        max_trade_size_percent = MAX_TRADE_SIZE_PERCENT,
        trailing_stop          = TRAILING_STOP,
        trailing_mult          = TRAILING_MULT,
    )

    equity_log: List[Dict]       = []
    last_analysis: Dict[str, Optional[pd.Timestamp]] = {p: None for p in active}
    MIN_BARS = EXTREMA_WINDOW * 2 + 5

    # ── Progress bar (optional) ────────────────────────────────
    try:
        from tqdm import tqdm
        ts_iter = tqdm(all_ts, desc="Simulating", unit="h", ncols=90)
    except ImportError:
        ts_iter = all_ts

    # ── Main loop ──────────────────────────────────────────────
    for ts in ts_iter:
        pf._sim_time = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        prices: Dict[str, float] = {}

        for pair in active:
            df_full = data_1h[pair]
            if ts not in df_full.index:
                continue

            # Candles UP TO current timestamp — zero look-ahead
            df = df_full.loc[:ts].tail(CANDLES_LIMIT)
            if len(df) < MIN_BARS:
                continue

            candle = df.iloc[-1]
            h, l, c = float(candle["high"]), float(candle["low"]), float(candle["close"])
            prices[pair] = c

            # Update correlation returns cache (same dict used by analysis.py)
            _returns_cache[pair] = df["close"].pct_change().dropna().tail(CORR_LOOKBACK)

            # ── Position checks every candle ───────────────────
            _position_checks(pf, pair, h, l, c)

            # ── Analysis every analysis_interval_h ────────────
            last = last_analysis[pair]
            if last is not None and (ts - last) < pd.Timedelta(hours=analysis_interval_h):
                continue
            last_analysis[pair] = ts

            # 4h slice (no look-ahead)
            df_4h: Optional[pd.DataFrame] = None
            raw_4h = data_4h.get(pair)
            if raw_4h is not None:
                sliced = raw_4h.loc[:ts].tail(max(HTF_SR_CANDLES, HTF_EMA_PERIOD + 30))
                if len(sliced) >= HTF_EMA_PERIOD:
                    df_4h = sliced

            # HTF confluence
            htf_trend, htf_ema, htf_sr, htf_structure, htf_rsi = _htf_from_df(df_4h)

            # S/R levels
            levels = find_support_resistance(df)

            # Confluence scoring
            ema_lvls  = _calc_ema_levels(df, EMA_CONFLUENCE_PERIODS)
            fvg_zones = _find_fvg_zones(df, FVG_LOOKBACK, FVG_MIN_GAP_PCT)
            _add_confluence_scores(levels, ema_lvls, fvg_zones, TOLERANCE_PERCENT)
            if htf_sr:
                _mark_htf_confirmed(levels, htf_sr, TOLERANCE_PERCENT)

            # Indicators
            adx_val, _, _ = _calc_adx(df, ADX_PERIOD)
            rsi_series    = _calc_rsi_series(df, RSI_PERIOD)
            rsi_raw       = float(rsi_series.iloc[-1])
            rsi_val       = rsi_raw if not np.isnan(rsi_raw) else 50.0
            atr_ratio     = _calc_atr_ratio(df, ATR_PERIOD)
            regime        = _detect_regime(rsi_series, atr_ratio)
            crash_low     = _find_crash_low(df, CRASH_LOW_LOOKBACK) if regime == "RECOVERY"   else None
            pump_high     = _find_pump_high(df, PUMP_HIGH_LOOKBACK) if regime == "CORRECTION" else None

            signal_levels = (
                [lv for lv in levels if lv.get("htf_confirmed")]
                if HTF_SR_REQUIRE_CONFIRM and htf_sr
                else levels
            )

            signals = find_entry_signals(
                df, signal_levels, c,
                htf_trend    = htf_trend,
                adx_val      = adx_val,
                regime       = regime,
                crash_low    = crash_low,
                pump_high    = pump_high,
                htf_structure= htf_structure,
                rsi_series   = rsi_series,
                htf_rsi      = htf_rsi,
            )

            _apply_orders(pf, pair, signals)

        # Equity snapshot
        pf._live_prices.update(prices)
        equity_log.append({
            "timestamp": ts.isoformat(),
            "equity":    pf.get_equity(prices),
            "balance":   pf.balance,
        })

    logger.info("\n=== PHASE 2 complete: %d closed / %d open trades ===\n",
                len(pf.closed_trades), len(pf.open_trades))
    return pf, equity_log


# ============================================================
# PHASE 3 — STATISTICS
# ============================================================

def _compute_stats(pf: BacktestPortfolio, equity_log: List[Dict]) -> Dict:
    closed = pf.closed_trades
    wins   = [t for t in closed if (t.get("pnl_usd") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl_usd") or 0) < 0]

    pnls      = [t.get("pnl_usd") or 0 for t in closed]
    total_pnl = sum(pnls)

    # Per-pair breakdown
    per_pair: Dict[str, Dict] = {}
    for t in closed:
        pair = t.get("pair", "?")
        d    = per_pair.setdefault(pair, {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        d["trades"] += 1
        if (t.get("pnl_usd") or 0) > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1
        d["pnl"] += t.get("pnl_usd") or 0

    # Monthly PnL
    monthly: Dict[str, float] = {}
    for t in closed:
        ts = (t.get("closed_at") or "")[:7]
        if ts:
            monthly[ts] = monthly.get(ts, 0.0) + (t.get("pnl_usd") or 0)

    # Equity curve metrics
    eq = np.array([e["equity"] for e in equity_log]) if equity_log else np.array([INITIAL_BALANCE])
    peak    = np.maximum.accumulate(eq)
    dd_pct  = (eq - peak) / np.where(peak > 0, peak, 1) * 100
    max_dd  = float(dd_pct.min())

    # Sharpe (annualised, hourly returns)
    sharpe = 0.0
    if len(eq) > 2:
        r = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        if r.std() > 0:
            sharpe = float(r.mean() / r.std() * np.sqrt(8760))

    # Average hold time
    hold_h = []
    for t in closed:
        try:
            opened  = datetime.strptime(t["opened_at"],  "%Y-%m-%d %H:%M:%S UTC")
            closed_ = datetime.strptime(t["closed_at"],  "%Y-%m-%d %H:%M:%S UTC")
            hold_h.append((closed_ - opened).total_seconds() / 3600)
        except Exception:
            pass

    # Close reason breakdown
    reasons: Dict[str, int] = {}
    for t in closed:
        r = t.get("close_reason") or "?"
        reasons[r] = reasons.get(r, 0) + 1

    final_eq  = float(eq[-1])
    win_gross = sum((t.get("pnl_usd") or 0) for t in wins)
    loss_gross= abs(sum((t.get("pnl_usd") or 0) for t in losses))

    return {
        "period": {
            "start": equity_log[0]["timestamp"][:10]  if equity_log else "",
            "end":   equity_log[-1]["timestamp"][:10] if equity_log else "",
        },
        "balance": {
            "initial":        INITIAL_BALANCE,
            "final_equity":   round(final_eq, 2),
            "total_pnl_usd":  round(total_pnl, 2),
            "total_return_pct": round((final_eq - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        },
        "risk": {
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio":     round(sharpe, 2),
        },
        "trades": {
            "total":          len(closed),
            "wins":           len(wins),
            "losses":         len(losses),
            "winrate_pct":    round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
            "profit_factor":  round(win_gross / loss_gross, 2) if loss_gross > 0 else None,
            "avg_win_usd":    round(float(np.mean([t.get("pnl_usd", 0) for t in wins])),  2) if wins   else 0.0,
            "avg_loss_usd":   round(float(np.mean([t.get("pnl_usd", 0) for t in losses])),2) if losses else 0.0,
            "avg_hold_hours": round(float(np.mean(hold_h)), 1) if hold_h else 0.0,
            "close_reasons":  reasons,
        },
        "orders": {
            "created":   pf.orders_created,
            "cancelled": pf.orders_cancelled,
            "triggered": len(pf.trades),
        },
        "monthly_pnl": {k: round(v, 2) for k, v in sorted(monthly.items())},
        "per_pair": {
            pair: {
                "trades":   d["trades"],
                "wins":     d["wins"],
                "losses":   d["losses"],
                "winrate":  round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0.0,
                "pnl_usd":  round(d["pnl"], 2),
            }
            for pair, d in sorted(per_pair.items(), key=lambda x: -x[1]["pnl"])
        },
    }


# ============================================================
# PHASE 3 — HTML REPORT
# ============================================================

def _generate_html(
    stats: Dict,
    equity_log: List[Dict],
    pf: BacktestPortfolio,
) -> str:
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fafafa"})

    def _b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    imgs: List[str] = []

    df_eq = pd.DataFrame(equity_log)
    df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])

    # ── 1. Equity curve + drawdown ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df_eq["timestamp"], df_eq["equity"],  color="#1976D2", lw=1.3, label="Equity")
    ax1.plot(df_eq["timestamp"], df_eq["balance"], color="#43A047", lw=0.8, alpha=0.6, label="Balance")
    ax1.axhline(stats["balance"]["initial"], color="#9E9E9E", ls="--", lw=0.7,
                label=f"Start ${stats['balance']['initial']:.0f}")
    ax1.set_ylabel("USD")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    eq_arr = np.array(df_eq["equity"])
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / np.where(peak > 0, peak, 1) * 100
    ax2.fill_between(df_eq["timestamp"], dd, 0, color="#E53935", alpha=0.55, label="Drawdown")
    ax2.set_ylabel("DD %")
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    imgs.append(_b64(fig))

    # ── 2. Monthly PnL ────────────────────────────────────────
    monthly = stats.get("monthly_pnl", {})
    if monthly:
        months = list(monthly.keys())
        mpnls  = [monthly[m] for m in months]
        colors = ["#43A047" if v >= 0 else "#E53935" for v in mpnls]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(months)), mpnls, color=colors, width=0.75, edgecolor="white", lw=0.4)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("PnL USD")
        ax.set_title("Monthly PnL", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        imgs.append(_b64(fig))

    # ── 3. Per-pair PnL ───────────────────────────────────────
    pp = stats.get("per_pair", {})
    if pp:
        items  = sorted(pp.items(), key=lambda x: x[1]["pnl_usd"])
        names  = [p for p, _ in items]
        ppnls  = [d["pnl_usd"] for _, d in items]
        pcolors= ["#43A047" if v >= 0 else "#E53935" for v in ppnls]
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
        ax.barh(names, ppnls, color=pcolors, height=0.65, edgecolor="white", lw=0.4)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("PnL USD")
        ax.set_title("PnL by Pair", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        imgs.append(_b64(fig))

    # ── 4. Win-rate by pair ───────────────────────────────────
    if pp:
        wr_items = sorted(pp.items(), key=lambda x: -x[1]["winrate"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([p for p, _ in wr_items], [d["winrate"] for _, d in wr_items],
               color="#FF8F00", width=0.65, edgecolor="white", lw=0.4)
        ax.axhline(50, color="#9E9E9E", ls="--", lw=0.8)
        overall_wr = stats["trades"]["winrate_pct"]
        ax.axhline(overall_wr, color="#1976D2", ls="--", lw=0.8,
                   label=f"Overall {overall_wr:.1f}%")
        ax.set_ylabel("Win Rate %")
        ax.set_title("Win Rate by Pair", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        imgs.append(_b64(fig))

    # ── 5. Trade PnL distribution ─────────────────────────────
    closed = pf.closed_trades
    if closed:
        pdist = [t.get("pnl_usd") or 0 for t in closed]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(pdist, bins=min(50, max(10, len(pdist) // 5)),
                color="#1976D2", edgecolor="white", lw=0.4, alpha=0.8)
        ax.axvline(0, color="#E53935", lw=1.2, ls="--")
        avg = np.mean(pdist)
        ax.axvline(avg, color="#43A047", lw=1, ls="--", label=f"Avg ${avg:.2f}")
        ax.set_xlabel("PnL USD")
        ax.set_ylabel("Count")
        ax.set_title("Trade PnL Distribution", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        imgs.append(_b64(fig))

    # ── 6. Cumulative trades timeline ─────────────────────────
    if closed:
        ts_list = []
        for t in sorted(closed, key=lambda x: x.get("closed_at") or ""):
            try:
                ts_list.append(datetime.strptime(t["closed_at"], "%Y-%m-%d %H:%M:%S UTC"))
            except Exception:
                pass
        if ts_list:
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(ts_list, range(1, len(ts_list) + 1), color="#7B1FA2", lw=1.2)
            ax.set_ylabel("Cumulative trades")
            ax.set_title("Trade Activity", fontweight="bold")
            ax.grid(alpha=0.25)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate(rotation=30)
            imgs.append(_b64(fig))

    # ── Assemble HTML ─────────────────────────────────────────
    s  = stats
    b  = s["balance"]
    tr = s["trades"]
    ri = s["risk"]
    or_= s["orders"]
    p  = s["period"]

    ret_color = "#2E7D32" if b["total_return_pct"] >= 0 else "#C62828"
    pf_val    = f"{tr['profit_factor']:.2f}" if tr["profit_factor"] else "N/A"

    def card(label, value, color="#333"):
        return (f'<div class="card"><div class="lbl">{label}</div>'
                f'<div class="val" style="color:{color}">{value}</div></div>')

    cards = "\n".join([
        card("Final Equity",   f"${b['final_equity']:,.2f}"),
        card("Total Return",   f"{b['total_return_pct']:+.2f}%",   ret_color),
        card("Total PnL",      f"${b['total_pnl_usd']:+.2f}",      ret_color),
        card("Max Drawdown",   f"{ri['max_drawdown_pct']:.2f}%",   "#C62828"),
        card("Sharpe Ratio",   f"{ri['sharpe_ratio']:.2f}"),
        card("Win Rate",       f"{tr['winrate_pct']:.1f}%"),
        card("Profit Factor",  pf_val),
        card("Total Trades",   str(tr["total"])),
        card("Avg Hold",       f"{tr['avg_hold_hours']}h"),
        card("Avg Win",        f"${tr['avg_win_usd']:.2f}",  "#2E7D32"),
        card("Avg Loss",       f"${tr['avg_loss_usd']:.2f}", "#C62828"),
        card("Orders Created", str(or_["created"])),
    ])

    monthly_rows = "".join(
        f"<tr><td>{m}</td>"
        f"<td style='background:{'#c8f7c5' if v >= 0 else '#ffc9c9'};text-align:right'>${v:+.2f}</td></tr>"
        for m, v in sorted(s.get("monthly_pnl", {}).items())
    )

    pair_rows = "".join(
        f"<tr><td>{pair}</td><td>{d['trades']}</td><td>{d['wins']}</td>"
        f"<td>{d['losses']}</td><td>{d['winrate']:.1f}%</td>"
        f"<td style='background:{'#c8f7c5' if d['pnl_usd'] >= 0 else '#ffc9c9'};text-align:right'>${d['pnl_usd']:+.2f}</td></tr>"
        for pair, d in s.get("per_pair", {}).items()
    )

    img_html = "\n".join(
        f'<img src="data:image/png;base64,{img}" style="width:100%;max-width:1100px;margin-bottom:18px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.12)">'
        for img in imgs
    )

    reasons_html = "".join(
        f"<tr><td>{r}</td><td>{n}</td></tr>"
        for r, n in sorted(tr.get("close_reasons", {}).items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Report</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px;background:#f0f2f5;color:#222}}
  h1{{margin:0 0 4px;font-size:26px}} h2{{font-size:16px;color:#555;border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:32px}}
  .meta{{color:#888;font-size:13px;margin-bottom:20px}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:24px}}
  .card{{background:#fff;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
  .lbl{{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:.5px}}
  .val{{font-size:20px;font-weight:700;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:20px}}
  th{{background:#37474F;color:#fff;padding:9px 14px;text-align:left;font-size:12px;font-weight:600}}
  td{{padding:7px 14px;border-bottom:1px solid #f0f0f0;font-size:13px}}
  tr:last-child td{{border:none}}
  .charts{{margin-top:8px}}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<div class="meta">
  Period: <b>{p['start']}</b> → <b>{p['end']}</b> &nbsp;·&nbsp;
  Pairs: <b>{len(s.get('per_pair', {}))} pairs</b> &nbsp;·&nbsp;
  Initial balance: <b>${b['initial']:,.2f}</b> &nbsp;·&nbsp;
  Leverage: <b>{LEVERAGE}×</b> &nbsp;·&nbsp;
  Risk/trade: <b>{RISK_PER_TRADE_PERCENT}%</b> &nbsp;·&nbsp;
  Commission: <b>{COMMISSION_RATE*100:.2f}%</b>
</div>

<div class="summary">{cards}</div>

<h2>Charts</h2>
<div class="charts">{img_html}</div>

<h2>Monthly PnL</h2>
<table><tr><th>Month</th><th>PnL USD</th></tr>{monthly_rows}</table>

<h2>Per-Pair Statistics</h2>
<table>
  <tr><th>Pair</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>PnL USD</th></tr>
  {pair_rows}
</table>

<h2>Close Reasons</h2>
<table><tr><th>Reason</th><th>Count</th></tr>{reasons_html}</table>

</body>
</html>"""


# ============================================================
# PHASE 3 — REPORT ENTRY POINT
# ============================================================

def phase3_report(pf: BacktestPortfolio, equity_log: List[Dict]) -> None:
    logger.info("=== PHASE 3: Generating reports ===")

    stats = _compute_stats(pf, equity_log)

    # trades.csv
    all_trades = pf.trades  # includes both OPEN and CLOSED
    if all_trades:
        pd.DataFrame(all_trades).to_csv(RESULTS_DIR / "trades.csv", index=False)
        logger.info("  trades.csv  — %d records", len(all_trades))

    # equity.csv
    if equity_log:
        pd.DataFrame(equity_log).to_csv(RESULTS_DIR / "equity.csv", index=False)
        logger.info("  equity.csv  — %d rows", len(equity_log))

    # stats.json
    with open(RESULTS_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("  stats.json  — saved")

    # report.html
    try:
        html = _generate_html(stats, equity_log, pf)
        (RESULTS_DIR / "report.html").write_text(html, encoding="utf-8")
        logger.info("  report.html — saved")
    except ImportError:
        logger.warning("  matplotlib not installed — skipping HTML report")
        logger.warning("  Install with: pip install matplotlib")
    except Exception as exc:
        logger.error("  HTML report failed: %s", exc)

    # Console summary
    b  = stats["balance"]
    tr = stats["trades"]
    ri = stats["risk"]
    print()
    print("=" * 58)
    print("  BACKTEST SUMMARY")
    print("=" * 58)
    print(f"  Period:        {stats['period']['start']} → {stats['period']['end']}")
    print(f"  Initial:       ${b['initial']:,.2f}")
    print(f"  Final Equity:  ${b['final_equity']:,.2f}")
    print(f"  Total Return:  {b['total_return_pct']:+.2f}%")
    print(f"  Total PnL:     ${b['total_pnl_usd']:+.2f}")
    print(f"  Max Drawdown:  {ri['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:  {ri['sharpe_ratio']:.2f}")
    print(f"  Trades:        {tr['total']}  ({tr['wins']}W / {tr['losses']}L)")
    print(f"  Win Rate:      {tr['winrate_pct']:.1f}%")
    print(f"  Profit Factor: {tr['profit_factor'] or 'N/A'}")
    print(f"  Avg Hold:      {tr['avg_hold_hours']}h")
    print(f"  Orders:        {stats['orders']['created']} created / "
          f"{stats['orders']['cancelled']} cancelled")
    print("=" * 58)
    print(f"\n  Results saved to: {RESULTS_DIR.resolve()}")
    print()

    logger.info("=== PHASE 3 complete ===")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward backtest for signaler_bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--download-only",  action="store_true",
                    help="Only download/refresh data, skip simulation")
    ap.add_argument("--skip-download",  action="store_true",
                    help="Skip download, use cached CSV files")
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download all data even if cached")
    ap.add_argument("--years",    type=int, default=4,
                    help="Years of history to use (default: 4)")
    ap.add_argument("--interval", type=int, default=4,
                    help="Analysis interval in hours (default: 4; use 1 for max fidelity)")
    ap.add_argument("--pairs",    type=str, default=None,
                    help="Comma-separated pairs, e.g. BTC/USDT,ETH/USDT")
    args = ap.parse_args()

    pairs = (
        [p.strip() for p in args.pairs.split(",") if p.strip()]
        if args.pairs
        else TRADING_PAIRS
    )

    if not args.skip_download:
        phase1_download(pairs, years=args.years, force=args.force_download)

    if args.download_only:
        logger.info("--download-only: done.")
        return

    pf, equity_log = phase2_simulate(pairs, analysis_interval_h=args.interval)
    phase3_report(pf, equity_log)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
