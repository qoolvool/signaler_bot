"""Форматирование Telegram-сообщений."""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from config import (
    BOT_VERSION, HTF_TIMEFRAME, HTF_EMA_PERIOD, EMA_PERIOD, ADX_MIN,
    RSI_OVERSOLD, RSI_RECOVERY_MAX, RSI_OVERBOUGHT, RSI_CORRECTION_MIN,
)
from paper_trader import PaperPortfolio

logger = logging.getLogger("signaler")


def _fp(price: float) -> str:
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
    htf_structure: Optional[str] = None,
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📊 <b>{pair}</b>  •  <code>{timeframe}</code>",
        f"💰 Цена: <b>{_fp(current_price)}</b>",
        f"🕒 {now}",
    ]
    htf_str = ""
    if htf_trend:
        ht = "↑ UP" if htf_trend == "UP" else "↓ DOWN"
        htf_str = f"  •  {HTF_TIMEFRAME} EMA{HTF_EMA_PERIOD}: <b>{ht}</b>"
        if htf_ema:
            htf_str += f" ({_fp(htf_ema)})"
    struct_str = ""
    if htf_structure:
        struct_map = {"BULLISH": "HH/HL ↑", "BEARISH": "LH/LL ↓", "RANGE": "↔ боковик"}
        struct_str = f"  •  Структура: <b>{struct_map.get(htf_structure, htf_structure)}</b>"
    adx_str = ""
    if adx is not None:
        label = "тренд" if adx >= ADX_MIN else "боковик"
        adx_str = f"  •  ADX: <b>{adx:.1f}</b> ({label})"
    if htf_str or struct_str or adx_str:
        lines.append(f"📡{htf_str}{struct_str}{adx_str}")

    rsi_str = ""
    if rsi is not None:
        if rsi < RSI_OVERSOLD:          rsi_label = "🔴 перепродан"
        elif rsi < RSI_RECOVERY_MAX:    rsi_label = "🟡 восст."
        elif rsi > RSI_OVERBOUGHT:      rsi_label = "🔴 перекуплен"
        elif rsi > RSI_CORRECTION_MIN:  rsi_label = "🟡 корр."
        else:                           rsi_label = "🟢 норма"
        rsi_str = f"RSI: <b>{rsi:.1f}</b> ({rsi_label})"
    ratio_str = f"ATR×: <b>{atr_ratio:.1f}</b>" if atr_ratio is not None else ""
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
            adx_s      = f"  •  ADX:{sig['adx']}" if sig.get("adx") is not None else ""
            htf_mark   = " ✨" if lvl.get("htf_confirmed") else ""
            div_mark   = " 📊" if sig.get("divergence") else ""
            patt_mark  = " 🕯" if sig.get("pattern_confirmed") else ""
            vol_mark   = " ⚡" if sig.get("volume_spike") else ""
            cf_tags    = lvl.get("confluence_tags", [])
            cf_mark    = f" [{'+'.join(cf_tags)}]" if cf_tags else ""
            lines.append(
                f"{de} <b>{sig['direction']}</b>  вблизи {_fp(lvl['price'])}"
                f"{htf_mark}{div_mark}{patt_mark}{vol_mark}{cf_mark}"
                f"  <i>({sig['distance_percent']}%)</i>"
            )
            lines.append(
                f"   {ta_30} EMA{EMA_PERIOD}={_fp(sig['ema'])}{ta_htf}{adx_s}"
                f"  •  ATR={_fp(sig['atr'])}"
            )
            trig_parts = []
            if sig.get("pattern"):
                patt_str = f"🕯 {sig['pattern']}" if sig.get("pattern_confirmed") else sig["pattern"]
                trig_parts.append(patt_str)
            if sig.get("vol_ratio") is not None:
                vol_str = (f"⚡ объём ×{sig['vol_ratio']}" if sig.get("volume_spike")
                           else f"объём ×{sig['vol_ratio']}")
                trig_parts.append(vol_str)
            if trig_parts:
                lines.append(f"   Триггер: {'  •  '.join(trig_parts)}")
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
        emoji    = "🔴" if lvl["type"] == "RESISTANCE" else "🟢"
        ru_type  = "Сопр." if lvl["type"] == "RESISTANCE" else "Подд."
        dist     = (lvl["price"] - current_price) / current_price * 100
        sign     = "+" if dist >= 0 else ""
        retest   = " ✅" if lvl.get("has_retest") else ""
        htf_mark = " ✨" if lvl.get("htf_confirmed") else ""
        cf_tags  = lvl.get("confluence_tags", [])
        cf_mark  = f" [{'+'.join(cf_tags)}]" if cf_tags else ""
        lines.append(
            f"{emoji} <b>{_fp(lvl['price'])}</b> — {ru_type} "
            f"(касаний: <b>{lvl['touches']}</b>, {sign}{dist:.2f}%{retest}{htf_mark}{cf_mark})"
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
    won      = (trade["pnl_usd"] or 0) > 0
    emoji    = "✅" if won else "❌"
    reason   = "ТЕЙК-ПРОФИТ 🎯" if trade["close_reason"] == "TP" else "СТОП-ЛОСС 🛑"
    sign     = "+" if (trade["pnl_usd"] or 0) >= 0 else ""
    comm     = trade.get("commission")
    comm_line = f"\nКомиссия: <b>-${comm}</b>" if comm else ""
    be_line  = "\n<i>SL был перенесён в безубыток</i>" if trade.get("sl_at_breakeven") else ""
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
            ntl   = t.get("notional", t["size_usd"])
            upnl  = ((cur - t["entry_price"]) / t["entry_price"] * ntl
                     if t["direction"] == "LONG"
                     else (t["entry_price"] - cur) / t["entry_price"] * ntl)
            upnl_pct = upnl / t["size_usd"] * 100
            sign     = "+" if upnl >= 0 else ""
            pnl_line = f"\n   PnL: <b>{sign}${upnl:.2f}  ({sign}{upnl_pct:.1f}%)</b>  •  Цена: {_fp(cur)}"
        else:
            pnl_line = ""
        ep     = t["entry_price"]
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
