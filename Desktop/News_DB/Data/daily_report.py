#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盤後綜合日報 — 整合所有已收集資料產生完整報告

包含:
  1. 績效總覽（今日/5日/20日準確率 + 方向分類）
  2. 系統健康度（修正因子、連勝連錯、出手率）
  3. AI 交易 PK + 績效評分
  4. 焦點股覆盤（含盤中買賣點、日內高低、支撐壓力）
  5. 每檔歷史命中率排行
  6. 信號效能統計

@author: rubylintu
"""

import os
import datetime
import logging
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

# Discord 顏色
COLOR_INFO = 0x3498DB
COLOR_GOOD = 0x2ECC71
COLOR_WARN = 0xFFAA00
COLOR_BAD = 0xE74C3C


# ============================================================
# 1. 績效總覽
# ============================================================

def build_accuracy_embed(today_metrics, advanced_metrics, correction):
    """
    績效總覽 Embed

    Args:
        today_metrics: get_tracking_metrics() 回傳
        advanced_metrics: calc_advanced_metrics() 回傳
        correction: calc_correction_factor() 回傳
    """
    today = datetime.date.today()
    hit = today_metrics.get('today_hit_rate', 0)
    color = COLOR_GOOD if hit >= 0.65 else COLOR_WARN if hit >= 0.50 else COLOR_BAD

    # 方向分類
    by_dir = advanced_metrics.get('by_direction', {})
    bull = by_dir.get('漲', {})
    bear = by_dir.get('跌', {})

    bull_str = f"🔴 漲: {bull.get('correct',0)}/{bull.get('count',0)}"
    if bull.get('count', 0) > 0:
        bull_str += f" ({bull['accuracy']:.0%})"
    bear_str = f"🟢 跌: {bear.get('correct',0)}/{bear.get('count',0)}"
    if bear.get('count', 0) > 0:
        bear_str += f" ({bear['accuracy']:.0%})"

    # 連勝/連錯
    streak = today_metrics.get('current_streak', 0)
    if streak > 0:
        streak_str = f"🔥 連對 {streak}"
    elif streak < 0:
        streak_str = f"💀 連錯 {abs(streak)}"
    else:
        streak_str = "—"

    # 修正因子
    bf = correction.get('bullish_factor', 1.0)
    bef = correction.get('bearish_factor', 1.0)
    corr_str = f"漲向 {bf:.2f} / 跌向 {bef:.2f}"
    if bf < 0.9 or bef < 0.9:
        corr_str += " ⚠️"

    fields = [
        {
            "name": "今日",
            "value": (f"**{today_metrics.get('today_correct',0)}"
                      f"/{today_metrics.get('today_predictions',0)}"
                      f" ({hit:.0%})**"),
            "inline": True,
        },
        {
            "name": "近20筆",
            "value": f"**{today_metrics.get('recent_20_hit_rate',0):.0%}**",
            "inline": True,
        },
        {
            "name": "整體",
            "value": f"**{advanced_metrics.get('overall_accuracy',0):.0%}**",
            "inline": True,
        },
        {
            "name": "方向分類",
            "value": f"{bull_str}\n{bear_str}",
            "inline": True,
        },
        {
            "name": "出手率 / 準度",
            "value": (f"📊 {advanced_metrics.get('coverage',0):.0%}"
                      f" / {advanced_metrics.get('precision',0):.0%}"),
            "inline": True,
        },
        {
            "name": "狀態",
            "value": (f"{streak_str}\n"
                      f"最大連錯: {today_metrics.get('max_consecutive_loss',0)}"),
            "inline": True,
        },
        {
            "name": "修正因子",
            "value": corr_str,
            "inline": False,
        },
    ]

    return {
        "title": f"📊 績效總覽 | {today}",
        "color": color,
        "fields": fields,
    }


# ============================================================
# 2. AI 交易 PK + 績效評分
# ============================================================

def build_pk_report_embed(gpt_trader, gemini_trader, closing_prices):
    """
    GPT vs Gemini PK 日報 + 績效評分

    Args:
        gpt_trader: AITrader instance
        gemini_trader: GeminiTrader instance
        closing_prices: {code: price} 收盤價
    """
    today = datetime.date.today()

    gpt_sum = gpt_trader.get_portfolio_summary(closing_prices)
    gem_sum = gemini_trader.get_portfolio_summary(closing_prices)

    # 績效評分
    try:
        gpt_report, gpt_score = gpt_trader.build_performance_report()
        gpt_grade = _score_to_grade(gpt_score)
    except Exception:
        gpt_score, gpt_grade, gpt_report = 0, '?', ''

    try:
        gem_report, gem_score = gemini_trader.build_performance_report()
        gem_grade = _score_to_grade(gem_score)
    except Exception:
        gem_score, gem_grade, gem_report = 0, '?', ''

    # 勝負判定
    gpt_ret = gpt_sum.get('total_return_pct', 0)
    gem_ret = gem_sum.get('total_return_pct', 0)
    if gpt_ret > gem_ret + 0.5:
        verdict = "🤖 GPT 領先"
    elif gem_ret > gpt_ret + 0.5:
        verdict = "🔷 Gemini 領先"
    else:
        verdict = "🤝 不分上下"

    gpt_line = (
        f"🤖 **GPT** {gpt_grade}({gpt_score}分)\n"
        f"  資產 ${gpt_sum['total_value']:,.0f} ({gpt_ret:+.1f}%)\n"
        f"  勝率 {gpt_sum['win_rate']:.0%}"
        f" ({gpt_sum['win_count']}勝{gpt_sum['loss_count']}敗)\n"
        f"  今日 ${gpt_sum['daily_pnl']:+,.0f}"
    )

    gem_line = (
        f"🔷 **Gemini** {gem_grade}({gem_score}分)\n"
        f"  資產 ${gem_sum['total_value']:,.0f} ({gem_ret:+.1f}%)\n"
        f"  勝率 {gem_sum['win_rate']:.0%}"
        f" ({gem_sum['win_count']}勝{gem_sum['loss_count']}敗)\n"
        f"  今日 ${gem_sum['daily_pnl']:+,.0f}"
    )

    fields = [
        {"name": verdict, "value": f"{gpt_line}\n\n{gem_line}", "inline": False},
    ]

    # 持倉對比（如果有）
    gpt_pos = gpt_sum.get('positions_detail', [])
    gem_pos = gem_sum.get('positions_detail', [])
    if gpt_pos or gem_pos:
        pos_lines = []
        all_codes = set(p['code'] for p in gpt_pos) | set(p['code'] for p in gem_pos)
        gpt_map = {p['code']: p for p in gpt_pos}
        gem_map = {p['code']: p for p in gem_pos}

        for code in sorted(all_codes):
            gp = gpt_map.get(code)
            gm = gem_map.get(code)
            name = (gp or gm)['name']
            g_str = f"{gp['pnl_pct']:+.1f}%" if gp else "—"
            m_str = f"{gm['pnl_pct']:+.1f}%" if gm else "—"
            pos_lines.append(f"{name}: GPT {g_str} | Gem {m_str}")

        if pos_lines:
            fields.append({
                "name": "持倉對比",
                "value": '\n'.join(pos_lines[:8]),
                "inline": False,
            })

    return {
        "title": f"🏆 AI 交易 PK | {today}",
        "color": COLOR_INFO,
        "fields": fields,
    }


def _score_to_grade(score):
    if score >= 90: return 'A+'
    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'


# ============================================================
# 3. 焦點股覆盤（含買賣點）
# ============================================================

def build_focus_review_embeds(focus_stocks, predictions, closing_data,
                              gpt_trader, gemini_trader, intraday_df=None):
    """
    焦點股逐檔覆盤 — 含日內買賣點、最高最低、AI 交易紀錄

    Args:
        focus_stocks: {code: {name, reason, ...}}
        predictions: PREMARKET_PREDICTIONS dict
        closing_data: msgArray from craw_realtime (盤後)
        gpt_trader: AITrader
        gemini_trader: GeminiTrader
        intraday_df: DataFrame from merge_intraday (optional)

    Returns:
        list of embed dicts (每檔一個 embed)
    """
    today = datetime.date.today()

    # 收盤價 dict
    close_map = {}
    for item in (closing_data or []):
        code = item.get('c', '')
        z = item.get('z', '-')
        y = item.get('y', '-')
        o = item.get('o', '-')
        h = item.get('h', '-')
        l = item.get('l', '-')
        close_map[code] = {
            'close': float(z) if z != '-' else None,
            'yesterday': float(y) if y != '-' else None,
            'open': float(o) if o != '-' else None,
            'high': float(h) if h != '-' else None,
            'low': float(l) if l != '-' else None,
        }

    # AI 交易紀錄
    gpt_trades = _get_trades_today(gpt_trader, today)
    gem_trades = _get_trades_today(gemini_trader, today)

    embeds = []
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']

    for i, (code, info) in enumerate(focus_stocks.items()):
        name = info['name']
        medal = medals[i] if i < len(medals) else f'{i+1}.'
        pred = predictions.get(code, {})
        mkt = close_map.get(code, {})

        close_p = mkt.get('close')
        yesterday_p = mkt.get('yesterday')
        open_p = mkt.get('open')
        high_p = mkt.get('high')
        low_p = mkt.get('low')

        # 漲跌幅
        if close_p and yesterday_p:
            change_pct = (close_p - yesterday_p) / yesterday_p * 100
            emoji = '🔴' if change_pct > 0.3 else '🟢' if change_pct < -0.3 else '⚪'
        else:
            change_pct = 0
            emoji = '⚪'

        # 預測 vs 實際
        pred_dir = pred.get('direction', '?')
        actual_dir = '漲' if change_pct > 0.5 else '跌' if change_pct < -0.5 else '盤整'
        correct = (pred_dir == actual_dir) or \
                  (pred_dir == '漲' and change_pct > 0) or \
                  (pred_dir == '跌' and change_pct < 0)
        if pred_dir == '觀望':
            mark = '⏸️'
        else:
            mark = '✓' if correct else '✗'

        # 日內價格統計
        price_line = ""
        if open_p and high_p and low_p and close_p:
            spread = high_p - low_p
            spread_pct = spread / yesterday_p * 100 if yesterday_p else 0
            price_line = (
                f"O {open_p:.1f} → H {high_p:.1f} / L {low_p:.1f} → C {close_p:.1f}\n"
                f"振幅 {spread_pct:.1f}%"
            )

        # 用 intraday 資料算買賣點
        buy_sell_line = ""
        if intraday_df is not None and not intraday_df.empty:
            stock_ticks = intraday_df[intraday_df['code'] == code]
            if not stock_ticks.empty:
                buy_sell_line = _calc_buy_sell_points(stock_ticks, yesterday_p)

        # AI 交易紀錄
        ai_line = ""
        gpt_t = [t for t in gpt_trades if t.get('stock_code') == code]
        gem_t = [t for t in gem_trades if t.get('stock_code') == code]

        if gpt_t:
            for t in gpt_t:
                action = '買' if t['action'] == 'buy' else '賣'
                ai_line += f"🤖 GPT {action} ${t['price']:.1f}"
                if t['action'] == 'sell':
                    ai_line += f" ({t.get('pnl_pct',0):+.1f}%)"
                ai_line += f" {t.get('reason','')[:20]}\n"

        if gem_t:
            for t in gem_t:
                action = '買' if t['action'] == 'buy' else '賣'
                ai_line += f"🔷 Gem {action} ${t['price']:.1f}"
                if t['action'] == 'sell':
                    ai_line += f" ({t.get('pnl_pct',0):+.1f}%)"
                ai_line += f" {t.get('reason','')[:20]}\n"

        # 信號摘要
        signals = pred.get('signals', {})
        sig_parts = []
        for key in ('foreign', 'ema', 'rsi', 'momentum'):
            val = signals.get(key, '')
            if val:
                sig_parts.append(val)
        sig_line = ' | '.join(sig_parts) if sig_parts else ''

        # 組合 fields
        fields = [
            {
                "name": f"預測 {pred_dir} {pred.get('confidence',0):.0%} → 實際 {change_pct:+.1f}% {mark}",
                "value": price_line or "—",
                "inline": False,
            },
        ]

        if buy_sell_line:
            fields.append({
                "name": "📍 盤中買賣點",
                "value": buy_sell_line,
                "inline": False,
            })

        if ai_line:
            fields.append({
                "name": "🤖 AI 交易",
                "value": ai_line.strip(),
                "inline": False,
            })

        if sig_line:
            fields.append({
                "name": "信號",
                "value": sig_line,
                "inline": False,
            })

        color = COLOR_GOOD if correct else COLOR_BAD
        if pred_dir == '觀望':
            color = COLOR_INFO

        embeds.append({
            "title": f"{medal} {emoji} {name}({code}) {change_pct:+.1f}%",
            "color": color,
            "fields": fields,
            "footer": {"text": info.get('reason', '')[:50]},
        })

    return embeds


def _get_trades_today(trader, today):
    """取出今日的交易紀錄"""
    trades = []
    today_str = today.isoformat()
    for t in getattr(trader, 'trade_history', []):
        sell_time = t.get('sell_time', '')
        if sell_time and sell_time[:10] == today_str:
            trades.append({**t, 'action': 'sell'})

    # 也看今日買入但尚未賣出的（在 positions 中）
    for code, pos in getattr(trader, 'positions', {}).items():
        buy_time = pos.get('buy_time', '')
        if buy_time and buy_time[:10] == today_str:
            trades.append({
                'stock_code': code,
                'stock_name': pos.get('name', code),
                'price': pos.get('buy_price', 0),
                'shares': pos.get('shares', 0),
                'reason': pos.get('reason', ''),
                'action': 'buy',
            })

    return trades


def _calc_buy_sell_points(ticks_df, yesterday_price):
    """
    從盤中 tick 資料計算買賣點

    策略:
    - 買點: 日內低點區間（最低價 ± 0.3%）的第一個時間
    - 賣點: 日內高點區間（最高價 ± 0.3%）的第一個時間
    - VWAP 交叉點
    - 量能爆發點（單筆成交量 > 平均 3 倍）

    Returns:
        str: 格式化的買賣點文字
    """
    if ticks_df.empty:
        return ""

    prices = ticks_df['price'].values
    timestamps = ticks_df['timestamp'].values
    trade_vols = ticks_df['trade_vol'].values

    high_p = prices.max()
    low_p = prices.min()
    close_p = prices[-1]

    lines = []

    # 最低點（買點）
    low_threshold = low_p * 1.003
    low_mask = prices <= low_threshold
    if low_mask.any():
        first_low_idx = np.argmax(low_mask)
        low_time = _extract_time(timestamps[first_low_idx])
        lines.append(f"🟢 買點: ${low_p:.1f} ({low_time})")

    # 最高點（賣點）
    high_threshold = high_p * 0.997
    high_mask = prices >= high_threshold
    if high_mask.any():
        first_high_idx = np.argmax(high_mask)
        high_time = _extract_time(timestamps[first_high_idx])
        lines.append(f"🔴 賣點: ${high_p:.1f} ({high_time})")

    # VWAP
    cum_vol = ticks_df['cum_vol'].values
    if cum_vol[-1] > 0 and yesterday_price:
        # 簡易 VWAP: 用累積量加權
        total_val = np.sum(prices * trade_vols)
        total_vol = np.sum(trade_vols)
        if total_vol > 0:
            vwap = total_val / total_vol
            vwap_vs_close = (close_p - vwap) / vwap * 100
            lines.append(f"📊 VWAP: ${vwap:.1f} (收盤{vwap_vs_close:+.1f}%)")

    # 量能爆發
    if len(trade_vols) > 10:
        avg_vol = np.mean(trade_vols[trade_vols > 0]) if np.any(trade_vols > 0) else 0
        if avg_vol > 0:
            spike_mask = trade_vols > avg_vol * 3
            if spike_mask.any():
                spike_idx = np.argmax(spike_mask)
                spike_time = _extract_time(timestamps[spike_idx])
                spike_price = prices[spike_idx]
                lines.append(f"💥 量能爆發: ${spike_price:.1f} ({spike_time})"
                             f" 量={trade_vols[spike_idx]}")

    return '\n'.join(lines) if lines else ""


def _extract_time(timestamp_str):
    """從 timestamp 字串取出 HH:MM"""
    try:
        ts = str(timestamp_str).strip()
        dt = datetime.datetime.fromisoformat(ts)
        return dt.strftime('%H:%M')
    except Exception:
        return '??:??'


# ============================================================
# 4. 每檔歷史命中率排行
# ============================================================

def build_stock_accuracy_embed(broadcast_report, predictions):
    """
    每檔股票歷史命中率排行

    Args:
        broadcast_report: generate_daily_report() 回傳
        predictions: PREMARKET_PREDICTIONS dict (含 name)
    """
    today = datetime.date.today()
    by_stock = broadcast_report.get('by_stock', {})

    if not by_stock:
        return None

    # 收集每檔結果
    results = []
    for code, data in by_stock.items():
        name = predictions.get(code, {}).get('name', code)
        results.append({
            'code': code,
            'name': name,
            'direction': data.get('direction', '?'),
            'confidence': data.get('confidence', 0),
            'actual': data.get('actual_direction', '?'),
            'correct': data.get('correct'),
        })

    # 分正確/錯誤
    correct_list = [r for r in results if r['correct'] is True]
    wrong_list = [r for r in results if r['correct'] is False]
    skip_list = [r for r in results if r['correct'] is None]

    lines = []
    if correct_list:
        lines.append("**✅ 預測正確:**")
        for r in correct_list:
            lines.append(f"  {r['name']}({r['code']}): "
                         f"預測{r['direction']} {r['confidence']:.0%} ✓")

    if wrong_list:
        lines.append("\n**❌ 預測錯誤:**")
        for r in wrong_list:
            lines.append(f"  {r['name']}({r['code']}): "
                         f"預測{r['direction']}→實際{r['actual']} ✗")

    if skip_list:
        lines.append(f"\n**⏸️ 觀望: {len(skip_list)} 檔**")

    accuracy = broadcast_report.get('accuracy', 0)

    return {
        "title": f"🎯 逐檔命中 | {today}",
        "color": COLOR_GOOD if accuracy >= 0.65 else COLOR_WARN,
        "description": '\n'.join(lines),
        "footer": {
            "text": (f"總計 {broadcast_report.get('total_broadcasts',0)} 檔 | "
                     f"有結果 {broadcast_report.get('with_outcome',0)} | "
                     f"準確率 {accuracy:.0%}")
        },
    }


# ============================================================
# 5. 信號效能統計
# ============================================================

def build_signal_effectiveness_embed(predictions, broadcast_report):
    """
    分析各信號組合的命中率

    用 predictions 的 signals dict + broadcast_report 的 correct 結果
    """
    today = datetime.date.today()
    by_stock = broadcast_report.get('by_stock', {})

    # 收集每個信號出現時的命中情況
    signal_stats = {}  # {signal_value: {'total': N, 'correct': N}}

    for code, pred in predictions.items():
        outcome = by_stock.get(code, {})
        correct = outcome.get('correct')
        if correct is None:
            continue  # 觀望，跳過

        signals = pred.get('signals', {})
        for key, value in signals.items():
            if not value:
                continue
            if key in ('correction', 'dampening'):
                continue  # 跳過系統內部信號

            if value not in signal_stats:
                signal_stats[value] = {'total': 0, 'correct': 0}
            signal_stats[value]['total'] += 1
            if correct:
                signal_stats[value]['correct'] += 1

    if not signal_stats:
        return None

    # 按命中率排序（至少出現 2 次）
    ranked = []
    for sig, stats in signal_stats.items():
        if stats['total'] >= 2:
            acc = stats['correct'] / stats['total']
            ranked.append((sig, stats['total'], stats['correct'], acc))

    ranked.sort(key=lambda x: x[3], reverse=True)

    lines = []
    if ranked:
        lines.append("**最強信號:**")
        for sig, total, correct, acc in ranked[:5]:
            bar = '█' * int(acc * 10)
            lines.append(f"  {bar} {acc:.0%} ({correct}/{total}) {sig}")

        if len(ranked) > 5:
            lines.append("\n**最弱信號:**")
            for sig, total, correct, acc in ranked[-3:]:
                bar = '░' * int(acc * 10)
                lines.append(f"  {bar} {acc:.0%} ({correct}/{total}) {sig}")

    if not lines:
        return None

    return {
        "title": f"📡 信號效能 | {today}",
        "color": COLOR_INFO,
        "description": '\n'.join(lines),
    }


# ============================================================
# 主函數: 產生完整盤後日報
# ============================================================

def generate_full_report(focus_stocks, predictions, closing_data,
                         gpt_trader, gemini_trader, closing_prices,
                         intraday_df=None):
    """
    產生完整盤後日報（多個 Discord Embed）

    Args:
        focus_stocks: FOCUS_STOCKS dict
        predictions: PREMARKET_PREDICTIONS dict
        closing_data: craw_realtime() 的 msgArray
        gpt_trader: AITrader instance
        gemini_trader: GeminiTrader instance
        closing_prices: {code: float} 收盤價
        intraday_df: (optional) merge_intraday 的 DataFrame

    Returns:
        list of embed dicts
    """
    embeds = []

    # 1. 績效總覽
    try:
        from prediction_history import (
            get_tracking_metrics, calc_advanced_metrics, calc_correction_factor
        )
        today_metrics = get_tracking_metrics()
        advanced_metrics = calc_advanced_metrics()
        correction = calc_correction_factor()

        embeds.append(build_accuracy_embed(
            today_metrics, advanced_metrics, correction
        ))
    except Exception as e:
        logger.error(f"績效總覽失敗: {e}")

    # 2. AI 交易 PK
    try:
        embeds.append(build_pk_report_embed(
            gpt_trader, gemini_trader, closing_prices
        ))
    except Exception as e:
        logger.error(f"AI PK 失敗: {e}")

    # 3. 焦點股覆盤（含買賣點）
    try:
        focus_embeds = build_focus_review_embeds(
            focus_stocks, predictions, closing_data,
            gpt_trader, gemini_trader, intraday_df
        )
        embeds.extend(focus_embeds)
    except Exception as e:
        logger.error(f"焦點股覆盤失敗: {e}")

    # 4. 逐檔命中率
    try:
        from broadcast_logger import generate_daily_report
        report = generate_daily_report()
        acc_embed = build_stock_accuracy_embed(report, predictions)
        if acc_embed:
            embeds.append(acc_embed)
    except Exception as e:
        logger.error(f"逐檔命中率失敗: {e}")

    # 5. 信號效能
    try:
        from broadcast_logger import generate_daily_report as gen_report
        report = gen_report()
        sig_embed = build_signal_effectiveness_embed(predictions, report)
        if sig_embed:
            embeds.append(sig_embed)
    except Exception as e:
        logger.error(f"信號效能失敗: {e}")

    return embeds
