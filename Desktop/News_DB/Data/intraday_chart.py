#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盤後即時曲線 PDF 報告

每檔股票一頁:
  - 日內價格曲線（黑線）
  - VWAP 均線（藍虛線）
  - 昨收參考線（灰虛線）
  - GPT 買賣點（紅▲ 綠▼）
  - Gemini 買賣點（橘▲ 紫▼）
  - 成交量副圖
  - 日內高低、開收標註

用法:
    python intraday_chart.py                        # 今日
    python intraday_chart.py --date 20260218        # 指定日期
    python intraday_chart.py --stock 2330           # 單一股票

@author: rubylintu
"""

import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 不需要 GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 中文字體
_FONT_PATH = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(_FONT_PATH):
    _CJK_FONT = fm.FontProperties(fname=_FONT_PATH, size=10)
    _CJK_FONT_TITLE = fm.FontProperties(fname=_FONT_PATH, size=13)
    _CJK_FONT_SMALL = fm.FontProperties(fname=_FONT_PATH, size=8)
else:
    _CJK_FONT = fm.FontProperties(size=10)
    _CJK_FONT_TITLE = fm.FontProperties(size=13)
    _CJK_FONT_SMALL = fm.FontProperties(size=8)


def load_intraday(date_str):
    """載入指定日期的 intraday 資料"""
    from merge_intraday import parse_intraday_file

    fpath = os.path.join(SCRIPT_DIR, 'intraday', f'{date_str}.txt')
    if not os.path.exists(fpath):
        print(f"找不到 {fpath}")
        return pd.DataFrame()

    df = parse_intraday_file(fpath)
    print(f"載入 {len(df)} 筆 tick ({date_str})")
    return df


def load_ai_trades(date_str):
    """
    從 AI trader portfolio 檔案載入當日交易紀錄

    Returns:
        gpt_trades: [{code, action, price, time, reason}, ...]
        gemini_trades: same
    """
    from config import PORTFOLIO_FILE, GEMINI_PORTFOLIO_FILE

    date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    gpt_trades = _extract_trades_from_portfolio(PORTFOLIO_FILE, date_iso)
    gem_trades = _extract_trades_from_portfolio(GEMINI_PORTFOLIO_FILE, date_iso)

    return gpt_trades, gem_trades


def _extract_trades_from_portfolio(portfolio_file, date_iso):
    """從 portfolio JSON 提取當日交易"""
    trades = []
    if not os.path.exists(portfolio_file):
        return trades

    try:
        with open(portfolio_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return trades

    # 持倉中的今日買入
    for code, pos in data.get('positions', {}).items():
        buy_time = pos.get('buy_time', '')
        if buy_time.startswith(date_iso):
            trades.append({
                'code': code,
                'name': pos.get('name', code),
                'action': 'buy',
                'price': pos.get('buy_price', 0),
                'time': buy_time,
                'reason': pos.get('reason', ''),
            })

    # 已完成的交易
    for t in data.get('trade_history', []):
        # 買入
        buy_time = t.get('buy_time', '')
        if buy_time.startswith(date_iso):
            trades.append({
                'code': t.get('stock_code', ''),
                'name': t.get('stock_name', ''),
                'action': 'buy',
                'price': t.get('buy_price', 0),
                'time': buy_time,
                'reason': t.get('reason', ''),
            })
        # 賣出
        sell_time = t.get('sell_time', '')
        if sell_time.startswith(date_iso):
            trades.append({
                'code': t.get('stock_code', ''),
                'name': t.get('stock_name', ''),
                'action': 'sell',
                'price': t.get('sell_price', 0),
                'time': sell_time,
                'reason': t.get('reason', ''),
                'pnl_pct': t.get('pnl_pct', 0),
            })

    return trades


def _load_predictions(date_str):
    """
    載入盤前粒子模型預測

    先嘗試 today_predictions.json（當日），
    再嘗試 broadcast log（歷史）

    Returns:
        dict: {code: {predicted_price, direction, confidence, bias, signals}}
    """
    predictions = {}

    # 1. 嘗試 today_predictions.json
    try:
        from config import PREDICTIONS_FILE
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 檢查日期
            file_date = data.get('date', '')
            date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            if file_date == date_iso:
                predictions = data.get('predictions', {})
                if predictions:
                    print(f"  從 today_predictions.json 載入 {len(predictions)} 筆預測")
                    return predictions
    except Exception:
        pass

    # 2. 嘗試 broadcast log
    try:
        from broadcast_logger import generate_daily_report
        date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        report = generate_daily_report(date_iso)
        by_stock = report.get('by_stock', {})
        for code, data in by_stock.items():
            predictions[code] = {
                'predicted_price': data.get('predicted_price', 0),
                'direction': data.get('direction', ''),
                'confidence': data.get('confidence', 0),
                'bias': data.get('bias', 0),
                'signals': data.get('signals', {}),
            }
        if predictions:
            print(f"  從 broadcast log 載入 {len(predictions)} 筆預測")
    except Exception:
        pass

    return predictions


def parse_timestamp(ts_str):
    """timestamp 字串 → datetime"""
    try:
        return datetime.datetime.fromisoformat(str(ts_str).strip())
    except Exception:
        return None


def plot_stock_page(ax_price, ax_vol, stock_df, code, name,
                    gpt_trades, gem_trades, prediction=None):
    """
    在給定的 axes 上繪製單一股票的日內圖

    Args:
        ax_price: 價格主圖 axes
        ax_vol: 成交量副圖 axes
        stock_df: 該股票的 intraday DataFrame
        code: 股票代號
        name: 股票名稱
        gpt_trades: GPT 交易 list (該股)
        gem_trades: Gemini 交易 list (該股)
        prediction: 粒子模型盤前預測 dict (optional)
                    {predicted_price, direction, confidence, bias, signals}
    """
    if stock_df.empty:
        return

    # 解析時間
    times = []
    for ts in stock_df['timestamp']:
        dt = parse_timestamp(ts)
        if dt:
            times.append(dt)
        else:
            times.append(None)

    stock_df = stock_df.copy()
    stock_df['dt'] = times
    stock_df = stock_df.dropna(subset=['dt'])

    if stock_df.empty:
        return

    prices = stock_df['price'].values
    dts = stock_df['dt'].values
    trade_vols = stock_df['trade_vol'].values
    yesterday = stock_df['yesterday'].iloc[0] if 'yesterday' in stock_df else None

    # === 價格主圖 ===
    ax_price.plot(dts, prices, color='#333333', linewidth=1.0, zorder=2)

    # 昨收參考線
    if yesterday and not np.isnan(yesterday):
        ax_price.axhline(y=yesterday, color='#999999', linestyle='--',
                         linewidth=0.7, alpha=0.7, zorder=1)
        ax_price.text(dts[0], yesterday, f' 昨收 {yesterday:.1f}',
                      fontproperties=_CJK_FONT_SMALL, color='#999999',
                      verticalalignment='bottom')

    # VWAP
    valid_vol = trade_vols.copy().astype(float)
    valid_vol[valid_vol <= 0] = np.nan
    cum_pv = np.nancumsum(prices * valid_vol)
    cum_v = np.nancumsum(valid_vol)
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
    ax_price.plot(dts, vwap, color='#3498DB', linestyle='--',
                  linewidth=0.8, alpha=0.8, label='VWAP', zorder=2)

    # === 粒子模型預測 ===
    if prediction:
        pred_price = prediction.get('predicted_price')
        pred_dir = prediction.get('direction', '')
        pred_conf = prediction.get('confidence', 0)
        pred_bias = prediction.get('bias', 0)

        if pred_price and pred_price > 0:
            # 預測目標價水平線（橘色點虛線）
            ax_price.axhline(y=pred_price, color='#FF6600', linestyle=':',
                             linewidth=1.2, alpha=0.85, zorder=3)

            # 標籤放在右側
            dir_emoji = '漲' if pred_dir == '漲' else '跌' if pred_dir == '跌' else pred_dir
            label = f'模型 ${pred_price:.1f} [{dir_emoji} {pred_conf:.0%}]'
            ax_price.text(dts[-1], pred_price, f'  {label}',
                          fontproperties=_CJK_FONT_SMALL,
                          color='#FF6600', verticalalignment='bottom',
                          fontweight='bold')

            # 如果預測價和收盤價差很大，畫一個箭頭指出偏差
            close_p_val = prices[-1]
            error_pct = abs(pred_price - close_p_val) / close_p_val * 100
            if error_pct > 1.0:
                ax_price.annotate(
                    f'誤差 {error_pct:.1f}%',
                    xy=(dts[-1], pred_price),
                    xytext=(0, -12 if pred_price > close_p_val else 12),
                    textcoords='offset points',
                    fontproperties=_CJK_FONT_SMALL,
                    color='#FF6600', alpha=0.7, ha='right',
                )

    # 日內高低標註
    high_idx = np.argmax(prices)
    low_idx = np.argmin(prices)
    high_p = prices[high_idx]
    low_p = prices[low_idx]

    ax_price.annotate(f'H {high_p:.1f}',
                      xy=(dts[high_idx], high_p),
                      xytext=(0, 8), textcoords='offset points',
                      fontproperties=_CJK_FONT_SMALL,
                      color='#E74C3C', ha='center',
                      arrowprops=dict(arrowstyle='-', color='#E74C3C', lw=0.5))

    ax_price.annotate(f'L {low_p:.1f}',
                      xy=(dts[low_idx], low_p),
                      xytext=(0, -12), textcoords='offset points',
                      fontproperties=_CJK_FONT_SMALL,
                      color='#2ECC71', ha='center',
                      arrowprops=dict(arrowstyle='-', color='#2ECC71', lw=0.5))

    # 開盤 / 收盤標註
    open_p = prices[0]
    close_p = prices[-1]
    ax_price.plot(dts[0], open_p, 'D', color='#3498DB', markersize=5, zorder=5)
    ax_price.plot(dts[-1], close_p, 's', color='#333333', markersize=5, zorder=5)

    # === GPT 買賣點 ===
    for t in gpt_trades:
        t_dt = parse_timestamp(t['time'])
        if t_dt is None:
            continue
        if t['action'] == 'buy':
            ax_price.plot(t_dt, t['price'], '^', color='#E74C3C',
                          markersize=12, markeredgecolor='white',
                          markeredgewidth=0.8, zorder=10)
            ax_price.annotate(f"GPT 買 ${t['price']:.0f}",
                              xy=(t_dt, t['price']),
                              xytext=(5, 12), textcoords='offset points',
                              fontproperties=_CJK_FONT_SMALL,
                              color='#E74C3C',
                              bbox=dict(boxstyle='round,pad=0.2',
                                        fc='white', ec='#E74C3C', alpha=0.8))
        elif t['action'] == 'sell':
            ax_price.plot(t_dt, t['price'], 'v', color='#2ECC71',
                          markersize=12, markeredgecolor='white',
                          markeredgewidth=0.8, zorder=10)
            pnl_str = f" {t.get('pnl_pct',0):+.1f}%" if 'pnl_pct' in t else ''
            ax_price.annotate(f"GPT 賣 ${t['price']:.0f}{pnl_str}",
                              xy=(t_dt, t['price']),
                              xytext=(5, -16), textcoords='offset points',
                              fontproperties=_CJK_FONT_SMALL,
                              color='#2ECC71',
                              bbox=dict(boxstyle='round,pad=0.2',
                                        fc='white', ec='#2ECC71', alpha=0.8))

    # === Gemini 買賣點 ===
    for t in gem_trades:
        t_dt = parse_timestamp(t['time'])
        if t_dt is None:
            continue
        if t['action'] == 'buy':
            ax_price.plot(t_dt, t['price'], '^', color='#E67E22',
                          markersize=10, markeredgecolor='white',
                          markeredgewidth=0.8, zorder=9)
            ax_price.annotate(f"Gem 買 ${t['price']:.0f}",
                              xy=(t_dt, t['price']),
                              xytext=(-5, 12), textcoords='offset points',
                              fontproperties=_CJK_FONT_SMALL,
                              color='#E67E22',
                              bbox=dict(boxstyle='round,pad=0.2',
                                        fc='white', ec='#E67E22', alpha=0.8))
        elif t['action'] == 'sell':
            ax_price.plot(t_dt, t['price'], 'v', color='#9B59B6',
                          markersize=10, markeredgecolor='white',
                          markeredgewidth=0.8, zorder=9)
            pnl_str = f" {t.get('pnl_pct',0):+.1f}%" if 'pnl_pct' in t else ''
            ax_price.annotate(f"Gem 賣 ${t['price']:.0f}{pnl_str}",
                              xy=(t_dt, t['price']),
                              xytext=(-5, -16), textcoords='offset points',
                              fontproperties=_CJK_FONT_SMALL,
                              color='#9B59B6',
                              bbox=dict(boxstyle='round,pad=0.2',
                                        fc='white', ec='#9B59B6', alpha=0.8))

    # 價格圖漲跌背景色
    if yesterday and not np.isnan(yesterday):
        change_pct = (close_p - yesterday) / yesterday * 100
        if change_pct > 0.3:
            ax_price.set_facecolor('#FFF5F5')
        elif change_pct < -0.3:
            ax_price.set_facecolor('#F5FFF5')

    # 標題
    if yesterday and not np.isnan(yesterday):
        change = (close_p - yesterday) / yesterday * 100
        arrow = '▲' if change > 0 else '▼' if change < 0 else '—'
        title = f"{name}({code})  {close_p:.1f} {arrow}{change:+.2f}%"
    else:
        title = f"{name}({code})  {close_p:.1f}"

    ax_price.set_title(title, fontproperties=_CJK_FONT_TITLE, pad=8)
    ax_price.grid(True, alpha=0.3)
    ax_price.tick_params(axis='x', labelsize=7)
    ax_price.tick_params(axis='y', labelsize=8)

    # X 軸格式
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_price.set_xlim(dts[0], dts[-1])

    # === 成交量副圖 ===
    colors = ['#E74C3C' if i > 0 and prices[i] >= prices[i-1] else '#2ECC71'
              for i in range(len(prices))]
    ax_vol.bar(dts, trade_vols, width=0.0003, color=colors, alpha=0.6)
    ax_vol.set_ylabel('量', fontproperties=_CJK_FONT_SMALL)
    ax_vol.tick_params(axis='x', labelsize=7)
    ax_vol.tick_params(axis='y', labelsize=7)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_vol.set_xlim(dts[0], dts[-1])
    ax_vol.grid(True, alpha=0.2)


def _draw_legend_page(pdf, date_fmt, num_stocks, num_gpt, num_gem):
    """繪製第一頁圖例說明"""
    fig = plt.figure(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.05)
    ax = fig.add_subplot(111)
    ax.axis('off')

    # 標題
    fig.text(0.5, 0.93,
             f'盤後即時曲線報告  {date_fmt}',
             ha='center', fontproperties=_CJK_FONT_TITLE,
             fontsize=18, color='#333333')
    fig.text(0.5, 0.88,
             f'{num_stocks} 檔股票 | GPT {num_gpt} 筆交易 | Gemini {num_gem} 筆交易',
             ha='center', fontproperties=_CJK_FONT,
             color='#666666')

    # 圖例項目：(y位置, 符號繪製函數, 說明文字)
    x_sym = 0.08   # 符號 x
    x_txt = 0.16   # 文字 x
    y_start = 0.78
    dy = 0.065

    items = [
        ('line',    '#333333', '━━',  '即時成交價',        '每 10~20 秒抓取一次的盤中成交價格'),
        ('dash',    '#3498DB', '- -', 'VWAP 均線',        '成交量加權平均價，收盤高於 VWAP 偏強，低於偏弱'),
        ('dash',    '#999999', '- -', '昨日收盤價',        '昨收參考線，判斷今日漲跌的基準'),
        ('marker',  '#3498DB', '◆',   '開盤價',           '當日第一筆成交價'),
        ('marker',  '#333333', '■',   '收盤價',           '當日最後一筆成交價'),
        ('annot',   '#E74C3C', 'H',   '日內最高價',        '當日盤中最高成交價及發生時間'),
        ('annot',   '#2ECC71', 'L',   '日內最低價',        '當日盤中最低成交價及發生時間'),
        ('marker',  '#E74C3C', '▲',   'GPT 買入點',       'GPT-4o AI 判斷買入的價位與時間'),
        ('marker',  '#2ECC71', '▼',   'GPT 賣出點',       'GPT-4o AI 判斷賣出的價位與損益'),
        ('marker',  '#E67E22', '▲',   'Gemini 買入點',    'Gemini AI 判斷買入的價位與時間'),
        ('marker',  '#9B59B6', '▼',   'Gemini 賣出點',    'Gemini AI 判斷賣出的價位與損益'),
        ('dash',    '#FF6600', '...',  '粒子模型預測價',    '盤前粒子模型預測的目標價位，含方向與信心度'),
        ('bar_r',   '#E74C3C', '█',   '成交量（價漲）',    '下方副圖，該筆成交時價格上漲'),
        ('bar_g',   '#2ECC71', '█',   '成交量（價跌）',    '下方副圖，該筆成交時價格下跌'),
        ('bg',      '#FFF5F5', '///',  '微紅背景',          '收盤高於昨收（今日上漲）'),
        ('bg',      '#F5FFF5', '///',  '微綠背景',          '收盤低於昨收（今日下跌）'),
    ]

    for i, (kind, color, symbol, label, desc) in enumerate(items):
        y = y_start - i * dy

        # 符號
        fig.text(x_sym, y, symbol,
                 fontproperties=_CJK_FONT, fontsize=13,
                 color=color, ha='center', va='center',
                 fontweight='bold')

        # 名稱
        fig.text(x_txt, y, label,
                 fontproperties=_CJK_FONT, fontsize=11,
                 color='#333333', va='center')

        # 說明
        fig.text(x_txt + 0.22, y, desc,
                 fontproperties=_CJK_FONT_SMALL, fontsize=9,
                 color='#888888', va='center')

    # 頁腳
    fig.text(0.5, 0.02,
             'News_DB AI 系統 — 每頁一檔股票，依代號排序',
             ha='center', fontproperties=_CJK_FONT_SMALL,
             color='#AAAAAA')

    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf(date_str=None, stock_filter=None, output_path=None):
    """
    產生盤後 PDF 報告

    Args:
        date_str: 日期 YYYYMMDD (預設今天)
        stock_filter: 只畫特定股票代號 (str or list)
        output_path: 輸出路徑 (預設 intraday/YYYYMMDD.pdf)

    Returns:
        str: 輸出 PDF 路徑
    """
    if date_str is None:
        date_str = datetime.date.today().strftime('%Y%m%d')

    df = load_intraday(date_str)
    if df.empty:
        print("無 intraday 資料，無法產生 PDF")
        return None

    # 載入 AI 交易
    gpt_trades, gem_trades = load_ai_trades(date_str)
    print(f"GPT 交易: {len(gpt_trades)} 筆, Gemini 交易: {len(gem_trades)} 筆")

    # 載入粒子模型盤前預測
    predictions = _load_predictions(date_str)
    print(f"粒子模型預測: {len(predictions)} 檔")

    # 取得所有股票
    stock_codes = sorted(df['code'].unique())

    if stock_filter:
        if isinstance(stock_filter, str):
            stock_filter = [stock_filter]
        stock_codes = [c for c in stock_codes if c in stock_filter]

    if not stock_codes:
        print("沒有符合的股票")
        return None

    # 輸出路徑
    if output_path is None:
        output_dir = os.path.join(SCRIPT_DIR, 'intraday')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{date_str}.pdf')

    date_fmt = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
    print(f"產生 PDF: {len(stock_codes)} 檔股票...")

    with PdfPages(output_path) as pdf:
        # === 第一頁：圖例說明 ===
        _draw_legend_page(pdf, date_fmt, len(stock_codes),
                          len(gpt_trades), len(gem_trades))

        for code in stock_codes:
            stock_df = df[df['code'] == code].copy()
            if stock_df.empty or len(stock_df) < 5:
                continue

            name = stock_df['name'].iloc[0]

            # 該股的 AI 交易
            gpt_t = [t for t in gpt_trades if t['code'] == code]
            gem_t = [t for t in gem_trades if t['code'] == code]

            # 該股的粒子模型預測
            pred = predictions.get(code)

            # 建立圖（價格 + 成交量 雙軸）
            fig, (ax_price, ax_vol) = plt.subplots(
                2, 1, figsize=(11, 6.5),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
            fig.subplots_adjust(hspace=0.08, left=0.08, right=0.85,
                                top=0.92, bottom=0.08)

            plot_stock_page(ax_price, ax_vol, stock_df, code, name,
                            gpt_t, gem_t, prediction=pred)

            # 頁腳
            fig.text(0.5, 0.01,
                     f'News_DB AI 系統 | {date_fmt} 盤後分析',
                     ha='center', fontproperties=_CJK_FONT_SMALL,
                     color='#999999')

            # 圖例（右上角）
            legend_items = ['● 價格', '-- VWAP', '-- 昨收']
            if gpt_t:
                legend_items.append('▲▼ GPT 買賣')
            if gem_t:
                legend_items.append('▲▼ Gemini 買賣')
            fig.text(0.95, 0.95, '  '.join(legend_items),
                     ha='right', fontproperties=_CJK_FONT_SMALL,
                     color='#666666')

            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {name}({code}) {'📍' if (gpt_t or gem_t) else '  '}"
                  f" {len(stock_df)} ticks")

    print(f"\nPDF 已產生: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='盤後即時曲線 PDF')
    parser.add_argument('--date', type=str, default=None,
                        help='日期 YYYYMMDD (預設今天)')
    parser.add_argument('--stock', type=str, default=None,
                        help='只畫特定股票代號')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出 PDF 路徑')
    args = parser.parse_args()

    date_str = args.date or datetime.date.today().strftime('%Y%m%d')
    generate_pdf(date_str, args.stock, args.output)


if __name__ == '__main__':
    main()
