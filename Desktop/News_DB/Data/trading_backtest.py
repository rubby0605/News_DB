#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Strategy Backtest v2 (Improved Model)

用改良版粒子模型（6大改善）預測曲線判斷買賣點，回測模擬交易績效
- 使用 calc_directional_bias（含衰減、RSI、成交量、大盤訊號）
- 信心度門檻 0.65 + 觀望機制
- 橫軸=時間，縱軸=價格
- 預測價格曲線 vs 實際價格曲線
- 買點/賣點標記在圖上
- 輸出 PDF 報告

@author: rubylintu
"""

import os
import sys
import datetime
import time
import requests
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from directional_particle_model import (
    calc_directional_bias, load_optimized_weights,
)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

STOCK_NAMES = {
    '2330': 'TSMC', '2454': 'MediaTek', '2344': 'Winbond',
    '3481': 'Innolux', '2313': 'Compeq',
}
TEST_STOCKS = ['2330', '2454', '2344', '3481', '2313']

# ── 交易參數 ──
INITIAL_CAPITAL = 1_000_000  # 初始資金 100 萬
BUY_THRESHOLD = 2.0          # bias > 此值 → 買入信號
SELL_THRESHOLD = -2.0         # bias < 此值 → 賣出信號
STOP_LOSS_PCT = -3.0          # 停損 %
TAKE_PROFIT_PCT = 5.0         # 停利 %
POSITION_SIZE = 0.3           # 每次用 30% 資金建倉
TAX_RATE = 0.00585            # 台股實際來回成本 0.585%（買0.1425%+賣0.1425%+證交稅0.3%）


# ============================================================
# 資料抓取（完整 OHLCV）
# ============================================================

def get_full_historical_data(stock_code, year_month):
    """取得完整 OHLCV 歷史價格"""
    date_str = f'{year_month}01'
    url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_code}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if data.get('stat') != 'OK' or 'data' not in data:
            return []

        result = []
        prev_close = None

        for row in data['data']:
            try:
                date = row[0]
                volume = int(row[1].replace(',', ''))
                open_p = float(row[3].replace(',', '')) if row[3] != '--' else 0
                high = float(row[4].replace(',', '')) if row[4] != '--' else 0
                low = float(row[5].replace(',', '')) if row[5] != '--' else 0
                close = float(row[6].replace(',', '')) if row[6] != '--' else 0

                if close > 0:
                    change = 0
                    if prev_close:
                        change = (close - prev_close) / prev_close * 100

                    result.append({
                        'date': date,
                        'open': open_p,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume,
                        'change': change
                    })
                    prev_close = close
            except:
                continue

        return result
    except:
        return []


def get_institutional_data(date_str):
    """取得法人資料（含外資、投信、自營商）"""
    url = f'https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if data.get('stat') != 'OK' or 'data' not in data:
            return None

        result = {}
        for row in data['data']:
            code = row[0].strip()
            foreign = int(row[4].replace(',', '')) // 1000 if row[4] != '--' else 0
            investment = int(row[10].replace(',', '')) // 1000 if row[10] != '--' else 0
            dealer = int(row[11].replace(',', '')) // 1000 if row[11] != '--' else 0

            result[code] = {
                'foreign': foreign,
                'investment': investment,
                'dealer': dealer,
                'total': foreign + investment + dealer
            }

        return result
    except:
        return None


def prepare_test_data(stock_codes, months):
    """準備測試資料（完整 OHLCV + 法人）"""
    print("Fetching test data (OHLCV + institutional)...")

    test_data = []
    inst_cache = {}

    for code in stock_codes:
        all_prices = []
        for month in months:
            prices = get_full_historical_data(code, month)
            all_prices.extend(prices)
            time.sleep(0.3)

        if len(all_prices) < 15:
            continue

        # 收集法人資料
        for i in range(10, len(all_prices)):
            date_parts = all_prices[i]['date'].split('/')
            if len(date_parts) != 3:
                continue

            roc_year = int(date_parts[0])
            month_num = int(date_parts[1])
            day = int(date_parts[2])
            date_str = f'{roc_year + 1911}{month_num:02d}{day:02d}'

            if date_str not in inst_cache:
                inst_data = get_institutional_data(date_str)
                if inst_data:
                    inst_cache[date_str] = inst_data
                time.sleep(0.2)

        test_data.append((code, all_prices, inst_cache))
        print(f"  {code}: {len(all_prices)} days")

    return test_data


# ============================================================
# 預測模擬（使用改良版模型）
# ============================================================

def simulate_predictions(stock_code, prices, inst_cache, n_particles=2000):
    """
    逐日模擬預測（使用改良版 calc_directional_bias）

    包含：衰減、RSI、成交量、信心度門檻
    不含：大盤即時訊號、GPT 情緒（回測無歷史資料）
    """
    weights = load_optimized_weights()
    conf_threshold = weights.get('confidence_threshold', 0.65)

    # 回測用的權重：關閉自動修正（無歷史預測紀錄）
    backtest_weights = dict(weights)
    backtest_weights['enable_auto_correction'] = False

    # 中性大盤訊號（回測無即時資料）
    neutral_market = {'taiex_signal': 0, 'sox_signal': 0,
                      'taiex_change': 0, 'sox_change': 0}

    days = []
    dir_correct = 0
    dir_total = 0

    for i in range(10, len(prices) - 1):
        date_parts = prices[i]['date'].split('/')
        if len(date_parts) != 3:
            continue

        roc_year = int(date_parts[0])
        month = int(date_parts[1])
        day = int(date_parts[2])
        date_str = f'{roc_year + 1911}{month:02d}{day:02d}'

        if date_str not in inst_cache:
            continue

        inst_data = inst_cache[date_str]

        # 準備歷史資料（到第 i 天為止）
        history = prices[:i + 1]

        # 使用新版 calc_directional_bias
        bias, signals = calc_directional_bias(
            stock_code, inst_data, history, backtest_weights,
            market_signal=neutral_market,
            external_bias=None
        )

        base_price = prices[i]['close']

        # 波動率
        if i >= 15:
            closes = [prices[j]['close'] for j in range(i - 10, i)]
            vol = np.std(closes) / np.mean(closes) * 100
            vol = max(1, min(5, vol))
        else:
            vol = 2.0

        # 粒子模擬
        mu = base_price * (bias / 100)
        sigma = base_price * (vol / 100)
        particles = np.random.normal(base_price + mu, sigma, n_particles)

        pred_mean = float(np.mean(particles))
        pred_std = float(np.std(particles))
        prob_up = float(np.sum(particles > base_price) / len(particles))
        prob_down = 1.0 - prob_up

        actual_next = prices[i + 1]['close']

        # 方向判斷（新版門檻 + 觀望）
        if prob_up > conf_threshold:
            direction = 'up'
        elif prob_down > conf_threshold:
            direction = 'down'
        elif max(prob_up, prob_down) > 0.55:
            direction = 'flat'
        else:
            direction = 'wait'

        # 方向準確率追蹤
        actual_change = (actual_next - base_price) / base_price * 100
        if direction in ('up', 'down'):
            dir_total += 1
            if (direction == 'up' and actual_change > 0) or \
               (direction == 'down' and actual_change < 0):
                dir_correct += 1

        days.append({
            'date': f'{month:02d}/{day:02d}',
            'date_full': date_str,
            'base_price': base_price,
            'pred_price': pred_mean,
            'pred_high': pred_mean + pred_std,
            'pred_low': pred_mean - pred_std,
            'actual_next': actual_next,
            'bias': bias,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'direction': direction,
        })

    dir_accuracy = dir_correct / dir_total * 100 if dir_total > 0 else 0
    print(f"  Direction accuracy: {dir_correct}/{dir_total} = {dir_accuracy:.1f}%")

    return days


def run_trading_simulation(days):
    """
    用預測信號模擬交易

    策略：
    - bias > BUY_THRESHOLD 且 prob_up > conf_threshold → 買入
    - bias < SELL_THRESHOLD 且 prob_down > conf_threshold → 賣出
    - direction == 'wait' → 不動作
    - 持有中碰到停損/停利 → 強制平倉
    """
    weights = load_optimized_weights()
    conf_threshold = weights.get('confidence_threshold', 0.65)

    buy_tax = TAX_RATE / 2   # 買入手續費（總成本的一半）
    sell_tax = TAX_RATE / 2  # 賣出手續費+交易稅（總成本的一半）
    total_tax_paid = 0       # 累計已付交易成本

    capital = INITIAL_CAPITAL
    position = 0         # 持有股數
    entry_cost = 0       # 進場總成本（含稅）
    entry_price = 0      # 進場均價（不含稅，用於停損停利判斷）
    trades = []          # 交易紀錄
    equity_curve = []    # 每日淨值
    buy_signals = []     # 買點 index
    sell_signals = []    # 賣點 index

    for i, d in enumerate(days):
        current_price = d['actual_next']

        # 觀望日不做任何操作（除非持有中觸發停損停利）
        is_wait = d.get('direction') == 'wait'

        # 持有中 → 檢查停損停利
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price * 100

            should_sell = False
            reason = ''

            if pnl_pct <= STOP_LOSS_PCT:
                should_sell = True
                reason = 'stop_loss'
            elif pnl_pct >= TAKE_PROFIT_PCT:
                should_sell = True
                reason = 'take_profit'
            elif not is_wait and d['bias'] < SELL_THRESHOLD and d['prob_down'] > conf_threshold:
                should_sell = True
                reason = 'signal_sell'

            if should_sell:
                gross_sell = position * current_price
                sell_fee = gross_sell * sell_tax
                net_sell = gross_sell - sell_fee
                total_tax_paid += sell_fee
                profit = net_sell - entry_cost
                pnl_pct_net = profit / entry_cost * 100
                capital += net_sell
                trades.append({
                    'type': 'SELL',
                    'day': i,
                    'date': d['date'],
                    'price': current_price,
                    'shares': position,
                    'profit': profit,
                    'pnl_pct': pnl_pct_net,
                    'tax': sell_fee,
                    'reason': reason,
                })
                sell_signals.append(i)
                position = 0
                entry_price = 0
                entry_cost = 0

        # 空手 → 檢查買入信號（觀望日不買）
        if position == 0 and not is_wait:
            if d['bias'] > BUY_THRESHOLD and d['prob_up'] > conf_threshold:
                invest = capital * POSITION_SIZE
                # 考慮買入稅後可買的股數
                effective_price = current_price * (1 + buy_tax)
                shares = int(invest / effective_price / 1000) * 1000  # 整張
                if shares > 0:
                    gross_cost = shares * current_price
                    buy_fee = gross_cost * buy_tax
                    total_cost = gross_cost + buy_fee
                    total_tax_paid += buy_fee
                    capital -= total_cost
                    position = shares
                    entry_price = current_price
                    entry_cost = total_cost
                    trades.append({
                        'type': 'BUY',
                        'day': i,
                        'date': d['date'],
                        'price': current_price,
                        'shares': shares,
                        'profit': 0,
                        'pnl_pct': 0,
                        'tax': buy_fee,
                        'reason': 'signal_buy',
                    })
                    buy_signals.append(i)

        # 計算當日淨值（持有部位以市價計，未扣未來賣出稅）
        portfolio_value = capital + position * current_price
        equity_curve.append(portfolio_value)

    # 最後如果還持有，強制平倉（含稅）
    if position > 0 and days:
        last_price = days[-1]['actual_next']
        gross_sell = position * last_price
        sell_fee = gross_sell * sell_tax
        net_sell = gross_sell - sell_fee
        total_tax_paid += sell_fee
        profit = net_sell - entry_cost
        pnl_pct_net = profit / entry_cost * 100
        capital += net_sell
        trades.append({
            'type': 'SELL',
            'day': len(days) - 1,
            'date': days[-1]['date'],
            'price': last_price,
            'shares': position,
            'profit': profit,
            'pnl_pct': pnl_pct_net,
            'tax': sell_fee,
            'reason': 'force_close',
        })
        sell_signals.append(len(days) - 1)
        position = 0
        equity_curve[-1] = capital

    # 統計
    final_value = capital
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    sell_trades = [t for t in trades if t['type'] == 'SELL']
    win_trades = [t for t in sell_trades if t['profit'] > 0]
    win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
    total_profit = sum(t['profit'] for t in sell_trades)
    avg_profit = np.mean([t['pnl_pct'] for t in sell_trades]) if sell_trades else 0

    # Buy & Hold 對照（也要扣稅：買一次+賣一次）
    if days:
        bh_gross = (days[-1]['actual_next'] - days[0]['base_price']) / days[0]['base_price'] * 100
        bh_return = bh_gross - TAX_RATE * 100  # B&H 也要付一次來回稅
    else:
        bh_return = 0

    # Max Drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': trades,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'equity_curve': equity_curve,
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': bh_return,
        'num_trades': len(sell_trades),
        'win_rate': win_rate,
        'avg_profit_pct': avg_profit,
        'total_profit': total_profit,
        'total_tax': total_tax_paid,
        'max_drawdown': max_dd,
    }


def main():
    weights = load_optimized_weights()
    conf_threshold = weights.get('confidence_threshold', 0.65)

    print("=" * 60)
    print("Trading Strategy Backtest v2 (Improved Model)")
    print("=" * 60)
    print(f"Model: calc_directional_bias (dampening + RSI + volume)")
    print(f"Confidence threshold: {conf_threshold:.0%}")
    print(f"Transaction tax: {TAX_RATE:.0%} per round-trip")

    # 準備資料
    today = datetime.date.today()
    months = []
    for i in range(2):
        target_month = today.month - i - 1
        target_year = today.year
        if target_month <= 0:
            target_month += 12
            target_year -= 1
        months.append(f'{target_year}{target_month:02d}')

    print(f"Stocks: {TEST_STOCKS}")
    print(f"Months: {months}")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Strategy: Buy when bias>{BUY_THRESHOLD} & P(up)>{conf_threshold:.0%}")
    print(f"          Sell when bias<{SELL_THRESHOLD} & P(down)>{conf_threshold:.0%}")
    print(f"          Stop loss: {STOP_LOSS_PCT}% | Take profit: {TAKE_PROFIT_PCT}%")

    test_data = prepare_test_data(TEST_STOCKS, months)
    if not test_data:
        print("No data")
        return

    output_path = os.path.join(SCRIPT_DIR, 'trading_backtest_report.pdf')

    all_summaries = []

    with PdfPages(output_path) as pdf:
        for stock_code, prices, inst_cache in test_data:
            name = STOCK_NAMES.get(stock_code, stock_code)
            print(f"\n{'='*50}")
            print(f"Backtesting {name} ({stock_code})...")

            days = simulate_predictions(stock_code, prices, inst_cache)
            if len(days) < 5:
                print("  Not enough data, skip")
                continue

            # 交易模擬
            result = run_trading_simulation(days)

            print(f"  Trades: {result['num_trades']}")
            print(f"  Win rate: {result['win_rate']:.1f}%")
            print(f"  Total return: {result['total_return']:+.2f}%")
            print(f"  Buy & Hold: {result['buy_hold_return']:+.2f}%")
            print(f"  Tax paid: ${result['total_tax']:,.0f}")
            print(f"  Max drawdown: {result['max_drawdown']:.2f}%")

            all_summaries.append({
                'name': name,
                'code': stock_code,
                'result': result,
                'days': days,
            })

            # ════════════ Page 1: 預測 vs 實際價格 + 買賣點 ════════════
            dates = [d['date'] for d in days]
            actual = [d['actual_next'] for d in days]
            predicted = [d['pred_price'] for d in days]
            pred_hi = [d['pred_high'] for d in days]
            pred_lo = [d['pred_low'] for d in days]
            x = np.arange(len(dates))

            fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                                     gridspec_kw={'height_ratios': [5, 2, 2]})
            fig.suptitle(f'{name} ({stock_code}) — Trading Backtest v2 (Improved)',
                         fontsize=16, fontweight='bold')

            # ── 上圖：價格走勢 + 買賣點 ──
            ax = axes[0]
            ax.plot(x, actual, '-', color='#2c3e50', linewidth=2,
                    label='Actual Price', zorder=4)
            ax.plot(x, predicted, '--', color='#3498db', linewidth=1.5,
                    label='Predicted Price', alpha=0.8, zorder=3)
            ax.fill_between(x, pred_lo, pred_hi, color='#3498db', alpha=0.08,
                            label='68% Confidence')

            # 買點
            for bi in result['buy_signals']:
                ax.annotate('BUY', xy=(bi, actual[bi]),
                            xytext=(bi, actual[bi] * 0.97),
                            fontsize=9, fontweight='bold', color='#e74c3c',
                            ha='center',
                            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
                ax.scatter(bi, actual[bi], color='#e74c3c', s=80, zorder=5,
                           marker='^', edgecolors='black', linewidths=0.5)

            # 賣點
            for si in result['sell_signals']:
                ax.annotate('SELL', xy=(si, actual[si]),
                            xytext=(si, actual[si] * 1.03),
                            fontsize=9, fontweight='bold', color='#27ae60',
                            ha='center',
                            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
                ax.scatter(si, actual[si], color='#27ae60', s=80, zorder=5,
                           marker='v', edgecolors='black', linewidths=0.5)

            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xticks(x[::2])
            ax.set_xticklabels([dates[j] for j in range(0, len(dates), 2)],
                               rotation=45, fontsize=8)

            # 右上角 stats
            stats_text = (
                f"Return: {result['total_return']:+.2f}%\n"
                f"B&H: {result['buy_hold_return']:+.2f}%\n"
                f"Trades: {result['num_trades']}\n"
                f"Win: {result['win_rate']:.0f}%\n"
                f"MaxDD: {result['max_drawdown']:.1f}%"
            )
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='#2c3e50', alpha=0.9),
                    fontfamily='monospace')

            # ── 中圖：Bias 信號 ──
            ax = axes[1]
            biases = [d['bias'] for d in days]
            colors = ['#e74c3c' if b > BUY_THRESHOLD else '#27ae60' if b < SELL_THRESHOLD
                       else '#bdc3c7' for b in biases]
            ax.bar(x, biases, color=colors, alpha=0.7, width=0.8)
            ax.axhline(y=BUY_THRESHOLD, color='#e74c3c', linestyle='--',
                       alpha=0.5, linewidth=1, label=f'Buy threshold ({BUY_THRESHOLD})')
            ax.axhline(y=SELL_THRESHOLD, color='#27ae60', linestyle='--',
                       alpha=0.5, linewidth=1, label=f'Sell threshold ({SELL_THRESHOLD})')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Bias', fontsize=11)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticks(x[::2])
            ax.set_xticklabels([dates[j] for j in range(0, len(dates), 2)],
                               rotation=45, fontsize=8)

            # ── 下圖：淨值曲線 ──
            ax = axes[2]
            eq = result['equity_curve']
            eq_color = '#27ae60' if eq[-1] >= INITIAL_CAPITAL else '#e74c3c'
            ax.plot(x, eq, '-', color=eq_color, linewidth=2)
            ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5,
                       linewidth=1, label=f'Initial ${INITIAL_CAPITAL:,.0f}')
            ax.fill_between(x, INITIAL_CAPITAL, eq,
                            where=[e >= INITIAL_CAPITAL for e in eq],
                            color='#27ae60', alpha=0.15)
            ax.fill_between(x, INITIAL_CAPITAL, eq,
                            where=[e < INITIAL_CAPITAL for e in eq],
                            color='#e74c3c', alpha=0.15)
            ax.set_ylabel('Portfolio ($)', fontsize=11)
            ax.set_xlabel('Date', fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xticks(x[::2])
            ax.set_xticklabels([dates[j] for j in range(0, len(dates), 2)],
                               rotation=45, fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close()

            # ════════════ Page 2: 交易明細 ════════════
            if result['trades']:
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.axis('off')
                fig.suptitle(f'{name} ({stock_code}) — Trade Log',
                             fontsize=14, fontweight='bold')

                header = ['#', 'Type', 'Date', 'Price', 'Shares', 'Tax', 'P&L', 'P&L%', 'Reason']
                rows = []
                for j, t in enumerate(result['trades']):
                    rows.append([
                        str(j + 1),
                        t['type'],
                        t['date'],
                        f"${t['price']:,.1f}",
                        f"{t['shares']:,}",
                        f"${t.get('tax', 0):,.0f}",
                        f"${t['profit']:+,.0f}" if t['type'] == 'SELL' else '-',
                        f"{t['pnl_pct']:+.2f}%" if t['type'] == 'SELL' else '-',
                        t['reason'],
                    ])

                table = ax.table(cellText=rows, colLabels=header,
                                 cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.1, 1.6)

                for j in range(len(header)):
                    table[0, j].set_facecolor('#2c3e50')
                    table[0, j].set_text_props(color='white', fontweight='bold')

                for idx, t in enumerate(result['trades']):
                    bg = '#fde8e8' if t['type'] == 'BUY' else '#e8fde8'
                    for j in range(len(header)):
                        table[idx + 1, j].set_facecolor(bg)

                plt.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig)
                plt.close()

        # ════════════ Summary Page ════════════
        if all_summaries:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle('Trading Backtest v2 Summary — Improved Model',
                         fontsize=16, fontweight='bold')

            names = [s['name'] for s in all_summaries]
            returns = [s['result']['total_return'] for s in all_summaries]
            bh_returns = [s['result']['buy_hold_return'] for s in all_summaries]
            win_rates = [s['result']['win_rate'] for s in all_summaries]

            # 左圖：Return comparison
            ax = axes[0]
            x = np.arange(len(names))
            width = 0.35
            b1 = ax.bar(x - width / 2, returns, width, label='Strategy',
                        color=['#27ae60' if r >= 0 else '#e74c3c' for r in returns],
                        edgecolor='black', alpha=0.8)
            b2 = ax.bar(x + width / 2, bh_returns, width, label='Buy & Hold',
                        color='#3498db', edgecolor='black', alpha=0.5)
            for bar, val in zip(b1, returns):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:+.1f}%', ha='center', fontsize=9, fontweight='bold')
            for bar, val in zip(b2, bh_returns):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:+.1f}%', ha='center', fontsize=9)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Return (%)', fontsize=12)
            ax.set_title('Strategy vs Buy & Hold', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # 右圖：Win rate + num trades
            ax = axes[1]
            num_trades = [s['result']['num_trades'] for s in all_summaries]
            bars = ax.bar(x, win_rates, width=0.5,
                          color=['#27ae60' if w >= 50 else '#e74c3c' for w in win_rates],
                          edgecolor='black', alpha=0.8)
            for bar, val, nt in zip(bars, win_rates, num_trades):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}%\n({nt} trades)', ha='center', fontsize=9, fontweight='bold')
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
            ax.set_ylabel('Win Rate (%)', fontsize=12)
            ax.set_title('Win Rate per Stock', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=10)
            ax.set_ylim(0, max(win_rates + [60]) + 15)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close()

    # 印出總結
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY (Improved Model v2)")
    print(f"{'='*60}")
    total_pnl = 0
    total_tax = 0
    for s in all_summaries:
        r = s['result']
        total_pnl += r['total_profit']
        total_tax += r['total_tax']
        alpha = r['total_return'] - r['buy_hold_return']
        print(f"  {s['name']:12s} | Return {r['total_return']:+6.2f}% | "
              f"B&H {r['buy_hold_return']:+6.2f}% | "
              f"Alpha {alpha:+5.2f}% | "
              f"Win {r['win_rate']:4.0f}% | "
              f"Trades {r['num_trades']:2d} | "
              f"Tax ${r['total_tax']:>8,.0f} | "
              f"MaxDD {r['max_drawdown']:5.2f}%")

    print(f"\n  Total P&L: ${total_pnl:+,.0f}")
    print(f"  Total Tax: ${total_tax:+,.0f}")
    print(f"  Tax Rate:  {TAX_RATE:.0%} per round-trip")
    print(f"\n  PDF: {output_path}")


if __name__ == '__main__':
    main()
