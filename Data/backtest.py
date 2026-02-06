#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子模型回測系統
驗證預測模型的歷史準確率

@author: rubylintu
"""

import os
import json
import datetime
import time
import requests
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_historical_institutional(date_str):
    """
    取得指定日期的法人買賣超資料

    Args:
        date_str: 日期字串 YYYYMMDD

    Returns:
        dict: {股票代號: {'foreign': 外資, 'total': 合計}}
    """
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

    except Exception as e:
        print(f"取得 {date_str} 法人資料失敗: {e}")
        return None


def get_historical_prices(stock_code, year_month):
    """
    取得指定月份的股票價格

    Args:
        stock_code: 股票代號
        year_month: YYYYMM

    Returns:
        list: [{'date': '...', 'close': ..., 'change': ...}, ...]
    """
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
                date = row[0]  # 民國年/月/日
                close = float(row[6].replace(',', '')) if row[6] != '--' else None

                if close:
                    change = 0
                    if prev_close:
                        change = (close - prev_close) / prev_close * 100

                    result.append({
                        'date': date,
                        'close': close,
                        'change': change
                    })
                    prev_close = close
            except:
                continue

        return result

    except Exception as e:
        print(f"取得 {stock_code} {year_month} 價格失敗: {e}")
        return []


def calc_prediction_bias(institutional_data, stock_code, prices, day_index):
    """
    根據法人資料計算預測偏移量（模擬粒子模型邏輯）

    Args:
        institutional_data: 法人資料
        stock_code: 股票代號
        prices: 價格列表
        day_index: 當天在列表中的索引

    Returns:
        float: 偏移量 (-10 到 +10)
    """
    bias = 0

    # 1. 法人買賣超 (40%)
    if stock_code in institutional_data:
        inst = institutional_data[stock_code]
        foreign = inst['foreign']

        if foreign > 3000:
            bias += 4
        elif foreign > 1000:
            bias += 2
        elif foreign < -3000:
            bias -= 4
        elif foreign < -1000:
            bias -= 2

        if inst['total'] > 5000:
            bias += 1
        elif inst['total'] < -5000:
            bias -= 1

    # 2. 價格動量 (30%)
    if day_index >= 5:
        momentum_5d = prices[day_index]['change']
        for i in range(1, 5):
            if day_index - i >= 0:
                momentum_5d += prices[day_index - i]['change']

        if momentum_5d > 5:
            bias += 2
        elif momentum_5d > 2:
            bias += 1
        elif momentum_5d < -5:
            bias -= 2
        elif momentum_5d < -2:
            bias -= 1

    # 3. 簡化的均線判斷 (20%)
    if day_index >= 10:
        recent_avg = sum(p['close'] for p in prices[day_index-5:day_index]) / 5
        longer_avg = sum(p['close'] for p in prices[day_index-10:day_index]) / 10
        current = prices[day_index]['close']

        if current > recent_avg > longer_avg:
            bias += 2
        elif current < recent_avg < longer_avg:
            bias -= 2

    # 限制範圍
    return max(-10, min(10, bias))


def predict_direction(bias):
    """根據偏移量預測方向"""
    if bias > 2:
        return '漲'
    elif bias < -2:
        return '跌'
    else:
        return '盤整'


def get_actual_direction(change_pct):
    """根據實際漲跌幅判斷方向"""
    if change_pct > 0.5:
        return '漲'
    elif change_pct < -0.5:
        return '跌'
    else:
        return '盤整'


def run_backtest(stock_codes, months_back=2):
    """
    執行回測

    Args:
        stock_codes: 股票代號列表 [(code, name), ...]
        months_back: 回測幾個月

    Returns:
        dict: 回測結果
    """
    print("=" * 60)
    print("開始回測...")
    print("=" * 60)

    # 計算要回測的月份
    today = datetime.date.today()
    months = []
    for i in range(months_back):
        target_month = today.month - i - 1
        target_year = today.year
        if target_month <= 0:
            target_month += 12
            target_year -= 1
        months.append(f'{target_year}{target_month:02d}')

    print(f"回測月份: {months}")
    print()

    all_results = []

    for code, name in stock_codes:
        print(f"回測 {name} ({code})...")

        # 取得歷史價格
        all_prices = []
        for month in reversed(months):
            prices = get_historical_prices(code, month)
            all_prices.extend(prices)
            time.sleep(0.3)

        if len(all_prices) < 15:
            print(f"  {name} 資料不足，跳過")
            continue

        stock_results = []

        # 從第 10 天開始預測（需要前面的資料算動量）
        for i in range(10, len(all_prices) - 1):
            # 取得當天日期
            date_parts = all_prices[i]['date'].split('/')
            if len(date_parts) == 3:
                roc_year = int(date_parts[0])
                month = int(date_parts[1])
                day = int(date_parts[2])
                ad_year = roc_year + 1911
                date_str = f'{ad_year}{month:02d}{day:02d}'
            else:
                continue

            # 取得當天法人資料
            inst_data = get_historical_institutional(date_str)
            if not inst_data:
                continue

            # 計算預測
            bias = calc_prediction_bias(inst_data, code, all_prices, i)
            pred_direction = predict_direction(bias)

            # 隔天實際結果
            actual_change = all_prices[i + 1]['change']
            actual_direction = get_actual_direction(actual_change)

            # 判斷是否正確
            correct = (pred_direction == actual_direction) or \
                     (pred_direction == '漲' and actual_change > 0) or \
                     (pred_direction == '跌' and actual_change < 0)

            stock_results.append({
                'date': all_prices[i]['date'],
                'stock': name,
                'code': code,
                'bias': bias,
                'pred_direction': pred_direction,
                'actual_change': actual_change,
                'actual_direction': actual_direction,
                'correct': correct
            })

            time.sleep(0.2)  # 避免請求太快

        if stock_results:
            correct_count = sum(1 for r in stock_results if r['correct'])
            accuracy = correct_count / len(stock_results) * 100
            print(f"  {name}: {correct_count}/{len(stock_results)} = {accuracy:.1f}%")
            all_results.extend(stock_results)

    return all_results


def analyze_results(results):
    """分析回測結果"""
    if not results:
        return None

    df = pd.DataFrame(results)

    # 整體準確率
    total_correct = df['correct'].sum()
    total_count = len(df)
    overall_accuracy = total_correct / total_count * 100

    # 各股票準確率
    stock_accuracy = df.groupby('stock').agg({
        'correct': ['sum', 'count']
    })
    stock_accuracy.columns = ['correct', 'total']
    stock_accuracy['accuracy'] = stock_accuracy['correct'] / stock_accuracy['total'] * 100
    stock_accuracy = stock_accuracy.sort_values('accuracy', ascending=False)

    # 預測方向分布
    pred_dist = df['pred_direction'].value_counts()

    # 各方向準確率
    direction_accuracy = {}
    for direction in ['漲', '跌', '盤整']:
        subset = df[df['pred_direction'] == direction]
        if len(subset) > 0:
            acc = subset['correct'].sum() / len(subset) * 100
            direction_accuracy[direction] = {
                'count': len(subset),
                'accuracy': acc
            }

    # 偏移量分析
    df['bias_abs'] = df['bias'].abs()
    high_confidence = df[df['bias_abs'] >= 4]
    if len(high_confidence) > 0:
        high_conf_accuracy = high_confidence['correct'].sum() / len(high_confidence) * 100
    else:
        high_conf_accuracy = 0

    return {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total_count,
        'total_correct': total_correct,
        'stock_accuracy': stock_accuracy,
        'direction_accuracy': direction_accuracy,
        'high_confidence_accuracy': high_conf_accuracy,
        'high_confidence_count': len(high_confidence)
    }


def print_report(analysis):
    """印出回測報告"""
    if not analysis:
        print("無回測結果")
        return

    print()
    print("=" * 60)
    print("📊 回測報告")
    print("=" * 60)

    print(f"\n🎯 整體準確率: {analysis['overall_accuracy']:.1f}%")
    print(f"   總預測次數: {analysis['total_predictions']}")
    print(f"   正確次數: {analysis['total_correct']}")

    print(f"\n🔥 高信心度預測 (|bias| >= 4):")
    print(f"   準確率: {analysis['high_confidence_accuracy']:.1f}%")
    print(f"   次數: {analysis['high_confidence_count']}")

    print("\n📈 各方向準確率:")
    for direction, data in analysis['direction_accuracy'].items():
        emoji = '🔴' if direction == '漲' else '🟢' if direction == '跌' else '⚪'
        print(f"   {emoji} {direction}: {data['accuracy']:.1f}% ({data['count']} 次)")

    print("\n📋 各股票準確率:")
    for stock, row in analysis['stock_accuracy'].iterrows():
        acc = row['accuracy']
        emoji = '✅' if acc >= 60 else '⚠️' if acc >= 50 else '❌'
        print(f"   {emoji} {stock}: {acc:.1f}% ({int(row['correct'])}/{int(row['total'])})")


def send_backtest_report(analysis):
    """發送回測報告到 Discord"""
    from notifier import send_discord

    lines = [
        '**📊 粒子模型回測報告**',
        '',
        f'**🎯 整體準確率: {analysis["overall_accuracy"]:.1f}%**',
        f'總預測: {analysis["total_predictions"]} 次 | 正確: {analysis["total_correct"]} 次',
        '',
        f'**🔥 高信心度預測:** {analysis["high_confidence_accuracy"]:.1f}% ({analysis["high_confidence_count"]} 次)',
        '',
        '**📈 各方向準確率：**'
    ]

    for direction, data in analysis['direction_accuracy'].items():
        emoji = '🔴' if direction == '漲' else '🟢' if direction == '跌' else '⚪'
        lines.append(f'{emoji} {direction}: {data["accuracy"]:.1f}% ({data["count"]} 次)')

    lines.append('')
    lines.append('**📋 各股票準確率 TOP 5：**')

    for i, (stock, row) in enumerate(analysis['stock_accuracy'].head(5).iterrows()):
        acc = row['accuracy']
        lines.append(f'• {stock}: {acc:.1f}%')

    # 結論
    lines.append('')
    if analysis['overall_accuracy'] >= 55:
        lines.append('✅ 模型表現優於隨機 (>55%)')
    elif analysis['overall_accuracy'] >= 50:
        lines.append('⚠️ 模型表現接近隨機 (~50%)')
    else:
        lines.append('❌ 模型表現不佳 (<50%)')

    message = '\n'.join(lines)
    send_discord(message, title='模型回測結果')


def main():
    """主程式"""
    from newslib import read_stock_list

    # 讀取股票清單
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    # 只測試部分股票（節省時間）
    test_stocks = [
        (str(dict_stock[name]), name)
        for name in ['景碩', '台積電', '聯發科', '長榮', '富邦金']
        if name in dict_stock
    ]

    print(f"測試股票: {[s[1] for s in test_stocks]}")

    # 執行回測
    results = run_backtest(test_stocks, months_back=2)

    # 分析結果
    analysis = analyze_results(results)

    # 印出報告
    print_report(analysis)

    # 詢問是否發送到 Discord
    print()
    send = input("發送報告到 Discord? (y/n): ")
    if send.lower() == 'y':
        send_backtest_report(analysis)
        print("已發送!")


if __name__ == "__main__":
    main()
