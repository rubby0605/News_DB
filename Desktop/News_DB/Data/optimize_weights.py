#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子模型權重優化
使用歷史資料找出最佳權重組合

@author: rubylintu
"""

import os
import json
import datetime
import time
import requests
import numpy as np
from itertools import product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'optimized_weights.json')


def get_historical_data(stock_code, year_month):
    """取得歷史價格"""
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
    except:
        return []


def get_institutional_data(date_str):
    """取得法人資料"""
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
            total = foreign + int(row[10].replace(',', '')) // 1000 + int(row[11].replace(',', '')) // 1000

            result[code] = {'foreign': foreign, 'total': total}

        return result
    except:
        return None


def calc_bias_with_weights(inst_data, stock_code, prices, day_index, weights):
    """
    使用指定權重計算偏移量（12 參數版）

    weights = {
        'foreign_large': 外資大買賣門檻,
        'foreign_medium': 外資中等門檻,
        'foreign_weight': 外資權重,
        'momentum_weight': 動量權重,
        'ema_weight': 均線權重,
        'momentum_threshold': 動量門檻,
        'dampening_threshold': 抑制門檻,
        'market_weight': 大盤權重,
        'gpt_weight': GPT 權重,
        'confidence_threshold': 信心度門檻,
        'volume_weight': 成交量權重,
        'decision_threshold': 決策門檻 (取代寫死的 > 2 / < -2),
    }
    """
    bias = 0

    # 法人
    if stock_code in inst_data:
        foreign = inst_data[stock_code]['foreign']

        if foreign > weights['foreign_large']:
            bias += weights['foreign_weight']
        elif foreign > weights['foreign_medium']:
            bias += weights['foreign_weight'] * 0.5
        elif foreign < -weights['foreign_large']:
            bias -= weights['foreign_weight']
        elif foreign < -weights['foreign_medium']:
            bias -= weights['foreign_weight'] * 0.5

    # 動量
    if day_index >= 5:
        momentum = sum(prices[day_index - i]['change'] for i in range(5) if day_index - i >= 0)

        if momentum > weights['momentum_threshold'] * 2:
            bias += weights['momentum_weight']
        elif momentum > weights['momentum_threshold']:
            bias += weights['momentum_weight'] * 0.5
        elif momentum < -weights['momentum_threshold'] * 2:
            bias -= weights['momentum_weight']
        elif momentum < -weights['momentum_threshold']:
            bias -= weights['momentum_weight'] * 0.5

    # 均線
    if day_index >= 10:
        recent_avg = sum(p['close'] for p in prices[day_index-5:day_index]) / 5
        longer_avg = sum(p['close'] for p in prices[day_index-10:day_index]) / 10
        current = prices[day_index]['close']

        if current > recent_avg > longer_avg:
            bias += weights['ema_weight']
        elif current < recent_avg < longer_avg:
            bias -= weights['ema_weight']

    # 成交量（量比）
    volume_weight = weights.get('volume_weight', 0.5)
    if day_index >= 5:
        volumes = [p.get('volume', 0) for p in prices[max(0, day_index-5):day_index]]
        avg_vol = sum(volumes) / len(volumes) if volumes and sum(volumes) > 0 else 0
        current_vol = prices[day_index].get('volume', 0)
        if avg_vol > 0 and current_vol > 0:
            vol_ratio = current_vol / avg_vol
            if vol_ratio > 1.5:
                # 放量 → 強化既有方向
                bias *= (1 + volume_weight * 0.3)
            elif vol_ratio < 0.5:
                # 縮量 → 衰減信號
                bias *= (1 - volume_weight * 0.2)

    # 抑制過大偏移
    dampening_threshold = weights.get('dampening_threshold', 3.0)
    if abs(bias) > dampening_threshold:
        bias = dampening_threshold * (1 if bias > 0 else -1) + \
               (bias - dampening_threshold * (1 if bias > 0 else -1)) * 0.3

    return bias


def evaluate_weights(weights, test_data):
    """評估權重的準確率"""
    correct = 0
    total = 0

    for stock_code, prices, inst_cache in test_data:
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
            bias = calc_bias_with_weights(inst_data, stock_code, prices, i, weights)

            # 預測（使用可調決策門檻）
            decision_threshold = weights.get('decision_threshold', 2.0)
            if bias > decision_threshold:
                pred = 1  # 漲
            elif bias < -decision_threshold:
                pred = -1  # 跌
            else:
                pred = 0  # 盤整

            # 實際
            actual_change = prices[i + 1]['change']
            if actual_change > 0.5:
                actual = 1
            elif actual_change < -0.5:
                actual = -1
            else:
                actual = 0

            # 判斷
            if pred == actual or (pred == 1 and actual_change > 0) or (pred == -1 and actual_change < 0):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


def prepare_test_data(stock_codes, months):
    """準備測試資料"""
    print("準備測試資料...")

    test_data = []
    inst_cache = {}

    for code in stock_codes:
        all_prices = []
        for month in months:
            prices = get_historical_data(code, month)
            all_prices.extend(prices)
            time.sleep(0.3)

        if len(all_prices) < 15:
            continue

        # 收集需要的法人資料日期
        for i in range(10, len(all_prices)):
            date_parts = all_prices[i]['date'].split('/')
            if len(date_parts) != 3:
                continue

            roc_year = int(date_parts[0])
            month = int(date_parts[1])
            day = int(date_parts[2])
            date_str = f'{roc_year + 1911}{month:02d}{day:02d}'

            if date_str not in inst_cache:
                inst_data = get_institutional_data(date_str)
                if inst_data:
                    inst_cache[date_str] = inst_data
                time.sleep(0.2)

        test_data.append((code, all_prices, inst_cache))
        print(f"  {code}: {len(all_prices)} 天資料")

    return test_data


def genetic_algorithm(test_data, population_size=50, generations=30, mutation_rate=0.2):
    """
    遺傳演算法優化權重

    Args:
        test_data: 測試資料
        population_size: 族群大小
        generations: 迭代代數
        mutation_rate: 突變率
    """
    print(f"\n🧬 開始遺傳演算法優化...")
    print(f"   族群大小: {population_size}")
    print(f"   迭代代數: {generations}")
    print(f"   突變率: {mutation_rate}")

    # 參數範圍（12 個參數）
    param_ranges = {
        'foreign_large': (1000, 8000),
        'foreign_medium': (200, 2000),
        'foreign_weight': (1, 8),
        'momentum_weight': (0.5, 5),
        'ema_weight': (0.5, 4),
        'momentum_threshold': (1, 8),
        'dampening_threshold': (1.5, 6.0),
        'market_weight': (0.3, 3.0),
        'gpt_weight': (-3.0, 3.0),
        'confidence_threshold': (0.5, 0.85),
        'volume_weight': (0.2, 2.0),
        'decision_threshold': (1.0, 4.0),
    }

    def enforce_constraints(ind):
        """強制邏輯 constraint"""
        # foreign_large 一定要 > foreign_medium
        if ind['foreign_medium'] >= ind['foreign_large']:
            ind['foreign_large'] = ind['foreign_medium'] + np.random.uniform(200, 1000)
            ind['foreign_large'] = np.clip(ind['foreign_large'], *param_ranges['foreign_large'])

        # decision_threshold 一定要 < dampening_threshold
        if ind['decision_threshold'] >= ind['dampening_threshold']:
            ind['decision_threshold'] = ind['dampening_threshold'] * np.random.uniform(0.4, 0.8)
            ind['decision_threshold'] = np.clip(ind['decision_threshold'], *param_ranges['decision_threshold'])

        return ind

    def random_individual():
        """產生隨機個體"""
        ind = {
            k: np.random.uniform(v[0], v[1])
            for k, v in param_ranges.items()
        }
        return enforce_constraints(ind)

    def crossover(parent1, parent2):
        """交叉"""
        child = {}
        for k in param_ranges.keys():
            if np.random.random() < 0.5:
                child[k] = parent1[k]
            else:
                child[k] = parent2[k]
        return enforce_constraints(child)

    def mutate(individual):
        """突變"""
        mutated = individual.copy()
        for k, (low, high) in param_ranges.items():
            if np.random.random() < mutation_rate:
                # 在當前值附近隨機調整
                delta = (high - low) * 0.3 * np.random.randn()
                mutated[k] = np.clip(mutated[k] + delta, low, high)
        return enforce_constraints(mutated)

    def select_parents(population, fitnesses, num_parents):
        """輪盤選擇"""
        # 將適應度轉為正數
        min_fit = min(fitnesses)
        adjusted = [f - min_fit + 0.01 for f in fitnesses]
        total = sum(adjusted)
        probs = [f / total for f in adjusted]

        indices = np.random.choice(len(population), size=num_parents, p=probs, replace=False)
        return [population[i] for i in indices]

    # 初始化族群
    population = [random_individual() for _ in range(population_size)]

    best_ever = None
    best_ever_fitness = 0
    history = []

    for gen in range(generations):
        # 評估適應度
        fitnesses = []
        for ind in population:
            weights = ind.copy()
            weights['foreign_large'] = int(weights['foreign_large'])
            weights['foreign_medium'] = int(weights['foreign_medium'])
            fitness = evaluate_weights(weights, test_data)
            fitnesses.append(fitness)

        # 記錄最佳
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best = population[gen_best_idx]

        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever = gen_best.copy()
            print(f"  🏆 第 {gen+1} 代: 新最佳 {best_ever_fitness:.1%}")

        history.append({
            'generation': gen + 1,
            'best': gen_best_fitness,
            'avg': np.mean(fitnesses)
        })

        if (gen + 1) % 5 == 0:
            print(f"  第 {gen+1} 代: 最佳 {gen_best_fitness:.1%}, 平均 {np.mean(fitnesses):.1%}")

        # 產生下一代
        new_population = []

        # 菁英保留（保留最好的 10%）
        elite_count = max(2, population_size // 10)
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        for i in elite_indices:
            new_population.append(population[i])

        # 交叉產生其餘個體
        while len(new_population) < population_size:
            parents = select_parents(population, fitnesses, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # 最終結果
    best_weights = {}
    for k, v in best_ever.items():
        if k in ('foreign_large', 'foreign_medium'):
            best_weights[k] = int(v)
        else:
            best_weights[k] = round(v, 2)

    return best_weights, best_ever_fitness, history


def grid_search(test_data):
    """網格搜尋（備用）"""
    # 直接用遺傳演算法
    return genetic_algorithm(test_data)


def save_weights(weights, accuracy):
    """儲存最佳權重"""
    data = {
        'weights': weights,
        'accuracy': accuracy,
        'updated': datetime.datetime.now().isoformat()
    }

    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n權重已儲存到 {WEIGHTS_FILE}")


def load_weights():
    """載入已儲存的權重"""
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def calc_weight_drift(old_weights, new_weights):
    """計算新舊權重的漂移程度（0~1，越大表示差異越大）"""
    if not old_weights or not new_weights:
        return 1.0

    param_ranges = {
        'foreign_large': (1000, 8000),
        'foreign_medium': (200, 2000),
        'foreign_weight': (1, 8),
        'momentum_weight': (0.5, 5),
        'ema_weight': (0.5, 4),
        'momentum_threshold': (1, 8),
        'dampening_threshold': (1.5, 6.0),
        'market_weight': (0.3, 3.0),
        'gpt_weight': (-3.0, 3.0),
        'confidence_threshold': (0.5, 0.85),
        'volume_weight': (0.2, 2.0),
        'decision_threshold': (1.0, 4.0),
    }

    drifts = []
    for k, (low, high) in param_ranges.items():
        old_v = old_weights.get(k, (low + high) / 2)
        new_v = new_weights.get(k, (low + high) / 2)
        normalized_drift = abs(new_v - old_v) / (high - low)
        drifts.append(normalized_drift)

    return sum(drifts) / len(drifts) if drifts else 0


def run_daily_optimization(stock_codes=None, rolling_days=40,
                           population_size=30, generations=20,
                           max_drift=0.25, min_improvement=0.005):
    """
    每日盤後 GA 優化（rolling window + 穩定性檢查）

    Args:
        stock_codes: 測試股票代碼，預設 5 檔
        rolling_days: rolling window 天數
        population_size: GA 族群大小
        generations: GA 迭代數
        max_drift: 權重最大允許漂移（0~1），超過則不更新
        min_improvement: 最小準確率提升，未達到則不更新

    Returns:
        dict: {'updated': bool, 'reason': str, 'new_acc': float, 'old_acc': float}
    """
    if stock_codes is None:
        stock_codes = ['2330', '3189', '2454', '2881', '2603']

    print(f"\n📊 每日 GA 優化（rolling {rolling_days} 天）")

    # 計算 rolling window 月份（涵蓋最近 N 天）
    today = datetime.date.today()
    months_needed = (rolling_days // 20) + 2  # 每月約 20 個交易日，多取 2 個月
    months = []
    for i in range(months_needed):
        target_month = today.month - i
        target_year = today.year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        months.append(f'{target_year}{target_month:02d}')

    months.reverse()
    print(f"  測試股票: {stock_codes}")
    print(f"  資料月份: {months}")

    # 準備資料
    test_data = prepare_test_data(stock_codes, months)
    if not test_data:
        return {'updated': False, 'reason': '無測試資料'}

    # 載入舊權重
    old_data = load_weights()
    old_weights = old_data.get('weights', {}) if old_data else {}
    old_acc = old_data.get('accuracy', 0) if old_data else 0

    # 跑 GA
    new_weights, new_acc, history = genetic_algorithm(
        test_data,
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2
    )

    # 穩定性檢查
    drift = calc_weight_drift(old_weights, new_weights)
    improvement = new_acc - old_acc

    print(f"\n  舊準確率: {old_acc:.1%}")
    print(f"  新準確率: {new_acc:.1%} (差異: {improvement:+.1%})")
    print(f"  權重漂移: {drift:.2%}")

    # 決定是否更新
    if drift > max_drift and improvement < min_improvement * 2:
        reason = f'漂移過大 ({drift:.1%}) 且改善不足 ({improvement:+.1%})，不更新'
        print(f"  ⚠️ {reason}")
        return {
            'updated': False, 'reason': reason,
            'new_acc': new_acc, 'old_acc': old_acc,
            'drift': drift, 'new_weights': new_weights
        }

    if improvement < -min_improvement:
        reason = f'準確率下降 ({improvement:+.1%})，不更新'
        print(f"  ⚠️ {reason}")
        return {
            'updated': False, 'reason': reason,
            'new_acc': new_acc, 'old_acc': old_acc,
            'drift': drift, 'new_weights': new_weights
        }

    # 更新權重
    save_weights(new_weights, new_acc)
    reason = f'準確率 {old_acc:.1%} → {new_acc:.1%}，漂移 {drift:.1%}，已更新'
    print(f"  ✅ {reason}")

    return {
        'updated': True, 'reason': reason,
        'new_acc': new_acc, 'old_acc': old_acc,
        'drift': drift, 'new_weights': new_weights
    }


def main():
    """主程式"""
    from newslib import read_stock_list

    print("=" * 60)
    print("粒子模型權重優化")
    print("=" * 60)

    # 讀取股票清單
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    # 選擇測試股票
    test_stocks = ['2330', '3189', '2454', '2881', '2603']  # 台積電、景碩、聯發科、富邦金、長榮

    # 計算月份
    today = datetime.date.today()
    months = []
    for i in range(2):  # 2個月
        target_month = today.month - i - 1
        target_year = today.year
        if target_month <= 0:
            target_month += 12
            target_year -= 1
        months.append(f'{target_year}{target_month:02d}')

    print(f"測試股票: {test_stocks}")
    print(f"測試月份: {months}")

    # 準備資料
    test_data = prepare_test_data(test_stocks, months)

    if not test_data:
        print("無測試資料")
        return

    # 網格搜尋
    best_weights, best_accuracy, top_results = grid_search(test_data)

    # 印出結果
    print("\n" + "=" * 60)
    print("優化結果")
    print("=" * 60)
    print(f"\n🏆 最佳準確率: {best_accuracy:.1%}")
    print("\n最佳權重:")
    for k, v in best_weights.items():
        print(f"  {k}: {v}")

    print("\n📊 TOP 10 組合:")
    for i, (acc, weights) in enumerate(top_results):
        print(f"  {i+1}. {acc:.1%} - foreign_weight={weights['foreign_weight']}, momentum_weight={weights['momentum_weight']}")

    # 儲存
    save_weights(best_weights, best_accuracy)

    # 發送到 Discord
    from notifier import send_discord

    # 比較新舊權重
    old_data = load_weights()
    old_weights = old_data.get('weights', {}) if old_data else {}
    old_acc = old_data.get('accuracy', 0) if old_data else 0

    param_labels = {
        'foreign_large': '外資大量門檻',
        'foreign_medium': '外資中量門檻',
        'foreign_weight': '外資權重',
        'momentum_weight': '動量權重',
        'ema_weight': '均線權重',
        'momentum_threshold': '動量門檻',
        'dampening_threshold': '抑制門檻',
        'market_weight': '大盤權重',
        'gpt_weight': 'GPT權重',
        'confidence_threshold': '信心門檻',
        'volume_weight': '成量權重',
        'decision_threshold': '決策門檻',
    }

    weight_lines = []
    for k, label in param_labels.items():
        new_v = best_weights.get(k, '-')
        old_v = old_weights.get(k, '-')
        if old_v != '-' and new_v != '-':
            delta = new_v - old_v if isinstance(new_v, (int, float)) and isinstance(old_v, (int, float)) else 0
            arrow = '↑' if delta > 0 else '↓' if delta < 0 else '→'
            weight_lines.append(f"• {label}: {new_v} {arrow}")
        else:
            weight_lines.append(f"• {label}: {new_v}")

    weights_text = '\n'.join(weight_lines)
    acc_delta = best_accuracy - old_acc
    acc_arrow = '↑' if acc_delta > 0 else '↓' if acc_delta < 0 else '→'

    from notifier import send_discord_embed, COLOR_INFO

    embed = {
        "title": "🧬 GA 權重優化完成",
        "color": COLOR_INFO,
        "fields": [
            {"name": "回測準確率", "value": f"**{best_accuracy:.1%}** (舊: {old_acc:.1%} {acc_arrow})", "inline": True},
            {"name": "測試股票", "value": ", ".join(test_stocks), "inline": True},
            {"name": "測試期間", "value": f"{months[0]} ~ {months[-1]}", "inline": True},
            {"name": "最佳參數 (12)", "value": weights_text, "inline": False},
        ],
    }
    send_discord_embed(embed)
    print("\n已發送到 Discord!")


if __name__ == "__main__":
    main()
