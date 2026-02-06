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
    使用指定權重計算偏移量

    weights = {
        'foreign_large': 外資大買賣門檻,
        'foreign_medium': 外資中等門檻,
        'foreign_weight': 外資權重,
        'momentum_weight': 動量權重,
        'ema_weight': 均線權重,
        'momentum_threshold': 動量門檻
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

            # 預測
            if bias > 2:
                pred = 1  # 漲
            elif bias < -2:
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

    # 參數範圍
    param_ranges = {
        'foreign_large': (1000, 8000),
        'foreign_medium': (200, 2000),
        'foreign_weight': (1, 8),
        'momentum_weight': (0.5, 5),
        'ema_weight': (0.5, 4),
        'momentum_threshold': (1, 8)
    }

    def random_individual():
        """產生隨機個體"""
        return {
            k: np.random.uniform(v[0], v[1])
            for k, v in param_ranges.items()
        }

    def crossover(parent1, parent2):
        """交叉"""
        child = {}
        for k in param_ranges.keys():
            if np.random.random() < 0.5:
                child[k] = parent1[k]
            else:
                child[k] = parent2[k]
        return child

    def mutate(individual):
        """突變"""
        mutated = individual.copy()
        for k, (low, high) in param_ranges.items():
            if np.random.random() < mutation_rate:
                # 在當前值附近隨機調整
                delta = (high - low) * 0.3 * np.random.randn()
                mutated[k] = np.clip(mutated[k] + delta, low, high)
        return mutated

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
            # 將浮點數轉為整數（某些參數需要）
            weights = {
                'foreign_large': int(ind['foreign_large']),
                'foreign_medium': int(ind['foreign_medium']),
                'foreign_weight': ind['foreign_weight'],
                'momentum_weight': ind['momentum_weight'],
                'ema_weight': ind['ema_weight'],
                'momentum_threshold': ind['momentum_threshold']
            }
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
    best_weights = {
        'foreign_large': int(best_ever['foreign_large']),
        'foreign_medium': int(best_ever['foreign_medium']),
        'foreign_weight': round(best_ever['foreign_weight'], 2),
        'momentum_weight': round(best_ever['momentum_weight'], 2),
        'ema_weight': round(best_ever['ema_weight'], 2),
        'momentum_threshold': round(best_ever['momentum_threshold'], 2)
    }

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

    message = f'''**🔧 權重優化完成**

**🏆 最佳準確率: {best_accuracy:.1%}**

**最佳權重:**
• 外資大量門檻: {best_weights['foreign_large']} 張
• 外資中量門檻: {best_weights['foreign_medium']} 張
• 外資權重: {best_weights['foreign_weight']}
• 動量權重: {best_weights['momentum_weight']}
• 均線權重: {best_weights['ema_weight']}
• 動量門檻: {best_weights['momentum_threshold']}%

測試股票: {", ".join(test_stocks)}
測試期間: {months[0]} ~ {months[-1]}'''

    send_discord(message, title='模型優化結果')
    print("\n已發送到 Discord!")


if __name__ == "__main__":
    main()
