#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA 關鍵字優化器

用遺傳演算法篩選哪些看漲/看跌關鍵字真正有效，
移除噪音關鍵字、保留高效關鍵字。

@author: rubylintu
"""

import os
import json
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZED_KEYWORDS_FILE = os.path.join(SCRIPT_DIR, 'optimized_keywords.json')


def load_keyword_lists():
    """載入原始關鍵字清單"""
    from hybrid_predictor import BULL_KEYWORDS, BEAR_KEYWORDS
    return list(BULL_KEYWORDS), list(BEAR_KEYWORDS)


def load_training_data():
    """
    載入歷史新聞 + 實際結果作為訓練資料

    Returns:
        list of (text, actual_direction)
        actual_direction: 1 (漲), -1 (跌), 0 (盤整)
    """
    # 從 prediction_history 和 news_data 目錄載入
    from prediction_history import load_history

    history = load_history()
    preds = [p for p in history['predictions'] if p['actual_direction'] is not None]

    if not preds:
        return []

    # 嘗試載入對應的新聞
    news_dir = os.path.join(SCRIPT_DIR, 'news_data')
    training_data = []

    for pred in preds:
        stock_code = pred['stock_code']
        date = pred['date']
        actual_dir = pred['actual_direction']

        actual_label = 1 if actual_dir == '漲' else (-1 if actual_dir == '跌' else 0)

        # 嘗試從 seen_news.json 或 news_data 找相關新聞
        news_file = os.path.join(news_dir, f'news_{date}.json')
        if os.path.exists(news_file):
            try:
                with open(news_file, 'r', encoding='utf-8') as f:
                    daily_news = json.load(f)
                # 找跟這檔股票相關的新聞
                for news in daily_news:
                    title = news.get('title', '')
                    if stock_code in title or pred.get('stock_name', '') in title:
                        training_data.append((title, actual_label))
            except Exception:
                pass

    return training_data


def keyword_score_with_mask(text, bull_keywords, bear_keywords, bull_mask, bear_mask):
    """使用遮罩後的關鍵字計算分數"""
    selected_bull = [kw for kw, m in zip(bull_keywords, bull_mask) if m]
    selected_bear = [kw for kw, m in zip(bear_keywords, bear_mask) if m]

    bull_count = sum(1 for kw in selected_bull if kw in text)
    bear_count = sum(1 for kw in selected_bear if kw in text)

    total = bull_count + bear_count
    if total == 0:
        return 0

    return (bull_count - bear_count) / total


def evaluate_keyword_mask(bull_keywords, bear_keywords, bull_mask, bear_mask, training_data):
    """評估關鍵字遮罩的準確率"""
    if not training_data:
        return 0

    correct = 0
    total = 0

    for text, actual in training_data:
        score = keyword_score_with_mask(text, bull_keywords, bear_keywords, bull_mask, bear_mask)

        if score > 0.1:
            pred = 1
        elif score < -0.1:
            pred = -1
        else:
            continue  # 無信號不計

        if pred == actual or (pred == 1 and actual >= 0) or (pred == -1 and actual <= 0):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


def optimize(pop_size=100, generations=50, mutation_rate=0.1):
    """
    用 GA 優化關鍵字選擇

    Returns:
        dict: {
            'selected_bull': list,
            'removed_bull': list,
            'selected_bear': list,
            'removed_bear': list,
            'accuracy': float,
        }
    """
    bull_kw, bear_kw = load_keyword_lists()
    training_data = load_training_data()

    if len(training_data) < 10:
        print(f"訓練資料不足 ({len(training_data)} 筆)，跳過關鍵字優化")
        return None

    n_bull = len(bull_kw)
    n_bear = len(bear_kw)
    total_genes = n_bull + n_bear

    print(f"🧬 關鍵字 GA 優化")
    print(f"   看漲關鍵字: {n_bull}, 看跌關鍵字: {n_bear}")
    print(f"   訓練樣本: {len(training_data)}")

    # 初始化族群（二進位遮罩，1=保留, 0=移除）
    population = []
    for _ in range(pop_size):
        # 偏向保留（80% 機率保留）
        mask = [1 if random.random() < 0.8 else 0 for _ in range(total_genes)]
        population.append(mask)

    best_ever = None
    best_fitness = 0

    for gen in range(generations):
        fitnesses = []
        for ind in population:
            bull_mask = ind[:n_bull]
            bear_mask = ind[n_bull:]
            fitness = evaluate_keyword_mask(bull_kw, bear_kw, bull_mask, bear_mask, training_data)
            # 加入稀疏性懲罰（太少關鍵字效果不好）
            kept = sum(ind) / total_genes
            if kept < 0.3:
                fitness *= 0.8
            fitnesses.append(fitness)

        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_ever = population[gen_best_idx][:]
            print(f"  🏆 第 {gen+1} 代: {best_fitness:.1%}")

        if (gen + 1) % 10 == 0:
            print(f"  第 {gen+1} 代: 最佳 {gen_best_fitness:.1%}, 平均 {np.mean(fitnesses):.1%}")

        # 下一代
        new_pop = []

        # 菁英
        elite_count = max(2, pop_size // 10)
        elite_idx = np.argsort(fitnesses)[-elite_count:]
        for i in elite_idx:
            new_pop.append(population[i][:])

        while len(new_pop) < pop_size:
            # 錦標賽選擇
            p1_idx = max(random.sample(range(pop_size), 3), key=lambda i: fitnesses[i])
            p2_idx = max(random.sample(range(pop_size), 3), key=lambda i: fitnesses[i])

            # 均勻交叉
            child = [
                population[p1_idx][i] if random.random() < 0.5 else population[p2_idx][i]
                for i in range(total_genes)
            ]

            # 突變
            for i in range(total_genes):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_pop.append(child)

        population = new_pop

    # 解析結果
    bull_mask = best_ever[:n_bull]
    bear_mask = best_ever[n_bull:]

    selected_bull = [kw for kw, m in zip(bull_kw, bull_mask) if m]
    removed_bull = [kw for kw, m in zip(bull_kw, bull_mask) if not m]
    selected_bear = [kw for kw, m in zip(bear_kw, bear_mask) if m]
    removed_bear = [kw for kw, m in zip(bear_kw, bear_mask) if not m]

    result = {
        'selected_bull': selected_bull,
        'removed_bull': removed_bull,
        'selected_bear': selected_bear,
        'removed_bear': removed_bear,
        'accuracy': best_fitness,
    }

    print(f"\n📊 結果:")
    print(f"   保留看漲: {len(selected_bull)}/{n_bull}")
    print(f"   保留看跌: {len(selected_bear)}/{n_bear}")
    print(f"   準確率: {best_fitness:.1%}")

    if removed_bull:
        print(f"   移除看漲: {', '.join(removed_bull)}")
    if removed_bear:
        print(f"   移除看跌: {', '.join(removed_bear)}")

    return result


def save_optimized_keywords(result):
    """儲存 GA 優化後的關鍵字"""
    if result is None:
        return

    import datetime
    data = {
        'bull_keywords': result['selected_bull'],
        'bear_keywords': result['selected_bear'],
        'removed_bull': result['removed_bull'],
        'removed_bear': result['removed_bear'],
        'accuracy': result['accuracy'],
        'updated': datetime.datetime.now().isoformat(),
    }

    with open(OPTIMIZED_KEYWORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已儲存到 {OPTIMIZED_KEYWORDS_FILE}")


def load_optimized_keywords():
    """
    載入 GA 優化後的關鍵字

    Returns:
        (bull_keywords, bear_keywords) 或 None（若檔案不存在）
    """
    if not os.path.exists(OPTIMIZED_KEYWORDS_FILE):
        return None

    try:
        with open(OPTIMIZED_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('bull_keywords'), data.get('bear_keywords')
    except Exception:
        return None


if __name__ == "__main__":
    result = optimize(pop_size=80, generations=40, mutation_rate=0.1)
    if result:
        save_optimized_keywords(result)

        # 發送 Discord 通知
        try:
            from notifier import send_discord_embed, COLOR_INFO

            embed = {
                "title": "🧬 關鍵字 GA 優化完成",
                "color": COLOR_INFO,
                "fields": [
                    {
                        "name": "準確率",
                        "value": f"**{result['accuracy']:.1%}**",
                        "inline": True,
                    },
                    {
                        "name": "保留",
                        "value": f"看漲 {len(result['selected_bull'])} / 看跌 {len(result['selected_bear'])}",
                        "inline": True,
                    },
                ],
            }

            if result['removed_bull']:
                embed['fields'].append({
                    "name": "移除看漲關鍵字",
                    "value": ', '.join(result['removed_bull'][:10]),
                    "inline": False,
                })
            if result['removed_bear']:
                embed['fields'].append({
                    "name": "移除看跌關鍵字",
                    "value": ', '.join(result['removed_bear'][:10]),
                    "inline": False,
                })

            send_discord_embed(embed)
            print("已發送 Discord 通知")
        except Exception as e:
            print(f"Discord 通知失敗: {e}")
