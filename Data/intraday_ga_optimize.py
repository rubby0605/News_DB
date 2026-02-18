#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日內粒子模型 GA 優化

用歷史 intraday 資料優化 7 個模型參數，
架構複用 optimize_weights.py（輪盤選擇、菁英保留、交叉、突變）。

用法:
    python intraday_ga_optimize.py                   # 用最近的 intraday 資料
    python intraday_ga_optimize.py --days 5           # 只用最近 5 天
    python intraday_ga_optimize.py --pop 50 --gen 30  # 調整 GA 參數

@author: rubylintu
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from intraday_particle_model import (
    IntradayParticleModel, IntradayParticleParams,
    PARAM_RANGES, DEFAULT_PARAMS,
)
from config import (
    INTRADAY_MODEL_PARAMS_FILE, INTRADAY_PARTICLES,
    INTRADAY_GA_POP, INTRADAY_GA_GEN, INTRADAY_TRAIN_DAYS,
)


# ============================================================
# 資料準備
# ============================================================

def load_training_data(days=None, date_list=None):
    """
    從 intraday/ 目錄載入訓練資料

    Args:
        days: 最近 N 天
        date_list: 指定日期列表 ['20220107', '20220111']

    Returns:
        list of (date_str, DataFrame)
    """
    from merge_intraday import parse_intraday_file

    intraday_dir = os.path.join(SCRIPT_DIR, 'intraday')
    if not os.path.exists(intraday_dir):
        print("intraday/ 目錄不存在")
        return []

    # 找到所有 .txt 檔
    all_files = sorted([
        f for f in os.listdir(intraday_dir)
        if f.endswith('.txt') and len(f) == 12  # YYYYMMDD.txt
    ])

    if date_list:
        all_files = [f for f in all_files if f.replace('.txt', '') in date_list]
    elif days:
        all_files = all_files[-days:]

    data = []
    for fname in all_files:
        fpath = os.path.join(intraday_dir, fname)
        date_str = fname.replace('.txt', '')
        df = parse_intraday_file(fpath)
        if not df.empty:
            data.append((date_str, df))
            print(f"  載入 {date_str}: {len(df)} ticks, "
                  f"{df['code'].nunique()} 檔股票")

    return data


# ============================================================
# 適應度評估
# ============================================================

def evaluate_individual(params_dict, training_data, n_particles=None):
    """
    評估一組參數的適應度

    fitness = direction_accuracy - mae_penalty

    Args:
        params_dict: 7 個參數 dict
        training_data: list of (date_str, DataFrame)
        n_particles: 粒子數

    Returns:
        float: 適應度分數（越高越好）
    """
    if n_particles is None:
        n_particles = INTRADAY_PARTICLES

    params = IntradayParticleParams(params_dict)
    model = IntradayParticleModel(params, n_particles=n_particles)

    total_dir_correct = 0
    total_dir_count = 0
    total_mae_pct = 0
    n_stocks = 0

    for date_str, df in training_data:
        stock_codes = df['code'].unique()
        for code in stock_codes:
            stock_df = df[df['code'] == code].copy()
            if len(stock_df) < 30:  # 至少 30 個 tick
                continue

            _, _, metrics = model.backtest_day(stock_df)
            if metrics['n_samples'] == 0:
                continue

            total_dir_correct += metrics['direction_accuracy'] * metrics['n_samples']
            total_dir_count += metrics['n_samples']
            total_mae_pct += metrics['mae_pct']
            n_stocks += 1

    if total_dir_count == 0 or n_stocks == 0:
        return 0.0

    overall_dir_acc = total_dir_correct / total_dir_count
    avg_mae_pct = total_mae_pct / n_stocks

    # 適應度：方向正確率為主，MAE 為輔（penalty）
    # direction_accuracy 0~1, mae_pct 通常 0.1~2%
    fitness = overall_dir_acc - avg_mae_pct * 0.1

    return fitness


# ============================================================
# GA 核心
# ============================================================

def enforce_constraints(ind):
    """強制參數在合理範圍內"""
    for k, (low, high) in PARAM_RANGES.items():
        ind[k] = np.clip(ind[k], low, high)
    # lookback_window 和 prediction_horizon 要是整數
    ind['lookback_window'] = int(round(ind['lookback_window']))
    ind['prediction_horizon'] = int(round(ind['prediction_horizon']))
    return ind


def random_individual():
    """產生隨機個體"""
    ind = {}
    for k, (low, high) in PARAM_RANGES.items():
        ind[k] = np.random.uniform(low, high)
    return enforce_constraints(ind)


def crossover(parent1, parent2):
    """均勻交叉"""
    child = {}
    for k in PARAM_RANGES:
        if np.random.random() < 0.5:
            child[k] = parent1[k]
        else:
            child[k] = parent2[k]
    return enforce_constraints(child)


def mutate(individual, mutation_rate=0.2):
    """高斯突變"""
    mutated = individual.copy()
    for k, (low, high) in PARAM_RANGES.items():
        if np.random.random() < mutation_rate:
            delta = (high - low) * 0.3 * np.random.randn()
            mutated[k] = np.clip(mutated[k] + delta, low, high)
    return enforce_constraints(mutated)


def select_parents(population, fitnesses, num_parents):
    """輪盤選擇"""
    min_fit = min(fitnesses)
    adjusted = [f - min_fit + 0.01 for f in fitnesses]
    total = sum(adjusted)
    probs = [f / total for f in adjusted]
    indices = np.random.choice(
        len(population), size=min(num_parents, len(population)),
        p=probs, replace=False
    )
    return [population[i] for i in indices]


def calc_param_drift(old_params, new_params):
    """計算新舊參數的漂移程度"""
    if not old_params or not new_params:
        return 1.0
    drifts = []
    for k, (low, high) in PARAM_RANGES.items():
        old_v = old_params.get(k, (low + high) / 2)
        new_v = new_params.get(k, (low + high) / 2)
        normalized_drift = abs(new_v - old_v) / (high - low)
        drifts.append(normalized_drift)
    return sum(drifts) / len(drifts) if drifts else 0


def genetic_algorithm(training_data, population_size=None, generations=None,
                      mutation_rate=0.2):
    """
    GA 優化日內粒子模型參數

    Args:
        training_data: list of (date_str, DataFrame)
        population_size: 族群大小
        generations: 迭代代數
        mutation_rate: 突變率

    Returns:
        best_params: dict
        best_fitness: float
        history: list of dicts
    """
    if population_size is None:
        population_size = INTRADAY_GA_POP
    if generations is None:
        generations = INTRADAY_GA_GEN

    print(f"\n  GA 優化開始")
    print(f"  族群: {population_size}, 代數: {generations}, 突變率: {mutation_rate}")
    print(f"  訓練資料: {len(training_data)} 天")

    # 初始化族群（包含一個預設參數個體）
    population = [DEFAULT_PARAMS.copy()]
    # 如果有已存的最佳參數，也加入
    if os.path.exists(INTRADAY_MODEL_PARAMS_FILE):
        try:
            old_params = IntradayParticleParams.load().to_dict()
            population.append(old_params)
        except Exception:
            pass
    # 其餘隨機
    while len(population) < population_size:
        population.append(random_individual())

    best_ever = None
    best_ever_fitness = -999
    history = []

    for gen in range(generations):
        # 評估適應度
        fitnesses = []
        for ind in population:
            fitness = evaluate_individual(ind, training_data)
            fitnesses.append(fitness)

        # 記錄最佳
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best = population[gen_best_idx]

        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever = gen_best.copy()
            print(f"  Gen {gen+1:3d}: NEW BEST {best_ever_fitness:.4f}")

        history.append({
            'generation': gen + 1,
            'best': round(gen_best_fitness, 4),
            'avg': round(float(np.mean(fitnesses)), 4),
        })

        if (gen + 1) % 5 == 0:
            print(f"  Gen {gen+1:3d}: best={gen_best_fitness:.4f} "
                  f"avg={np.mean(fitnesses):.4f}")

        # 產生下一代
        new_population = []

        # 菁英保留 top 10%
        elite_count = max(2, population_size // 10)
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        for i in elite_indices:
            new_population.append(population[i])

        # 交叉 + 突變
        while len(new_population) < population_size:
            parents = select_parents(population, fitnesses, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    # 整理最佳結果
    best_params = enforce_constraints(best_ever.copy())
    for k in ('decay_rate', 'momentum_weight', 'mean_reversion_weight',
              'vol_sensitivity', 'particle_spread'):
        best_params[k] = round(best_params[k], 4)

    return best_params, best_ever_fitness, history


# ============================================================
# 儲存 & 報告
# ============================================================

def save_params(params, fitness, history=None):
    """儲存最佳參數到 JSON"""
    data = {
        'params': params,
        'fitness': round(fitness, 4),
        'updated': datetime.datetime.now().isoformat(),
    }
    if history:
        data['ga_history'] = history

    with open(INTRADAY_MODEL_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  參數已儲存到 {INTRADAY_MODEL_PARAMS_FILE}")


def send_optimization_report(params, fitness, old_fitness=None):
    """發送 Discord 通知"""
    try:
        from notifier import send_discord_embed
        from config import COLOR_INFO

        param_lines = []
        labels = {
            'decay_rate': '衰減率',
            'momentum_weight': '動量權重',
            'mean_reversion_weight': '均值回歸',
            'vol_sensitivity': '波動敏感度',
            'particle_spread': '粒子擴散',
            'lookback_window': '回看窗口',
            'prediction_horizon': '預測步長',
        }
        for k, label in labels.items():
            param_lines.append(f"{label}: {params[k]}")

        acc_text = f"**{fitness:.4f}**"
        if old_fitness is not None:
            delta = fitness - old_fitness
            arrow = '+' if delta > 0 else ''
            acc_text += f" (prev: {old_fitness:.4f} {arrow}{delta:.4f})"

        embed = {
            "title": "日內粒子模型 GA 優化完成",
            "color": COLOR_INFO,
            "fields": [
                {"name": "適應度", "value": acc_text, "inline": True},
                {"name": "參數", "value": '\n'.join(param_lines), "inline": False},
            ],
        }
        send_discord_embed(embed)
    except Exception as e:
        print(f"  Discord 通知失敗: {e}")


# ============================================================
# 每日優化入口
# ============================================================

def run_daily_optimization(days=None, max_drift=0.3, min_improvement=0.005):
    """
    每日盤後 GA 優化（供 daily_stock_job.py 呼叫）

    Args:
        days: 訓練天數
        max_drift: 最大允許漂移
        min_improvement: 最小改善門檻

    Returns:
        dict: {updated, reason, fitness, params}
    """
    if days is None:
        days = INTRADAY_TRAIN_DAYS

    print(f"\n  日內粒子模型 GA 優化（最近 {days} 天）")

    training_data = load_training_data(days=days)
    if not training_data:
        return {'updated': False, 'reason': '無訓練資料'}

    # 載入舊參數
    old_fitness = None
    old_params = None
    if os.path.exists(INTRADAY_MODEL_PARAMS_FILE):
        try:
            with open(INTRADAY_MODEL_PARAMS_FILE, 'r') as f:
                old_data = json.load(f)
            old_params = old_data.get('params', {})
            old_fitness = old_data.get('fitness')
        except Exception:
            pass

    # 跑 GA
    new_params, new_fitness, history = genetic_algorithm(training_data)

    # 穩定性檢查
    drift = calc_param_drift(old_params, new_params)
    improvement = new_fitness - old_fitness if old_fitness is not None else 1.0

    print(f"\n  新適應度: {new_fitness:.4f}")
    if old_fitness is not None:
        print(f"  舊適應度: {old_fitness:.4f} (差: {improvement:+.4f})")
    print(f"  參數漂移: {drift:.2%}")

    if old_fitness is not None:
        if drift > max_drift and improvement < min_improvement * 2:
            reason = f'漂移過大 ({drift:.1%}) 且改善不足，不更新'
            print(f"  {reason}")
            return {'updated': False, 'reason': reason,
                    'fitness': new_fitness, 'params': new_params}

        if improvement < -min_improvement:
            reason = f'適應度下降 ({improvement:+.4f})，不更新'
            print(f"  {reason}")
            return {'updated': False, 'reason': reason,
                    'fitness': new_fitness, 'params': new_params}

    # 更新
    save_params(new_params, new_fitness, history)
    send_optimization_report(new_params, new_fitness, old_fitness)

    reason = f'適應度 {new_fitness:.4f}，漂移 {drift:.1%}，已更新'
    print(f"  {reason}")

    return {'updated': True, 'reason': reason,
            'fitness': new_fitness, 'params': new_params}


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='日內粒子模型 GA 優化')
    parser.add_argument('--days', type=int, default=None,
                        help='使用最近 N 天資料')
    parser.add_argument('--dates', type=str, nargs='+', default=None,
                        help='指定日期 (e.g. 20220107 20220111)')
    parser.add_argument('--pop', type=int, default=None,
                        help='GA 族群大小')
    parser.add_argument('--gen', type=int, default=None,
                        help='GA 代數')
    args = parser.parse_args()

    print("=" * 60)
    print("日內粒子模型 GA 優化")
    print("=" * 60)

    # 載入訓練資料
    if args.dates:
        training_data = load_training_data(date_list=args.dates)
    else:
        training_data = load_training_data(days=args.days or INTRADAY_TRAIN_DAYS)

    if not training_data:
        print("無訓練資料，結束")
        return

    # 跑 GA
    pop = args.pop or INTRADAY_GA_POP
    gen = args.gen or INTRADAY_GA_GEN

    best_params, best_fitness, history = genetic_algorithm(
        training_data, population_size=pop, generations=gen
    )

    # 結果
    print(f"\n{'='*60}")
    print(f"最佳適應度: {best_fitness:.4f}")
    print(f"最佳參數:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 儲存
    save_params(best_params, best_fitness, history)


if __name__ == '__main__':
    main()
