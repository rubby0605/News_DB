#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II 多目標基因演算法優化粒子模型參數

同時最小化：
  1. 方向錯誤率 (1 - accuracy)
  2. 漲跌幅 MAE (mean absolute error)

@author: rubylintu
"""

import os
import sys
import datetime
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from optimize_weights import (
    prepare_test_data,
    calc_bias_with_weights,
    save_weights,
)
from notifier import send_discord

# ── 參數範圍 ──────────────────────────────────────────────
PARAM_NAMES = [
    'foreign_large',
    'foreign_medium',
    'foreign_weight',
    'momentum_weight',
    'ema_weight',
    'momentum_threshold',
]

PARAM_RANGES = {
    'foreign_large':      (1000, 8000),
    'foreign_medium':     (200, 2000),
    'foreign_weight':     (1, 8),
    'momentum_weight':    (0.5, 5),
    'ema_weight':         (0.5, 4),
    'momentum_threshold': (1, 8),
}

# ── 雙目標評估 ────────────────────────────────────────────
def evaluate_objectives(weights, test_data):
    """
    回傳 (direction_error_rate, mae)，兩個都要 minimize。
    """
    correct = 0
    total = 0
    abs_errors = []

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

            # 預測方向
            if bias > 2:
                pred_dir = 1   # 漲
            elif bias < -2:
                pred_dir = -1  # 跌
            else:
                pred_dir = 0   # 盤整

            # 實際方向
            actual_change = prices[i + 1]['change']
            if actual_change > 0.5:
                actual_dir = 1
            elif actual_change < -0.5:
                actual_dir = -1
            else:
                actual_dir = 0

            # 方向是否正確
            if (pred_dir == actual_dir
                    or (pred_dir == 1 and actual_change > 0)
                    or (pred_dir == -1 and actual_change < 0)):
                correct += 1
            total += 1

            # bias → expected_change%（簡單線性映射）
            expected_change = bias * 0.5
            abs_errors.append(abs(expected_change - actual_change))

    if total == 0:
        return (1.0, 10.0)

    direction_error = 1 - correct / total
    mae = float(np.mean(abs_errors))
    return (direction_error, mae)


# ── 個體操作 ──────────────────────────────────────────────
def to_array(ind):
    """dict → numpy array（按 PARAM_NAMES 順序）"""
    return np.array([ind[k] for k in PARAM_NAMES])


def to_dict(arr):
    """numpy array → dict"""
    return {k: float(v) for k, v in zip(PARAM_NAMES, arr)}


def to_weights(ind):
    """dict → 可餵給 calc_bias_with_weights 的 weights dict"""
    return {
        'foreign_large':      int(ind['foreign_large']),
        'foreign_medium':     int(ind['foreign_medium']),
        'foreign_weight':     ind['foreign_weight'],
        'momentum_weight':    ind['momentum_weight'],
        'ema_weight':         ind['ema_weight'],
        'momentum_threshold': ind['momentum_threshold'],
    }


def random_individual():
    return {k: np.random.uniform(*PARAM_RANGES[k]) for k in PARAM_NAMES}


def clip_individual(ind):
    """把個體限制在合法範圍內"""
    return {k: np.clip(ind[k], *PARAM_RANGES[k]) for k in PARAM_NAMES}


# ── SBX 交叉 ─────────────────────────────────────────────
def sbx_crossover(p1, p2, eta=20, prob=0.9):
    """Simulated Binary Crossover"""
    if np.random.random() > prob:
        return p1.copy(), p2.copy()

    a1 = to_array(p1)
    a2 = to_array(p2)
    c1, c2 = a1.copy(), a2.copy()

    for i in range(len(PARAM_NAMES)):
        if np.random.random() > 0.5:
            continue
        if abs(a1[i] - a2[i]) < 1e-14:
            continue

        lo, hi = PARAM_RANGES[PARAM_NAMES[i]]
        x1, x2 = min(a1[i], a2[i]), max(a1[i], a2[i])

        rand = np.random.random()
        beta = 1.0 + (2.0 * (x1 - lo) / (x2 - x1 + 1e-14))
        alpha = 2.0 - beta ** (-(eta + 1))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        c1[i] = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (hi - x2) / (x2 - x1 + 1e-14))
        alpha = 2.0 - beta ** (-(eta + 1))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        c2[i] = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1[i] = np.clip(c1[i], lo, hi)
        c2[i] = np.clip(c2[i], lo, hi)

    return clip_individual(to_dict(c1)), clip_individual(to_dict(c2))


# ── 多項式突變 ────────────────────────────────────────────
def polynomial_mutation(ind, eta=20, prob=None):
    """Polynomial Mutation"""
    n = len(PARAM_NAMES)
    if prob is None:
        prob = 1.0 / n

    arr = to_array(ind)
    for i in range(n):
        if np.random.random() > prob:
            continue
        lo, hi = PARAM_RANGES[PARAM_NAMES[i]]
        delta1 = (arr[i] - lo) / (hi - lo + 1e-14)
        delta2 = (hi - arr[i]) / (hi - lo + 1e-14)
        rand = np.random.random()
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
            deltaq = val ** (1.0 / (eta + 1)) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
            deltaq = 1.0 - val ** (1.0 / (eta + 1))
        arr[i] += deltaq * (hi - lo)
        arr[i] = np.clip(arr[i], lo, hi)

    return to_dict(arr)


# ── 非支配排序 ────────────────────────────────────────────
def non_dominated_sort(objectives):
    """
    Fast Non-Dominated Sort (Deb 2002)
    objectives: list of (obj1, obj2), 都要 minimize
    回傳 list of fronts, 每個 front 是 index list
    """
    n = len(objectives)
    domination_count = [0] * n   # 被幾個支配
    dominated_set = [[] for _ in range(n)]  # 支配哪些
    rank = [0] * n
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            # p 是否支配 q
            if (objectives[p][0] <= objectives[q][0] and
                    objectives[p][1] <= objectives[q][1] and
                    (objectives[p][0] < objectives[q][0] or
                     objectives[p][1] < objectives[q][1])):
                dominated_set[p].append(q)
            elif (objectives[q][0] <= objectives[p][0] and
                  objectives[q][1] <= objectives[p][1] and
                  (objectives[q][0] < objectives[p][0] or
                   objectives[q][1] < objectives[p][1])):
                domination_count[p] += 1

        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # 移除最後的空 front
    if not fronts[-1]:
        fronts.pop()

    return fronts, rank


# ── 擁擠距離 ─────────────────────────────────────────────
def crowding_distance(front_indices, objectives):
    """
    計算 front 內每個個體的 crowding distance
    """
    n = len(front_indices)
    if n <= 2:
        return {idx: float('inf') for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    num_objectives = 2

    for m in range(num_objectives):
        sorted_indices = sorted(front_indices, key=lambda i: objectives[i][m])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
        if obj_range < 1e-14:
            continue

        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                (objectives[sorted_indices[i + 1]][m] - objectives[sorted_indices[i - 1]][m])
                / obj_range
            )

    return distances


# ── 錦標賽選擇 ────────────────────────────────────────────
def tournament_selection(population, ranks, crowd_dist, pool_size=2):
    """二元錦標賽：先比 rank，再比 crowding distance"""
    candidates = np.random.choice(len(population), size=pool_size, replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best]:
            best = c
        elif ranks[c] == ranks[best] and crowd_dist.get(c, 0) > crowd_dist.get(best, 0):
            best = c
    return population[best]


# ── NSGA-II 主流程 ────────────────────────────────────────
def nsga2(test_data, population_size=100, generations=50):
    """
    NSGA-II 多目標優化

    Returns:
        pareto_front: list of (weights_dict, (err_rate, mae))
        knee_point:   (weights_dict, (err_rate, mae))
    """
    print(f"\n{'='*60}")
    print("NSGA-II 多目標優化")
    print(f"{'='*60}")
    print(f"  族群大小: {population_size}")
    print(f"  迭代代數: {generations}")
    print(f"  目標 1: 方向錯誤率 (minimize)")
    print(f"  目標 2: 漲跌幅 MAE (minimize)")

    # 1. 初始化族群
    population = [random_individual() for _ in range(population_size)]

    # 評估初始族群
    objectives = []
    for ind in population:
        obj = evaluate_objectives(to_weights(ind), test_data)
        objectives.append(obj)

    for gen in range(generations):
        # 2. 產生子代
        offspring = []
        # 非支配排序 + 擁擠距離（用於選擇）
        fronts, ranks = non_dominated_sort(objectives)
        crowd_dist = {}
        for front in fronts:
            cd = crowding_distance(front, objectives)
            crowd_dist.update(cd)

        while len(offspring) < population_size:
            p1 = tournament_selection(population, ranks, crowd_dist)
            p2 = tournament_selection(population, ranks, crowd_dist)
            c1, c2 = sbx_crossover(p1, p2)
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)

        # 評估子代
        offspring_objectives = []
        for ind in offspring:
            obj = evaluate_objectives(to_weights(ind), test_data)
            offspring_objectives.append(obj)

        # 3. 合併父代 + 子代
        combined = population + offspring
        combined_obj = objectives + offspring_objectives

        # 4. 非支配排序
        fronts, _ = non_dominated_sort(combined_obj)

        # 5. 選出下一代
        new_population = []
        new_objectives = []

        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                for idx in front:
                    new_population.append(combined[idx])
                    new_objectives.append(combined_obj[idx])
            else:
                # 需要截斷：用 crowding distance 排序
                cd = crowding_distance(front, combined_obj)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                remaining = population_size - len(new_population)
                for idx in sorted_front[:remaining]:
                    new_population.append(combined[idx])
                    new_objectives.append(combined_obj[idx])
                break

        population = new_population
        objectives = new_objectives

        # 印出進度
        best_err = min(o[0] for o in objectives)
        best_mae = min(o[1] for o in objectives)
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  第 {gen+1:3d} 代 | 最佳方向準確率: {1 - best_err:.1%} | 最佳 MAE: {best_mae:.3f}")

    # ── 取得最終 Pareto front ──
    fronts, _ = non_dominated_sort(objectives)
    pareto_indices = fronts[0]

    pareto_front = []
    for idx in pareto_indices:
        w = to_weights(population[idx])
        pareto_front.append((w, objectives[idx]))

    # 按方向準確率排序
    pareto_front.sort(key=lambda x: x[1][0])

    return pareto_front


# ── Knee point 選擇 ──────────────────────────────────────
def select_knee_point(pareto_front):
    """
    在 Pareto front 上選 knee point：
    歸一化兩個目標後，找離「理想點(0,0)」距離最小的解。
    """
    if len(pareto_front) == 1:
        return pareto_front[0]

    errs = [pf[1][0] for pf in pareto_front]
    maes = [pf[1][1] for pf in pareto_front]

    err_min, err_max = min(errs), max(errs)
    mae_min, mae_max = min(maes), max(maes)

    err_range = err_max - err_min if err_max > err_min else 1.0
    mae_range = mae_max - mae_min if mae_max > mae_min else 1.0

    best_dist = float('inf')
    best_pf = pareto_front[0]

    for pf in pareto_front:
        norm_err = (pf[1][0] - err_min) / err_range
        norm_mae = (pf[1][1] - mae_min) / mae_range
        dist = np.sqrt(norm_err ** 2 + norm_mae ** 2)
        if dist < best_dist:
            best_dist = dist
            best_pf = pf

    return best_pf


# ── 主程式 ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NSGA-II 粒子模型多目標優化")
    print("=" * 60)

    # 測試股票
    test_stocks = ['2330', '2454', '2344', '3481', '2313']
    stock_names = {
        '2330': '台積電',
        '2454': '聯發科',
        '2344': '華邦電子',
        '3481': '群創',
        '2313': '華通',
    }

    # 計算近 2 個月
    today = datetime.date.today()
    months = []
    for i in range(2):
        target_month = today.month - i - 1
        target_year = today.year
        if target_month <= 0:
            target_month += 12
            target_year -= 1
        months.append(f'{target_year}{target_month:02d}')

    print(f"測試股票: {[f'{c}({stock_names[c]})' for c in test_stocks]}")
    print(f"測試月份: {months}")

    # 準備資料
    test_data = prepare_test_data(test_stocks, months)
    if not test_data:
        print("無測試資料，結束")
        return

    # ── NSGA-II 優化 ──
    start = time.time()
    pareto_front = nsga2(test_data, population_size=100, generations=50)
    elapsed = time.time() - start

    # ── 印出 Pareto front ──
    print(f"\n{'='*60}")
    print(f"Pareto Front ({len(pareto_front)} 個解)  |  耗時 {elapsed:.1f} 秒")
    print(f"{'='*60}")
    print(f"{'#':>3}  {'準確率':>8}  {'MAE':>8}  {'外資大量':>8}  {'外資中量':>8}  "
          f"{'外資權重':>8}  {'動量權重':>8}  {'均線權重':>8}  {'動量門檻':>8}")
    print("-" * 90)

    for i, (w, (err, mae)) in enumerate(pareto_front):
        print(f"{i+1:>3}  {1-err:>7.1%}  {mae:>8.3f}  "
              f"{w['foreign_large']:>8}  {w['foreign_medium']:>8}  "
              f"{w['foreign_weight']:>8.2f}  {w['momentum_weight']:>8.2f}  "
              f"{w['ema_weight']:>8.2f}  {w['momentum_threshold']:>8.2f}")

    # ── 選 Knee Point ──
    knee_weights, (knee_err, knee_mae) = select_knee_point(pareto_front)
    knee_acc = 1 - knee_err

    print(f"\n{'='*60}")
    print(f"Knee Point（最佳平衡解）")
    print(f"{'='*60}")
    print(f"  方向準確率: {knee_acc:.1%}")
    print(f"  漲跌幅 MAE: {knee_mae:.3f}")
    print(f"  權重:")
    for k, v in knee_weights.items():
        print(f"    {k}: {v}")

    # ── 儲存最佳權重 ──
    save_weights(knee_weights, knee_acc)

    # ── Discord 通知（預設不自動發送，加 --discord 才發）──
    if '--discord' in sys.argv:
        pf_lines = []
        for i, (w, (err, mae)) in enumerate(pareto_front):
            pf_lines.append(f"  {i+1}. 準確率 {1-err:.1%} | MAE {mae:.3f}")

        message = f"""**NSGA-II 多目標優化完成**

**Pareto Front ({len(pareto_front)} 個解):**
{chr(10).join(pf_lines)}

**Knee Point（最佳平衡解）:**
- 方向準確率: {knee_acc:.1%}
- 漲跌幅 MAE: {knee_mae:.3f}

**最佳權重:**
- 外資大量門檻: {knee_weights['foreign_large']} 張
- 外資中量門檻: {knee_weights['foreign_medium']} 張
- 外資權重: {knee_weights['foreign_weight']:.2f}
- 動量權重: {knee_weights['momentum_weight']:.2f}
- 均線權重: {knee_weights['ema_weight']:.2f}
- 動量門檻: {knee_weights['momentum_threshold']:.2f}%

測試股票: {', '.join(f'{c}({stock_names[c]})' for c in test_stocks)}
測試月份: {months[0]} ~ {months[-1]}
耗時: {elapsed:.1f} 秒"""

        send_discord(message, title='NSGA-II 優化結果')
        print("\n已發送到 Discord!")
    else:
        print("\n(加 --discord 參數才會發送到 Discord)")


if __name__ == "__main__":
    main()
