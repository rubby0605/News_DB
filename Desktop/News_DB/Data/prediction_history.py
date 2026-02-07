#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預測歷史追蹤與自動修正模組

記錄每日預測與實際結果，計算系統偏差修正因子。

@author: rubylintu
"""

import os
import json
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(SCRIPT_DIR, 'prediction_history.json')


def load_history():
    """載入預測歷史"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'predictions': []}


def save_history(history):
    """儲存預測歷史"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_prediction(stock_code, direction, confidence, bias):
    """
    記錄盤前預測

    Args:
        stock_code: 股票代號
        direction: 預測方向 ('漲'/'跌'/'盤整'/'觀望')
        confidence: 信心度 (0-1)
        bias: 偏移量
    """
    history = load_history()
    today = datetime.date.today().isoformat()

    # 避免重複記錄
    for p in history['predictions']:
        if p['stock_code'] == stock_code and p['date'] == today:
            p['direction'] = direction
            p['confidence'] = confidence
            p['bias'] = bias
            save_history(history)
            return

    history['predictions'].append({
        'date': today,
        'stock_code': stock_code,
        'direction': direction,
        'confidence': confidence,
        'bias': bias,
        'actual_direction': None,
        'actual_change': None,
    })

    # 只保留近 30 天
    cutoff = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    history['predictions'] = [p for p in history['predictions'] if p['date'] >= cutoff]

    save_history(history)


def record_outcome(stock_code, actual_direction, actual_change):
    """
    記錄盤後實際結果

    Args:
        stock_code: 股票代號
        actual_direction: 實際方向 ('漲'/'跌'/'盤整')
        actual_change: 實際漲跌幅 (%)
    """
    history = load_history()
    today = datetime.date.today().isoformat()

    for pred in reversed(history['predictions']):
        if pred['stock_code'] == stock_code and pred['date'] == today:
            pred['actual_direction'] = actual_direction
            pred['actual_change'] = actual_change
            break

    save_history(history)


def calc_correction_factor():
    """
    計算系統偏差修正因子

    根據近期預測準確率，自動調整 bias 方向的強度。

    Returns:
        dict: {
            'bullish_factor': float (0.7-1.0),
            'bearish_factor': float (0.7-1.0),
            'bullish_accuracy': float,
            'bearish_accuracy': float,
            'sample_size': int,
        }
    """
    history = load_history()

    # 過濾有結果的預測（最近 50 筆）
    recent = [p for p in history['predictions']
              if p['actual_direction'] is not None
              and p['direction'] in ('漲', '跌')][-50:]

    if len(recent) < 10:
        return {'bullish_factor': 1.0, 'bearish_factor': 1.0, 'sample_size': len(recent)}

    bullish_preds = [p for p in recent if p['direction'] == '漲']
    bearish_preds = [p for p in recent if p['direction'] == '跌']

    bullish_correct = sum(
        1 for p in bullish_preds
        if p['actual_direction'] == '漲' or
        (p['actual_change'] is not None and p['actual_change'] > 0)
    )
    bearish_correct = sum(
        1 for p in bearish_preds
        if p['actual_direction'] == '跌' or
        (p['actual_change'] is not None and p['actual_change'] < 0)
    )

    bullish_accuracy = bullish_correct / len(bullish_preds) if bullish_preds else 0.5
    bearish_accuracy = bearish_correct / len(bearish_preds) if bearish_preds else 0.5

    # 準確率低於 30% → 衰減 30%，低於 40% → 衰減 15%
    if bullish_accuracy < 0.30:
        bullish_factor = 0.7
    elif bullish_accuracy < 0.40:
        bullish_factor = 0.85
    else:
        bullish_factor = 1.0

    if bearish_accuracy < 0.30:
        bearish_factor = 0.7
    elif bearish_accuracy < 0.40:
        bearish_factor = 0.85
    else:
        bearish_factor = 1.0

    return {
        'bullish_factor': bullish_factor,
        'bearish_factor': bearish_factor,
        'bullish_accuracy': bullish_accuracy,
        'bearish_accuracy': bearish_accuracy,
        'sample_size': len(recent),
    }


def get_tracking_metrics(stock_code=None):
    """
    取得追蹤指標（供 embed 使用）

    Args:
        stock_code: 篩選特定股票（None = 全部）

    Returns:
        dict: {
            'today_predictions': int,
            'today_correct': int,
            'today_hit_rate': float,
            'recent_20_hit_rate': float,
            'max_consecutive_loss': int,
            'current_streak': int,  # 正=連對, 負=連錯
        }
    """
    history = load_history()
    today = datetime.date.today().isoformat()

    preds = history['predictions']
    if stock_code:
        preds = [p for p in preds if p['stock_code'] == stock_code]

    # 今日指標
    today_preds = [p for p in preds if p['date'] == today and p['direction'] in ('漲', '跌')]
    today_with_outcome = [p for p in today_preds if p['actual_direction'] is not None]
    today_correct = sum(
        1 for p in today_with_outcome
        if p['direction'] == p['actual_direction'] or
        (p['direction'] == '漲' and p.get('actual_change', 0) and p['actual_change'] > 0) or
        (p['direction'] == '跌' and p.get('actual_change', 0) and p['actual_change'] < 0)
    )

    # 近 20 筆有結果的預測
    recent_with_outcome = [
        p for p in preds
        if p['actual_direction'] is not None and p['direction'] in ('漲', '跌')
    ][-20:]

    recent_correct = sum(
        1 for p in recent_with_outcome
        if p['direction'] == p['actual_direction'] or
        (p['direction'] == '漲' and p.get('actual_change', 0) and p['actual_change'] > 0) or
        (p['direction'] == '跌' and p.get('actual_change', 0) and p['actual_change'] < 0)
    )

    # 連勝/連敗 + 最大連錯
    max_loss = 0
    current_loss = 0
    current_streak = 0  # 正=連對, 負=連錯

    for p in recent_with_outcome:
        correct = (
            p['direction'] == p['actual_direction'] or
            (p['direction'] == '漲' and p.get('actual_change', 0) and p['actual_change'] > 0) or
            (p['direction'] == '跌' and p.get('actual_change', 0) and p['actual_change'] < 0)
        )
        if correct:
            current_loss = 0
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
        else:
            current_loss += 1
            max_loss = max(max_loss, current_loss)
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1

    return {
        'today_predictions': len(today_preds),
        'today_correct': today_correct,
        'today_hit_rate': today_correct / len(today_with_outcome) if today_with_outcome else 0,
        'recent_20_hit_rate': recent_correct / len(recent_with_outcome) if recent_with_outcome else 0,
        'max_consecutive_loss': max_loss,
        'current_streak': current_streak,
    }


def calc_advanced_metrics(days=20):
    """
    計算進階績效指標

    Args:
        days: 回看天數

    Returns:
        dict: {
            'coverage': float,       # 出手率 = 有信心預測 / 全部機會
            'precision': float,      # 出手準度 = 出手正確 / 出手次數
            'overall_accuracy': float,
            'max_drawdown': int,     # 最大連續錯誤
            'current_streak': int,
            'by_direction': {
                '漲': {'count', 'correct', 'accuracy'},
                '跌': {'count', 'correct', 'accuracy'},
            }
        }
    """
    history = load_history()
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    recent = [p for p in history['predictions'] if p['date'] >= cutoff]

    # 出手率: 有信心預測（漲/跌） / 全部（含盤整/觀望）
    total_opportunities = len(recent)
    confident_preds = [p for p in recent if p['direction'] in ('漲', '跌')]
    coverage = len(confident_preds) / total_opportunities if total_opportunities else 0

    # 有結果的出手
    with_outcome = [p for p in confident_preds if p['actual_direction'] is not None]

    def _is_correct(p):
        return (
            p['direction'] == p['actual_direction'] or
            (p['direction'] == '漲' and p.get('actual_change', 0) and p['actual_change'] > 0) or
            (p['direction'] == '跌' and p.get('actual_change', 0) and p['actual_change'] < 0)
        )

    correct_count = sum(1 for p in with_outcome if _is_correct(p))
    precision = correct_count / len(with_outcome) if with_outcome else 0

    # 全部有結果的預測（含觀望、盤整）
    all_with_outcome = [p for p in recent if p['actual_direction'] is not None]
    all_correct = sum(
        1 for p in all_with_outcome
        if p['direction'] == p['actual_direction'] or
        (p['direction'] == '漲' and p.get('actual_change', 0) and p['actual_change'] > 0) or
        (p['direction'] == '跌' and p.get('actual_change', 0) and p['actual_change'] < 0)
    )
    overall_accuracy = all_correct / len(all_with_outcome) if all_with_outcome else 0

    # 最大連錯 + 目前連勝/連敗
    max_dd = 0
    current_loss = 0
    streak = 0

    for p in with_outcome:
        if _is_correct(p):
            current_loss = 0
            streak = streak + 1 if streak >= 0 else 1
        else:
            current_loss += 1
            max_dd = max(max_dd, current_loss)
            streak = streak - 1 if streak <= 0 else -1

    # 方向分佈
    by_direction = {}
    for d in ('漲', '跌'):
        d_preds = [p for p in with_outcome if p['direction'] == d]
        d_correct = sum(1 for p in d_preds if _is_correct(p))
        by_direction[d] = {
            'count': len(d_preds),
            'correct': d_correct,
            'accuracy': d_correct / len(d_preds) if d_preds else 0,
        }

    return {
        'coverage': coverage,
        'precision': precision,
        'overall_accuracy': overall_accuracy,
        'max_drawdown': max_dd,
        'current_streak': streak,
        'by_direction': by_direction,
    }


if __name__ == "__main__":
    print("預測歷史追蹤模組")
    history = load_history()
    total = len(history['predictions'])
    with_outcome = sum(1 for p in history['predictions'] if p['actual_direction'] is not None)
    print(f"總預測數: {total}, 已有結果: {with_outcome}")

    correction = calc_correction_factor()
    print(f"修正因子: {correction}")

    metrics = get_tracking_metrics()
    print(f"追蹤指標: {metrics}")

    advanced = calc_advanced_metrics()
    print(f"進階指標: {advanced}")
