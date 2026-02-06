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


if __name__ == "__main__":
    print("預測歷史追蹤模組")
    history = load_history()
    total = len(history['predictions'])
    with_outcome = sum(1 for p in history['predictions'] if p['actual_direction'] is not None)
    print(f"總預測數: {total}, 已有結果: {with_outcome}")

    correction = calc_correction_factor()
    print(f"修正因子: {correction}")
