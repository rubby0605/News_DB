#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
廣播紀錄模組

每次推播都存完整快照到 broadcast_log.jsonl，
盤後可回填實際結果，用於回測與績效追蹤。

@author: rubylintu
"""

import os
import json
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def _log_file(date=None):
    """取得當日 broadcast log 檔案路徑"""
    if date is None:
        date = datetime.date.today().isoformat()
    return os.path.join(LOG_DIR, f'broadcast_{date}.jsonl')


def log_broadcast(stock_code, prediction, news_titles=None,
                  signals=None, bias=None, warnings=None):
    """
    記錄一次推播快照

    Args:
        stock_code: 股票代號
        prediction: dict from DirectionalParticleModel.predict()
        news_titles: list of str（相關新聞標題）
        signals: dict（信號分解）
        bias: float（偏移量）
        warnings: list of str（風險警示）
    """
    now = datetime.datetime.now()
    record = {
        'timestamp': now.isoformat(),
        'stock_code': stock_code,
        'stock_name': prediction.get('stock_name', ''),
        'direction': prediction.get('direction', ''),
        'confidence': prediction.get('confidence', 0),
        'current_price': prediction.get('current_price', 0),
        'predicted_price': prediction.get('predicted_price', 0),
        'expected_change': prediction.get('expected_change', 0),
        'bias': bias or prediction.get('bias', 0),
        'signals': signals or prediction.get('signals', {}),
        'news_titles': news_titles or [],
        'warnings': warnings or [],
        'actual_direction': None,
        'actual_close': None,
        'actual_change': None,
        'correct': None,
    }

    filepath = _log_file()
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def update_outcomes(date, outcomes):
    """
    盤後填入實際結果

    Args:
        date: str 'YYYY-MM-DD'
        outcomes: dict {stock_code: {
            'actual_direction': '漲'/'跌'/'盤整',
            'actual_close': float,
            'actual_change': float (%)
        }}
    """
    filepath = _log_file(date)
    if not os.path.exists(filepath):
        return

    updated_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                updated_lines.append(line)
                continue

            code = record.get('stock_code', '')
            if code in outcomes:
                o = outcomes[code]
                record['actual_direction'] = o.get('actual_direction')
                record['actual_close'] = o.get('actual_close')
                record['actual_change'] = o.get('actual_change')
                # 判斷正確性
                pred_dir = record.get('direction', '')
                actual_dir = o.get('actual_direction', '')
                if pred_dir in ('漲', '跌') and actual_dir:
                    record['correct'] = (pred_dir == actual_dir)

            updated_lines.append(json.dumps(record, ensure_ascii=False))

    with open(filepath, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')


def generate_daily_report(date=None):
    """
    產生當日精準度/覆蓋率報告

    Args:
        date: str 'YYYY-MM-DD'（預設今天）

    Returns:
        dict: {
            'date': str,
            'total_broadcasts': int,
            'with_outcome': int,
            'correct': int,
            'incorrect': int,
            'accuracy': float,
            'by_stock': dict
        }
    """
    if date is None:
        date = datetime.date.today().isoformat()

    filepath = _log_file(date)
    if not os.path.exists(filepath):
        return {'date': date, 'total_broadcasts': 0}

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(records)
    with_outcome = [r for r in records if r.get('correct') is not None]
    correct = sum(1 for r in with_outcome if r['correct'])
    incorrect = len(with_outcome) - correct

    # 每檔統計（取最後一次推播）
    by_stock = {}
    for r in records:
        code = r.get('stock_code', '')
        by_stock[code] = {
            'direction': r.get('direction'),
            'confidence': r.get('confidence'),
            'actual_direction': r.get('actual_direction'),
            'correct': r.get('correct'),
        }

    return {
        'date': date,
        'total_broadcasts': total,
        'with_outcome': len(with_outcome),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': correct / len(with_outcome) if with_outcome else 0,
        'by_stock': by_stock,
    }


if __name__ == "__main__":
    print("廣播紀錄模組")
    report = generate_daily_report()
    print(f"日期: {report['date']}")
    print(f"總推播數: {report['total_broadcasts']}")
    if report.get('with_outcome'):
        print(f"已有結果: {report['with_outcome']}")
        print(f"正確: {report['correct']}, 錯誤: {report['incorrect']}")
        print(f"準確率: {report['accuracy']:.0%}")
