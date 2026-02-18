#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日內粒子模型回測工具

用歷史 intraday 資料驗證模型效果，
產生 per-stock 和 overall 指標。

用法:
    python intraday_backtest.py --date 20220107
    python intraday_backtest.py --date 20220107 --stock 2330
    python intraday_backtest.py --days 5 --csv results.csv

@author: rubylintu
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from intraday_particle_model import IntradayParticleModel, IntradayParticleParams


def backtest_single_day(date_str, stock_filter=None, params=None):
    """
    回測單日所有股票

    Returns:
        list of dicts (per-stock metrics)
    """
    from merge_intraday import parse_intraday_file

    fpath = os.path.join(SCRIPT_DIR, 'intraday', f'{date_str}.txt')
    if not os.path.exists(fpath):
        print(f"找不到 {fpath}")
        return []

    df = parse_intraday_file(fpath)
    if df.empty:
        return []

    stock_codes = sorted(df['code'].unique())
    if stock_filter:
        stock_codes = [c for c in stock_codes if c == stock_filter]

    if params is None:
        params = IntradayParticleParams.load()

    model = IntradayParticleModel(params)
    results = []

    for code in stock_codes:
        stock_df = df[df['code'] == code].copy()
        if len(stock_df) < 30:
            continue

        name = stock_df['name'].iloc[0]
        preds, actuals, metrics = model.backtest_day(stock_df)

        metrics['date'] = date_str
        metrics['code'] = code
        metrics['name'] = name
        metrics['n_ticks'] = len(stock_df)
        results.append(metrics)

    return results


def backtest_multi_day(days=None, date_list=None, stock_filter=None, params=None):
    """
    回測多日

    Returns:
        list of dicts (per-stock-per-day metrics)
    """
    intraday_dir = os.path.join(SCRIPT_DIR, 'intraday')
    all_files = sorted([
        f for f in os.listdir(intraday_dir)
        if f.endswith('.txt') and len(f) == 12
    ])

    if date_list:
        all_files = [f for f in all_files if f.replace('.txt', '') in date_list]
    elif days:
        all_files = all_files[-days:]

    all_results = []
    for fname in all_files:
        date_str = fname.replace('.txt', '')
        results = backtest_single_day(date_str, stock_filter, params)
        all_results.extend(results)

    return all_results


def print_results(results):
    """印出回測結果摘要"""
    if not results:
        print("無回測結果")
        return

    print(f"\n{'='*70}")
    print(f"{'股票':>10} {'日期':>10} {'MAE%':>8} {'方向':>6} {'相關':>8} {'最大誤差':>10} {'ticks':>6}")
    print(f"{'='*70}")

    for r in results:
        print(f"  {r['name']}({r['code']}) {r['date']} "
              f"{r['mae_pct']:7.3f}% {r['direction_accuracy']:5.1%} "
              f"{r['correlation']:7.4f} {r['max_error']:9.2f} {r['n_ticks']:5d}")

    # 總結
    avg_mae = np.mean([r['mae_pct'] for r in results])
    avg_dir = np.mean([r['direction_accuracy'] for r in results])
    avg_corr = np.mean([r['correlation'] for r in results])
    total_ticks = sum(r['n_ticks'] for r in results)

    print(f"{'='*70}")
    print(f"  平均: MAE={avg_mae:.3f}%, 方向={avg_dir:.1%}, "
          f"相關={avg_corr:.4f}, 共 {total_ticks} ticks, {len(results)} 筆")


def export_csv(results, output_path):
    """匯出 CSV"""
    df = pd.DataFrame(results)
    cols = ['date', 'code', 'name', 'mae', 'mae_pct',
            'direction_accuracy', 'correlation', 'max_error',
            'n_ticks', 'n_samples']
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已匯出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='日內粒子模型回測')
    parser.add_argument('--date', type=str, default=None,
                        help='單日回測 YYYYMMDD')
    parser.add_argument('--days', type=int, default=None,
                        help='回測最近 N 天')
    parser.add_argument('--stock', type=str, default=None,
                        help='只回測特定股票')
    parser.add_argument('--csv', type=str, default=None,
                        help='匯出 CSV 路徑')
    parser.add_argument('--params', type=str, default=None,
                        help='參數 JSON 檔路徑')
    args = parser.parse_args()

    # 載入參數
    if args.params:
        params = IntradayParticleParams.from_json(args.params)
    else:
        params = IntradayParticleParams.load()
    print(f"模型參數: {params.to_dict()}")

    # 回測
    if args.date:
        results = backtest_single_day(args.date, args.stock, params)
    elif args.days:
        results = backtest_multi_day(days=args.days,
                                     stock_filter=args.stock, params=params)
    else:
        # 預設：所有可用資料
        results = backtest_multi_day(stock_filter=args.stock, params=params)

    print_results(results)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == '__main__':
    main()
