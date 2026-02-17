#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併每日盤中即時資料 — 供粒子模型回測使用

讀取 intraday/ 目錄下的每日 YYYYMMDD.txt，
合併指定日期範圍內的所有 tick 資料，
輸出 per-stock 連續時間序列。

用法:
    python merge_intraday.py                     # 預設最近 30 天
    python merge_intraday.py --days 60           # 最近 60 天
    python merge_intraday.py --start 20260101 --end 20260215
    python merge_intraday.py --stock 2330        # 只輸出單一股票
    python merge_intraday.py --output merged.csv # 指定輸出檔

@author: rubylintu
"""

import os
import sys
import argparse
import datetime
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(SCRIPT_DIR, 'intraday')

COLUMNS = ['code', 'name', 'price', 'trade_vol', 'cum_vol',
           'open', 'high', 'low', 'yesterday', 'timestamp']


def parse_intraday_file(filepath):
    """
    解析單日 intraday txt 檔，回傳 DataFrame

    格式: tab-separated, 第一行為 header (新格式) 或無 header (舊格式)
    舊格式欄位: code, name, price, trade_vol, cum_vol, open, high, low, yesterday, timestamp
    """
    basename = os.path.basename(filepath)
    date_str = basename.replace('.txt', '')

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return pd.DataFrame(columns=COLUMNS)

    # 判斷是否有 header
    first_line = lines[0].strip()
    start_idx = 1 if first_line.startswith('code') else 0

    for line in lines[start_idx:]:
        parts = line.strip().split('\t')
        # 過濾空行和格式不對的行
        # 舊格式開頭有空 tab，所以第一個 part 可能是空字串
        parts = [p for p in parts if p]
        if len(parts) < 9:
            continue

        code = parts[0]
        name = parts[1]
        price_str = parts[2]
        trade_vol = parts[3]
        cum_vol = parts[4]
        open_p = parts[5]
        high = parts[6]
        low = parts[7]
        yesterday = parts[8]
        timestamp = parts[9] if len(parts) > 9 else ''

        # 跳過尚未成交的 tick
        if price_str == '-':
            continue

        try:
            rows.append({
                'code': code,
                'name': name,
                'price': float(price_str),
                'trade_vol': int(trade_vol) if trade_vol != '-' else 0,
                'cum_vol': int(cum_vol) if cum_vol != '-' else 0,
                'open': float(open_p) if open_p != '-' else np.nan,
                'high': float(high) if high != '-' else np.nan,
                'low': float(low) if low != '-' else np.nan,
                'yesterday': float(yesterday) if yesterday != '-' else np.nan,
                'timestamp': timestamp,
                'date': date_str,
            })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(rows)


def convert_old_trace_db(trace_file):
    """
    將舊 trace_stock_DB.txt 轉換為每日 intraday/YYYYMMDD.txt 檔案

    舊格式: \\tcode\\tname\\tprice\\ttrade_vol\\tcum_vol\\topen\\thigh\\tlow\\tyesterdayTIMESTAMP\\n
    注意: yesterday 和 timestamp 之間沒有 tab（format bug）

    Returns:
        dict: {date_str: num_lines} 統計每天轉了多少筆
    """
    import re

    os.makedirs(INTRADAY_DIR, exist_ok=True)
    stats = {}
    output_files = {}

    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            parts = [p for p in parts if p]
            if len(parts) < 9:
                continue

            code = parts[0]
            name = parts[1]
            price_str = parts[2]
            trade_vol = parts[3]
            cum_vol = parts[4]
            open_p = parts[5]
            high = parts[6]
            low = parts[7]

            # 舊格式: yesterday 和 timestamp 粘在一起
            # e.g. "283.50002022-01-11 09:00:35.225939"
            last_field = parts[8] if len(parts) == 9 else parts[8]
            timestamp = parts[9] if len(parts) > 9 else ''

            if not timestamp:
                # 嘗試從 last_field 拆分 yesterday + timestamp
                m = re.match(r'^([\d.]+)(\d{4}-\d{2}-\d{2}\s.+)$', last_field)
                if m:
                    yesterday = m.group(1)
                    timestamp = m.group(2).strip()
                else:
                    yesterday = last_field
                    timestamp = ''
            else:
                yesterday = last_field

            # 從 timestamp 取得日期
            date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', timestamp)
            if not date_match:
                continue

            date_str = date_match.group(1) + date_match.group(2) + date_match.group(3)

            # 開檔（lazy open）
            if date_str not in output_files:
                fpath = os.path.join(INTRADAY_DIR, f'{date_str}.txt')
                output_files[date_str] = open(fpath, 'w', encoding='utf-8')
                output_files[date_str].write(
                    'code\tname\tprice\ttrade_vol\tcum_vol\t'
                    'open\thigh\tlow\tyesterday\ttimestamp\n'
                )
                stats[date_str] = 0

            output_files[date_str].write(
                f'{code}\t{name}\t{price_str}\t{trade_vol}\t{cum_vol}\t'
                f'{open_p}\t{high}\t{low}\t{yesterday}\t{timestamp}\n'
            )
            stats[date_str] += 1

    # 關閉所有檔案
    for fh in output_files.values():
        fh.close()

    for date_str, count in sorted(stats.items()):
        print(f"  {date_str}: {count} 筆")

    print(f"\n轉換完成: {sum(stats.values())} 筆 → {len(stats)} 個日期檔案")
    return stats


def list_intraday_files(start_date=None, end_date=None, days=30):
    """
    列出 intraday/ 目錄下符合日期範圍的檔案

    Returns:
        list of (date_str, filepath) sorted by date
    """
    if not os.path.isdir(INTRADAY_DIR):
        print(f"找不到 intraday 目錄: {INTRADAY_DIR}")
        return []

    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=days)

    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    files = []
    for fname in sorted(os.listdir(INTRADAY_DIR)):
        if not fname.endswith('.txt'):
            continue
        date_part = fname.replace('.txt', '')
        if len(date_part) != 8 or not date_part.isdigit():
            continue
        if start_str <= date_part <= end_str:
            files.append((date_part, os.path.join(INTRADAY_DIR, fname)))

    return files


def merge_intraday(start_date=None, end_date=None, days=30,
                   stock_code=None):
    """
    合併多日 intraday 資料

    Args:
        start_date: 開始日期 (datetime.date)
        end_date:   結束日期 (datetime.date)
        days:       若未指定 start_date，往前幾天
        stock_code: 只篩選特定股票代號 (str)

    Returns:
        DataFrame with columns:
        code, name, price, trade_vol, cum_vol, open, high, low, yesterday,
        timestamp, date, elapsed_sec (從每日開盤起算的秒數),
        day_index (第幾個交易日, 0-based)
    """
    files = list_intraday_files(start_date, end_date, days)

    if not files:
        print("沒有找到符合條件的 intraday 資料")
        return pd.DataFrame()

    print(f"找到 {len(files)} 天的資料: {files[0][0]} ~ {files[-1][0]}")

    dfs = []
    for i, (date_str, fpath) in enumerate(files):
        df = parse_intraday_file(fpath)
        if df.empty:
            continue
        df['day_index'] = i
        dfs.append(df)
        print(f"  {date_str}: {len(df)} 筆 tick")

    if not dfs:
        print("所有檔案都是空的")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # 計算每日開盤起算的秒數
    def calc_elapsed(ts_str):
        """從 timestamp 字串計算當日 09:00 起算的秒數"""
        try:
            # 格式: 2026-02-13 09:05:12.345678
            dt = datetime.datetime.fromisoformat(ts_str.strip())
            market_open = dt.replace(hour=9, minute=0, second=0, microsecond=0)
            return (dt - market_open).total_seconds()
        except Exception:
            return np.nan

    merged['elapsed_sec'] = merged['timestamp'].apply(calc_elapsed)

    # 篩選特定股票
    if stock_code:
        merged = merged[merged['code'] == stock_code]

    # 排序: 日期 → 股票代號 → 時間
    merged = merged.sort_values(['day_index', 'code', 'elapsed_sec'],
                                ignore_index=True)

    print(f"\n合併完成: {len(merged)} 筆 tick, "
          f"{merged['code'].nunique()} 檔股票, "
          f"{merged['day_index'].nunique()} 個交易日")

    return merged


def build_stock_series(merged_df, stock_code):
    """
    從合併資料中取出單一股票的連續時間序列

    用途: 供粒子模型回測 — 連續價格曲線 + 成交量

    Returns:
        DataFrame with columns:
        date, elapsed_sec, continuous_sec, price, trade_vol, cum_vol, yesterday
        continuous_sec: 跨日連續秒數（每天 4.5 小時 = 16200 秒）
    """
    df = merged_df[merged_df['code'] == stock_code].copy()
    if df.empty:
        return df

    DAY_SECONDS = 4.5 * 3600  # 台股每日交易 4.5 小時

    df['continuous_sec'] = df['day_index'] * DAY_SECONDS + df['elapsed_sec']
    df = df.sort_values('continuous_sec', ignore_index=True)

    return df[['date', 'elapsed_sec', 'continuous_sec', 'price',
               'trade_vol', 'cum_vol', 'yesterday', 'name']]


def export_for_backtest(merged_df, output_path=None):
    """
    輸出適合粒子模型回測的 CSV

    每列: date, code, name, price, trade_vol, cum_vol, open, high, low,
          yesterday, elapsed_sec, day_index
    """
    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, 'intraday_merged.csv')

    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"已輸出: {output_path} ({len(merged_df)} 筆)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='合併每日盤中即時資料')
    parser.add_argument('--days', type=int, default=30,
                        help='往前幾天 (預設 30)')
    parser.add_argument('--start', type=str, default=None,
                        help='開始日期 YYYYMMDD')
    parser.add_argument('--end', type=str, default=None,
                        help='結束日期 YYYYMMDD')
    parser.add_argument('--stock', type=str, default=None,
                        help='只輸出特定股票代號')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出 CSV 路徑')
    parser.add_argument('--summary', action='store_true',
                        help='只印出每日摘要，不輸出 CSV')
    parser.add_argument('--convert', type=str, default=None,
                        help='轉換舊 trace_stock_DB.txt 到 intraday/ 日期檔')
    args = parser.parse_args()

    # 轉換舊檔模式
    if args.convert:
        if not os.path.exists(args.convert):
            print(f"找不到檔案: {args.convert}")
            return
        print(f"轉換舊格式: {args.convert}")
        convert_old_trace_db(args.convert)
        return

    start_date = (datetime.datetime.strptime(args.start, '%Y%m%d').date()
                  if args.start else None)
    end_date = (datetime.datetime.strptime(args.end, '%Y%m%d').date()
                if args.end else None)

    merged = merge_intraday(start_date=start_date, end_date=end_date,
                            days=args.days, stock_code=args.stock)

    if merged.empty:
        return

    if args.summary:
        # 每日每股摘要
        print("\n=== 每日摘要 ===")
        for date, grp in merged.groupby('date'):
            print(f"\n{date}:")
            for code, sg in grp.groupby('code'):
                name = sg['name'].iloc[0]
                o = sg['price'].iloc[0]
                c = sg['price'].iloc[-1]
                h = sg['price'].max()
                l = sg['price'].min()
                chg = (c - sg['yesterday'].iloc[0]) / sg['yesterday'].iloc[0] * 100
                print(f"  {name}({code}): "
                      f"O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} "
                      f"({chg:+.2f}%) ticks={len(sg)}")
        return

    # 輸出 CSV
    export_for_backtest(merged, args.output)

    # 如果指定單一股票，也印出連續序列統計
    if args.stock:
        series = build_stock_series(merged, args.stock)
        if not series.empty:
            name = series['name'].iloc[0]
            print(f"\n{name}({args.stock}) 連續序列:")
            print(f"  總 ticks: {len(series)}")
            print(f"  價格範圍: {series['price'].min():.1f} ~ {series['price'].max():.1f}")
            total_hours = series['continuous_sec'].iloc[-1] / 3600
            print(f"  涵蓋時間: {total_hours:.1f} 小時 ({total_hours/4.5:.1f} 個交易日)")


if __name__ == '__main__':
    main()
