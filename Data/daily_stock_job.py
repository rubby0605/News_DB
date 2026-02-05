#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日股票資料自動抓取排程
- 08:00 執行基本面資料抓取
- 09:00-13:30 即時股價監控（盤中）
- 13:30 自動停止

@author: rubylintu
"""

import os
import sys
import time
import datetime
import random
import logging

# 設定工作目錄為腳本所在位置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 匯入自訂模組
from newslib import (
    read_stock_list,
    getGoodInfo,
    craw_realtime,
    get_stock_info
)
from news_collector import collect_all_news

# 設定日誌
LOG_FILE = os.path.join(SCRIPT_DIR, 'logs', f'stock_job_{datetime.date.today()}.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_fundamental_data():
    """
    抓取股票基本面資料並存成 CSV
    優先使用 GoodInfo，若失敗則使用證交所 API
    """
    logger.info("=== 開始抓取基本面資料 ===")

    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    output_file = os.path.join(SCRIPT_DIR, 'Data', 'stock_data.csv')

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dict_stock = read_stock_list(stock_list_file)
    stock_list_str = dict_stock.keys()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Name,code,price,open,high,low,yesterday,volume\n')

        for i, stock in enumerate(stock_list_str, 1):
            try:
                num = dict_stock[stock]
                # 使用證交所即時 API
                url, data = get_stock_info(num)
                # data = [代號, 名稱, 成交價, 成交量, 累積量, 開盤價, 最高價, 最低價, 昨收價]
                code, name, price, tv, volume, open_p, high, low, yesterday = data
                f.write(f'{stock},{code},{price},{open_p},{high},{low},{yesterday},{volume}\n')
                logger.info(f"[{i}/{len(dict_stock)}] {stock}({code}): ${price}")
                time.sleep(0.3)  # 避免請求過快
            except Exception as e:
                logger.error(f"抓取 {stock} 失敗: {e}")

    logger.info(f"基本面資料已儲存至: {output_file}")


def monitor_realtime_prices():
    """
    即時股價監控
    只在台股開盤時間（09:00-13:30）執行
    """
    logger.info("=== 開始即時股價監控 ===")

    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    db_file = os.path.join(SCRIPT_DIR, 'Data', 'trace_stock_DB.txt')

    columns = ['c', 'n', 'z', 'tv', 'v', 'o', 'h', 'l', 'y']
    # ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']

    dict_stock = read_stock_list(stock_list_file)
    stock_list = [int(dict_stock[stock]) for stock in dict_stock.keys()]

    iteration = 0

    with open(db_file, 'a', encoding='utf-8') as fi:
        while True:
            now = datetime.datetime.now()
            current_time = now.time()

            # 台股交易時間：09:00 - 13:30
            market_open = datetime.time(9, 0)
            market_close = datetime.time(13, 30)

            # 檢查是否為週末
            if now.weekday() >= 5:  # 週六=5, 週日=6
                logger.info("今天是週末，停止監控")
                break

            # 檢查是否超過收盤時間
            if current_time > market_close:
                logger.info("已過收盤時間 (13:30)，停止監控")
                break

            # 如果還沒開盤，等待
            if current_time < market_open:
                wait_seconds = (datetime.datetime.combine(now.date(), market_open) - now).seconds
                logger.info(f"等待開盤... ({wait_seconds} 秒後)")
                time.sleep(min(wait_seconds, 60))  # 最多等 60 秒後再檢查
                continue

            # 抓取即時資料
            try:
                data = craw_realtime(stock_list)

                if 'msgArray' not in data or len(data['msgArray']) == 0:
                    logger.warning("無法取得即時資料，等待重試...")
                    time.sleep(10)
                    continue

                for i in range(min(len(dict_stock) - 1, len(data['msgArray']))):
                    line = ''
                    for column in columns:
                        value = data['msgArray'][i].get(column, '-')
                        line = line + '\t' + str(value)
                    line = line + '\t' + str(now) + '\n'
                    fi.write(line)

                fi.flush()  # 確保寫入磁碟
                iteration += 1

                if iteration % 10 == 0:
                    logger.info(f"已執行 {iteration} 次，時間: {now.strftime('%H:%M:%S')}")

            except Exception as e:
                logger.error(f"抓取即時資料錯誤: {e}")

            # 隨機等待 10-20 秒
            time.sleep(10 + random.random() * 10)

    logger.info(f"即時監控結束，共執行 {iteration} 次")


def main():
    """主程式"""
    logger.info("=" * 50)
    logger.info("每日股票資料抓取程式啟動")
    logger.info(f"日期: {datetime.date.today()}")
    logger.info("=" * 50)

    now = datetime.datetime.now()

    # 檢查是否為週末
    if now.weekday() >= 5:
        logger.info("今天是週末，不執行")
        return

    # 1. 抓取基本面資料
    try:
        fetch_fundamental_data()
    except Exception as e:
        logger.error(f"基本面資料抓取失敗: {e}")

    # 2. 收集新聞資料（用於 AI 訓練）
    try:
        logger.info("=== 開始收集新聞 ===")
        collect_all_news()
    except Exception as e:
        logger.error(f"新聞收集失敗: {e}")

    # 3. 即時股價監控（等到 9:00 開盤後開始）
    try:
        monitor_realtime_prices()
    except Exception as e:
        logger.error(f"即時監控失敗: {e}")

    logger.info("今日任務完成")


if __name__ == "__main__":
    main()
