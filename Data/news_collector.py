#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞收集系統 - 用於 AI 股票分析回測

功能：
1. 抓取關注股票的相關新聞
2. 標記股票代號、日期
3. 儲存為結構化 CSV 供後續訓練

@author: rubylintu
"""

import os
import re
import json
import time
import random
import datetime
import logging
import csv
from pathlib import Path

# 設定工作目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from newslib import (
    read_stock_list,
    scrapBingNews,
    scrapGoogleNews,
    craw_one_month,
    getPage
)

# 設定日誌
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'news_collector_{datetime.date.today()}.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 資料儲存目錄
NEWS_DATA_DIR = os.path.join(SCRIPT_DIR, 'news_data')
os.makedirs(NEWS_DATA_DIR, exist_ok=True)


def clean_text(text):
    """清理新聞文字"""
    if not text:
        return ""
    # 移除多餘空白和換行
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字元
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、：；「」『』（）]', '', text)
    return text.strip()


def extract_news_from_bing(keyword, stock_code):
    """從 Bing 抓取新聞"""
    news_list = []
    try:
        url, title, body, bs = scrapBingNews(keyword)

        if bs is None:
            return news_list

        # 尋找新聞卡片
        news_cards = bs.find_all('div', class_='news-card')
        if not news_cards:
            news_cards = bs.find_all('div', class_='newsitem')
        if not news_cards:
            # 嘗試其他選擇器
            news_cards = bs.find_all('a', class_='title')

        for card in news_cards[:10]:  # 最多取 10 則
            try:
                # 嘗試提取標題
                title_elem = card.find('a', class_='title') or card.find('a')
                if title_elem:
                    news_title = clean_text(title_elem.get_text())
                    news_url = title_elem.get('href', '')

                    if news_title and len(news_title) > 10:
                        news_list.append({
                            'title': news_title,
                            'url': news_url,
                            'source': 'Bing',
                            'stock_code': stock_code,
                            'keyword': keyword
                        })
            except Exception as e:
                continue

    except Exception as e:
        logger.error(f"Bing 抓取失敗 ({keyword}): {e}")

    return news_list


def extract_news_from_google(keyword, stock_code):
    """從 Google News 抓取新聞"""
    news_list = []
    try:
        url, title, body, bs = scrapGoogleNews(keyword)

        if bs is None:
            return news_list

        # Google 搜尋結果
        results = bs.find_all('div', class_='BNeawe')

        seen_titles = set()
        for result in results[:20]:
            try:
                text = clean_text(result.get_text())
                # 過濾太短或重複的
                if text and len(text) > 15 and text not in seen_titles:
                    seen_titles.add(text)
                    news_list.append({
                        'title': text[:200],  # 限制長度
                        'url': '',
                        'source': 'Google',
                        'stock_code': stock_code,
                        'keyword': keyword
                    })
            except:
                continue

    except Exception as e:
        logger.error(f"Google 抓取失敗 ({keyword}): {e}")

    return news_list


def get_stock_price_change(stock_code, date_str):
    """
    取得股票在指定日期的漲跌幅
    date_str 格式: YYYYMMDD
    """
    try:
        data = craw_one_month(stock_code, date_str)

        if '很抱歉' in str(data.get('stat', '')):
            return None, None, None

        df_data = data.get('data', [])
        if not df_data:
            return None, None, None

        # 取最後一筆（最新）
        latest = df_data[-1]
        # [日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數]
        close_price = float(latest[6].replace(',', ''))
        change = latest[7]  # 漲跌價差

        # 計算漲跌幅
        if '+' in change or (change.replace('.', '').replace('-', '').isdigit() and float(change) > 0):
            change_val = float(change.replace('+', '').replace(',', ''))
        elif '-' in change:
            change_val = float(change.replace(',', ''))
        else:
            change_val = 0

        prev_close = close_price - change_val
        if prev_close > 0:
            change_pct = (change_val / prev_close) * 100
        else:
            change_pct = 0

        return close_price, change_val, round(change_pct, 2)

    except Exception as e:
        logger.error(f"取得股價失敗 ({stock_code}): {e}")
        return None, None, None


def collect_news_for_stock(stock_name, stock_code):
    """收集單一股票的新聞"""
    all_news = []

    # 用股票名稱搜尋
    keywords = [stock_name, f"{stock_code} 股票"]

    for keyword in keywords:
        # Bing
        news = extract_news_from_bing(keyword, stock_code)
        all_news.extend(news)
        time.sleep(random.uniform(1, 2))

        # Google
        news = extract_news_from_google(keyword, stock_code)
        all_news.extend(news)
        time.sleep(random.uniform(1, 2))

    # 去重
    seen = set()
    unique_news = []
    for item in all_news:
        title = item['title']
        if title not in seen:
            seen.add(title)
            unique_news.append(item)

    return unique_news


def save_daily_news(news_data, stock_prices):
    """儲存每日新聞資料"""
    today = datetime.date.today()
    date_str = today.strftime('%Y%m%d')

    # CSV 檔案路徑
    csv_file = os.path.join(NEWS_DATA_DIR, f'news_{date_str}.csv')

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'stock_code', 'stock_name', 'keyword', 'source',
            'title', 'url', 'close_price', 'price_change', 'change_pct'
        ])
        writer.writeheader()

        for item in news_data:
            code = item['stock_code']
            price_info = stock_prices.get(code, {})

            writer.writerow({
                'date': date_str,
                'stock_code': code,
                'stock_name': item.get('keyword', ''),
                'keyword': item.get('keyword', ''),
                'source': item['source'],
                'title': item['title'],
                'url': item.get('url', ''),
                'close_price': price_info.get('close', ''),
                'price_change': price_info.get('change', ''),
                'change_pct': price_info.get('change_pct', '')
            })

    logger.info(f"新聞資料已儲存: {csv_file} ({len(news_data)} 則)")

    # 也存一份到彙總檔案
    master_file = os.path.join(NEWS_DATA_DIR, 'news_history.csv')
    file_exists = os.path.exists(master_file)

    with open(master_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'stock_code', 'stock_name', 'keyword', 'source',
            'title', 'url', 'close_price', 'price_change', 'change_pct'
        ])

        if not file_exists:
            writer.writeheader()

        for item in news_data:
            code = item['stock_code']
            price_info = stock_prices.get(code, {})

            writer.writerow({
                'date': date_str,
                'stock_code': code,
                'stock_name': item.get('keyword', ''),
                'keyword': item.get('keyword', ''),
                'source': item['source'],
                'title': item['title'],
                'url': item.get('url', ''),
                'close_price': price_info.get('close', ''),
                'price_change': price_info.get('change', ''),
                'change_pct': price_info.get('change_pct', '')
            })

    return csv_file


def collect_all_news():
    """主函式：收集所有關注股票的新聞"""
    logger.info("=" * 50)
    logger.info("開始收集新聞資料")
    logger.info(f"日期: {datetime.date.today()}")
    logger.info("=" * 50)

    # 讀取股票清單
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    all_news = []
    stock_prices = {}

    # 取得當月日期字串
    today = datetime.date.today()
    date_str = today.strftime('%Y%m01')

    total = len(dict_stock)

    for i, (stock_name, stock_code) in enumerate(dict_stock.items(), 1):
        logger.info(f"[{i}/{total}] 處理: {stock_name} ({stock_code})")

        # 收集新聞
        news = collect_news_for_stock(stock_name, stock_code)
        all_news.extend(news)
        logger.info(f"  找到 {len(news)} 則新聞")

        # 取得股價
        close, change, change_pct = get_stock_price_change(stock_code, date_str)
        if close:
            stock_prices[stock_code] = {
                'close': close,
                'change': change,
                'change_pct': change_pct
            }
            logger.info(f"  股價: {close}, 漲跌: {change_pct}%")

        # 避免請求過快
        time.sleep(random.uniform(2, 4))

    # 儲存資料
    if all_news:
        csv_file = save_daily_news(all_news, stock_prices)
        logger.info(f"總共收集 {len(all_news)} 則新聞")
    else:
        logger.warning("沒有收集到任何新聞")

    logger.info("新聞收集完成")
    return all_news, stock_prices


def get_news_statistics():
    """取得新聞收集統計"""
    master_file = os.path.join(NEWS_DATA_DIR, 'news_history.csv')

    if not os.path.exists(master_file):
        return {"total": 0, "days": 0, "stocks": 0}

    dates = set()
    stocks = set()
    total = 0

    with open(master_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            dates.add(row.get('date', ''))
            stocks.add(row.get('stock_code', ''))

    return {
        "total": total,
        "days": len(dates),
        "stocks": len(stocks)
    }


if __name__ == "__main__":
    collect_all_news()
