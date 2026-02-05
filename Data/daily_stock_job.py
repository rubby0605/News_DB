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
from notifier import send_daily_report, send_discord

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


def send_prediction_notification(stock_prices, clf, vectorizer, now):
    """
    發送股票預測通知到 Discord
    - 只顯示漲跌幅大的股票
    - 加入重要新聞標題
    - 加入粒子模型預測
    """
    from hybrid_predictor import hybrid_predict
    from newslib import scrapBingNews, scrapGoogleNews
    import re

    logger.info("發送 15 分鐘預測通知...")

    # 優先關注的股票
    PRIORITY_STOCKS = ['群聯', '景碩']
    CHANGE_THRESHOLD = 1.5  # 漲跌幅超過 1.5% 才顯示

    # 載入粒子模型（每日只抓一次法人資料）
    particle_predictions = {}
    try:
        from directional_particle_model import DirectionalParticleModel
        particle_model = DirectionalParticleModel(n_particles=500)

        # 只預測優先股票（節省時間）
        from newslib import read_stock_list
        stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
        dict_stock = read_stock_list(stock_list_file)

        for name in PRIORITY_STOCKS:
            if name in dict_stock:
                code = str(dict_stock[name])
                result = particle_model.predict(code, name)
                if 'error' not in result:
                    particle_predictions[name] = result
    except Exception as e:
        logger.warning(f"粒子模型載入失敗: {e}")

    # 建立通知內容
    lines = [
        f"**{now.strftime('%H:%M')} 盤中快報**",
    ]

    # 計算每檔股票的漲跌幅
    stock_changes = []
    for s in stock_prices:
        if s['price'] == '-':
            continue
        try:
            price = float(s['price'])
            yesterday = float(s['yesterday']) if s['yesterday'] != '-' else price
            change_pct = ((price - yesterday) / yesterday) * 100
            stock_changes.append({
                'name': s['name'],
                'code': s['code'],
                'price': price,
                'change_pct': change_pct,
                'is_priority': s['name'] in PRIORITY_STOCKS
            })
        except:
            continue

    # 篩選：優先股 + 漲跌幅大的
    priority = [s for s in stock_changes if s['is_priority']]
    big_movers = [s for s in stock_changes if abs(s['change_pct']) >= CHANGE_THRESHOLD and not s['is_priority']]
    big_movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)

    # 顯示優先關注股票
    if priority:
        lines.append("")
        lines.append("**⭐ 重點關注：**")
        for s in priority:
            emoji = "🔴" if s['change_pct'] < 0 else "🟢" if s['change_pct'] > 0 else "⚪"
            lines.append(f"{emoji} {s['name']}: ${s['price']:.1f} ({s['change_pct']:+.1f}%)")

    # 顯示粒子模型預測
    if particle_predictions:
        lines.append("")
        lines.append("**🎯 AI預測（法人+技術面）：**")
        for name, pred in particle_predictions.items():
            emoji = "🟢" if pred['direction'] == '漲' else "🔴" if pred['direction'] == '跌' else "⚪"
            # 顯示主要信號
            signal = pred['signals'].get('foreign', '')
            lines.append(f"{emoji} {name}: ${pred['current_price']:.0f}→${pred['predicted_price']:.0f} ({pred['expected_change']:+.1f}%) [{pred['direction']} {pred['confidence']:.0%}]")
            if signal:
                lines.append(f"   └ {signal}")

    # 顯示漲跌幅大的股票（最多 5 檔）
    if big_movers:
        lines.append("")
        lines.append("**📊 大幅波動：**")
        for s in big_movers[:5]:
            emoji = "🔴" if s['change_pct'] < 0 else "🟢"
            lines.append(f"{emoji} {s['name']}: ${s['price']:.1f} ({s['change_pct']:+.1f}%)")

    # 抓取重要新聞並分析
    if clf and vectorizer:
        lines.append("")
        lines.append("**📰 重要新聞：**")

        news_items = []
        # 針對優先股票抓新聞
        for stock_name in PRIORITY_STOCKS[:2]:
            try:
                url, title, body, bs = scrapBingNews(stock_name)
                if body:
                    # 提取新聞句子
                    sentences = re.split(r'[。！？\n]', body)
                    for sent in sentences[:3]:
                        sent = sent.strip()
                        if len(sent) > 15 and stock_name in sent:
                            pred, conf, _ = hybrid_predict(sent, clf, vectorizer)
                            news_items.append({
                                'text': sent[:50] + '...' if len(sent) > 50 else sent,
                                'prediction': pred,
                                'stock': stock_name
                            })
                            break
            except:
                continue

        if news_items:
            for item in news_items[:3]:
                emoji = "🟢" if item['prediction'] == '漲' else "🔴" if item['prediction'] == '跌' else "⚪"
                lines.append(f"{emoji} [{item['stock']}] {item['text']}")
        else:
            lines.append("（暫無重大新聞）")

    # 統計摘要
    bull_count = sum(1 for s in stock_changes if s['change_pct'] > 0)
    bear_count = sum(1 for s in stock_changes if s['change_pct'] < 0)
    lines.append("")
    lines.append(f"📈 上漲: {bull_count} 檔 | 📉 下跌: {bear_count} 檔")

    message = "\n".join(lines)

    try:
        send_discord(message, title="盤中即時更新")
        logger.info("Discord 通知已發送")
    except Exception as e:
        logger.error(f"發送通知失敗: {e}")


def monitor_realtime_prices():
    """
    即時股價監控
    只在台股開盤時間（09:00-13:30）執行
    每 15 分鐘發送 Discord 通知
    """
    logger.info("=== 開始即時股價監控 ===")

    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    db_file = os.path.join(SCRIPT_DIR, 'Data', 'trace_stock_DB.txt')

    columns = ['c', 'n', 'z', 'tv', 'v', 'o', 'h', 'l', 'y']
    # ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']

    dict_stock = read_stock_list(stock_list_file)
    stock_list = [int(dict_stock[stock]) for stock in dict_stock.keys()]
    stock_names = {v: k for k, v in dict_stock.items()}  # 代號 -> 名稱

    iteration = 0
    last_notify_time = None  # 上次通知時間

    # 載入預測模型
    try:
        from hybrid_predictor import hybrid_predict, load_ml_model
        clf, vectorizer = load_ml_model()
    except:
        clf, vectorizer = None, None

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

                # 收集股價資料
                stock_prices = []
                for i in range(min(len(dict_stock) - 1, len(data['msgArray']))):
                    item = data['msgArray'][i]
                    line = ''
                    for column in columns:
                        value = item.get(column, '-')
                        line = line + '\t' + str(value)
                    line = line + '\t' + str(now) + '\n'
                    fi.write(line)

                    # 記錄股價資訊
                    code = item.get('c', '')
                    name = item.get('n', stock_names.get(code, code))
                    price = item.get('z', '-')
                    yesterday = item.get('y', '-')
                    stock_prices.append({
                        'code': code,
                        'name': name,
                        'price': price,
                        'yesterday': yesterday
                    })

                fi.flush()  # 確保寫入磁碟
                iteration += 1

                # 每 15 分鐘發送 Discord 通知
                should_notify = False
                if last_notify_time is None:
                    should_notify = True
                elif (now - last_notify_time).total_seconds() >= 900:  # 900秒 = 15分鐘
                    should_notify = True

                if should_notify:
                    send_prediction_notification(stock_prices, clf, vectorizer, now)
                    last_notify_time = now

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

    # 3. 發送 Discord 通知（開盤前）
    try:
        send_discord(
            f"**每日排程啟動**\n\n"
            f"日期: {now.strftime('%Y-%m-%d')}\n"
            f"時間: {now.strftime('%H:%M')}\n\n"
            f"即將開始盤中監控 (09:00-13:30)",
            title="股票系統通知"
        )
    except Exception as e:
        logger.error(f"發送通知失敗: {e}")

    # 4. 即時股價監控（等到 9:00 開盤後開始）
    try:
        monitor_realtime_prices()
    except Exception as e:
        logger.error(f"即時監控失敗: {e}")

    # 5. 收盤後發送每日報告
    try:
        send_daily_report(news_count=0)  # TODO: 傳入實際收集數量
    except Exception as e:
        logger.error(f"發送每日報告失敗: {e}")

    logger.info("今日任務完成")


if __name__ == "__main__":
    main()
