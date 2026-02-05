#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日股票資料自動抓取排程
- 08:00 盤前分析（粒子模型預測）
- 09:00-13:30 即時股價監控（盤中）
- 13:30 盤後誤差分析

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

# 儲存盤前預測結果（供盤後比較）
PREMARKET_PREDICTIONS = {}


def send_premarket_analysis():
    """
    盤前分析 - 使用粒子模型預測並發送到 Discord
    """
    global PREMARKET_PREDICTIONS
    logger.info("=== 開始盤前分析 ===")

    try:
        from directional_particle_model import DirectionalParticleModel
        from newslib import read_stock_list

        model = DirectionalParticleModel(n_particles=1000)
        stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
        dict_stock = read_stock_list(stock_list_file)

        results = []
        for name, code in dict_stock.items():
            result = model.predict(str(code), name)
            if 'error' not in result:
                results.append(result)
                # 儲存預測供盤後比較
                PREMARKET_PREDICTIONS[result['stock_code']] = {
                    'name': result['stock_name'],
                    'predicted_price': result['predicted_price'],
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'current_price': result['current_price']
                }

        # 分類
        bulls = [r for r in results if r['direction'] == '漲']
        bears = [r for r in results if r['direction'] == '跌']
        neutral = [r for r in results if r['direction'] == '盤整']

        bulls.sort(key=lambda x: x['expected_change'], reverse=True)
        bears.sort(key=lambda x: x['expected_change'])

        # 組合訊息
        now = datetime.datetime.now()
        lines = [
            '**📊 盤前分析報告**',
            f'📅 {now.strftime("%Y/%m/%d")} {now.strftime("%H:%M")}',
            '',
            '**🟢 看漲 TOP 5：**'
        ]

        for r in bulls[:5]:
            foreign = r['signals'].get('foreign', '')
            foreign_info = f' [{foreign}]' if '買超' in foreign or '大買' in foreign else ''
            lines.append(f"• {r['stock_name']}: ${r['current_price']:.0f}→${r['predicted_price']:.0f} ({r['expected_change']:+.1f}%){foreign_info}")

        lines.append('')
        lines.append('**🔴 看跌 TOP 5：**')

        for r in bears[:5]:
            foreign = r['signals'].get('foreign', '')
            foreign_info = f' [{foreign}]' if '賣超' in foreign or '大賣' in foreign else ''
            lines.append(f"• {r['stock_name']}: ${r['current_price']:.0f}→${r['predicted_price']:.0f} ({r['expected_change']:+.1f}%){foreign_info}")

        # 重點關注
        lines.append('')
        lines.append('**⭐ 重點關注：**')
        for r in results:
            if r['stock_name'] in ['群聯', '景碩']:
                foreign = r['signals'].get('foreign', '')
                momentum = r['signals'].get('momentum', '')
                lines.append(f"• {r['stock_name']}: ${r['current_price']:.0f}→${r['predicted_price']:.0f} ({r['expected_change']:+.1f}%) [{r['direction']} {r['confidence']:.0%}]")
                lines.append(f"  └ {foreign}, {momentum}")

        no_data = [name for name in ['群聯', '景碩'] if name not in [r['stock_name'] for r in results]]
        for name in no_data:
            lines.append(f'• {name}: 無歷史資料')

        lines.append('')
        lines.append(f'**📈 統計：** 看漲 {len(bulls)} 檔 | 看跌 {len(bears)} 檔 | 盤整 {len(neutral)} 檔')

        message = '\n'.join(lines)
        send_discord(message, title='盤前 AI 分析')
        logger.info(f"盤前分析完成，預測 {len(results)} 檔股票")

    except Exception as e:
        logger.error(f"盤前分析失敗: {e}")


def send_postmarket_analysis():
    """
    盤後誤差分析 - 比較預測 vs 實際收盤價
    """
    global PREMARKET_PREDICTIONS
    logger.info("=== 開始盤後誤差分析 ===")

    if not PREMARKET_PREDICTIONS:
        logger.warning("沒有盤前預測資料，跳過誤差分析")
        return

    try:
        from newslib import read_stock_list, craw_realtime

        stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
        dict_stock = read_stock_list(stock_list_file)
        stock_list = [int(dict_stock[stock]) for stock in dict_stock.keys()]

        # 抓取收盤價
        data = craw_realtime(stock_list)

        if 'msgArray' not in data or len(data['msgArray']) == 0:
            logger.error("無法取得收盤資料")
            return

        # 比較預測與實際
        results = []
        correct_direction = 0
        total_compared = 0

        for item in data['msgArray']:
            code = item.get('c', '')
            actual_price = item.get('z', '-')
            yesterday = item.get('y', '-')

            if code in PREMARKET_PREDICTIONS and actual_price != '-':
                pred = PREMARKET_PREDICTIONS[code]
                actual_price = float(actual_price)
                yesterday_price = float(yesterday) if yesterday != '-' else pred['current_price']

                # 計算實際漲跌
                actual_change = (actual_price - yesterday_price) / yesterday_price * 100
                actual_direction = '漲' if actual_change > 0.5 else '跌' if actual_change < -0.5 else '盤整'

                # 計算預測誤差
                pred_error = abs(pred['predicted_price'] - actual_price) / actual_price * 100

                # 方向是否正確
                direction_correct = (pred['direction'] == actual_direction) or \
                                   (pred['direction'] == '漲' and actual_change > 0) or \
                                   (pred['direction'] == '跌' and actual_change < 0)

                if direction_correct:
                    correct_direction += 1
                total_compared += 1

                results.append({
                    'name': pred['name'],
                    'code': code,
                    'predicted': pred['predicted_price'],
                    'actual': actual_price,
                    'pred_direction': pred['direction'],
                    'actual_direction': actual_direction,
                    'actual_change': actual_change,
                    'error': pred_error,
                    'correct': direction_correct
                })

        # 計算準確率
        accuracy = correct_direction / total_compared * 100 if total_compared > 0 else 0

        # 按誤差排序
        results.sort(key=lambda x: x['error'])

        # 組合訊息
        now = datetime.datetime.now()
        lines = [
            '**📊 盤後誤差分析報告**',
            f'📅 {now.strftime("%Y/%m/%d")} 收盤',
            '',
            f'**🎯 方向準確率: {accuracy:.1f}%** ({correct_direction}/{total_compared})',
            '',
            '**✅ 預測正確 TOP 5：**'
        ]

        correct_results = [r for r in results if r['correct']]
        for r in correct_results[:5]:
            emoji = '🟢' if r['actual_change'] > 0 else '🔴' if r['actual_change'] < 0 else '⚪'
            lines.append(f"{emoji} {r['name']}: 預測{r['pred_direction']} → 實際{r['actual_change']:+.1f}% ✓")

        lines.append('')
        lines.append('**❌ 預測錯誤：**')

        wrong_results = [r for r in results if not r['correct']]
        for r in wrong_results[:5]:
            emoji = '🟢' if r['actual_change'] > 0 else '🔴' if r['actual_change'] < 0 else '⚪'
            lines.append(f"{emoji} {r['name']}: 預測{r['pred_direction']} → 實際{r['actual_change']:+.1f}% ✗")

        # 重點關注的誤差
        lines.append('')
        lines.append('**⭐ 重點關注結果：**')
        for r in results:
            if r['name'] in ['群聯', '景碩']:
                status = '✓' if r['correct'] else '✗'
                lines.append(f"• {r['name']}: 預測${r['predicted']:.0f} → 實際${r['actual']:.0f} (誤差 {r['error']:.1f}%) {status}")

        # 統計
        avg_error = sum(r['error'] for r in results) / len(results) if results else 0
        lines.append('')
        lines.append(f'**📈 平均價格誤差: {avg_error:.1f}%**')

        message = '\n'.join(lines)
        send_discord(message, title='盤後誤差分析')
        logger.info(f"盤後分析完成，準確率 {accuracy:.1f}%")

        # 清空預測資料
        PREMARKET_PREDICTIONS = {}

    except Exception as e:
        logger.error(f"盤後分析失敗: {e}")


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

    # 3. 盤前分析（粒子模型預測）→ 發送 Discord
    try:
        send_premarket_analysis()
    except Exception as e:
        logger.error(f"盤前分析失敗: {e}")

    # 4. 即時股價監控（等到 9:00 開盤後開始）
    try:
        monitor_realtime_prices()
    except Exception as e:
        logger.error(f"即時監控失敗: {e}")

    # 5. 盤後誤差分析 → 發送 Discord
    try:
        send_postmarket_analysis()
    except Exception as e:
        logger.error(f"盤後分析失敗: {e}")

    # 6. 收盤後發送每日報告
    try:
        send_daily_report(news_count=0)
    except Exception as e:
        logger.error(f"發送每日報告失敗: {e}")

    logger.info("今日任務完成")


if __name__ == "__main__":
    main()
