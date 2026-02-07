#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性全股票預測腳本
走完「新聞收集 → 選股 → 粒子模型預測 → GPT 情緒分析 → 輸出報告」完整流程

用法:
    cd /Users/rubylintu/Desktop/News_DB/Data
    /opt/anaconda3/bin/python run_prediction.py

@author: rubylintu
"""

import os
import sys
import time
import datetime
import logging

# 設定工作目錄為腳本所在位置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 匯入自訂模組
from newslib import read_stock_list, get_stock_info
from news_collector import collect_all_news
from news_stock_selector import select_focus_stocks_from_news
from directional_particle_model import DirectionalParticleModel
from gpt_sentiment import analyze_stock_with_news
from notifier import send_discord

# 設定日誌
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f'run_prediction_{datetime.date.today()}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    now = datetime.datetime.now()

    logger.info("=" * 60)
    logger.info("一次性全股票預測腳本啟動")
    logger.info(f"日期: {now.strftime('%Y/%m/%d %H:%M')}")
    logger.info("=" * 60)

    # ── 1. 讀取股票清單 ──
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)
    logger.info(f"[1/7] 讀取股票清單: {len(dict_stock)} 檔")

    # ── 2. 抓取基本面（即時價格） ──
    logger.info("[2/7] 抓取即時股價...")
    stock_prices = {}
    for i, (name, code) in enumerate(dict_stock.items(), 1):
        try:
            url, data = get_stock_info(code)
            if data:
                price = data[2]  # 成交價
                stock_prices[str(code)] = {
                    'name': name,
                    'price': price,
                    'yesterday': data[8],  # 昨收價
                }
                logger.info(f"  [{i}/{len(dict_stock)}] {name}({code}): ${price}")
            else:
                logger.warning(f"  [{i}/{len(dict_stock)}] {name}({code}): 無資料")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"  [{i}/{len(dict_stock)}] {name}({code}) 失敗: {e}")

    # ── 3. 收集新聞 ──
    logger.info("[3/7] 收集新聞...")
    try:
        all_news, _ = collect_all_news()
        news_count = len(all_news) if all_news else 0
        logger.info(f"  收集到 {news_count} 則新聞")
    except Exception as e:
        logger.error(f"  新聞收集失敗: {e}")
        news_count = 0

    # ── 4. 新聞選股（GPT 選 5 檔焦點） ──
    logger.info("[4/7] 新聞選股...")
    focus_stocks = {}
    try:
        selected = select_focus_stocks_from_news(num_stocks=5)
        if selected:
            for s in selected:
                focus_stocks[s['code']] = {
                    'name': s['name'],
                    'reason': s['reason'],
                    'news_count': s['news_count'],
                    'sentiment_score': s['sentiment_score'],
                }
            logger.info(f"  選出 {len(selected)} 檔焦點股票")
        else:
            logger.warning("  新聞選股未選出任何股票")
    except Exception as e:
        logger.error(f"  新聞選股失敗: {e}")

    focus_names = {v['name'] for v in focus_stocks.values()} if focus_stocks else set()
    focus_codes = set(focus_stocks.keys()) if focus_stocks else set()

    # ── 5. 粒子模型預測（全部 31 檔） ──
    logger.info("[5/7] 粒子模型預測...")
    model = DirectionalParticleModel(n_particles=1000)
    results = []
    for name, code in dict_stock.items():
        try:
            result = model.predict(str(code), name)
            if 'error' not in result:
                results.append(result)
                logger.info(
                    f"  {name}({code}): ${result['current_price']:.0f}→"
                    f"${result['predicted_price']:.0f} "
                    f"({result['expected_change']:+.1f}%) "
                    f"[{result['direction']} {result['confidence']:.0%}]"
                )
            else:
                logger.warning(f"  {name}({code}): {result['error']}")
        except Exception as e:
            logger.error(f"  {name}({code}) 預測失敗: {e}")

    # ── 6. GPT 情緒分析（焦點 5 檔） ──
    logger.info("[6/7] GPT 情緒分析（焦點股票）...")
    gpt_sentiments = {}
    for name in focus_names:
        try:
            result = analyze_stock_with_news(name)
            gpt_sentiments[name] = result
            logger.info(
                f"  {name}: {result.get('sentiment')} "
                f"({result.get('confidence', 0):.0%}) "
                f"{result.get('reason', '')}"
            )
        except Exception as e:
            logger.error(f"  {name} GPT 分析失敗: {e}")

    # ── 7. 組合報告 ──
    logger.info("[7/7] 組合報告...")

    # 分出焦點與其餘
    focus_results = [
        r for r in results
        if r['stock_code'] in focus_codes or r['stock_name'] in focus_names
    ]
    other_results = [
        r for r in results
        if r['stock_code'] not in focus_codes and r['stock_name'] not in focus_names
    ]

    other_bulls = sorted(
        [r for r in other_results if r['direction'] == '漲'],
        key=lambda x: x['expected_change'], reverse=True
    )
    other_bears = sorted(
        [r for r in other_results if r['direction'] == '跌'],
        key=lambda x: x['expected_change']
    )

    all_bulls = [r for r in results if r['direction'] == '漲']
    all_bears = [r for r in results if r['direction'] == '跌']
    all_neutral = [r for r in results if r['direction'] == '盤整']

    # 建立報告文字
    lines = [
        '**📊 盤前分析報告**',
        f'📅 {now.strftime("%Y/%m/%d")} {now.strftime("%H:%M")}',
    ]

    # ⭐ 新聞精選焦點區
    if focus_stocks:
        lines.append('')
        lines.append(f'**⭐ 新聞精選 {len(focus_stocks)} 檔（完整分析）：**')
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        for i, (code, info) in enumerate(focus_stocks.items()):
            medal = medals[i] if i < len(medals) else f'{i+1}.'
            name = info['name']

            pred = next(
                (r for r in focus_results
                 if r['stock_code'] == code or r['stock_name'] == name),
                None
            )
            if pred:
                foreign = pred['signals'].get('foreign', '')
                foreign_info = ''
                if any(kw in foreign for kw in ['買超', '大買', '賣超', '大賣']):
                    foreign_info = f' [{foreign}]'
                lines.append(
                    f"{medal} {name}({code}): "
                    f"${pred['current_price']:.0f}→${pred['predicted_price']:.0f} "
                    f"({pred['expected_change']:+.1f}%) "
                    f"[{pred['direction']} {pred['confidence']:.0%}]"
                    f"{foreign_info}"
                )
            else:
                lines.append(f"{medal} {name}({code}): 無預測資料")

            lines.append(f"   └ 選股理由：{info['reason']}")

            gpt = gpt_sentiments.get(name)
            if gpt:
                sentiment = gpt.get('sentiment', '中性')
                confidence = gpt.get('confidence', 0)
                reason = gpt.get('reason', '')
                emoji = "🔴" if sentiment == '漲' else "🟢" if sentiment == '跌' else "⚪"
                lines.append(
                    f"   └ GPT情緒: {emoji} {sentiment} ({confidence:.0%}) {reason}"
                )

    # 🔴 其餘看漲 TOP 5
    lines.append('')
    lines.append('**🔴 其餘看漲 TOP 5：**')
    for r in other_bulls[:5]:
        lines.append(
            f"• {r['stock_name']}: {r['expected_change']:+.1f}% "
            f"[{r['direction']} {r['confidence']:.0%}]"
        )

    # 🟢 其餘看跌 TOP 5
    lines.append('')
    lines.append('**🟢 其餘看跌 TOP 5：**')
    for r in other_bears[:5]:
        lines.append(
            f"• {r['stock_name']}: {r['expected_change']:+.1f}% "
            f"[{r['direction']} {r['confidence']:.0%}]"
        )

    # 📈 統計
    lines.append('')
    lines.append(
        f'**📈 統計：** 看漲 {len(all_bulls)} 檔 | '
        f'看跌 {len(all_bears)} 檔 | 盤整 {len(all_neutral)} 檔'
    )

    elapsed = time.time() - start_time
    lines.append('')
    lines.append(f'⏱ 執行時間: {elapsed:.0f} 秒 | 新聞 {news_count} 則')

    report = '\n'.join(lines)

    # 印到終端
    print('\n' + '=' * 60)
    print(report)
    print('=' * 60)

    # 發送 Discord
    try:
        send_discord(report, title='盤前 AI 分析')
        logger.info("Discord 通知已發送")
    except Exception as e:
        logger.error(f"Discord 發送失敗: {e}")

    logger.info(f"完成！共預測 {len(results)} 檔，耗時 {elapsed:.0f} 秒")


if __name__ == "__main__":
    main()
