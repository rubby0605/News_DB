#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盤前新聞選股模組
掃描全部股票新聞，用 GPT 智慧選出當天最值得關注的 5 檔

流程：
1. 對 31 檔股票各抓取 Bing + Google 新聞標題
2. 計算每檔股票的新聞數量和關鍵字情緒分數
3. 將新聞摘要送給 GPT，選出 5 檔最有交易機會的股票

@author: rubylintu
"""

import os
import sys
import time
import random
import logging
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from newslib import read_stock_list
from news_collector import collect_news_for_stock, load_seen_news
from hybrid_predictor import keyword_score, BULL_KEYWORDS, BEAR_KEYWORDS
from gpt_sentiment import select_top_stocks

logger = logging.getLogger(__name__)


def scan_all_stock_news(dict_stock):
    """
    掃描所有股票的新聞標題

    Args:
        dict_stock: {股票名稱: 股票代號, ...}

    Returns:
        dict: {股票名稱: {'code': '2330', 'news': [{'title': ..., 'source': ...}, ...], 'news_count': N}, ...}
    """
    all_stock_news = {}
    seen_data = load_seen_news()

    total = len(dict_stock)
    for i, (name, code) in enumerate(dict_stock.items(), 1):
        logger.info(f"[{i}/{total}] 掃描新聞: {name} ({code})")

        try:
            news_list = collect_news_for_stock(name, str(code), seen_data=None)
            all_stock_news[name] = {
                'code': str(code),
                'news': news_list,
                'news_count': len(news_list)
            }
            logger.info(f"  找到 {len(news_list)} 則新聞")
        except Exception as e:
            logger.error(f"  掃描 {name} 新聞失敗: {e}")
            all_stock_news[name] = {
                'code': str(code),
                'news': [],
                'news_count': 0
            }

        # 避免請求過快
        time.sleep(random.uniform(1, 2))

    return all_stock_news


def calculate_sentiment_scores(all_stock_news):
    """
    用關鍵字規則計算每檔股票的情緒分數

    Args:
        all_stock_news: scan_all_stock_news() 的輸出

    Returns:
        dict: {股票名稱: {'code': ..., 'news_count': N, 'sentiment_score': float, 'bull_count': N, 'bear_count': N, 'titles': [...]}, ...}
    """
    scored_stocks = {}

    for name, data in all_stock_news.items():
        news_list = data['news']
        titles = [n['title'] for n in news_list if n.get('title')]

        # 合併所有標題計算情緒
        combined_text = ' '.join(titles)
        score = keyword_score(combined_text) if combined_text else 0

        # 個別統計看漲/看跌關鍵字命中數
        bull_hits = sum(1 for kw in BULL_KEYWORDS for t in titles if kw in t)
        bear_hits = sum(1 for kw in BEAR_KEYWORDS for t in titles if kw in t)

        scored_stocks[name] = {
            'code': data['code'],
            'news_count': data['news_count'],
            'sentiment_score': score,
            'bull_count': bull_hits,
            'bear_count': bear_hits,
            'titles': titles[:10]  # 最多保留 10 則標題
        }

    return scored_stocks


def build_gpt_summary(scored_stocks):
    """
    建立送給 GPT 的新聞摘要

    Args:
        scored_stocks: calculate_sentiment_scores() 的輸出

    Returns:
        dict: {股票名稱: [新聞標題列表], ...}（只包含有新聞的股票）
    """
    summary = {}
    for name, data in scored_stocks.items():
        if data['news_count'] > 0:
            summary[name] = data['titles']
    return summary


def select_focus_stocks_from_news(num_stocks=5):
    """
    盤前新聞選股主流程

    Returns:
        list: [{'code': '2330', 'name': '台積電', 'reason': '...', 'news_count': 5, 'sentiment_score': 0.8}, ...]
    """
    logger.info("=== 盤前新聞選股開始 ===")

    # 1. 讀取股票清單
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)
    logger.info(f"共 {len(dict_stock)} 檔股票")

    # 2. 掃描所有股票新聞
    all_stock_news = scan_all_stock_news(dict_stock)

    # 3. 計算關鍵字情緒分數
    scored_stocks = calculate_sentiment_scores(all_stock_news)

    # 4. 建立 GPT 摘要
    news_summary = build_gpt_summary(scored_stocks)

    if not news_summary:
        logger.warning("沒有收集到任何新聞，無法選股")
        # 回退：用新聞數量 + 情緒分數排序
        return _fallback_selection(scored_stocks, num_stocks)

    # 5. 用 GPT 選出焦點股票
    try:
        gpt_picks = select_top_stocks(news_summary, num_stocks=num_stocks)
        logger.info(f"GPT 選出 {len(gpt_picks)} 檔焦點股票")
    except Exception as e:
        logger.error(f"GPT 選股失敗: {e}，使用回退方案")
        return _fallback_selection(scored_stocks, num_stocks)

    # 6. 合併 GPT 結果與情緒分數
    selected = []
    for pick in gpt_picks:
        name = pick.get('name', '')
        score_data = scored_stocks.get(name, {})
        selected.append({
            'code': pick.get('code', score_data.get('code', '')),
            'name': name,
            'reason': pick.get('reason', ''),
            'news_count': score_data.get('news_count', 0),
            'sentiment_score': score_data.get('sentiment_score', 0),
            'bull_count': score_data.get('bull_count', 0),
            'bear_count': score_data.get('bear_count', 0),
        })

    logger.info(f"=== 盤前新聞選股完成，選出 {len(selected)} 檔 ===")
    return selected


def _fallback_selection(scored_stocks, num_stocks=5):
    """
    回退選股：用新聞數量 * 情緒絕對值排序

    Args:
        scored_stocks: calculate_sentiment_scores() 的輸出
        num_stocks: 要選幾檔

    Returns:
        list: 同 select_focus_stocks_from_news 格式
    """
    logger.info("使用回退方案選股（新聞數量 + 情緒強度）")

    ranked = []
    for name, data in scored_stocks.items():
        # 綜合分數 = 新聞數量 * (1 + 情緒絕對值)
        composite = data['news_count'] * (1 + abs(data['sentiment_score']))
        ranked.append((name, data, composite))

    ranked.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for name, data, composite in ranked[:num_stocks]:
        sentiment_desc = '偏多' if data['sentiment_score'] > 0 else '偏空' if data['sentiment_score'] < 0 else '中性'
        selected.append({
            'code': data['code'],
            'name': name,
            'reason': f'新聞數 {data["news_count"]} 則，情緒{sentiment_desc}（回退選股）',
            'news_count': data['news_count'],
            'sentiment_score': data['sentiment_score'],
            'bull_count': data.get('bull_count', 0),
            'bear_count': data.get('bear_count', 0),
        })

    return selected


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    print("=" * 60)
    print("盤前新聞選股測試")
    print(f"日期: {datetime.date.today()}")
    print("=" * 60)

    selected = select_focus_stocks_from_news()

    print("\n" + "=" * 60)
    print(f"選出 {len(selected)} 檔焦點股票：")
    print("=" * 60)

    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    for i, s in enumerate(selected):
        medal = medals[i] if i < len(medals) else f'{i+1}.'
        print(f"{medal} {s['name']} ({s['code']})")
        print(f"   └ 理由：{s['reason']}")
        print(f"   └ 新聞數: {s['news_count']}，情緒分數: {s['sentiment_score']:.2f}")
        print()
