#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT 新聞情緒分析模組
使用 OpenAI API 分析股票新聞對股價的影響

@author: rubylintu
"""

import os
import json
import time
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'openai_config.json')


def load_config():
    """載入 OpenAI 設定"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(config):
    """儲存設定"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def set_api_key(api_key):
    """設定 OpenAI API Key"""
    config = load_config()
    config['api_key'] = api_key
    save_config(config)
    print("OpenAI API Key 已設定！")


def get_client():
    """取得 OpenAI Client"""
    config = load_config()
    api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("尚未設定 OpenAI API Key。請執行: python gpt_sentiment.py --setup YOUR_API_KEY")

    return OpenAI(api_key=api_key)


def analyze_sentiment(news_text, stock_name, model="gpt-3.5-turbo"):
    """
    分析單則新聞的情緒

    Args:
        news_text: 新聞內容
        stock_name: 股票名稱
        model: 使用的模型 (gpt-3.5-turbo / gpt-4o-mini)

    Returns:
        dict: {
            'sentiment': '漲' / '跌' / '中性',
            'confidence': 0.0-1.0,
            'reason': '分析理由'
        }
    """
    client = get_client()

    prompt = f"""你是一位專業的台股分析師。請分析以下關於「{stock_name}」的新聞對股價的短期影響。

新聞內容：
{news_text[:500]}

請用以下 JSON 格式回答（只回答 JSON，不要其他文字）：
{{
    "sentiment": "漲" 或 "跌" 或 "中性",
    "confidence": 0.0 到 1.0 之間的數字（表示信心程度）,
    "reason": "簡短說明理由（20字以內）"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()

        # 解析 JSON
        # 處理可能的 markdown 格式
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]

        result = json.loads(result_text)

        return {
            'sentiment': result.get('sentiment', '中性'),
            'confidence': float(result.get('confidence', 0.5)),
            'reason': result.get('reason', ''),
            'model': model
        }

    except json.JSONDecodeError:
        # 如果 JSON 解析失敗，嘗試從文字中提取
        if '漲' in result_text:
            return {'sentiment': '漲', 'confidence': 0.6, 'reason': '關鍵字判斷', 'model': model}
        elif '跌' in result_text:
            return {'sentiment': '跌', 'confidence': 0.6, 'reason': '關鍵字判斷', 'model': model}
        else:
            return {'sentiment': '中性', 'confidence': 0.5, 'reason': '無法判斷', 'model': model}

    except Exception as e:
        print(f"GPT 分析錯誤: {e}")
        return {'sentiment': '中性', 'confidence': 0.0, 'reason': f'錯誤: {e}', 'model': model}


def analyze_multiple_news(news_list, stock_name, model="gpt-3.5-turbo"):
    """
    分析多則新聞並綜合判斷

    Args:
        news_list: [{'title': '...', 'content': '...'}, ...]
        stock_name: 股票名稱

    Returns:
        dict: 綜合分析結果
    """
    if not news_list:
        return {'sentiment': '中性', 'confidence': 0.0, 'reason': '無新聞資料'}

    client = get_client()

    # 組合新聞摘要
    news_summary = "\n".join([
        f"- {news.get('title', news.get('content', '')[:50])}"
        for news in news_list[:5]  # 最多 5 則
    ])

    prompt = f"""你是一位專業的台股分析師。請綜合分析以下關於「{stock_name}」的多則新聞，判斷對股價的短期影響。

近期新聞標題：
{news_summary}

請用以下 JSON 格式回答（只回答 JSON）：
{{
    "sentiment": "漲" 或 "跌" 或 "中性",
    "confidence": 0.0 到 1.0,
    "reason": "綜合分析理由（30字以內）",
    "key_news": "最重要的一則新聞標題"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()

        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]

        result = json.loads(result_text)

        return {
            'sentiment': result.get('sentiment', '中性'),
            'confidence': float(result.get('confidence', 0.5)),
            'reason': result.get('reason', ''),
            'key_news': result.get('key_news', ''),
            'model': model,
            'news_count': len(news_list)
        }

    except Exception as e:
        print(f"GPT 綜合分析錯誤: {e}")
        return {'sentiment': '中性', 'confidence': 0.0, 'reason': f'錯誤: {e}'}


def analyze_stock_with_news(stock_name, model="gpt-3.5-turbo"):
    """
    自動抓取新聞並分析股票情緒

    Args:
        stock_name: 股票名稱

    Returns:
        dict: 分析結果
    """
    from newslib import scrapBingNews, scrapGoogleNews

    news_list = []

    # 嘗試從 Bing 抓新聞
    try:
        url, title, body, bs = scrapBingNews(stock_name)
        if title:
            news_list.append({'title': title, 'content': body[:200] if body else ''})
    except:
        pass

    # 嘗試從 Google 抓新聞
    try:
        url, title, body, bs = scrapGoogleNews(stock_name)
        if title and title not in [n['title'] for n in news_list]:
            news_list.append({'title': title, 'content': body[:200] if body else ''})
    except:
        pass

    if not news_list:
        return {
            'stock': stock_name,
            'sentiment': '中性',
            'confidence': 0.0,
            'reason': '無法取得新聞'
        }

    result = analyze_multiple_news(news_list, stock_name, model)
    result['stock'] = stock_name

    return result


def batch_analyze_stocks(stock_names, model="gpt-3.5-turbo", delay=1.0):
    """
    批次分析多檔股票

    Args:
        stock_names: 股票名稱列表
        model: 使用的模型
        delay: 每次請求間隔（秒）

    Returns:
        list: 分析結果列表
    """
    results = []

    for i, name in enumerate(stock_names):
        print(f"[{i+1}/{len(stock_names)}] 分析 {name}...")

        result = analyze_stock_with_news(name, model)
        results.append(result)

        if i < len(stock_names) - 1:
            time.sleep(delay)

    return results


def test_api():
    """測試 API 連線"""
    print("測試 OpenAI API 連線...")

    try:
        client = get_client()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "說 OK"}],
            max_tokens=5
        )

        print(f"API 連線成功！回應: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"API 連線失敗: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup' and len(sys.argv) > 2:
            set_api_key(sys.argv[2])
        elif sys.argv[1] == '--test':
            test_api()
        elif sys.argv[1] == '--analyze' and len(sys.argv) > 2:
            stock_name = sys.argv[2]
            print(f"分析 {stock_name}...")
            result = analyze_stock_with_news(stock_name)
            print(f"\n結果:")
            print(f"  情緒: {result['sentiment']}")
            print(f"  信心度: {result.get('confidence', 0):.0%}")
            print(f"  理由: {result.get('reason', '')}")
        else:
            print("用法:")
            print("  設定 API Key: python gpt_sentiment.py --setup YOUR_API_KEY")
            print("  測試連線:     python gpt_sentiment.py --test")
            print("  分析股票:     python gpt_sentiment.py --analyze 台積電")
    else:
        print("GPT 新聞情緒分析模組")
        print("")
        print("設定方式:")
        print("  python gpt_sentiment.py --setup YOUR_OPENAI_API_KEY")
        print("")
        print("測試連線:")
        print("  python gpt_sentiment.py --test")
        print("")
        print("分析股票:")
        print("  python gpt_sentiment.py --analyze 台積電")
