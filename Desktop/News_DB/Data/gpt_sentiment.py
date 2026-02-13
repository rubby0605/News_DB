#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞情緒分析模組
使用 Gemini 2.5 Flash（免費）分析股票新聞對股價的影響
GPT client 保留供 ai_trader.py 使用

@author: rubylintu
"""

import os
import json
import re
import time
import urllib.request
import urllib.error
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


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def _call_gemini(prompt):
    """用 Gemini REST API 做情緒分析（免費）"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY 環境變數未設定")

    url = GEMINI_API_URL + f"?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 500,
            "responseMimeType": "application/json",
        },
    }

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    text = result['candidates'][0]['content']['parts'][0]['text']
    # 清理 trailing comma
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


def analyze_sentiment(news_text, stock_name, model="gemini"):
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
    prompt = f"""你是一位專業的台股分析師。請分析以下關於「{stock_name}」的新聞對股價的短期影響。

新聞內容：
{news_text[:500]}

請用以下 JSON 格式回答：
{{"sentiment": "漲" 或 "跌" 或 "中性", "confidence": 0.0到1.0, "reason": "簡短理由20字以內"}}"""

    try:
        result = _call_gemini(prompt)
        return {
            'sentiment': result.get('sentiment', '中性'),
            'confidence': float(result.get('confidence', 0.5)),
            'reason': result.get('reason', ''),
            'model': 'gemini-2.5-flash'
        }
    except Exception as e:
        print(f"Gemini 情緒分析錯誤: {e}")
        return {'sentiment': '中性', 'confidence': 0.0, 'reason': f'錯誤: {e}', 'model': 'gemini-error'}


def analyze_multiple_news(news_list, stock_name, model="gemini"):
    """
    分析多則新聞並綜合判斷（使用 Gemini）
    """
    if not news_list:
        return {'sentiment': '中性', 'confidence': 0.0, 'reason': '無新聞資料'}

    news_summary = "\n".join([
        f"- {news.get('title', news.get('content', '')[:50])}"
        for news in news_list[:5]
    ])

    prompt = f"""你是一位專業的台股分析師。請綜合分析以下關於「{stock_name}」的多則新聞，判斷對股價的短期影響。

近期新聞標題：
{news_summary}

請用 JSON 格式回答：
{{"sentiment": "漲"或"跌"或"中性", "confidence": 0.0到1.0, "reason": "綜合理由30字以內", "key_news": "最重要的一則新聞標題"}}"""

    try:
        result = _call_gemini(prompt)
        return {
            'sentiment': result.get('sentiment', '中性'),
            'confidence': float(result.get('confidence', 0.5)),
            'reason': result.get('reason', ''),
            'key_news': result.get('key_news', ''),
            'model': 'gemini-2.5-flash',
            'news_count': len(news_list)
        }
    except Exception as e:
        print(f"Gemini 綜合分析錯誤: {e}")
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


def batch_analyze_stocks(stock_names, model="gemini", delay=1.0):
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


def select_top_stocks(all_news_summary, num_stocks=5, model="gemini"):
    """
    從所有股票的新聞摘要中選出最值得關注的股票（使用 Gemini）
    """
    stock_code_map = {}
    try:
        from newslib import read_stock_list
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stock_list_file = os.path.join(script_dir, 'stock_list_less.txt')
        dict_stock = read_stock_list(stock_list_file)
        stock_code_map = {name: str(code) for name, code in dict_stock.items()}
    except Exception:
        pass

    summary_lines = []
    for name, titles in all_news_summary.items():
        code = stock_code_map.get(name, '?')
        if titles:
            title_text = '；'.join(titles[:5])
            summary_lines.append(f"• {name}({code}) [{len(titles)}則]: {title_text}")

    if not summary_lines:
        return []

    news_text = '\n'.join(summary_lines)

    prompt = f"""你是一位專業的台股短線交易分析師。以下是今天 {len(all_news_summary)} 檔股票的新聞標題摘要。

請從中選出 {num_stocks} 檔「今天最有短線交易機會」的股票。

選股標準：
1. 新聞熱度（新聞數量多代表市場關注度高）
2. 情緒強度（明顯利多或利空，而非中性新聞）
3. 異常信號（突發事件、重大消息、法人動向等）
4. 優先選擇有明確方向性的股票（不論漲跌都算）

各股票新聞摘要：
{news_text}

請用 JSON 格式回答：
{{"selected": [{{"name": "股票名稱", "code": "股票代號", "reason": "選中理由30字以內"}}]}}
共選 {num_stocks} 檔。"""

    try:
        result = _call_gemini(prompt)
        selected = result.get('selected', [])

        for item in selected:
            if not item.get('code') or item['code'] == '?':
                item['code'] = stock_code_map.get(item.get('name', ''), '')

        return selected[:num_stocks]

    except Exception as e:
        print(f"Gemini 選股錯誤: {e}")
        return []


def test_api():
    """測試 Gemini API 連線"""
    print("測試 Gemini API 連線...")

    try:
        result = _call_gemini('回覆 JSON: {"status": "ok"}')
        print(f"Gemini API 連線成功！回應: {result}")
        return True
    except Exception as e:
        print(f"Gemini API 連線失敗: {e}")
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
