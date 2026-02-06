#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票漲跌預測

使用訓練好的情緒分析模型，即時預測股票走勢

@author: rubylintu
"""

import os
import sys
import pickle
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from newslib import read_stock_list, scrapBingNews, scrapGoogleNews
from news_collector import extract_news_from_bing, extract_news_from_google, clean_text
from hybrid_predictor import hybrid_predict, analyze_news, load_ml_model

MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')


def predict_stock(stock_name, stock_code=None):
    """
    預測單一股票的漲跌（使用混合模型）

    Args:
        stock_name: 股票名稱（如 "台積電"）
        stock_code: 股票代號（可選）

    Returns:
        prediction: "漲" 或 "跌"
        confidence: 信心度
        news_list: 分析的新聞列表
    """
    # 載入模型
    clf, vectorizer = load_ml_model()

    # 收集新聞
    print(f"\n搜尋 {stock_name} 相關新聞...")

    news_list = []
    news_list.extend(extract_news_from_bing(stock_name, stock_code or ''))
    news_list.extend(extract_news_from_google(stock_name, stock_code or ''))

    if not news_list:
        print("找不到相關新聞")
        return None, None, []

    print(f"找到 {len(news_list)} 則新聞\n")

    # 使用混合預測
    news_texts = [n['title'] for n in news_list if len(n.get('title', '')) > 5]

    if not news_texts:
        print("無法分析新聞")
        return None, None, []

    # 分析每則新聞
    predictions = []
    bull_count = 0
    bear_count = 0

    for title in news_texts:
        pred, conf, details = hybrid_predict(title, clf, vectorizer)
        predictions.append({
            'title': title,
            'prediction': pred,
            'confidence': conf,
            'keyword_score': details['keyword_score']
        })

        if pred == '漲':
            bull_count += 1
        elif pred == '跌':
            bear_count += 1

    # 綜合判斷
    total = bull_count + bear_count
    if total == 0:
        final_prediction = '持平'
        final_confidence = 0.5
    elif bull_count > bear_count * 1.2:
        final_prediction = '漲'
        final_confidence = bull_count / total
    elif bear_count > bull_count * 1.2:
        final_prediction = '跌'
        final_confidence = bear_count / total
    else:
        final_prediction = '持平'
        final_confidence = 0.5

    # 輸出結果
    print("=" * 60)
    print(f"【{stock_name}】預測結果")
    print("=" * 60)
    print(f"  預測走勢: {final_prediction}")
    print(f"  信心度: {final_confidence:.1%}")
    print(f"  看漲新聞: {bull_count} 則")
    print(f"  看跌新聞: {bear_count} 則")
    print("=" * 60)

    # 顯示重要新聞
    print("\n重要新聞:")
    # 按關鍵字分數排序
    sorted_preds = sorted(predictions, key=lambda x: abs(x['keyword_score']), reverse=True)[:5]
    for i, p in enumerate(sorted_preds, 1):
        kw = p['keyword_score']
        print(f"{i}. [{p['prediction']}] (關鍵字:{kw:+.1f}) {p['title'][:45]}...")

    return final_prediction, final_confidence, predictions


def predict_all_stocks():
    """預測所有關注的股票"""
    # 讀取股票清單
    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    results = []

    print("\n" + "=" * 60)
    print(f"股票漲跌預測 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    for stock_name, stock_code in dict_stock.items():
        print(f"\n處理: {stock_name} ({stock_code})")
        prediction, confidence, _ = predict_stock(stock_name, stock_code)

        if prediction:
            results.append({
                'name': stock_name,
                'code': stock_code,
                'prediction': prediction,
                'confidence': confidence
            })

    # 彙總報告
    print("\n" + "=" * 60)
    print("彙總報告")
    print("=" * 60)

    # 按信心度排序
    results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)

    print("\n【看漲股票】(信心度 > 60%)")
    for r in results_sorted:
        if r['prediction'] == '漲' and r['confidence'] > 0.6:
            print(f"  {r['name']} ({r['code']}): {r['confidence']:.1%}")

    print("\n【看跌股票】(信心度 > 60%)")
    for r in results_sorted:
        if r['prediction'] == '跌' and r['confidence'] > 0.6:
            print(f"  {r['name']} ({r['code']}): {r['confidence']:.1%}")

    return results


def interactive_mode():
    """互動模式"""
    print("\n股票漲跌預測系統")
    print("=" * 40)
    print("輸入股票名稱進行預測")
    print("輸入 'all' 預測所有關注股票")
    print("輸入 'quit' 退出")
    print("=" * 40)

    while True:
        try:
            user_input = input("\n請輸入股票名稱: ").strip()

            if user_input.lower() == 'quit':
                print("再見！")
                break
            elif user_input.lower() == 'all':
                predict_all_stocks()
            elif user_input:
                predict_stock(user_input)

        except KeyboardInterrupt:
            print("\n再見！")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            predict_all_stocks()
        else:
            stock_name = ' '.join(sys.argv[1:])
            predict_stock(stock_name)
    else:
        interactive_mode()
