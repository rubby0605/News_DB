#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合預測模型
結合 ML 模型 + 關鍵字規則，提高預測準確度

@author: rubylintu
"""

import os
import re
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 看漲關鍵字
BULL_KEYWORDS = [
    '創新高', '大漲', '漲停', '利多', '看好', '加碼', '買超', '獲利',
    '成長', '突破', '強勢', '熱銷', '訂單', '營收增', '獲大單', '擴產',
    '上調', '優於預期', '利好', '反彈', '站穩', '新高', '飆漲', '噴出',
    '多頭', '牛市', '爆發', '亮眼', '佳績', '需求強', '供不應求',
    '法人看好', '外資買', '投信買', '主力進場', '底部訊號'
]

# 看跌關鍵字
BEAR_KEYWORDS = [
    '下跌', '跌停', '利空', '衰退', '虧損', '裁員', '暴跌', '崩盤',
    '賣超', '減持', '下修', '警訊', '危機', '低迷', '疲軟', '不如預期',
    '產業寒冬', '需求減', '庫存高', '價格下跌', '毛利降', '營收衰',
    '外資賣', '法人調節', '主力出貨', '空頭', '熊市', '破底',
    '跳水', '殺盤', '恐慌', '風險', '泡沫', '清算'
]


def _get_keywords():
    """取得關鍵字（優先使用 GA 優化版）"""
    try:
        from keyword_optimizer import load_optimized_keywords
        result = load_optimized_keywords()
        if result:
            return result  # (bull_keywords, bear_keywords)
    except Exception:
        pass
    return BULL_KEYWORDS, BEAR_KEYWORDS


def keyword_score(text):
    """
    計算關鍵字分數
    正分 = 看漲，負分 = 看跌
    """
    bull_kw, bear_kw = _get_keywords()
    bull_count = sum(1 for kw in bull_kw if kw in text)
    bear_count = sum(1 for kw in bear_kw if kw in text)

    # 計算分數 (-1 到 1)
    total = bull_count + bear_count
    if total == 0:
        return 0

    score = (bull_count - bear_count) / total
    return score


def load_ml_model():
    """載入 ML 模型"""
    MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
    model_path = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'sentiment_model_vectorizer.pkl')

    if not os.path.exists(model_path):
        return None, None

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return clf, vectorizer


def hybrid_predict(text, clf=None, vectorizer=None):
    """
    混合預測
    結合 ML 模型 (40%) + 關鍵字規則 (60%)
    """
    # 1. 關鍵字分數
    kw_score = keyword_score(text)

    # 2. ML 預測
    ml_score = 0
    ml_confidence = 0.5

    if clf is not None and vectorizer is not None:
        try:
            X = vectorizer.transform([text])
            pred = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]

            ml_score = 1 if pred == 1 else -1
            ml_confidence = max(proba)
        except:
            pass

    # 3. 混合分數
    # 關鍵字權重 60%，ML 權重 40%
    if kw_score != 0:
        # 有關鍵字匹配時，關鍵字為主
        final_score = kw_score * 0.6 + (ml_score * ml_confidence) * 0.4
    else:
        # 無關鍵字匹配時，使用 ML
        final_score = ml_score * ml_confidence

    # 計算最終信心度
    confidence = min(abs(final_score) * 0.8 + 0.5, 0.95)

    # 判斷漲跌
    if final_score > 0.1:
        prediction = '漲'
    elif final_score < -0.1:
        prediction = '跌'
    else:
        prediction = '持平'

    return prediction, confidence, {
        'keyword_score': kw_score,
        'ml_score': ml_score,
        'ml_confidence': ml_confidence,
        'final_score': final_score
    }


def analyze_news(news_list):
    """
    分析多則新聞，給出綜合判斷
    """
    clf, vectorizer = load_ml_model()

    results = []
    bull_count = 0
    bear_count = 0
    neutral_count = 0

    for news in news_list:
        pred, conf, details = hybrid_predict(news, clf, vectorizer)
        results.append({
            'text': news,
            'prediction': pred,
            'confidence': conf,
            'details': details
        })

        if pred == '漲':
            bull_count += 1
        elif pred == '跌':
            bear_count += 1
        else:
            neutral_count += 1

    # 綜合判斷
    total = len(news_list)
    if total == 0:
        return None, 0, []

    if bull_count > bear_count * 1.5:
        overall = '看漲'
        overall_conf = bull_count / total
    elif bear_count > bull_count * 1.5:
        overall = '看跌'
        overall_conf = bear_count / total
    else:
        overall = '中性'
        overall_conf = 0.5

    return overall, overall_conf, results


def main():
    """測試混合預測"""
    print("=" * 60)
    print("混合預測模型測試")
    print("=" * 60)

    clf, vectorizer = load_ml_model()

    test_cases = [
        "台積電獲得蘋果大單，營收創歷史新高",
        "公司宣布大規模裁員，股價承壓下跌",
        "法人持續看好後市，外資連續買超",
        "營收大幅衰退，獲利不如預期",
        "新產品上市熱銷，訂單供不應求",
        "面板報價持續下跌，產業前景悲觀",
        "公司今日召開股東會",
        "外資大舉賣超，股價跌停",
        "突破歷史新高，多頭氣勢強勁",
        "庫存水位過高，需求疲軟"
    ]

    print("\n預測結果:")
    print("-" * 60)

    for news in test_cases:
        pred, conf, details = hybrid_predict(news, clf, vectorizer)
        kw = details['keyword_score']
        print(f"[{pred}] {conf:.0%} (關鍵字:{kw:+.1f}) - {news[:35]}...")

    print("\n" + "=" * 60)
    print("綜合分析測試")
    print("=" * 60)

    # 測試綜合分析
    bull_news = [
        "台積電營收創新高",
        "外資連續買超",
        "法人看好後市"
    ]

    bear_news = [
        "公司裁員",
        "營收衰退",
        "股價暴跌"
    ]

    overall, conf, _ = analyze_news(bull_news)
    print(f"\n看漲新聞組: {overall} ({conf:.0%})")

    overall, conf, _ = analyze_news(bear_news)
    print(f"看跌新聞組: {overall} ({conf:.0%})")


if __name__ == "__main__":
    main()
