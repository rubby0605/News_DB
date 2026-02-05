#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速訓練模型
- 抓取當前新聞
- 取得過去一個月股價
- 建立訓練資料並訓練模型

@author: rubylintu
"""

import os
import sys
import time
import random
import datetime
import csv
import pickle
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from newslib import (
    read_stock_list,
    scrapBingNews,
    scrapGoogleNews,
    craw_one_month,
    getPage
)

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re

# 目錄設定
NEWS_DATA_DIR = os.path.join(SCRIPT_DIR, 'news_data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(NEWS_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def clean_text(text):
    """清理文字"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_stock_monthly_data(stock_code):
    """取得股票過去一個月的資料"""
    today = datetime.date.today()
    date_str = today.strftime('%Y%m01')

    try:
        data = craw_one_month(stock_code, date_str)

        if '很抱歉' in str(data.get('stat', '')):
            return None

        df_data = data.get('data', [])
        if not df_data or len(df_data) < 2:  # 至少需要 2 天資料
            return None

        # 計算整月漲跌
        try:
            first_close = float(str(df_data[0][6]).replace(',', ''))
            last_close = float(str(df_data[-1][6]).replace(',', ''))
        except (ValueError, IndexError) as e:
            print(f"  解析價格失敗: {e}")
            return None

        if first_close <= 0:
            return None

        monthly_change = ((last_close - first_close) / first_close) * 100

        # 計算波動性
        closes = []
        for d in df_data:
            try:
                closes.append(float(str(d[6]).replace(',', '')))
            except:
                continue

        if len(closes) < 2:
            return None

        volatility = np.std(closes) / np.mean(closes) * 100

        # 最近趨勢
        if len(closes) >= 5:
            recent_trend = ((closes[-1] - closes[-5]) / closes[-5]) * 100
        elif len(closes) >= 2:
            recent_trend = ((closes[-1] - closes[0]) / closes[0]) * 100
        else:
            recent_trend = monthly_change

        return {
            'monthly_change': monthly_change,
            'recent_trend': recent_trend,
            'volatility': volatility,
            'last_close': last_close,
            'data_points': len(df_data)
        }
    except Exception as e:
        print(f"  取得股價失敗: {e}")
        return None


def collect_news_simple(keyword):
    """簡單收集新聞"""
    news_list = []

    try:
        # Bing
        url, title, body, bs = scrapBingNews(keyword)
        if body:
            # 分割成句子
            sentences = re.split(r'[。！？\n]', body)
            for s in sentences:
                s = clean_text(s)
                if len(s) > 15 and keyword.replace(' ', '') in s or len(s) > 30:
                    news_list.append(s[:200])
    except:
        pass

    try:
        # Google
        url, title, body, bs = scrapGoogleNews(keyword)
        if body:
            sentences = re.split(r'[。！？\n]', body)
            for s in sentences:
                s = clean_text(s)
                if len(s) > 15:
                    news_list.append(s[:200])
    except:
        pass

    # 去重
    unique_news = list(set(news_list))
    return unique_news[:20]  # 最多 20 則


def build_training_data():
    """建立訓練資料"""
    print("=" * 60)
    print("建立訓練資料集")
    print("=" * 60)

    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    training_data = []

    total = len(dict_stock)

    for i, (stock_name, stock_code) in enumerate(dict_stock.items(), 1):
        print(f"\n[{i}/{total}] 處理: {stock_name} ({stock_code})")

        # 1. 取得股價資料
        print("  取得股價...")
        price_data = get_stock_monthly_data(stock_code)

        if price_data is None:
            print("  跳過（無股價資料）")
            continue

        monthly_change = price_data['monthly_change']
        recent_trend = price_data['recent_trend']

        # 判斷漲跌標籤
        # 使用最近趨勢作為標籤
        if recent_trend > 1:
            label = 1  # 漲
        elif recent_trend < -1:
            label = 0  # 跌
        else:
            print(f"  跳過（趨勢不明顯: {recent_trend:.1f}%）")
            continue

        print(f"  月漲跌: {monthly_change:.1f}%, 近期趨勢: {recent_trend:.1f}%, 標籤: {'漲' if label else '跌'}")

        # 2. 收集新聞
        print("  收集新聞...")
        news = collect_news_simple(stock_name)
        print(f"  找到 {len(news)} 則新聞")

        # 3. 加入訓練資料
        for n in news:
            if len(n) > 10:
                training_data.append({
                    'text': n,
                    'label': label,
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'change': recent_trend
                })

        # 避免請求過快
        time.sleep(random.uniform(1, 2))

    print(f"\n總共收集 {len(training_data)} 筆訓練資料")

    # 儲存
    if training_data:
        df = pd.DataFrame(training_data)
        save_path = os.path.join(NEWS_DATA_DIR, 'training_data.csv')
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"訓練資料已儲存: {save_path}")

    return training_data


def train_model(training_data):
    """訓練模型"""
    print("\n" + "=" * 60)
    print("訓練情緒分析模型")
    print("=" * 60)

    if len(training_data) < 20:
        print("訓練資料不足（需要至少 20 筆）")
        return None, None

    df = pd.DataFrame(training_data)

    # 統計
    print(f"\n資料統計:")
    print(f"  總樣本數: {len(df)}")
    print(f"  漲: {sum(df['label']==1)}")
    print(f"  跌: {sum(df['label']==0)}")

    X = df['text'].values
    y = df['label'].values

    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    print("\n建立 TF-IDF 向量...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 訓練多個模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=0.1)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, clf in models.items():
        print(f"\n訓練 {name}...")
        clf.fit(X_train_tfidf, y_train)

        y_pred = clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  準確率: {accuracy:.1%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_name = name

    print(f"\n{'='*60}")
    print(f"最佳模型: {best_name}")
    print(f"準確率: {best_accuracy:.1%}")
    print(f"{'='*60}")

    # 詳細報告
    y_pred = best_model.predict(X_test_tfidf)
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '漲']))

    # 儲存模型
    model_path = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'sentiment_model_vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n模型已儲存: {model_path}")

    # 顯示重要關鍵字
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_[0]
    else:
        importances = None

    if importances is not None:
        feature_names = vectorizer.get_feature_names_out()
        indices = np.argsort(importances)

        print("\n看漲關鍵字 (Top 10):")
        for i in indices[-10:][::-1]:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")

        print("\n看跌關鍵字 (Top 10):")
        for i in indices[:10]:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")

    return best_model, vectorizer


def quick_predict(text, clf, vectorizer):
    """快速預測"""
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    return '漲' if pred == 1 else '跌', max(proba)


def main():
    print("=" * 60)
    print("快速訓練 AI 股票預測模型")
    print(f"時間: {datetime.datetime.now()}")
    print("=" * 60)

    # 1. 建立訓練資料
    training_data = build_training_data()

    if not training_data:
        print("\n無法建立訓練資料")
        return

    # 2. 訓練模型
    clf, vectorizer = train_model(training_data)

    if clf is None:
        print("\n訓練失敗")
        return

    # 3. 測試預測
    print("\n" + "=" * 60)
    print("測試預測")
    print("=" * 60)

    test_news = [
        "台積電獲大單，營收創新高",
        "公司宣布裁員，股價承壓",
        "法人看好後市，持續加碼",
        "營收衰退，獲利不如預期",
        "新產品上市，市場反應熱烈"
    ]

    for news in test_news:
        pred, conf = quick_predict(news, clf, vectorizer)
        print(f"  [{pred}] {conf:.0%} - {news}")

    print("\n✅ 訓練完成！")
    print("現在可以執行: python predict_stock.py 台積電")


if __name__ == "__main__":
    main()
