#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新聞情緒分析模型訓練

功能：
1. 讀取歷史新聞 + 股價資料
2. 建立訓練資料集（新聞 → 漲/跌）
3. 訓練 ML 模型
4. 評估準確率

@author: rubylintu
"""

import os
import re
import csv
import pickle
import datetime
import logging
import numpy as np
import pandas as pd
from collections import Counter

# ML 相關
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# 設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DATA_DIR = os.path.join(SCRIPT_DIR, 'news_data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_news_data():
    """載入所有新聞歷史資料"""
    master_file = os.path.join(NEWS_DATA_DIR, 'news_history.csv')

    if not os.path.exists(master_file):
        logger.error(f"找不到資料檔案: {master_file}")
        logger.info("請先執行 news_collector.py 收集資料")
        return None

    df = pd.read_csv(master_file, encoding='utf-8')
    logger.info(f"載入 {len(df)} 筆新聞資料")
    return df


def preprocess_text(text):
    """文字預處理"""
    if pd.isna(text):
        return ""

    # 轉小寫（英文）
    text = str(text).lower()

    # 移除 URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # 移除數字（保留中文數字詞彙）
    text = re.sub(r'\d+', '', text)

    # 移除標點符號（保留中文）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)

    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def create_training_dataset(df, threshold=0.5):
    """
    建立訓練資料集

    Args:
        df: 原始資料
        threshold: 漲跌分類閾值（%）
            漲幅 > threshold → 1 (看漲)
            跌幅 < -threshold → 0 (看跌)
            其他 → 排除
    """
    logger.info("建立訓練資料集...")

    # 過濾有效資料
    df_valid = df.dropna(subset=['title', 'change_pct'])
    df_valid = df_valid[df_valid['title'].str.len() > 10]

    # 標記漲跌
    def label_change(pct):
        try:
            pct = float(pct)
            if pct > threshold:
                return 1  # 漲
            elif pct < -threshold:
                return 0  # 跌
            else:
                return -1  # 持平，排除
        except:
            return -1

    df_valid['label'] = df_valid['change_pct'].apply(label_change)

    # 排除持平
    df_labeled = df_valid[df_valid['label'] >= 0].copy()

    # 預處理文字
    df_labeled['clean_title'] = df_labeled['title'].apply(preprocess_text)
    df_labeled = df_labeled[df_labeled['clean_title'].str.len() > 5]

    logger.info(f"有效訓練樣本: {len(df_labeled)}")
    logger.info(f"漲: {sum(df_labeled['label']==1)}, 跌: {sum(df_labeled['label']==0)}")

    return df_labeled


def train_model(df_train, model_type='random_forest'):
    """
    訓練情緒分析模型

    Args:
        df_train: 訓練資料
        model_type: 模型類型
            - 'random_forest': 隨機森林
            - 'logistic': 邏輯迴歸
            - 'naive_bayes': 樸素貝葉斯
            - 'gradient_boost': 梯度提升
    """
    logger.info(f"訓練模型: {model_type}")

    X = df_train['clean_title'].values
    y = df_train['label'].values

    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigram + bigram
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 選擇模型
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'logistic': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'naive_bayes': MultinomialNB(alpha=0.1),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }

    clf = models.get(model_type, models['random_forest'])

    # 訓練
    logger.info("開始訓練...")
    clf.fit(X_train_tfidf, y_train)

    # 評估
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"模型準確率: {accuracy:.2%}")
    logger.info(f"{'='*50}")
    logger.info("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '漲']))

    logger.info("\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))

    return clf, vectorizer, accuracy


def save_model(clf, vectorizer, model_name='sentiment_model'):
    """儲存模型"""
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, f'{model_name}_vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    logger.info(f"模型已儲存: {model_path}")
    return model_path


def load_model(model_name='sentiment_model'):
    """載入模型"""
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, f'{model_name}_vectorizer.pkl')

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return clf, vectorizer


def predict_sentiment(news_title, clf=None, vectorizer=None):
    """
    預測新聞情緒

    Returns:
        prediction: 0=跌, 1=漲
        probability: 預測機率
    """
    if clf is None or vectorizer is None:
        clf, vectorizer = load_model()

    clean_text = preprocess_text(news_title)
    X = vectorizer.transform([clean_text])

    prediction = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    return prediction, proba


def analyze_important_words(clf, vectorizer, top_n=20):
    """分析最重要的多空關鍵字"""
    if hasattr(clf, 'feature_importances_'):
        # Random Forest / Gradient Boosting
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        # Logistic Regression
        importances = clf.coef_[0]
    else:
        logger.warning("此模型不支援特徵重要性分析")
        return None, None

    feature_names = vectorizer.get_feature_names_out()

    # 排序
    indices = np.argsort(importances)

    # 看漲關鍵字（正向）
    bull_indices = indices[-top_n:][::-1]
    bull_words = [(feature_names[i], importances[i]) for i in bull_indices]

    # 看跌關鍵字（負向，對於 logistic regression）
    bear_indices = indices[:top_n]
    bear_words = [(feature_names[i], importances[i]) for i in bear_indices]

    logger.info(f"\n{'='*50}")
    logger.info("看漲關鍵字 (Top 20):")
    for word, score in bull_words:
        logger.info(f"  {word}: {score:.4f}")

    logger.info(f"\n看跌關鍵字 (Top 20):")
    for word, score in bear_words:
        logger.info(f"  {word}: {score:.4f}")

    return bull_words, bear_words


def run_backtest(df, clf, vectorizer):
    """
    執行回測
    模擬根據模型預測進行交易
    """
    logger.info("\n開始回測...")

    df_test = df.copy()
    df_test['clean_title'] = df_test['title'].apply(preprocess_text)
    df_test = df_test[df_test['clean_title'].str.len() > 5]

    X = vectorizer.transform(df_test['clean_title'])
    predictions = clf.predict(X)
    probas = clf.predict_proba(X)

    df_test['prediction'] = predictions
    df_test['confidence'] = [max(p) for p in probas]

    # 只看高信心度的預測
    df_confident = df_test[df_test['confidence'] > 0.6]

    # 計算收益
    # 預測漲 → 買入，實際漲 → 賺
    # 預測跌 → 賣出/不買，實際跌 → 賺

    correct = 0
    total_return = 0

    for _, row in df_confident.iterrows():
        try:
            actual_change = float(row['change_pct'])
            predicted = row['prediction']

            if predicted == 1:  # 預測漲，買入
                total_return += actual_change
                if actual_change > 0:
                    correct += 1
            else:  # 預測跌，做空或不買
                total_return -= actual_change  # 做空收益
                if actual_change < 0:
                    correct += 1
        except:
            continue

    if len(df_confident) > 0:
        accuracy = correct / len(df_confident)
        avg_return = total_return / len(df_confident)
    else:
        accuracy = 0
        avg_return = 0

    logger.info(f"\n{'='*50}")
    logger.info("回測結果:")
    logger.info(f"  總預測次數: {len(df_confident)}")
    logger.info(f"  正確次數: {correct}")
    logger.info(f"  準確率: {accuracy:.2%}")
    logger.info(f"  平均收益: {avg_return:.2f}%")
    logger.info(f"{'='*50}")

    return accuracy, avg_return


def main():
    """主程式"""
    logger.info("=" * 60)
    logger.info("新聞情緒分析模型訓練")
    logger.info("=" * 60)

    # 1. 載入資料
    df = load_news_data()
    if df is None or len(df) < 100:
        logger.error("資料不足，請先收集至少 100 筆新聞資料")
        logger.info("執行: python news_collector.py")
        return

    # 2. 建立訓練資料集
    df_train = create_training_dataset(df, threshold=0.5)

    if len(df_train) < 50:
        logger.error("訓練樣本不足 (需要至少 50 筆)")
        return

    # 3. 訓練多個模型比較
    results = {}

    for model_type in ['random_forest', 'logistic', 'naive_bayes']:
        logger.info(f"\n{'='*50}")
        logger.info(f"訓練 {model_type} 模型")
        logger.info(f"{'='*50}")

        clf, vectorizer, accuracy = train_model(df_train, model_type)
        results[model_type] = (clf, vectorizer, accuracy)

    # 4. 選擇最佳模型
    best_model = max(results, key=lambda x: results[x][2])
    clf, vectorizer, accuracy = results[best_model]

    logger.info(f"\n最佳模型: {best_model} (準確率: {accuracy:.2%})")

    # 5. 儲存最佳模型
    save_model(clf, vectorizer)

    # 6. 分析關鍵字
    if best_model in ['random_forest', 'logistic']:
        analyze_important_words(clf, vectorizer)

    # 7. 回測
    run_backtest(df, clf, vectorizer)

    logger.info("\n訓練完成！")


if __name__ == "__main__":
    main()
