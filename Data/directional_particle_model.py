#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向性粒子預測模型
基於原本的粒子模擬，加入趨勢信號產生方向偏移

@author: rubylintu
"""

import math
import random
import json
import numpy as np
import pandas as pd
import requests
import datetime
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 資料抓取函數
# ============================================================

def get_institutional_data(date=None, retry=0):
    """
    抓取三大法人買賣超資料
    來源：證交所 API

    Returns:
        dict: {股票代號: {'foreign': 外資, 'investment': 投信, 'dealer': 自營商}}
    """
    if retry > 30:  # 最多嘗試 30 天
        print("無法取得法人資料（嘗試超過30天）")
        return {}

    if date is None:
        # 自動找最近有資料的交易日（從今天往前找）
        today = datetime.date.today()
        date = today

    date_str = date.strftime('%Y%m%d')
    url = f'https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json'

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if data.get('stat') != 'OK' or 'data' not in data:
            # 嘗試前一天
            prev_date = date - datetime.timedelta(days=1)
            return get_institutional_data(prev_date, retry + 1)

        result = {}
        for row in data['data']:
            code = row[0].strip()
            # 原始資料是「股」，轉換成「張」(1張=1000股)
            foreign = int(row[4].replace(',', '')) // 1000 if row[4] != '--' else 0
            investment = int(row[10].replace(',', '')) // 1000 if row[10] != '--' else 0
            dealer = int(row[11].replace(',', '')) // 1000 if row[11] != '--' else 0

            result[code] = {
                'foreign': foreign,
                'investment': investment,
                'dealer': dealer,
                'total': foreign + investment + dealer
            }

        print(f"取得 {len(result)} 檔股票法人資料 ({date_str})")
        return result

    except Exception as e:
        print(f"抓取法人資料失敗: {e}")
        return {}


def get_stock_history(stock_code, days=20):
    """
    抓取股票歷史價格
    同時支援上市(TWSE)與上櫃(TPEX)

    Returns:
        list: [{'date', 'open', 'high', 'low', 'close', 'volume'}, ...]
    """
    result = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    # 自動計算最近兩個月（從今天往前）
    today = datetime.date.today()
    base_dates = []
    for month_offset in range(2):
        target_month = today.month - month_offset
        target_year = today.year
        if target_month <= 0:
            target_month += 12
            target_year -= 1
        base_dates.append(f'{target_year}{target_month:02d}01')

    # 1. 先試 TWSE (上市)
    for date_str in base_dates:
        url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_code}'

        try:
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            if data.get('stat') == 'OK' and 'data' in data:
                for row in data['data']:
                    try:
                        date = row[0]
                        volume = int(row[1].replace(',', ''))
                        open_p = float(row[3].replace(',', '')) if row[3] != '--' else 0
                        high = float(row[4].replace(',', '')) if row[4] != '--' else 0
                        low = float(row[5].replace(',', '')) if row[5] != '--' else 0
                        close = float(row[6].replace(',', '')) if row[6] != '--' else 0

                        if close > 0:
                            result.append({
                                'date': date,
                                'open': open_p,
                                'high': high,
                                'low': low,
                                'close': close,
                                'volume': volume
                            })
                    except:
                        continue

            time.sleep(0.3)

        except Exception as e:
            print(f"TWSE {stock_code} {date_str}: {e}")
            continue

    # 如果 TWSE 沒資料，試 TPEX (上櫃)
    if not result:
        # 自動計算 TPEX 日期格式
        tpex_dates = []
        for month_offset in range(2):
            target_month = today.month - month_offset
            target_year = today.year
            if target_month <= 0:
                target_month += 12
                target_year -= 1
            tpex_dates.append(f'{target_year}/{target_month:02d}/01')

        for date_str in tpex_dates:
            url = f'https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock?id={stock_code}&date={date_str}'

            try:
                response = requests.get(url, headers=headers, timeout=10)
                data = response.json()

                if data.get('stat') == 'ok':
                    tables = data.get('tables', [{}])
                    rows = tables[0].get('data', []) if tables else []

                    for row in rows:
                        try:
                            # TPEX 格式: [日期, 成交仟股, 成交仟元, 開盤, 最高, 最低, 收盤, 漲跌, 筆數]
                            date = row[0]
                            volume = int(float(row[1].replace(',', '')) * 1000) if row[1] != '--' else 0
                            open_p = float(row[3].replace(',', '')) if row[3] != '--' else 0
                            high = float(row[4].replace(',', '')) if row[4] != '--' else 0
                            low = float(row[5].replace(',', '')) if row[5] != '--' else 0
                            close = float(row[6].replace(',', '')) if row[6] != '--' else 0

                            if close > 0:
                                result.append({
                                    'date': date,
                                    'open': open_p,
                                    'high': high,
                                    'low': low,
                                    'close': close,
                                    'volume': volume
                                })
                        except:
                            continue

                time.sleep(0.3)

            except Exception as e:
                print(f"TPEX {stock_code} {date_str}: {e}")
                continue

    # 按日期排序，取最近 N 天
    result.sort(key=lambda x: x['date'])
    return result[-days:] if len(result) > days else result


# ============================================================
# 技術指標計算
# ============================================================

def calc_ema(prices, period):
    """計算指數移動平均線"""
    if len(prices) < period:
        return prices[-1] if prices else 0

    multiplier = 2 / (period + 1)
    ema = prices[0]

    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calc_rsi(prices, period=14):
    """計算 RSI 指標"""
    if len(prices) < period + 1:
        return 50  # 預設中性

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calc_momentum(prices, days=5):
    """計算價格動量 (%)"""
    if len(prices) < days:
        return 0

    return (prices[-1] - prices[-days]) / prices[-days] * 100


def calc_volatility(prices, days=10):
    """計算波動率 (標準差)"""
    if len(prices) < days:
        return 0

    recent = prices[-days:]
    return np.std(recent) / np.mean(recent) * 100


# ============================================================
# 權重載入
# ============================================================

WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'optimized_weights.json')

def load_optimized_weights():
    """載入優化後的權重"""
    default_weights = {
        'foreign_large': 3000,
        'foreign_medium': 1000,
        'foreign_weight': 4,
        'momentum_weight': 2,
        'ema_weight': 2,
        'momentum_threshold': 3
    }

    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                weights = data.get('weights', default_weights)
                print(f"使用優化權重 (準確率: {data.get('accuracy', 0):.1%})")
                return weights
        except:
            pass

    print("使用預設權重")
    return default_weights


# ============================================================
# 方向偏移計算
# ============================================================

def calc_directional_bias(stock_code, institutional_data, history, weights=None):
    """
    計算方向偏移量（使用優化權重）

    Returns:
        float: 偏移量 (-10 到 +10)
        dict: 詳細信號
    """
    # 載入權重
    if weights is None:
        weights = load_optimized_weights()

    bias = 0
    signals = {}

    # 1. 法人買賣超
    if stock_code in institutional_data:
        inst = institutional_data[stock_code]
        foreign = inst['foreign']
        total = inst['total']

        # 使用優化後的門檻和權重
        if foreign > weights['foreign_large']:
            bias += weights['foreign_weight']
            signals['foreign'] = f'外資大買 +{foreign} 張'
        elif foreign > weights['foreign_medium']:
            bias += weights['foreign_weight'] * 0.5
            signals['foreign'] = f'外資買超 +{foreign} 張'
        elif foreign < -weights['foreign_large']:
            bias -= weights['foreign_weight']
            signals['foreign'] = f'外資大賣 {foreign} 張'
        elif foreign < -weights['foreign_medium']:
            bias -= weights['foreign_weight'] * 0.5
            signals['foreign'] = f'外資賣超 {foreign} 張'
        else:
            signals['foreign'] = f'外資 {foreign:+d} 張 (中性)'

        # 三大法人合計
        if total > 5000:
            bias += 1
        elif total < -5000:
            bias -= 1
    else:
        signals['foreign'] = '無法人資料'

    # 2. 價格動量
    if history:
        closes = [d['close'] for d in history]

        momentum_5d = calc_momentum(closes, 5)
        momentum_10d = calc_momentum(closes, 10)

        # 使用優化後的動量權重和門檻
        if momentum_5d > weights['momentum_threshold'] * 2:
            bias += weights['momentum_weight']
        elif momentum_5d > weights['momentum_threshold']:
            bias += weights['momentum_weight'] * 0.5
        elif momentum_5d < -weights['momentum_threshold'] * 2:
            bias -= weights['momentum_weight']
        elif momentum_5d < -weights['momentum_threshold']:
            bias -= weights['momentum_weight'] * 0.5

        signals['momentum'] = f'5日動量 {momentum_5d:+.1f}%'

        # 10日動量加成
        if momentum_10d > 10:
            bias += 0.5
        elif momentum_10d < -10:
            bias -= 0.5

    # 3. 均線排列
    if history and len(history) >= 20:
        closes = [d['close'] for d in history]

        ema5 = calc_ema(closes, 5)
        ema10 = calc_ema(closes, 10)
        ema20 = calc_ema(closes, 20)
        current = closes[-1]

        # 多頭排列: 股價 > EMA5 > EMA10 > EMA20（使用優化後的均線權重）
        if current > ema5 > ema10 > ema20:
            bias += weights['ema_weight']
            signals['ema'] = '多頭排列'
        elif current > ema5 > ema10:
            bias += weights['ema_weight'] * 0.5
            signals['ema'] = '短多排列'
        # 空頭排列: 股價 < EMA5 < EMA10 < EMA20
        elif current < ema5 < ema10 < ema20:
            bias -= weights['ema_weight']
            signals['ema'] = '空頭排列'
        elif current < ema5 < ema10:
            bias -= weights['ema_weight'] * 0.5
            signals['ema'] = '短空排列'
        else:
            signals['ema'] = '均線糾結'

    # 4. RSI 指標 (權重 10%)
    if history and len(history) >= 14:
        closes = [d['close'] for d in history]
        rsi = calc_rsi(closes)

        if rsi > 70:
            bias -= 0.5  # 超買，可能回檔
            signals['rsi'] = f'RSI={rsi:.0f} (超買)'
        elif rsi > 50:
            bias += 0.5
            signals['rsi'] = f'RSI={rsi:.0f} (偏多)'
        elif rsi < 30:
            bias += 0.5  # 超賣，可能反彈
            signals['rsi'] = f'RSI={rsi:.0f} (超賣)'
        elif rsi < 50:
            bias -= 0.5
            signals['rsi'] = f'RSI={rsi:.0f} (偏空)'

    # 限制在 -10 到 +10
    bias = max(-10, min(10, bias))

    return bias, signals


# ============================================================
# 粒子模型
# ============================================================

class DirectionalParticle:
    """方向性粒子"""

    def __init__(self, base_price, bias=0, volatility=2):
        """
        Args:
            base_price: 基準價格
            bias: 方向偏移 (-10 到 +10)
            volatility: 波動率
        """
        self.base_price = base_price
        self.bias = bias
        self.volatility = volatility
        self.predicted_price = None
        self.generate()

    def generate(self):
        """生成粒子預測價格"""
        # μ = bias% 的基準價格
        mu = self.base_price * (self.bias / 100)

        # σ = volatility% 的基準價格
        sigma = self.base_price * (self.volatility / 100)

        # 高斯隨機偏移
        offset = random.gauss(mu, sigma)

        self.predicted_price = self.base_price + offset


class DirectionalParticleModel:
    """方向性粒子預測模型"""

    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.institutional_data = None
        self.last_fetch_date = None

    def fetch_market_data(self):
        """抓取市場資料"""
        # 只在需要時抓取（一次）
        if self.institutional_data is None:
            print("抓取三大法人資料...")
            self.institutional_data = get_institutional_data()
            self.last_fetch_date = datetime.date.today()

    def predict(self, stock_code, stock_name=None, current_price=None):
        """
        預測股票價格

        Args:
            stock_code: 股票代號
            stock_name: 股票名稱（可選）
            current_price: 當前價格（可選，會自動抓取）

        Returns:
            dict: 預測結果
        """
        self.fetch_market_data()

        # 抓取歷史資料
        history = get_stock_history(stock_code, days=30)

        if not history:
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'error': '無法取得歷史資料'
            }

        # 取得當前價格
        if current_price is None:
            current_price = history[-1]['close']

        # 計算方向偏移
        bias, signals = calc_directional_bias(
            stock_code,
            self.institutional_data or {},
            history
        )

        # 計算波動率
        closes = [d['close'] for d in history]
        volatility = calc_volatility(closes, 10)
        volatility = max(1, min(5, volatility))  # 限制在 1-5%

        # 生成粒子
        particles = []
        for _ in range(self.n_particles):
            p = DirectionalParticle(current_price, bias, volatility)
            particles.append(p.predicted_price)

        # 統計預測結果
        particles = np.array(particles)
        predicted_mean = np.mean(particles)
        predicted_std = np.std(particles)

        # 計算機率
        prob_up = np.sum(particles > current_price) / len(particles)
        prob_down = np.sum(particles < current_price) / len(particles)

        # 預測方向
        if prob_up > 0.6:
            direction = '漲'
            confidence = prob_up
        elif prob_down > 0.6:
            direction = '跌'
            confidence = prob_down
        else:
            direction = '盤整'
            confidence = max(prob_up, prob_down)

        # 預測價格區間 (68% 信賴區間)
        price_low = predicted_mean - predicted_std
        price_high = predicted_mean + predicted_std

        # 預測漲跌幅
        expected_change = (predicted_mean - current_price) / current_price * 100

        return {
            'stock_code': stock_code,
            'stock_name': stock_name or stock_code,
            'current_price': current_price,
            'predicted_price': round(predicted_mean, 2),
            'price_range': (round(price_low, 2), round(price_high, 2)),
            'expected_change': round(expected_change, 2),
            'direction': direction,
            'confidence': round(confidence, 2),
            'bias': round(bias, 2),
            'volatility': round(volatility, 2),
            'signals': signals,
            'prob_up': round(prob_up, 2),
            'prob_down': round(prob_down, 2)
        }


# ============================================================
# 主程式
# ============================================================

def predict_all_stocks():
    """預測所有關注股票"""
    from newslib import read_stock_list

    stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
    dict_stock = read_stock_list(stock_list_file)

    model = DirectionalParticleModel(n_particles=1000)

    results = []
    for name, code in dict_stock.items():
        print(f"\n預測 {name} ({code})...")
        result = model.predict(str(code), name)
        results.append(result)
        time.sleep(0.5)  # 避免請求太快

    return results


def print_prediction(result):
    """印出預測結果"""
    if 'error' in result:
        print(f"  錯誤: {result['error']}")
        return

    print(f"\n{'='*50}")
    print(f"  {result['stock_name']} ({result['stock_code']})")
    print(f"{'='*50}")
    print(f"  現價: ${result['current_price']}")
    print(f"  預測: ${result['predicted_price']} ({result['expected_change']:+.1f}%)")
    print(f"  區間: ${result['price_range'][0]} ~ ${result['price_range'][1]}")
    print(f"")
    print(f"  方向: {result['direction']} (信心度 {result['confidence']:.0%})")
    print(f"  上漲機率: {result['prob_up']:.0%}")
    print(f"  下跌機率: {result['prob_down']:.0%}")
    print(f"")
    print(f"  偏移量: {result['bias']:+.1f}")
    print(f"  波動率: {result['volatility']:.1f}%")
    print(f"")
    print(f"  信號:")
    for key, value in result['signals'].items():
        print(f"    - {value}")


def main():
    """主程式"""
    import sys

    model = DirectionalParticleModel(n_particles=1000)

    if len(sys.argv) > 1:
        # 指定股票
        stock_input = sys.argv[1]

        # 判斷是代號還是名稱
        if stock_input.isdigit():
            result = model.predict(stock_input)
        else:
            # 從股票清單找代號
            from newslib import read_stock_list
            stock_list_file = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')
            dict_stock = read_stock_list(stock_list_file)

            if stock_input in dict_stock:
                code = dict_stock[stock_input]
                result = model.predict(str(code), stock_input)
            else:
                print(f"找不到股票: {stock_input}")
                return

        print_prediction(result)

    else:
        # 預測所有股票
        print("方向性粒子預測模型")
        print("=" * 50)

        results = predict_all_stocks()

        # 排序：按預期漲幅
        results.sort(key=lambda x: x.get('expected_change', 0), reverse=True)

        print("\n\n" + "=" * 60)
        print("預測摘要 (按預期漲幅排序)")
        print("=" * 60)

        for r in results:
            if 'error' in r:
                continue

            emoji = '🟢' if r['direction'] == '漲' else '🔴' if r['direction'] == '跌' else '⚪'
            print(f"{emoji} {r['stock_name']}: ${r['current_price']} → ${r['predicted_price']} ({r['expected_change']:+.1f}%) [{r['direction']} {r['confidence']:.0%}]")


if __name__ == "__main__":
    main()
