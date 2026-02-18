#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日內粒子模型 — tick-by-tick 動態預測

每收到一個新 tick 就跑一次粒子模擬，產生隨時間演化的預測曲線。
7 個參數可由 GA 優化。

用法:
    python intraday_particle_model.py --date 20220107
    python intraday_particle_model.py --date 20220107 --stock 2330

@author: rubylintu
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 參數封裝
# ============================================================

PARAM_RANGES = {
    'decay_rate':           (0.80, 0.99),
    'momentum_weight':      (0.1, 3.0),
    'mean_reversion_weight':(0.1, 2.0),
    'vol_sensitivity':      (0.5, 3.0),
    'particle_spread':      (0.3, 2.0),
    'lookback_window':      (5, 30),
    'prediction_horizon':   (3, 15),
}

DEFAULT_PARAMS = {
    'decay_rate':            0.92,
    'momentum_weight':       1.0,
    'mean_reversion_weight': 0.5,
    'vol_sensitivity':       1.5,
    'particle_spread':       0.8,
    'lookback_window':       10,
    'prediction_horizon':    5,
}


class IntradayParticleParams:
    """封裝日內粒子模型的 7 個可調參數"""

    def __init__(self, params=None):
        p = DEFAULT_PARAMS.copy()
        if params:
            p.update(params)
        self.decay_rate = float(p['decay_rate'])
        self.momentum_weight = float(p['momentum_weight'])
        self.mean_reversion_weight = float(p['mean_reversion_weight'])
        self.vol_sensitivity = float(p['vol_sensitivity'])
        self.particle_spread = float(p['particle_spread'])
        self.lookback_window = int(round(p['lookback_window']))
        self.prediction_horizon = int(round(p['prediction_horizon']))

    def to_dict(self):
        return {
            'decay_rate': round(self.decay_rate, 4),
            'momentum_weight': round(self.momentum_weight, 4),
            'mean_reversion_weight': round(self.mean_reversion_weight, 4),
            'vol_sensitivity': round(self.vol_sensitivity, 4),
            'particle_spread': round(self.particle_spread, 4),
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data.get('params', data))

    def save(self, filepath):
        data = {
            'params': self.to_dict(),
            'updated': datetime.datetime.now().isoformat(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath=None):
        if filepath is None:
            from config import INTRADAY_MODEL_PARAMS_FILE
            filepath = INTRADAY_MODEL_PARAMS_FILE
        if os.path.exists(filepath):
            return cls.from_json(filepath)
        return cls()


# ============================================================
# 核心模型
# ============================================================

class IntradayParticleModel:
    """
    日內粒子預測模型

    每個 tick 呼叫 update()，內部維護價格歷史和 VWAP，
    用動量 + 均值回歸計算 bias，生成粒子並指數平滑。
    """

    def __init__(self, params=None, n_particles=None):
        """
        Args:
            params: IntradayParticleParams 或 dict 或 None（用預設）
            n_particles: 粒子數，預設從 config 讀取
        """
        if isinstance(params, IntradayParticleParams):
            self.params = params
        elif isinstance(params, dict):
            self.params = IntradayParticleParams(params)
        else:
            self.params = IntradayParticleParams()

        if n_particles is None:
            try:
                from config import INTRADAY_PARTICLES
                n_particles = INTRADAY_PARTICLES
            except ImportError:
                n_particles = 200
        self.n_particles = n_particles

        self.reset()

    def reset(self):
        """清除所有狀態（新的一天）"""
        self.prices = []
        self.volumes = []
        self.timestamps = []
        self.cum_pv = 0.0        # cumulative price * volume
        self.cum_vol = 0.0       # cumulative volume
        self.predictions = []    # [(timestamp, predicted_price)]
        self._prev_prediction = None

    def update(self, price, volume, timestamp=None):
        """
        收到新 tick，更新模型並產生預測

        Args:
            price: 成交價
            volume: 成交量
            timestamp: 時間戳（可選）

        Returns:
            float: 預測價格（prediction_horizon ticks 後的價格）
        """
        price = float(price)
        volume = float(volume) if volume and volume > 0 else 0.0

        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)

        # 更新 VWAP
        if volume > 0:
            self.cum_pv += price * volume
            self.cum_vol += volume

        n = len(self.prices)
        p = self.params

        # 需要足夠的歷史才能計算
        if n < max(3, p.lookback_window):
            self.predictions.append((timestamp, price))
            self._prev_prediction = price
            return price

        # --- 計算特徵 ---

        # 1. 動量
        lb = min(p.lookback_window, n - 1)
        past_price = self.prices[n - 1 - lb]
        momentum = (price - past_price) / past_price if past_price > 0 else 0

        # 2. VWAP
        vwap = self.cum_pv / self.cum_vol if self.cum_vol > 0 else price

        # 3. 滾動波動率（用最近 lookback 個 tick 的 return std）
        recent_prices = self.prices[-lb:]
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                returns.append(
                    (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                )
        rolling_vol = np.std(returns) if len(returns) > 2 else 0.005

        # 4. 量能比（最近 volume vs 平均 volume）
        recent_vols = self.volumes[-lb:]
        avg_vol = np.mean(recent_vols) if recent_vols else 1.0
        current_vol = volume if volume > 0 else avg_vol
        vol_surge = current_vol / avg_vol if avg_vol > 0 else 1.0

        # --- 計算 bias ---
        trend_bias = momentum * p.momentum_weight
        reversion_bias = (vwap - price) / price * p.mean_reversion_weight if price > 0 else 0
        bias = trend_bias + reversion_bias

        # 量能放大：放量時 bias 加強，縮量時衰減
        if vol_surge > 1.5:
            bias *= (1 + 0.2 * min(vol_surge - 1, 2))
        elif vol_surge < 0.5:
            bias *= 0.8

        # --- 生成粒子 ---
        mu = price * (1 + bias)
        sigma = price * rolling_vol * p.vol_sensitivity * (p.particle_spread / 100)
        sigma = max(sigma, price * 0.0001)  # 最小擴散

        particles = np.random.normal(mu, sigma, self.n_particles)
        particle_mean = np.mean(particles)

        # --- 指數平滑 ---
        if self._prev_prediction is not None:
            predicted = (p.decay_rate * self._prev_prediction +
                         (1 - p.decay_rate) * particle_mean)
        else:
            predicted = particle_mean

        self._prev_prediction = predicted
        self.predictions.append((timestamp, predicted))
        return predicted

    def get_prediction_curve(self):
        """
        回傳所有歷史預測

        Returns:
            list: [(timestamp, predicted_price), ...]
        """
        return list(self.predictions)

    def backtest_day(self, ticks_df):
        """
        對整天的 tick 資料跑回測

        Args:
            ticks_df: DataFrame with columns [price, trade_vol, timestamp]

        Returns:
            predictions: list of predicted prices
            actual: list of actual prices
            metrics: dict with MAE, direction_accuracy, correlation, max_error
        """
        self.reset()
        horizon = self.params.prediction_horizon

        predictions = []
        actual_prices = list(ticks_df['price'].values)

        for _, row in ticks_df.iterrows():
            pred = self.update(row['price'], row.get('trade_vol', 0),
                               row.get('timestamp'))
            predictions.append(pred)

        # 計算指標：比較 prediction[t] vs actual[t + horizon]
        n = len(predictions)
        valid_preds = []
        valid_actuals = []
        direction_correct = 0
        direction_total = 0

        for t in range(n - horizon):
            pred_price = predictions[t]
            actual_future = actual_prices[t + horizon]
            current_price = actual_prices[t]

            valid_preds.append(pred_price)
            valid_actuals.append(actual_future)

            # 方向判斷
            pred_dir = 1 if pred_price > current_price else -1
            actual_dir = 1 if actual_future > current_price else -1
            if pred_dir == actual_dir:
                direction_correct += 1
            direction_total += 1

        if not valid_preds:
            return predictions, actual_prices, {
                'mae': 0, 'direction_accuracy': 0,
                'correlation': 0, 'max_error': 0,
                'n_samples': 0,
            }

        errors = np.abs(np.array(valid_preds) - np.array(valid_actuals))
        mae = np.mean(errors)
        max_error = np.max(errors)

        # 相關係數
        if len(valid_preds) > 2:
            corr = np.corrcoef(valid_preds, valid_actuals)[0, 1]
        else:
            corr = 0

        dir_acc = direction_correct / direction_total if direction_total > 0 else 0

        # 正規化 MAE（相對於價格水準）
        avg_price = np.mean(actual_prices)
        mae_pct = mae / avg_price * 100 if avg_price > 0 else 0

        metrics = {
            'mae': round(float(mae), 4),
            'mae_pct': round(float(mae_pct), 4),
            'direction_accuracy': round(float(dir_acc), 4),
            'correlation': round(float(corr), 4),
            'max_error': round(float(max_error), 4),
            'n_samples': direction_total,
        }

        return predictions, actual_prices, metrics


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='日內粒子模型測試')
    parser.add_argument('--date', type=str, default=None,
                        help='日期 YYYYMMDD')
    parser.add_argument('--stock', type=str, default=None,
                        help='只測特定股票代號')
    parser.add_argument('--params', type=str, default=None,
                        help='參數 JSON 檔路徑')
    args = parser.parse_args()

    date_str = args.date or datetime.date.today().strftime('%Y%m%d')

    # 載入參數
    if args.params:
        params = IntradayParticleParams.from_json(args.params)
    else:
        params = IntradayParticleParams.load()
    print(f"模型參數: {params.to_dict()}")

    # 載入 intraday 資料
    from merge_intraday import parse_intraday_file
    fpath = os.path.join(SCRIPT_DIR, 'intraday', f'{date_str}.txt')
    if not os.path.exists(fpath):
        print(f"找不到 {fpath}")
        return

    df = parse_intraday_file(fpath)
    print(f"載入 {len(df)} 筆 tick ({date_str})")

    # 取得股票列表
    stock_codes = sorted(df['code'].unique())
    if args.stock:
        stock_codes = [c for c in stock_codes if c == args.stock]

    model = IntradayParticleModel(params)

    print(f"\n{'='*60}")
    print(f"日內粒子模型回測 {date_str}")
    print(f"{'='*60}")

    all_metrics = []
    for code in stock_codes:
        stock_df = df[df['code'] == code].copy()
        if len(stock_df) < 20:
            continue

        name = stock_df['name'].iloc[0]
        preds, actuals, metrics = model.backtest_day(stock_df)

        print(f"\n  {name}({code}): {len(stock_df)} ticks")
        print(f"    MAE: {metrics['mae']:.2f} ({metrics['mae_pct']:.3f}%)")
        print(f"    方向正確率: {metrics['direction_accuracy']:.1%}")
        print(f"    相關係數: {metrics['correlation']:.4f}")
        print(f"    最大誤差: {metrics['max_error']:.2f}")

        metrics['code'] = code
        metrics['name'] = name
        all_metrics.append(metrics)

    if all_metrics:
        avg_mae = np.mean([m['mae_pct'] for m in all_metrics])
        avg_dir = np.mean([m['direction_accuracy'] for m in all_metrics])
        avg_corr = np.mean([m['correlation'] for m in all_metrics])
        print(f"\n{'='*60}")
        print(f"  平均 MAE%: {avg_mae:.3f}%")
        print(f"  平均方向正確率: {avg_dir:.1%}")
        print(f"  平均相關係數: {avg_corr:.4f}")


if __name__ == '__main__':
    main()
