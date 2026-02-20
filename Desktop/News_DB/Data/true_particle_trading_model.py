#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真正的粒子交易模擬模型
每個粒子 = 一次完整的交易情境（買→持有→賣）

樹懶的原始想法：
- 模擬 1000 次可能的交易
- 每次交易有不同的持有時間（高斯分布）
- 每次交易有不同的結果（賺或虧）
- 統計：勝率、平均報酬、最大回撤

@author: rubylintu (樹懶)
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt


class TradingParticle:
    """一次交易情境模擬"""

    def __init__(self, entry_price, bias=0, volatility=2, avg_holding_days=5,
                 pdf_params=None):
        """
        Args:
            entry_price: 買入價格
            bias: 方向偏移 (-10 到 +10)
            volatility: 波動率 (%)
            avg_holding_days: 平均持有天數
            pdf_params: PDF 參數 dict，None 則用高斯
                - 'df': Student-t 自由度 (2-30)
                - 'jump_intensity': 跳躍機率 (0-0.15)
                - 'jump_std': 跳躍標準差 (0.01-0.10)
                - 'mixture_calm_prob': 平靜日機率 (0.7-0.98)
                - 'mixture_vol_mult': 波動日倍數 (2-8)
        """
        self.entry_price = entry_price
        self.bias = bias
        self.volatility = volatility
        self.avg_holding_days = avg_holding_days
        self.pdf_params = pdf_params or {}

        # 模擬結果
        self.holding_days = 0
        self.exit_price = 0
        self.return_pct = 0
        self.profit = 0

        self.simulate()

    def _generate_daily_return(self, mu, sigma):
        """根據 pdf_params 生成單日報酬"""
        p = self.pdf_params

        if not p:
            # 原始高斯
            return random.gauss(mu, sigma)

        # Student-t 基底
        df = p.get('df', 30)
        t_random = np.random.standard_t(df)
        base_return = mu + sigma * t_random

        # Jump Diffusion 疊加
        jump_intensity = p.get('jump_intensity', 0)
        if jump_intensity > 0 and np.random.random() < jump_intensity:
            jump_std = p.get('jump_std', 0.03)
            base_return += np.random.normal(0, jump_std)

        # Mixture: 波動日放大
        calm_prob = p.get('mixture_calm_prob', 1.0)
        if calm_prob < 1.0 and np.random.random() > calm_prob:
            vol_mult = p.get('mixture_vol_mult', 3.0)
            base_return = mu + sigma * vol_mult * np.random.standard_t(df)

        return base_return

    def simulate(self):
        """模擬一次完整的交易"""
        # 1. 持有時間（高斯分布）
        self.holding_days = max(1, int(random.gauss(self.avg_holding_days, 2)))

        # 2. 每天價格隨機遊走
        current_price = self.entry_price
        daily_returns = []

        for day in range(self.holding_days):
            daily_bias = self.bias / 100 / self.avg_holding_days
            daily_vol = self.volatility / 100 / np.sqrt(252)

            daily_return = self._generate_daily_return(daily_bias, daily_vol)
            current_price *= (1 + daily_return)
            daily_returns.append(daily_return)

        # 3. 計算最終結果
        self.exit_price = current_price
        self.return_pct = (self.exit_price - self.entry_price) / self.entry_price * 100

        # 假設買 1 張（1000 股）
        self.profit = (self.exit_price - self.entry_price) * 1000

        # 扣除交易成本（0.585%）
        self.profit -= self.entry_price * 1000 * 0.00585


class TrueParticleModel:
    """粒子交易模擬模型"""

    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.particles = []

    def simulate_trading(self, entry_price, bias=0, volatility=2, avg_holding_days=5,
                         pdf_params=None):
        """
        模擬 N 次交易情境

        Args:
            entry_price: 現在的買入價格
            bias: 方向偏移（從你的多因子模型來）
            volatility: 波動率
            avg_holding_days: 平均持有天數
            pdf_params: PDF 參數（df, jump_intensity 等），由 GA 優化

        Returns:
            dict: 統計結果
        """
        self.particles = []

        # 生成 N 個粒子（交易情境）
        for _ in range(self.n_particles):
            particle = TradingParticle(entry_price, bias, volatility, avg_holding_days,
                                       pdf_params=pdf_params)
            self.particles.append(particle)

        # 統計結果
        returns = [p.return_pct for p in self.particles]
        profits = [p.profit for p in self.particles]
        holding_days = [p.holding_days for p in self.particles]

        win_count = sum(1 for r in returns if r > 0)
        lose_count = sum(1 for r in returns if r < 0)

        avg_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)

        max_return = max(returns)
        min_return = min(returns)

        avg_profit = np.mean(profits)
        total_profit_if_all_trades = sum(profits)

        win_rate = win_count / self.n_particles

        avg_holding = np.mean(holding_days)

        # VaR (Value at Risk): 95% 信賴區間的最壞情況
        var_95 = np.percentile(returns, 5)

        return {
            'entry_price': entry_price,
            'bias': bias,
            'volatility': volatility,
            'n_simulations': self.n_particles,

            # 報酬統計
            'avg_return_pct': avg_return,
            'median_return_pct': median_return,
            'std_return': std_return,
            'max_return': max_return,
            'min_return': min_return,

            # 勝率
            'win_rate': win_rate,
            'win_count': win_count,
            'lose_count': lose_count,

            # 獲利
            'avg_profit_per_trade': avg_profit,
            'total_profit_if_all_trades': total_profit_if_all_trades,

            # 持有時間
            'avg_holding_days': avg_holding,

            # 風險指標
            'var_95': var_95,  # 95% 的情況下，最多虧這麼多

            # 建議
            'recommendation': self._make_recommendation(win_rate, avg_return, var_95)
        }

    def _make_recommendation(self, win_rate, avg_return, var_95):
        """根據模擬結果給建議"""
        if win_rate > 0.65 and avg_return > 3 and var_95 > -5:
            return "✅ 強烈建議買入：高勝率、高報酬、低風險"
        elif win_rate > 0.55 and avg_return > 2:
            return "✅ 建議買入：勝率和報酬都不錯"
        elif win_rate > 0.50 and avg_return > 1:
            return "⚠️ 謹慎考慮：略有優勢但不明顯"
        elif win_rate < 0.45 or avg_return < 0:
            return "❌ 不建議買入：勝率太低或預期虧損"
        else:
            return "⏸️ 觀望：沒有明顯優勢"

    def plot_results(self, save_path=None):
        """繪製模擬結果"""
        if not self.particles:
            print("請先執行 simulate_trading()")
            return

        returns = [p.return_pct for p in self.particles]
        holding_days = [p.holding_days for p in self.particles]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 報酬率分布
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='損益平衡')
        axes[0, 0].set_xlabel('報酬率 (%)')
        axes[0, 0].set_ylabel('次數')
        axes[0, 0].set_title('📊 報酬率分布（1000 次模擬）')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. 持有時間分布
        axes[0, 1].hist(holding_days, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('持有天數')
        axes[0, 1].set_ylabel('次數')
        axes[0, 1].set_title('⏱️ 持有時間分布（高斯）')
        axes[0, 1].grid(alpha=0.3)

        # 3. 報酬 vs 持有時間
        axes[1, 0].scatter(holding_days, returns, alpha=0.3, s=10)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('持有天數')
        axes[1, 0].set_ylabel('報酬率 (%)')
        axes[1, 0].set_title('📈 持有時間 vs 報酬率')
        axes[1, 0].grid(alpha=0.3)

        # 4. 累積報酬分布（排序後）
        sorted_returns = sorted(returns)
        cumulative = np.cumsum(sorted_returns)
        axes[1, 1].plot(range(len(sorted_returns)), sorted_returns, label='個別報酬')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('模擬次數（由低到高排序）')
        axes[1, 1].set_ylabel('報酬率 (%)')
        axes[1, 1].set_title('📊 報酬率分布（排序）')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已儲存: {save_path}")
        else:
            plt.show()


def print_results(result):
    """印出模擬結果"""
    print("\n" + "="*60)
    print("🎲 粒子交易模擬結果")
    print("="*60)
    print(f"\n📊 基本資訊:")
    print(f"  買入價格: ${result['entry_price']:.2f}")
    print(f"  方向偏移: {result['bias']:+.1f}")
    print(f"  波動率: {result['volatility']:.1f}%")
    print(f"  模擬次數: {result['n_simulations']} 次")

    print(f"\n💰 報酬統計:")
    print(f"  平均報酬: {result['avg_return_pct']:+.2f}%")
    print(f"  中位數報酬: {result['median_return_pct']:+.2f}%")
    print(f"  標準差: {result['std_return']:.2f}%")
    print(f"  最佳情況: {result['max_return']:+.2f}%")
    print(f"  最差情況: {result['min_return']:+.2f}%")

    print(f"\n🎯 勝率:")
    print(f"  賺錢次數: {result['win_count']} ({result['win_rate']:.1%})")
    print(f"  虧錢次數: {result['lose_count']} ({1-result['win_rate']:.1%})")

    print(f"\n💵 獲利 (以 1 張 = 1000 股計算):")
    print(f"  每次平均獲利: {result['avg_profit_per_trade']:+,.0f} 元")
    print(f"  如果全部交易: {result['total_profit_if_all_trades']:+,.0f} 元")

    print(f"\n⏱️ 持有時間:")
    print(f"  平均持有: {result['avg_holding_days']:.1f} 天")

    print(f"\n⚠️ 風險指標:")
    print(f"  VaR (95%): {result['var_95']:.2f}%")
    print(f"  (95% 的情況下，最多虧損不超過 {abs(result['var_95']):.2f}%)")

    print(f"\n💡 建議:")
    print(f"  {result['recommendation']}")
    print("="*60 + "\n")


# ============================================================
# 分布圖生成（給 AI Agent Vision 用）
# ============================================================

def generate_distribution_chart(predictions, pdf_params=None, n_particles=500):
    """
    為多檔股票跑粒子模擬，產生報酬分布圖（base64 PNG）。
    讓 GPT-4o / Gemini 的 vision 模型看到肥尾分布形狀。

    Args:
        predictions: list[dict] 每檔股票的預測結果
            需要: stock_code, stock_name, current_price (或 predicted_price), bias, volatility
        pdf_params: dict 肥尾 PDF 參數（df, jump_intensity 等），None 則用高斯
        n_particles: 每檔模擬的粒子數

    Returns:
        str: base64 編碼的 PNG 圖片，失敗回傳 None
    """
    import io
    import base64

    # 過濾掉觀望和缺少資料的
    stocks = []
    for p in predictions:
        direction = p.get('direction', '')
        if direction in ('觀望', ''):
            continue
        price = p.get('current_price') or p.get('predicted_price', 0)
        if not price or price <= 0:
            continue
        stocks.append(p)

    if not stocks:
        return None

    # 最多顯示 6 檔（2x3 grid）
    stocks = stocks[:6]
    n = len(stocks)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    model = TrueParticleModel(n_particles=n_particles)

    for i, p in enumerate(stocks):
        ax = axes[i]
        code = p.get('stock_code', '?')
        name = p.get('stock_name', code)
        price = p.get('current_price') or p.get('predicted_price', 0)
        bias = p.get('bias', 0)
        vol = p.get('volatility', 2)

        result = model.simulate_trading(
            entry_price=price, bias=bias, volatility=vol,
            avg_holding_days=1, pdf_params=pdf_params
        )

        returns = [pt.return_pct for pt in model.particles]
        win_rate = result['win_rate']
        avg_ret = result['avg_return_pct']
        var95 = result['var_95']

        # 繪圖
        color = '#26a641' if bias > 0 else '#f85149' if bias < 0 else '#8b949e'
        ax.hist(returns, bins=40, alpha=0.75, color=color, edgecolor='#161b22')
        ax.axvline(0, color='#f0f6fc', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(avg_ret, color='#58a6ff', linestyle='-', linewidth=1.2, label=f'avg={avg_ret:+.2f}%')

        direction_str = p.get('direction', '')
        ax.set_title(f'{code} {name} ({direction_str})', fontsize=10, color='#f0f6fc')
        ax.set_xlabel('Return %', fontsize=8, color='#8b949e')

        # 標注
        stats_text = f'WR={win_rate:.0%}  VaR95={var95:.1f}%\nbias={bias:+.1f}  vol={vol:.1f}%'
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, va='top', ha='right', color='#c9d1d9',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', alpha=0.8))
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.set_facecolor('#0d1117')

    # 隱藏多餘的 subplot
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.patch.set_facecolor('#0d1117')
    title = '粒子模擬報酬分布'
    if pdf_params:
        df_val = pdf_params.get('df', 30)
        ji = pdf_params.get('jump_intensity', 0)
        title += f'  (df={df_val:.0f}, jump={ji:.2f})'
    fig.suptitle(title, fontsize=12, color='#f0f6fc', y=1.02)
    fig.tight_layout()

    # 轉 base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_pdf_params_from_weights():
    """從 optimized_weights.json 載入 PDF 參數（如果有的話）"""
    weights_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimized_weights.json')
    try:
        with open(weights_file, 'r') as f:
            data = json.load(f)
        w = data.get('weights', {})
        params = {}
        for key in ('df', 'jump_intensity', 'jump_std', 'mixture_calm_prob', 'mixture_vol_mult'):
            if key in w:
                params[key] = w[key]
        return params if params else None
    except Exception:
        return None


# ============================================================
# 使用範例
# ============================================================

if __name__ == "__main__":
    # 創建模型
    model = TrueParticleModel(n_particles=1000)

    # 案例 1: 強烈看多（bias +5）
    print("【案例 1】強烈看多的股票（台積電）")
    result1 = model.simulate_trading(
        entry_price=1880,
        bias=5.0,  # 強烈看多
        volatility=2.5,
        avg_holding_days=5
    )
    print_results(result1)
    model.plot_results(save_path='/Users/rubylintu/Desktop/News_DB/Data/simulation_bullish.png')

    # 案例 2: 中性（bias 0）
    print("\n【案例 2】中性的股票")
    result2 = model.simulate_trading(
        entry_price=1880,
        bias=0.0,  # 中性
        volatility=2.5,
        avg_holding_days=5
    )
    print_results(result2)

    # 案例 3: 看空（bias -3）
    print("\n【案例 3】看空的股票")
    result3 = model.simulate_trading(
        entry_price=1880,
        bias=-3.0,  # 看空
        volatility=2.5,
        avg_holding_days=5
    )
    print_results(result3)

    print("\n✅ 這才是樹懶想要的「粒子交易模型」！")
    print("   每個粒子 = 一次完整的交易情境（買→持有→賣）")
    print("   持有時間符合高斯分布")
    print("   可以統計：勝率、平均報酬、風險")
