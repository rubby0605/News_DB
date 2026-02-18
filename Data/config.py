#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中設定檔 — 所有常數和路徑統一管理

@author: rubylintu
"""

import os

# === 腳本根目錄 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === 檔案路徑 ===
PORTFOLIO_FILE = os.path.join(SCRIPT_DIR, 'ai_portfolio.json')
GEMINI_PORTFOLIO_FILE = os.path.join(SCRIPT_DIR, 'gemini_portfolio.json')
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'optimized_weights.json')
PREDICTIONS_FILE = os.path.join(SCRIPT_DIR, 'today_predictions.json')
FOCUS_STOCKS_FILE = os.path.join(SCRIPT_DIR, 'today_focus_stocks.json')
STOCK_LIST_FILE = os.path.join(SCRIPT_DIR, 'stock_list_less.txt')

# === 台股交易成本 ===
BROKER_FEE_RATE = 0.001425    # 0.1425%（買賣都收）
SECURITIES_TAX_RATE = 0.003   # 0.3%（僅賣出）
LOT_SIZE = 1000               # 1 張 = 1000 股

# === 交易參數 ===
INITIAL_CAPITAL = 1_000_000   # 100 萬
MAX_POSITIONS = 5             # 最多同時持有
POSITION_WEIGHT = 0.20        # 每檔 20% 總資產
STOP_LOSS_PCT = -3.0          # 停損 %
TAKE_PROFIT_PCT = 5.0         # 停利 %
BUY_CONFIDENCE = 0.70         # 買入最低信心度
BUY_BIAS = 3.0                # 買入最低 bias
SELL_CONFIDENCE = 0.65        # 賣出（反轉）信心度
MAX_WARNINGS = 1              # 買入時最多允許幾個警示
MIN_HOLD_DAYS = 3             # 最少持有天數（停損除外）
COOLDOWN_HOURS = 24           # 同一支賣出後冷卻時間（小時）

# === Discord 顏色（台股慣例：紅漲綠跌）===
COLOR_BULLISH = 0xFF4444   # 紅（漲）
COLOR_BEARISH = 0x44FF44   # 綠（跌）
COLOR_NEUTRAL = 0x808080   # 灰（盤整/觀望）
COLOR_INFO = 0x3498DB      # 藍（資訊）
COLOR_WARNING = 0xFFAA00   # 橘（警告）
COLOR_PROFIT = 0x2ECC71    # 綠（獲利）
COLOR_LOSS = 0xE74C3C      # 紅（虧損）

# === Discord 頻道 ===
DISCORD_CHANNEL = 'release'
AI_TRADE_CHANNEL = 'test'
GEMINI_TRADE_CHANNEL = 'test'

# === 回測參數（trading_backtest.py 用）===
BACKTEST_BUY_THRESHOLD = 2.0      # bias > 此值 → 買入信號
BACKTEST_SELL_THRESHOLD = -2.0    # bias < 此值 → 賣出信號
BACKTEST_POSITION_SIZE = 0.3      # 每次用 30% 資金建倉
BACKTEST_TAX_RATE = 0.00585       # 台股實際來回成本 0.585%

# === 日內粒子模型 ===
INTRADAY_MODEL_PARAMS_FILE = os.path.join(SCRIPT_DIR, 'intraday_model_params.json')
INTRADAY_PARTICLES = 200       # 粒子數（每 tick）
INTRADAY_GA_POP = 30           # GA 族群
INTRADAY_GA_GEN = 20           # GA 代數
INTRADAY_TRAIN_DAYS = 10       # 訓練用天數
