# News_DB

用 AI 分析新聞情緒預測台股漲跌

> Taiwan Stock Market Intelligence System - AI-powered news sentiment analysis for stock prediction

## 功能特色

- 自動收集股票新聞（Bing/Google）
- AI 情緒分析預測漲跌（ML + 關鍵字 + GPT + 粒子模型）
- 每日定時執行（08:00 自動啟動）
- 盤中即時監控（09:00-13:30，每 15 分鐘推送）
- Discord 結構化 Embed 通知（訊號分解、風險警示、追蹤指標）
- 通知去重 / 冷卻機制（避免重複推播）
- 新聞自動去重（不重複抓取）
- GA 遺傳演算法自動優化模型參數 + 關鍵字
- 盤後自動誤差分析 + 績效追蹤
- 預測歷史記錄 + 系統偏差自動修正

---

## 專案目錄

```
News_DB/
├── README.md
├── .env.example                     # 環境變數範例
├── Data/
│   ├── daily_stock_job.py           # 主排程：盤前/盤中/盤後全流程
│   ├── newslib.py                   # 核心函式庫（爬蟲、API、資料處理）
│   ├── news_collector.py            # 新聞收集（自動去重）
│   ├── news_stock_selector.py       # 新聞焦點股篩選
│   │
│   ├── ## AI 預測模型
│   ├── directional_particle_model.py  # 方向性粒子預測模型（核心）
│   ├── hybrid_predictor.py          # 混合預測（60% 關鍵字 + 40% ML）
│   ├── gpt_sentiment.py             # GPT 新聞情緒分析
│   ├── predict_stock.py             # 單一股票預測 CLI
│   │
│   ├── ## 優化 & 回測
│   ├── optimize_weights.py          # GA 遺傳演算法（12 參數優化）
│   ├── keyword_optimizer.py         # GA 關鍵字篩選
│   ├── backtest.py                  # 回測框架
│   ├── prediction_history.py        # 預測歷史 + 修正因子 + 進階指標
│   │
│   ├── ## 通知 & 保護
│   ├── notifier.py                  # Discord 通知（Embed + 訊號分解）
│   ├── notification_guard.py        # 通知去重 / 冷卻機制
│   ├── broadcast_logger.py          # 廣播紀錄（可回測 JSONL）
│   │
│   ├── ## 訓練
│   ├── quick_train.py               # 快速訓練 ML 模型
│   ├── train_sentiment_model.py     # 完整訓練流程
│   │
│   ├── ## 工具
│   ├── analyze_logs.py              # 日誌分析器
│   ├── plot_accuracy.py             # 準確度圖表
│   │
│   ├── ## 資料
│   ├── stock_list_less.txt          # 監控股票清單（34 檔）
│   ├── optimized_weights.json       # GA 最佳權重
│   ├── optimized_keywords.json      # GA 最佳關鍵字
│   ├── prediction_history.json      # 預測歷史紀錄
│   ├── notify_config.json           # Discord webhook 設定
│   ├── news_data/                   # 收集的新聞資料
│   ├── models/                      # 訓練好的 AI 模型
│   └── logs/                        # 執行日誌
│       ├── stock_job_YYYY-MM-DD.log
│       ├── news_collector_YYYY-MM-DD.log
│       └── broadcast_YYYY-MM-DD.jsonl
```

---

## 系統架構

### 每日流程

```
08:00 盤前分析
├── 抓取基本面資料
├── 收集新聞（自動去重）
├── 選出新聞焦點 5 檔
├── 粒子模型盤前預測
├── GPT 新聞情緒分析
└── 發送 Discord「排程啟動」

09:00-13:30 盤中監控
├── 每 15 分鐘抓取即時股價
├── 焦點 5 檔：結構化 Embed（訊號分解 + 風險 + 指標）
├── 其餘股票：即時漲跌摘要
├── 通知去重（同方向 + 低信心差 = 不重發）
└── 廣播日誌記錄（broadcast_logger）

13:30 盤後分析
├── 比較預測 vs 實際收盤
├── 計算方向準確率 + 價格誤差
├── 回填廣播日誌實際結果
├── 發送每日績效 Embed（出手率 / 準度 / 連勝）
└── 系統偏差自動修正

每週一 GA 優化
├── 12 參數遺傳演算法
├── 關鍵字 GA 篩選
└── 發送優化結果 Embed
```

### 預測模型

```
粒子模型 bias = Σ(訊號 × 權重)
├── 外資買賣超     [GA 優化權重]
├── 5 日/10 日動量  [GA 優化權重]
├── 均線排列        [GA 優化權重]
├── RSI
├── 大盤加權指數
├── 費半/SOX
├── GPT 情緒偏移
├── 成交量比
├── 系統修正因子
└── 抑制過大偏移

混合預測 = 關鍵字 (60%) + ML 模型 (40%)
└── 關鍵字經 GA 優化篩選
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install pandas numpy scikit-learn beautifulsoup4 requests matplotlib openai
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入你的 API Key
```

### 3. 設定 Discord 通知

```bash
cd Data
python notifier.py --setup "YOUR_DISCORD_WEBHOOK_URL"
python notifier.py --test        # 測試基本通知
python notifier.py --test-embed  # 測試結構化 Embed
```

### 4. 訓練 AI 模型

```bash
python quick_train.py
```

### 5. 手動執行預測

```bash
python predict_stock.py 台積電
python predict_stock.py --all
```

### 6. 設定每日自動執行（cron）

```bash
crontab -l  # 查看排程
# 每週一至五 08:00 自動執行
```

---

## 常用指令

```bash
# 手動執行每日任務
python daily_stock_job.py

# 測試模式（發到測試頻道）
python daily_stock_job.py --test

# 訓練 AI 模型
python quick_train.py

# 預測單一股票
python predict_stock.py 台積電

# 測試 Discord Embed
python notifier.py --test-embed

# 分析日誌
python analyze_logs.py --latest
python analyze_logs.py 2026-02-06

# GA 權重優化
python optimize_weights.py

# GA 關鍵字優化
python keyword_optimizer.py

# 查看預測歷史指標
python prediction_history.py

# 查看廣播日誌報告
python broadcast_logger.py
```

---

## Discord 通知範例

### 盤中結構化 Embed

每個焦點股獨立 Embed，包含：
- 方向 + 信心度 + 預測價格
- 訊號分解（8 個信號各自貢獻）
- 新聞佐證 Top 3
- 風險警示（高波動 / 與大盤背離 / 資料不足）
- 追蹤指標（今日命中率 / 近 20 筆 / 連勝連敗）

### 盤後績效 Embed

- 今日表現：預測 N 檔 / 正確 N 檔 / 命中率
- 出手率 (Coverage) + 出手準度 (Precision)
- 最大連錯 + 目前連勝/連敗
- 方向分佈（看漲/看跌各自準確率）

---

## 免責聲明

本專案僅供研究/學習，不構成投資建議。股票投資有風險，請謹慎評估。

---

## 作者

- **rubby0605** - [GitHub](https://github.com/rubby0605)

## 協作

- Claude Opus 4.6 (AI Assistant)
