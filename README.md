# News_DB

用 AI 分析新聞情緒預測台股漲跌

> Taiwan Stock Market Intelligence System - AI-powered news sentiment analysis for stock prediction

## 功能特色

- 自動收集股票新聞（Bing/Google）
- AI 情緒分析預測漲跌（ML + 關鍵字 + GPT + 粒子模型）
- 每日定時執行（08:00 自動啟動）
- 盤中即時監控（09:00-13:30，每 10~20 秒拉取、每 15 分鐘推送）
- Discord 結構化 Embed 通知（訊號分解、風險警示、追蹤指標）
- 通知去重 / 冷卻機制（避免重複推播）
- 新聞自動去重（不重複抓取）
- GA / NSGA-II 遺傳演算法自動優化模型參數 + 關鍵字
- 盤後自動誤差分析 + 績效追蹤
- 預測歷史記錄 + 系統偏差自動修正
- AI 自動交易引擎（GPT / Gemini / 人工三軌並行）
- 策略引擎 + 風險引擎（止損 / 加碼 / 強制平倉）
- 盤中 Tick 級回測 + Walk-forward 驗證
- 情報監控系統（RSS / GitHub / 網頁搜尋，Gemini 評分 ≥7 才推播）
- Web UI 儀表板（Flask，含預測、交易、新聞、情報頁面）
- Discord Bot 互動指令

---

## 專案目錄

```
News_DB/
├── README.md
├── .env.example                        # 環境變數範例
├── Data/
│   ├── config.py                       # 全域設定（API key、RSS、閾值等）
│   ├── daily_stock_job.py              # 主排程：盤前/盤中/盤後全流程
│   ├── newslib.py                      # 核心函式庫（爬蟲、API、資料處理）
│   ├── news_collector.py               # 新聞收集（自動去重）
│   ├── news_stock_selector.py          # 新聞焦點股篩選
│   │
│   ├── ## AI 預測模型
│   ├── directional_particle_model.py   # 方向性粒子預測模型（核心）
│   ├── intraday_particle_model.py      # 盤中即時粒子模型（Tick 級）
│   ├── intraday_gpt_predictor.py       # 盤中 GPT 預測（每 30 分鐘）
│   ├── hybrid_predictor.py             # 混合預測（60% 關鍵字 + 40% ML）
│   ├── gpt_sentiment.py                # GPT / Gemini 新聞情緒分析
│   ├── predict_stock.py                # 單一股票預測 CLI
│   │
│   ├── ## 肥尾分布模型
│   ├── fat_tail_trading_model.py       # 4 種肥尾 PDF（Student-t / Levy / Jump / Mixture）
│   ├── true_particle_trading_model.py  # 粒子交易模擬（支援肥尾 PDF 參數）
│   ├── improved_directional_particle.py # 改良粒子模型比較
│   │
│   ├── ## AI 交易引擎
│   ├── ai_trader.py                    # GPT AI 交易引擎（策略→風險→LLM 決策）
│   ├── gemini_trader.py                # Gemini AI 交易引擎
│   ├── human_trader.py                 # 人工交易紀錄器
│   ├── strategy_engine.py              # 策略引擎（多策略訊號產生）
│   ├── risk_engine.py                  # 風險引擎（止損/加碼/強制平倉）
│   │
│   ├── ## 優化 & 回測
│   ├── optimize_weights.py             # GA 遺傳演算法（17 參數：12 信號 + 5 肥尾 PDF）
│   ├── nsga2_optimize.py               # NSGA-II 多目標優化
│   ├── keyword_optimizer.py            # GA 關鍵字篩選
│   ├── backtest.py                     # 日級回測框架
│   ├── trading_backtest.py             # 交易策略回測
│   ├── intraday_backtest.py            # 盤中 Tick 級回測
│   ├── intraday_eval.py                # 盤中策略評估（Walk-forward）
│   ├── intraday_ga_optimize.py         # 盤中參數 GA 優化
│   ├── prediction_history.py           # 預測歷史 + 修正因子 + 進階指標
│   │
│   ├── ## 盤中資料工具
│   ├── intraday_chart.py               # 盤中走勢圖 PDF 報告
│   ├── merge_intraday.py               # 合併多日 Tick 資料
│   ├── package_intraday_data.py        # 打包 Tick → 1min/5min K 線
│   ├── daily_report.py                 # 每日綜合報告
│   ├── plot_tsmc_forecast.py           # 台積電預測視覺化
│   │
│   ├── ## 通知 & 保護
│   ├── notifier.py                     # Discord 通知（Embed + 訊號分解）
│   ├── notification_guard.py           # 通知去重 / 冷卻機制
│   ├── broadcast_logger.py             # 廣播紀錄（可回測 JSONL）
│   ├── discord_bot.py                  # Discord Bot 互動指令
│   │
│   ├── ## 情報監控
│   ├── intel_monitor.py                # 全方位情報監控（RSS/GitHub/搜尋）
│   │
│   ├── ## 訓練
│   ├── quick_train.py                  # 快速訓練 ML 模型
│   ├── train_sentiment_model.py        # 完整訓練流程
│   │
│   ├── ## 工具 & 驗證
│   ├── analyze_logs.py                 # 日誌分析器
│   ├── plot_accuracy.py                # 準確度圖表
│   ├── plot_validation_results.py      # 驗證結果視覺化
│   ├── validate_with_intraday_data.py  # 盤中資料驗證
│   ├── verify_exponential_distribution.py # 分布驗證
│   ├── test_fat_tail_integration.py    # 肥尾整合測試
│   │
│   ├── ## Web UI
│   ├── web/
│   │   ├── app.py                      # Flask 主程式
│   │   └── templates/
│   │       ├── base.html               # 共用版型 + 導覽列
│   │       ├── dashboard.html          # 儀表板總覽
│   │       ├── predictions.html        # 預測結果
│   │       ├── trading.html            # AI 交易紀錄
│   │       ├── stock_detail.html       # 個股詳情
│   │       ├── news.html               # 新聞列表
│   │       ├── volume_profile.html     # 成交量分佈
│   │       ├── intel.html              # 情報監控
│   │       └── about.html              # 關於
│   │
│   ├── ## 資料檔
│   ├── stock_list_less.txt             # 監控股票清單（35 檔）
│   ├── config.py                       # 全域設定
│   ├── optimized_weights.json          # GA 最佳權重
│   ├── optimized_keywords.json         # GA 最佳關鍵字
│   ├── prediction_history.json         # 預測歷史紀錄
│   ├── today_predictions.json          # 今日預測快取
│   ├── ai_portfolio.json               # GPT 交易持倉
│   ├── gemini_portfolio.json           # Gemini 交易持倉
│   ├── intraday_model_params.json      # 盤中模型參數
│   ├── notification_guard_state.json   # 通知去重狀態
│   ├── notify_config.json              # Discord webhook 設定
│   ├── news_data/                      # 收集的新聞資料
│   │   ├── news_YYYYMMDD.csv
│   │   ├── news_history.csv
│   │   └── seen_news.json
│   ├── intraday/                       # 盤中 Tick 資料
│   │   └── YYYYMMDD.txt
│   ├── models/                         # 訓練好的 AI 模型
│   ├── logs/                           # 執行日誌
│   │   ├── stock_job_YYYY-MM-DD.log
│   │   ├── news_collector_YYYY-MM-DD.log
│   │   └── broadcast_YYYY-MM-DD.jsonl
│   └── archive/                        # 已棄用的舊檔案
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
├── 每 10~20 秒拉取即時股價（TWSE API batch）
├── 即時 Tick 寫入 intraday/YYYYMMDD.txt
├── 每 15 分鐘推送：
│   ├── 焦點 5 檔：結構化 Embed（訊號分解 + 風險 + 指標）
│   ├── 其餘股票：即時漲跌摘要
│   ├── 盤中粒子模型即時預測
│   └── GPT 每 30 分鐘重新評估
├── AI 交易引擎：
│   ├── 策略引擎產生訊號（動量/均線/量能/突破）
│   ├── 風險引擎過濾（止損/加碼/強平規則）
│   └── GPT + Gemini 各自決策買賣
├── 通知去重（同方向 + 低信心差 = 不重發）
└── 廣播日誌記錄（broadcast_logger）

13:30 盤後分析
├── 比較預測 vs 實際收盤
├── 計算方向準確率 + 價格誤差
├── ADR 預測驗證
├── AI 交易 P&L 結算
├── 回填廣播日誌實際結果
├── 發送每日績效 Embed（出手率 / 準度 / 連勝）
├── 發送盤中走勢圖 PDF
├── 發送每日綜合報告
└── 系統偏差自動修正

每週一 GA 優化
├── 17 參數遺傳演算法（12 信號權重 + 5 肥尾 PDF）
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

粒子模擬 PDF（肥尾分布）[GA 優化參數]
├── Student-t 自由度 (df)       → 控制尾部厚度
├── Jump Diffusion 跳躍機率     → 模擬黑天鵝事件
├── Jump 跳躍標準差             → 跳躍幅度
├── Mixture 平靜日機率          → 市場狀態切換
└── Mixture 波動日放大倍數      → 極端日波動

混合預測 = 關鍵字 (60%) + ML 模型 (40%)
└── 關鍵字經 GA 優化篩選
```

### AI 交易引擎

```
即時股價 → 策略引擎 → 風險引擎 → LLM 決策 → 執行 / 放棄
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
               GPT Trader     Gemini Trader    Human Trader
              (ai_trader)    (gemini_trader)  (human_trader)
                    │               │               │
              ai_portfolio    gemini_portfolio     手動
```

### 為什麼要肥尾？

真實股市的日報酬不是高斯分布 — 極端事件（黑天鵝）發生的頻率遠高於常態分布的預測。本系統用 GA 從歷史回測中自動 fit 出最佳的 PDF 參數，而非手動假設。

| 參數 | 範圍 | 說明 |
|------|------|------|
| `df` | 2-30 | Student-t 自由度，越小尾巴越厚（df=30 約等於高斯） |
| `jump_intensity` | 0-0.15 | 每日跳躍機率（0 = 無黑天鵝） |
| `jump_std` | 0.01-0.10 | 跳躍大小的標準差 |
| `mixture_calm_prob` | 0.70-0.98 | 平靜日佔比 |
| `mixture_vol_mult` | 2-8 | 波動日的波動率放大倍數 |

---

## 快速開始

### 1. 安裝依賴

```bash
pip install pandas numpy scikit-learn beautifulsoup4 requests matplotlib openai flask
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入你的 API Key（OpenAI / Gemini / Discord）
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

### 主排程

```bash
# 每日全流程（盤前→盤中→盤後）
python daily_stock_job.py

# 測試模式（發到測試頻道）
python daily_stock_job.py --test
```

### 情報監控

```bash
# 24/7 情報監控（每小時掃描 RSS/GitHub/搜尋，≥7 分推 Discord）
python daily_stock_job.py --intel

# 單次情報掃描
python daily_stock_job.py --intel-once

# 獨立執行情報監控
python intel_monitor.py --test           # 推到測試頻道
python intel_monitor.py --dry-run        # 不推播，只看結果
python intel_monitor.py --sources rss,github  # 只掃指定來源
```

### AI 預測

```bash
# 預測單一股票
python predict_stock.py 台積電

# 預測全部股票
python predict_stock.py --all

# 方向性粒子模型（單一股票）
python directional_particle_model.py 2330

# 方向性粒子模型（全部）
python directional_particle_model.py
```

### 盤中系統

```bash
# 盤中粒子模型測試
python intraday_particle_model.py --date 20260302 --stock 2330

# 盤中走勢圖 PDF
python intraday_chart.py --date 20260302
python intraday_chart.py --stock 2330 --output my_chart.pdf

# 合併多日 Tick 資料
python merge_intraday.py --days 30
python merge_intraday.py --start 20260201 --end 20260228 --stock 2330 --output merged.csv
python merge_intraday.py --summary   # 只看每日摘要

# 打包 Tick → K 線
python package_intraday_data.py --date 20260302
```

### 回測 & 策略評估

```bash
# 盤中回測
python intraday_backtest.py --days 30
python intraday_backtest.py --date 20260302 --stock 2330 --csv result.csv

# 盤中策略評估（含 Walk-forward）
python intraday_eval.py --report                     # 完整 4 策略報告
python intraday_eval.py --all                        # 全排列組合測試
python intraday_eval.py --walk-forward --train-days 5
python intraday_eval.py --signal RULE --exit STRATEGY --stock 2330 --csv out.csv
python intraday_eval.py --cost-sensitivity           # 手續費敏感度

# 盤中參數 GA 優化
python intraday_ga_optimize.py --days 20 --pop 50 --gen 30

# 日級回測
python backtest.py
python trading_backtest.py
```

### 優化

```bash
# GA 權重優化（17 參數）
python optimize_weights.py

# NSGA-II 多目標優化
python nsga2_optimize.py
python nsga2_optimize.py --discord    # 結果推送 Discord

# GA 關鍵字優化
python keyword_optimizer.py
```

### 訓練

```bash
# 快速訓練 ML 模型
python quick_train.py

# 完整訓練流程
python train_sentiment_model.py
```

### 通知 & 日誌

```bash
# 設定 Discord webhook
python notifier.py --setup "YOUR_WEBHOOK_URL"

# 測試通知
python notifier.py --test
python notifier.py --test-embed

# 查看通知去重狀態
python notification_guard.py

# 查看廣播日誌報告
python broadcast_logger.py

# 分析日誌
python analyze_logs.py --latest
python analyze_logs.py 2026-02-06

# 查看預測歷史指標
python prediction_history.py
```

### AI 交易引擎

```bash
# 測試 GPT 交易引擎
python ai_trader.py

# 測試 Gemini 交易引擎
python gemini_trader.py
```

### Web UI

```bash
# 啟動 Web 儀表板
python web/app.py
python web/app.py --port 8080 --debug
python web/app.py --no-ssl              # 搭配 ngrok 用

# 頁面：
#   /              → 儀表板總覽
#   /predictions   → 預測結果
#   /trading       → AI 交易紀錄
#   /stock/<code>  → 個股詳情
#   /news          → 新聞列表
#   /volume        → 成交量分佈
#   /intel         → 情報監控
#   /about         → 關於

# API：
#   /api/predictions    → 預測資料 JSON
#   /api/trading        → 交易紀錄 JSON
#   /api/intel          → 情報資料（?category=AI&min_score=7&limit=20）
#   /api/intel/stats    → 情報統計
```

### Discord Bot（管理小助手）

```bash
# 啟動 Bot
python discord_bot.py

# 需要設定 Bot Token（二擇一）：
#   方式 1: 環境變數
export DISCORD_BOT_TOKEN='你的token'
#   方式 2: 寫入 notify_config.json
#   { "discord_bot_token": "你的token" }
```

#### Slash Commands 一覽

**紙上交易 PK**（盤中 09:00~13:30 才能交易）

| 指令 | 說明 | 範例 |
|------|------|------|
| `/buy <stock> <lots>` | 買入股票（即時價成交） | `/buy 2330 2` 買 2 張台積電 |
| `/sell <stock> <lots>` | 賣出股票（顯示損益） | `/sell 2330 1` 賣 1 張 |
| `/status` | 查看持倉 + 未實現損益 + 勝率統計 | |
| `/pk` | 人類 vs GPT vs Gemini PK 計分板 | |
| `/leaderboard` | 全玩家排行榜（含 AI） | |
| `/history` | 近 10 筆交易紀錄 | |
| `/reset` | 歸零回到 100 萬（需確認） | |
| `/predict <stock>` | 粒子模型預測 + 15 分鐘扇形圖 | `/predict 2330` |
| `/help` | 顯示所有指令說明 | |

> `/buy`、`/sell`、`/status`、`/predict` 都支援 `private` 參數（私密模式，只有自己看得到）

**音樂工具**

| 指令 | 說明 |
|------|------|
| `/mp3 <url>` | 下載 YouTube MP3 |
| `/video <url>` | 下載 YouTube MP4 影片 |
| `/dep <url>` | Demucs 四軌分離（人聲 / 鼓 / 貝斯 / 其他） |
| `/pdf <url>` | YouTube → 鋼琴樂譜 PDF + MIDI |
| `/midi <url>` | YouTube → MIDI（人聲+伴奏合併） |
| `/jianpu <url>` | YouTube → 簡譜 PDF（數字譜+和弦+歌詞） |

**自動觸發**（不需打指令，直接操作即可）

| 動作 | Bot 反應 |
|------|----------|
| 貼 YouTube 連結 | 自動下載 MP3 回傳 |
| 上傳 PDF / DOCX | AI 摘要 → 語音 MP3 回傳 |
| 上傳影片（MP4/MKV 等） | Whisper 語音辨識 → SRT 字幕回傳 |

#### 交易規則

- 初始資金：$1,000,000
- 手續費：0.1425%（買入時扣）
- 證交稅：0.3%（賣出時扣）
- 即時價格來源：TWSE API
- 每位 Discord 用戶獨立持倉（依 user ID）
- GPT / Gemini AI 交易引擎同場競技

### 視覺化 & 驗證

```bash
# 準確度圖表
python plot_accuracy.py

# 台積電預測圖
python plot_tsmc_forecast.py

# 驗證結果視覺化
python plot_validation_results.py

# 盤中資料驗證
python validate_with_intraday_data.py

# 分布驗證
python verify_exponential_distribution.py

# 肥尾整合測試
python test_fat_tail_integration.py
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

### AI 交易通知

- 買入/賣出決策 + 理由
- 策略引擎訊號 + 風險引擎判斷
- GPT / Gemini 各自持倉 + 損益

### 盤後績效 Embed

- 今日表現：預測 N 檔 / 正確 N 檔 / 命中率
- 出手率 (Coverage) + 出手準度 (Precision)
- 最大連錯 + 目前連勝/連敗
- 方向分佈（看漲/看跌各自準確率）
- AI 交易 P&L 結算

### 情報推播

- 分類標籤（股票=紅 / AI=紫 / 半導體=藍）
- Gemini 評分星等（≥7 才推播）
- 摘要 + 來源連結

---

## 免責聲明

本專案僅供研究/學習，不構成投資建議。股票投資有風險，請謹慎評估。

---

## 作者

- **rubby0605** - [GitHub](https://github.com/rubby0605)

## 協作

- Claude Opus 4.6 (AI Assistant)
