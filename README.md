# News_DB

用 AI 分析新聞情緒預測台股漲跌

> Taiwan Stock Market Intelligence System - AI-powered news sentiment analysis for stock prediction

## 功能特色

- 自動收集股票新聞（Bing/Google）
- AI 情緒分析預測漲跌
- 每日定時執行（08:00 自動啟動）
- 盤中即時監控（09:00-13:30）
- Discord 通知（每 15 分鐘推送預測）
- 新聞自動去重（不重複抓取）

---

## 系統架構

### 核心檔案

| 檔案 | 功能 | 用法 |
|------|------|------|
| `daily_stock_job.py` | **主排程程式** | 每天 08:00 自動執行 |
| `newslib.py` | 核心函式庫 | 爬蟲、API、資料處理 |
| `hybrid_predictor.py` | AI 預測模型 | ML + 關鍵字混合預測 |
| `notifier.py` | Discord 通知 | 發送即時通知 |

### AI 訓練相關

| 檔案 | 功能 | 用法 |
|------|------|------|
| `quick_train.py` | 快速訓練模型 | `python quick_train.py` |
| `train_sentiment_model.py` | 完整訓練流程 | `python train_sentiment_model.py` |
| `news_collector.py` | 新聞收集（自動去重） | 被 daily_job 呼叫 |

### 預測與分析

| 檔案 | 功能 | 用法 |
|------|------|------|
| `predict_stock.py` | 單一股票預測 | `python predict_stock.py 台積電` |
| `plot_accuracy.py` | 準確度圖表 | `python plot_accuracy.py` |

### 資料檔案

| 位置 | 內容 |
|------|------|
| `stock_list_less.txt` | 監控的 34 檔股票清單 |
| `news_data/` | 收集的新聞資料 |
| `news_data/seen_news.json` | 已抓取新聞記錄（去重用） |
| `models/` | 訓練好的 AI 模型 |

---

## 自動化排程

```
每週一到週五 08:00 自動執行：
├── 1. 抓取股票基本面資料
├── 2. 收集新聞（自動去重）
├── 3. 發送 Discord 通知「排程啟動」
├── 4. 09:00-13:30 盤中監控
│   └── 每 15 分鐘發送 Discord 預測通知
└── 5. 收盤後發送每日報告
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install pandas numpy scikit-learn beautifulsoup4 requests matplotlib
```

### 2. 設定 Discord 通知

```bash
cd Data
python notifier.py --setup "YOUR_DISCORD_WEBHOOK_URL"
python notifier.py --test  # 測試通知
```

### 3. 訓練 AI 模型

```bash
python quick_train.py
```

### 4. 手動執行預測

```bash
python predict_stock.py 台積電
python predict_stock.py --all  # 預測所有股票
```

### 5. 設定每日自動執行（cron）

已設定：每週一至五 08:00 自動執行

```bash
crontab -l  # 查看排程
```

---

## 常用指令

```bash
# 手動執行每日任務
python daily_stock_job.py

# 訓練 AI 模型
python quick_train.py

# 預測單一股票
python predict_stock.py 台積電

# 測試 Discord 通知
python notifier.py --test

# 生成準確度圖表
python plot_accuracy.py

# 收集新聞（手動）
python news_collector.py
```

---

## AI 預測模型

### 混合預測策略

```
最終預測 = 關鍵字規則 (60%) + ML 模型 (40%)
```

### 看漲關鍵字
創新高、大漲、漲停、利多、看好、加碼、買超、獲利、成長、突破...

### 看跌關鍵字
下跌、跌停、利空、衰退、虧損、裁員、暴跌、崩盤、賣超...

---

## 免責聲明

本專案僅供研究/學習，不構成投資建議。股票投資有風險，請謹慎評估。

---

## 作者

- **rubby0605** - [GitHub](https://github.com/rubby0605)

## 協作

- Claude Opus 4.5 (AI Assistant)
