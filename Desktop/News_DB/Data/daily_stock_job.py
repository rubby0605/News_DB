#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日股票資料自動抓取排程
- 08:00 盤前分析（粒子模型預測）
- 09:00-13:30 即時股價監控（盤中）
- 13:30 盤後誤差分析

@author: rubylintu
"""

import os
import sys
import time
import datetime
import random
import logging
import json

# 設定工作目錄為腳本所在位置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ⚠️ 必須在 import 自訂模組之前設定 logging，
#    否則 news_collector.py 會先呼叫 basicConfig，搶走 root logger。
LOG_FILE = os.path.join(SCRIPT_DIR, 'logs', f'stock_job_{datetime.date.today()}.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 匯入自訂模組（必須在 logging 設定之後）
from newslib import (
    read_stock_list,
    getGoodInfo,
    craw_realtime,
    get_stock_info
)
from news_collector import collect_all_news
from news_stock_selector import select_focus_stocks_from_news
from notifier import (
    send_daily_report, send_discord, send_multi_embed,
    build_prediction_embed, format_signal_breakdown
)
from notification_guard import NotificationGuard
from broadcast_logger import log_broadcast
from prediction_history import get_tracking_metrics
from ai_trader import (
    AITrader, build_buy_embed, build_sell_embed, build_daily_portfolio_embed,
    build_buy_signal_embed, build_sell_signal_embed
)
from gemini_trader import (
    GeminiTrader, build_gemini_buy_embed, build_gemini_sell_embed,
    build_gemini_daily_portfolio_embed, build_pk_scoreboard_embed
)

# 儲存盤前預測結果（供盤後比較）
PREMARKET_PREDICTIONS = {}

from config import (
    PREDICTIONS_FILE, FOCUS_STOCKS_FILE, STOCK_LIST_FILE,
    DISCORD_CHANNEL as _DEFAULT_DISCORD_CHANNEL,
    AI_TRADE_CHANNEL as _DEFAULT_AI_TRADE_CHANNEL,
    GEMINI_TRADE_CHANNEL as _DEFAULT_GEMINI_TRADE_CHANNEL,
    INITIAL_CAPITAL, COLOR_INFO, COLOR_WARNING,
)

# 盤前新聞選股結果（精追 5 檔）
# {'2330': {'name': '台積電', 'reason': '...', 'news_count': N, 'sentiment_score': 0.8}, ...}
FOCUS_STOCKS = {}

# Discord 頻道：'release' 正式 / 'test' 測試
DISCORD_CHANNEL = _DEFAULT_DISCORD_CHANNEL

# AI 紙上交易系統（100 萬虛擬資金）
AI_TRADER = AITrader(initial_capital=INITIAL_CAPITAL)
AI_TRADE_CHANNEL = _DEFAULT_AI_TRADE_CHANNEL

# Gemini 紙上交易系統（100 萬虛擬資金，獨立帳戶 PK）
GEMINI_TRADER = GeminiTrader(initial_capital=INITIAL_CAPITAL)
GEMINI_TRADE_CHANNEL = _DEFAULT_GEMINI_TRADE_CHANNEL


def save_predictions_to_file():
    """將盤前預測存到 JSON 檔（供盤後讀取）"""
    data = {
        'date': datetime.date.today().isoformat(),
        'predictions': PREMARKET_PREDICTIONS,
        'focus_stocks': FOCUS_STOCKS,
    }
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"預測結果已存檔 ({len(PREMARKET_PREDICTIONS)} 檔)")


def load_predictions_from_file():
    """從 JSON 檔讀取今日盤前預測"""
    global PREMARKET_PREDICTIONS, FOCUS_STOCKS
    if not os.path.exists(PREDICTIONS_FILE):
        return False
    try:
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('date') != datetime.date.today().isoformat():
            logger.warning("預測檔案非今日資料，跳過")
            return False
        PREMARKET_PREDICTIONS = data.get('predictions', {})
        if not FOCUS_STOCKS and data.get('focus_stocks'):
            FOCUS_STOCKS = data['focus_stocks']
        logger.info(f"從檔案載入 {len(PREMARKET_PREDICTIONS)} 筆盤前預測")
        return True
    except Exception as e:
        logger.error(f"載入預測檔案失敗: {e}")
        return False


def select_focus_stocks():
    """
    盤前新聞選股 - 從 31 檔中選出 5 檔今日焦點
    """
    global FOCUS_STOCKS
    logger.info("=== 盤前新聞選股 ===")

    try:
        selected = select_focus_stocks_from_news(num_stocks=5)

        if not selected:
            logger.warning("新聞選股未選出任何股票")
            return

        # 存入全域變數
        FOCUS_STOCKS = {}
        for s in selected:
            FOCUS_STOCKS[s['code']] = {
                'name': s['name'],
                'reason': s['reason'],
                'news_count': s['news_count'],
                'sentiment_score': s['sentiment_score'],
            }

        # 發送 Discord 通知
        now = datetime.datetime.now()
        lines = [
            f'**📰 今日新聞精選 {len(selected)} 檔**',
            f'📅 {now.strftime("%Y/%m/%d")} {now.strftime("%H:%M")}',
            '',
        ]

        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        for i, s in enumerate(selected):
            medal = medals[i] if i < len(medals) else f'{i+1}.'
            lines.append(f"{medal} {s['name']} ({s['code']})")
            lines.append(f"   └ 理由：{s['reason']}")

        remaining = 31 - len(selected)
        lines.append('')
        lines.append(f'📊 其餘 {remaining} 檔無重大新聞異動')

        message = '\n'.join(lines)
        send_discord(message, title='盤前新聞選股', channel=DISCORD_CHANNEL)
        logger.info(f"盤前選股完成，選出 {len(selected)} 檔焦點股票")

    except Exception as e:
        logger.error(f"盤前新聞選股失敗: {e}")


def send_premarket_analysis():
    """
    盤前分析 - 使用粒子模型 + GPT 新聞情緒預測並發送到 Discord
    以新聞精選 5 檔為主角，其餘以摘要呈現
    """
    global PREMARKET_PREDICTIONS
    logger.info("=== 開始盤前分析 ===")

    # 焦點股票名稱集合（從 FOCUS_STOCKS 取得）
    focus_names = {v['name'] for v in FOCUS_STOCKS.values()} if FOCUS_STOCKS else {'群聯', '景碩'}
    focus_codes = set(FOCUS_STOCKS.keys()) if FOCUS_STOCKS else set()

    try:
        from directional_particle_model import DirectionalParticleModel
        from newslib import read_stock_list

        model = DirectionalParticleModel(n_particles=1000)
        stock_list_file = STOCK_LIST_FILE
        dict_stock = read_stock_list(stock_list_file)

        # GPT 新聞情緒分析（焦點股票）
        gpt_sentiments = {}
        try:
            from gpt_sentiment import analyze_stock_with_news
            for name in focus_names:
                result = analyze_stock_with_news(name)
                gpt_sentiments[name] = result
                logger.info(f"GPT 盤前分析 {name}: {result.get('sentiment')} ({result.get('confidence', 0):.0%})")
        except Exception as e:
            logger.warning(f"GPT 分析失敗: {e}")

        # 對全部股票做粒子模型預測（焦點股整合 GPT 情緒 + 肥尾模型）
        results = []
        for name, code in dict_stock.items():
            gpt_data = gpt_sentiments.get(name) if name in focus_names else None
            # 焦點股使用肥尾模型（更精確但較慢）
            is_focus = (name in focus_names) or (str(code) in focus_codes)
            result = model.predict(str(code), name, gpt_sentiment=gpt_data,
                                  use_fat_tail=is_focus)
            if 'error' not in result:
                results.append(result)
                PREMARKET_PREDICTIONS[result['stock_code']] = {
                    'name': result['stock_name'],
                    'predicted_price': result['predicted_price'],
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'current_price': result['current_price'],
                    'is_focus': result['stock_code'] in focus_codes,
                    'has_gpt': gpt_data is not None,
                    'bias': result.get('bias', 0),
                    'signals': result.get('signals', {}),
                    'warnings': result.get('warnings', []),
                }
                # 記錄預測（供系統偏差自動修正用）
                try:
                    from prediction_history import record_prediction
                    record_prediction(result['stock_code'], result['direction'],
                                      result['confidence'], result['bias'])
                except Exception:
                    pass

        # 分出焦點股票和其餘股票的預測結果
        focus_results = [r for r in results if r['stock_code'] in focus_codes or r['stock_name'] in focus_names]
        other_results = [r for r in results if r['stock_code'] not in focus_codes and r['stock_name'] not in focus_names]

        # 其餘股票分類
        other_bulls = [r for r in other_results if r['direction'] == '漲']
        other_bears = [r for r in other_results if r['direction'] == '跌']
        other_neutral = [r for r in other_results if r['direction'] in ('盤整', '觀望')]

        other_bulls.sort(key=lambda x: x['expected_change'], reverse=True)
        other_bears.sort(key=lambda x: x['expected_change'])

        # 全部分類（統計用）
        all_bulls = [r for r in results if r['direction'] == '漲']
        all_bears = [r for r in results if r['direction'] == '跌']
        all_neutral = [r for r in results if r['direction'] == '盤整']
        all_wait = [r for r in results if r['direction'] == '觀望']

        # 組合訊息
        now = datetime.datetime.now()
        lines = [
            '**📊 盤前分析報告**',
            f'📅 {now.strftime("%Y/%m/%d")} {now.strftime("%H:%M")}',
        ]

        # === 新聞精選焦點區 ===
        if FOCUS_STOCKS:
            lines.append('')
            lines.append(f'**⭐ 新聞精選 {len(FOCUS_STOCKS)} 檔（完整分析）：**')
            medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
            for i, (code, info) in enumerate(FOCUS_STOCKS.items()):
                medal = medals[i] if i < len(medals) else f'{i+1}.'
                name = info['name']

                # 找到對應的粒子模型預測（焦點股用肥尾模型）
                pred = next((r for r in focus_results if r['stock_code'] == code or r['stock_name'] == name), None)
                if pred:
                    foreign = pred['signals'].get('foreign', '')
                    foreign_info = f' [{foreign}]' if '買超' in foreign or '大買' in foreign or '賣超' in foreign or '大賣' in foreign else ''
                    fat_tail_mark = ' 🎯' if True else ''  # 焦點股都用肥尾模型
                    lines.append(f"{medal} {name}({code}): ${pred['current_price']:.0f}→${pred['predicted_price']:.0f} ({pred['expected_change']:+.1f}%) [{pred['direction']} {pred['confidence']:.0%}]{foreign_info}{fat_tail_mark}")
                else:
                    lines.append(f"{medal} {name}({code}): 無預測資料")

                lines.append(f"   └ 選股理由：{info['reason']}")

                # GPT 情緒
                gpt = gpt_sentiments.get(name)
                if gpt:
                    sentiment = gpt.get('sentiment', '中性')
                    confidence = gpt.get('confidence', 0)
                    reason = gpt.get('reason', '')
                    emoji = "🔴" if sentiment == '漲' else "🟢" if sentiment == '跌' else "⚪"
                    lines.append(f"   └ GPT情緒: {emoji} {sentiment} ({confidence:.0%}) {reason}")
        else:
            # 沒有焦點股票時，維持舊的重點關注
            lines.append('')
            lines.append('**⭐ 重點關注：**')
            for r in results:
                if r['stock_name'] in ['群聯', '景碩']:
                    foreign = r['signals'].get('foreign', '')
                    momentum = r['signals'].get('momentum', '')
                    lines.append(f"• {r['stock_name']}: ${r['current_price']:.0f}→${r['predicted_price']:.0f} ({r['expected_change']:+.1f}%) [{r['direction']} {r['confidence']:.0%}]")
                    lines.append(f"  └ {foreign}, {momentum}")

        # === 其餘看漲/看跌摘要 ===
        lines.append('')
        lines.append('**🔴 其餘看漲 TOP 5：**')
        for r in other_bulls[:5]:
            lines.append(f"• {r['stock_name']}: {r['expected_change']:+.1f}% [{r['direction']} {r['confidence']:.0%}]")

        lines.append('')
        lines.append('**🟢 其餘看跌 TOP 5：**')
        for r in other_bears[:5]:
            lines.append(f"• {r['stock_name']}: {r['expected_change']:+.1f}% [{r['direction']} {r['confidence']:.0%}]")

        lines.append('')
        wait_str = f' | 觀望 {len(all_wait)} 檔' if all_wait else ''
        lines.append(f'**📈 統計：** 看漲 {len(all_bulls)} 檔 | 看跌 {len(all_bears)} 檔 | 盤整 {len(all_neutral)} 檔{wait_str}')

        message = '\n'.join(lines)
        send_discord(message, title='盤前 AI 分析', channel=DISCORD_CHANNEL)
        logger.info(f"盤前分析完成，預測 {len(results)} 檔股票")

        # 存檔供盤後比對
        save_predictions_to_file()

    except Exception as e:
        logger.error(f"盤前分析失敗: {e}")


def send_postmarket_analysis():
    """
    盤後誤差分析 - 比較預測 vs 實際收盤價
    """
    global PREMARKET_PREDICTIONS
    logger.info("=== 開始盤後誤差分析 ===")

    # 記憶體沒有預測資料 → 從檔案載入
    if not PREMARKET_PREDICTIONS:
        load_predictions_from_file()

    if not PREMARKET_PREDICTIONS:
        logger.warning("沒有盤前預測資料（記憶體和檔案都沒有），跳過誤差分析")
        return

    try:
        from newslib import read_stock_list, craw_realtime

        stock_list_file = STOCK_LIST_FILE
        dict_stock = read_stock_list(stock_list_file)
        stock_list = [int(dict_stock[stock]) for stock in dict_stock.keys()]

        # 抓取收盤價
        data = craw_realtime(stock_list)

        if not data or 'msgArray' not in data or len(data['msgArray']) == 0:
            logger.error("無法取得收盤資料")
            return

        # 比較預測與實際
        results = []
        correct_direction = 0
        total_compared = 0

        for item in data['msgArray']:
            code = item.get('c', '')
            actual_price = item.get('z', '-')
            yesterday = item.get('y', '-')

            if code in PREMARKET_PREDICTIONS and actual_price != '-':
                pred = PREMARKET_PREDICTIONS[code]
                actual_price = float(actual_price)
                yesterday_price = float(yesterday) if yesterday != '-' else pred['current_price']

                # 計算實際漲跌
                actual_change = (actual_price - yesterday_price) / yesterday_price * 100
                actual_direction = '漲' if actual_change > 0.5 else '跌' if actual_change < -0.5 else '盤整'

                # 記錄結果（供系統偏差自動修正用）
                try:
                    from prediction_history import record_outcome
                    record_outcome(code, actual_direction, actual_change)
                except Exception:
                    pass

                # 廣播日誌回填實際結果
                try:
                    from broadcast_logger import update_outcomes
                    update_outcomes(
                        datetime.date.today().isoformat(),
                        {code: {
                            'actual_direction': actual_direction,
                            'actual_close': actual_price,
                            'actual_change': actual_change,
                        }}
                    )
                except Exception:
                    pass

                # 計算預測誤差
                pred_error = abs(pred['predicted_price'] - actual_price) / actual_price * 100

                # 觀望類別不計入方向準確率
                if pred['direction'] == '觀望':
                    results.append({
                        'name': pred['name'],
                        'code': code,
                        'predicted': pred['predicted_price'],
                        'actual': actual_price,
                        'pred_direction': pred['direction'],
                        'actual_direction': actual_direction,
                        'actual_change': actual_change,
                        'error': pred_error,
                        'correct': None  # 觀望不判斷對錯
                    })
                    continue

                # 方向是否正確
                direction_correct = (pred['direction'] == actual_direction) or \
                                   (pred['direction'] == '漲' and actual_change > 0) or \
                                   (pred['direction'] == '跌' and actual_change < 0)

                if direction_correct:
                    correct_direction += 1
                total_compared += 1

                results.append({
                    'name': pred['name'],
                    'code': code,
                    'predicted': pred['predicted_price'],
                    'actual': actual_price,
                    'pred_direction': pred['direction'],
                    'actual_direction': actual_direction,
                    'actual_change': actual_change,
                    'error': pred_error,
                    'correct': direction_correct
                })

        # 計算準確率
        accuracy = correct_direction / total_compared * 100 if total_compared > 0 else 0

        # 分出焦點股票和其餘股票
        focus_codes = set(FOCUS_STOCKS.keys()) if FOCUS_STOCKS else set()
        focus_names = {v['name'] for v in FOCUS_STOCKS.values()} if FOCUS_STOCKS else set()
        focus_results = [r for r in results if r['code'] in focus_codes or r['name'] in focus_names]
        other_results = [r for r in results if r['code'] not in focus_codes and r['name'] not in focus_names]

        # 焦點股票準確率（排除觀望）
        focus_judged = [r for r in focus_results if r['correct'] is not None]
        focus_correct = sum(1 for r in focus_judged if r['correct'])
        focus_total = len(focus_judged)
        focus_accuracy = focus_correct / focus_total * 100 if focus_total > 0 else 0

        # 按誤差排序
        results.sort(key=lambda x: x['error'])

        # 組合訊息
        now = datetime.datetime.now()
        lines = [
            '**📊 盤後誤差分析報告**',
            f'📅 {now.strftime("%Y/%m/%d")} 收盤',
            '',
            f'**🎯 整體方向準確率: {accuracy:.1f}%** ({correct_direction}/{total_compared})',
        ]

        # 焦點股票表現
        if focus_results:
            lines.append(f'**🎯 焦點股準確率: {focus_accuracy:.1f}%** ({focus_correct}/{focus_total})')
            lines.append('')
            lines.append('**⭐ 新聞焦點股表現：**')
            for r in focus_results:
                emoji = '🔴' if r['actual_change'] > 0 else '🟢' if r['actual_change'] < 0 else '⚪'
                status = '✓' if r['correct'] else ('—' if r['correct'] is None else '✗')
                lines.append(f"{emoji} {r['name']}: 預測{r['pred_direction']}→實際${r['actual']:.0f} ({r['actual_change']:+.1f}%) 誤差{r['error']:.1f}% {status}")

        lines.append('')
        lines.append('**✅ 預測正確 TOP 5：**')

        correct_results = [r for r in results if r['correct'] is True]
        for r in correct_results[:5]:
            emoji = '🔴' if r['actual_change'] > 0 else '🟢' if r['actual_change'] < 0 else '⚪'
            focus_tag = ' ⭐' if (r['code'] in focus_codes or r['name'] in focus_names) else ''
            lines.append(f"{emoji} {r['name']}: 預測{r['pred_direction']} → 實際{r['actual_change']:+.1f}% ✓{focus_tag}")

        lines.append('')
        lines.append('**❌ 預測錯誤：**')

        wrong_results = [r for r in results if r['correct'] is False]
        for r in wrong_results[:5]:
            emoji = '🔴' if r['actual_change'] > 0 else '🟢' if r['actual_change'] < 0 else '⚪'
            focus_tag = ' ⭐' if (r['code'] in focus_codes or r['name'] in focus_names) else ''
            lines.append(f"{emoji} {r['name']}: 預測{r['pred_direction']} → 實際{r['actual_change']:+.1f}% ✗{focus_tag}")

        # 觀望結果
        wait_results = [r for r in results if r['correct'] is None]
        if wait_results:
            lines.append('')
            lines.append(f'**⏸️ 觀望 {len(wait_results)} 檔（不計入準確率）：**')
            for r in wait_results[:5]:
                emoji = '🔴' if r['actual_change'] > 0 else '🟢' if r['actual_change'] < 0 else '⚪'
                lines.append(f"{emoji} {r['name']}: 實際{r['actual_change']:+.1f}%")

        # 統計
        avg_error = sum(r['error'] for r in results) / len(results) if results else 0
        lines.append('')
        lines.append(f'**📈 平均價格誤差: {avg_error:.1f}%**')
        if wait_results:
            lines.append(f'**⏸️ 觀望: {len(wait_results)} 檔** | 有效預測: {total_compared} 檔')

        message = '\n'.join(lines)
        send_discord(message, title='盤後誤差分析', channel=DISCORD_CHANNEL)
        logger.info(f"盤後分析完成，準確率 {accuracy:.1f}%")

        # 發送每日績效 Embed
        send_daily_metrics_summary()

        # AI 紙上交易：盤後日報（交易決策已在盤中每 15 分鐘執行）
        try:
            closing_prices = {}
            for item in data['msgArray']:
                code = item.get('c', '')
                price_str = item.get('z', '-')
                if price_str != '-':
                    closing_prices[code] = float(price_str)

            # 更新持倉現價（用收盤價）
            for code, pos in AI_TRADER.positions.items():
                if code in closing_prices:
                    pos['current_price'] = closing_prices[code]

            # 發送每日投資組合日報
            from notifier import send_discord_embed
            portfolio_embed = build_daily_portfolio_embed(AI_TRADER, closing_prices)
            send_discord_embed(portfolio_embed, channel=AI_TRADE_CHANNEL)
            logger.info("GPT 每日交易日報已發送")

            # Gemini 盤後日報
            for code, pos in GEMINI_TRADER.positions.items():
                if code in closing_prices:
                    pos['current_price'] = closing_prices[code]
            gemini_embed = build_gemini_daily_portfolio_embed(GEMINI_TRADER, closing_prices)
            send_discord_embed(gemini_embed, channel=GEMINI_TRADE_CHANNEL)
            logger.info("Gemini 每日交易日報已發送")

            # GPT vs Gemini PK 計分板
            gpt_summary = AI_TRADER.get_portfolio_summary(closing_prices)
            gemini_summary = GEMINI_TRADER.get_portfolio_summary(closing_prices)
            pk_embed = build_pk_scoreboard_embed(gpt_summary, gemini_summary)
            send_discord_embed(pk_embed, channel=AI_TRADE_CHANNEL)
            logger.info("GPT vs Gemini PK 計分板已發送")
        except Exception as e:
            logger.error(f"AI 交易日報發送失敗: {e}")

        # 清空預測資料
        PREMARKET_PREDICTIONS = {}

    except Exception as e:
        logger.error(f"盤後分析失敗: {e}")


def send_daily_metrics_summary():
    """盤後發送每日績效追蹤 Embed"""
    try:
        from prediction_history import get_tracking_metrics, calc_advanced_metrics
        from notifier import build_metrics_embed, send_discord_embed

        today_metrics = get_tracking_metrics()
        advanced_metrics = calc_advanced_metrics()

        embed = build_metrics_embed(today_metrics, advanced_metrics)
        send_discord_embed(embed, channel=DISCORD_CHANNEL)
        logger.info("每日績效 Embed 已發送")
    except Exception as e:
        logger.error(f"每日績效 Embed 發送失敗: {e}")


def fetch_fundamental_data():
    """
    抓取股票基本面資料並存成 CSV
    優先使用 GoodInfo，若失敗則使用證交所 API
    """
    logger.info("=== 開始抓取基本面資料 ===")

    stock_list_file = STOCK_LIST_FILE
    output_file = os.path.join(SCRIPT_DIR, 'Data', 'stock_data.csv')

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dict_stock = read_stock_list(stock_list_file)
    stock_list_str = dict_stock.keys()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Name,code,price,open,high,low,yesterday,volume\n')

        for i, stock in enumerate(stock_list_str, 1):
            try:
                num = dict_stock[stock]
                # 使用證交所即時 API
                url, data = get_stock_info(num)
                # data = [代號, 名稱, 成交價, 成交量, 累積量, 開盤價, 最高價, 最低價, 昨收價]
                code, name, price, tv, volume, open_p, high, low, yesterday = data
                f.write(f'{stock},{code},{price},{open_p},{high},{low},{yesterday},{volume}\n')
                logger.info(f"[{i}/{len(dict_stock)}] {stock}({code}): ${price}")
                time.sleep(0.3)  # 避免請求過快
            except Exception as e:
                logger.error(f"抓取 {stock} 失敗: {e}")

    logger.info(f"基本面資料已儲存至: {output_file}")


def send_prediction_notification(stock_prices, clf, vectorizer, now, taiex_info=None):
    """
    發送股票預測通知到 Discord
    - 只顯示漲跌幅大的股票
    - 加入重要新聞標題
    - 顯示大盤即時點數
    """
    from hybrid_predictor import hybrid_predict
    from newslib import scrapBingNews, scrapGoogleNews
    import re

    logger.info("發送 15 分鐘預測通知...")

    # 優先關注的股票
    PRIORITY_STOCKS = ['群聯', '景碩']
    CHANGE_THRESHOLD = 1.5  # 漲跌幅超過 1.5% 才顯示

    # 建立通知內容
    lines = [
        f"**{now.strftime('%H:%M')} 盤中快報**",
    ]

    # 大盤即時點數
    if taiex_info:
        try:
            idx_price = float(taiex_info.get('z', 0))
            idx_yesterday = float(taiex_info.get('y', 0))
            if idx_price > 0 and idx_yesterday > 0:
                idx_change = idx_price - idx_yesterday
                idx_pct = (idx_change / idx_yesterday) * 100
                idx_emoji = "🔴" if idx_change > 0 else "🟢" if idx_change < 0 else "⚪"
                lines.append(f"{idx_emoji} 加權指數: **{idx_price:,.2f}** ({idx_change:+,.2f} / {idx_pct:+.2f}%)")
            else:
                # 盤中 z 可能是 '-'，用最高/最低估算
                lines.append(f"📊 加權指數: 等待成交...")
        except Exception:
            pass

    # 計算每檔股票的漲跌幅
    stock_changes = []
    for s in stock_prices:
        if s['price'] == '-':
            continue
        try:
            price = float(s['price'])
            yesterday = float(s['yesterday']) if s['yesterday'] != '-' else price
            change_pct = ((price - yesterday) / yesterday) * 100
            stock_changes.append({
                'name': s['name'],
                'code': s['code'],
                'price': price,
                'change_pct': change_pct,
                'is_priority': s['name'] in PRIORITY_STOCKS
            })
        except Exception:
            continue

    # 依漲跌幅排序（漲最多在前）
    stock_changes.sort(key=lambda x: x['change_pct'], reverse=True)

    # 顯示全部股票
    if stock_changes:
        lines.append("")
        for s in stock_changes:
            emoji = "🔴" if s['change_pct'] > 0 else "🟢" if s['change_pct'] < 0 else "⚪"
            priority_tag = " ⭐" if s['is_priority'] else ""
            lines.append(f"{emoji} {s['name']}({s['code']}): ${s['price']:.1f} ({s['change_pct']:+.1f}%){priority_tag}")

    # 抓取重要新聞並分析
    if clf and vectorizer:
        lines.append("")
        lines.append("**📰 重要新聞：**")

        news_items = []
        # 針對優先股票抓新聞
        for stock_name in PRIORITY_STOCKS[:2]:
            try:
                url, title, body, bs = scrapBingNews(stock_name)
                if body:
                    # 提取新聞句子
                    sentences = re.split(r'[。！？\n]', body)
                    for sent in sentences[:3]:
                        sent = sent.strip()
                        if len(sent) > 15 and stock_name in sent:
                            pred, conf, _ = hybrid_predict(sent, clf, vectorizer)
                            news_items.append({
                                'text': sent[:50] + '...' if len(sent) > 50 else sent,
                                'prediction': pred,
                                'stock': stock_name
                            })
                            break
            except Exception:
                continue

        if news_items:
            for item in news_items[:3]:
                emoji = "🟢" if item['prediction'] == '漲' else "🔴" if item['prediction'] == '跌' else "⚪"
                lines.append(f"{emoji} [{item['stock']}] {item['text']}")
        else:
            lines.append("（暫無重大新聞）")

    # 統計摘要
    bull_count = sum(1 for s in stock_changes if s['change_pct'] > 0)
    bear_count = sum(1 for s in stock_changes if s['change_pct'] < 0)
    lines.append("")
    lines.append(f"📈 上漲: {bull_count} 檔 | 📉 下跌: {bear_count} 檔")

    message = "\n".join(lines)

    try:
        send_discord(message, title="盤中即時更新", channel=DISCORD_CHANNEL)
        logger.info("Discord 通知已發送")
    except Exception as e:
        logger.error(f"發送通知失敗: {e}")


def monitor_realtime_prices():
    """
    即時股價監控
    只在台股開盤時間（09:00-13:30）執行
    每 15 分鐘發送 Discord 通知
    """
    logger.info("=== 開始即時股價監控 ===")

    stock_list_file = STOCK_LIST_FILE
    db_file = os.path.join(SCRIPT_DIR, 'Data', 'trace_stock_DB.txt')

    columns = ['c', 'n', 'z', 'tv', 'v', 'o', 'h', 'l', 'y']
    # ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']

    dict_stock = read_stock_list(stock_list_file)
    stock_list = [int(dict_stock[stock]) for stock in dict_stock.keys()]
    stock_names = {v: k for k, v in dict_stock.items()}  # 代號 -> 名稱

    iteration = 0
    last_notify_time = None  # 上次通知時間

    # 載入預測模型
    try:
        from hybrid_predictor import hybrid_predict, load_ml_model
        clf, vectorizer = load_ml_model()
    except Exception:
        clf, vectorizer = None, None

    with open(db_file, 'a', encoding='utf-8') as fi:
        while True:
            now = datetime.datetime.now()
            current_time = now.time()

            # 台股交易時間：09:00 - 13:30
            market_open = datetime.time(9, 0)
            market_close = datetime.time(13, 30)

            # 檢查是否為週末
            if now.weekday() >= 5:  # 週六=5, 週日=6
                logger.info("今天是週末，停止監控")
                break

            # 檢查是否超過收盤時間
            if current_time > market_close:
                logger.info("已過收盤時間 (13:30)，停止監控")
                break

            # 如果還沒開盤，等待
            if current_time < market_open:
                wait_seconds = (datetime.datetime.combine(now.date(), market_open) - now).seconds
                logger.info(f"等待開盤... ({wait_seconds} 秒後)")
                time.sleep(min(wait_seconds, 60))  # 最多等 60 秒後再檢查
                continue

            # 抓取即時資料
            try:
                data = craw_realtime(stock_list)

                if 'msgArray' not in data or len(data['msgArray']) == 0:
                    logger.warning("無法取得即時資料，等待重試...")
                    time.sleep(10)
                    continue

                # 收集股價資料
                stock_prices = []
                for i in range(min(len(dict_stock), len(data['msgArray']))):
                    item = data['msgArray'][i]
                    line = ''
                    for column in columns:
                        value = item.get(column, '-')
                        line = line + '\t' + str(value)
                    line = line + '\t' + str(now) + '\n'
                    fi.write(line)

                    # 記錄股價資訊
                    code = item.get('c', '')
                    name = item.get('n', stock_names.get(code, code))
                    price = item.get('z', '-')
                    yesterday = item.get('y', '-')
                    open_price = item.get('o', '-')
                    stock_prices.append({
                        'code': code,
                        'name': name,
                        'price': price,
                        'yesterday': yesterday,
                        'open': open_price
                    })

                fi.flush()  # 確保寫入磁碟
                iteration += 1

                # 每 15 分鐘發送 Discord 通知
                should_notify = False
                if last_notify_time is None:
                    should_notify = True
                elif (now - last_notify_time).total_seconds() >= 900:  # 900秒 = 15分鐘
                    should_notify = True

                if should_notify:
                    # 抓取大盤加權指數
                    taiex_info = None
                    try:
                        from urllib.request import urlopen as _urlopen
                        taiex_url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw"
                        taiex_data = json.loads(_urlopen(taiex_url).read())
                        if 'msgArray' in taiex_data and len(taiex_data['msgArray']) > 0:
                            taiex_info = taiex_data['msgArray'][0]
                    except Exception as e:
                        logger.warning(f"抓取大盤指數失敗: {e}")

                    send_prediction_notification(stock_prices, clf, vectorizer, now, taiex_info)

                    # GPT Agent 盤中即時決策（每 15 分鐘）
                    try:
                        # 建立即時價格 dict
                        realtime_prices = {}
                        for sp in stock_prices:
                            if sp['price'] != '-':
                                try:
                                    realtime_prices[sp['code']] = float(sp['price'])
                                except (ValueError, TypeError):
                                    pass

                        # 準備焦點股預測資料
                        focus_preds = []
                        for code, pred in PREMARKET_PREDICTIONS.items():
                            if code in realtime_prices:
                                pred_copy = dict(pred)
                                pred_copy['stock_code'] = code
                                pred_copy['stock_name'] = pred.get('name', code)
                                focus_preds.append(pred_copy)

                        if focus_preds:
                            recent_accuracy = None
                            try:
                                metrics = get_tracking_metrics()
                                if metrics and metrics.get('accuracy_5d'):
                                    recent_accuracy = metrics['accuracy_5d'] / 100.0
                            except Exception:
                                pass

                            # 生成完整技術分析報告給 GPT-4o
                            ta_reports = []
                            try:
                                from directional_particle_model import (
                                    build_ta_report, get_stock_history,
                                    get_institutional_data
                                )
                                inst_data = get_institutional_data()
                                for pred in focus_preds:
                                    code = pred.get('stock_code', '')
                                    name = pred.get('stock_name', code)
                                    try:
                                        hist = get_stock_history(code, days=30)
                                        if hist:
                                            report = build_ta_report(
                                                code, name, hist, inst_data
                                            )
                                            ta_reports.append(report)
                                    except Exception as e:
                                        logger.warning(f"TA report {code} 失敗: {e}")
                            except Exception as e:
                                logger.warning(f"TA reports 生成失敗: {e}")

                            trade_results = AI_TRADER.evaluate_all_with_gpt(
                                focus_preds, realtime_prices, recent_accuracy,
                                ta_reports=ta_reports if ta_reports else None
                            )

                            # 發送交易結果到 Discord
                            from notifier import send_discord_embed
                            for result in trade_results:
                                if result['action'] == 'buy':
                                    embed = build_buy_embed(result)
                                    send_discord_embed(embed, channel=AI_TRADE_CHANNEL)
                                elif result['action'] == 'sell':
                                    embed = build_sell_embed(result)
                                    send_discord_embed(embed, channel=AI_TRADE_CHANNEL)

                            if trade_results:
                                logger.info(f"盤中 GPT Agent 執行 {len(trade_results)} 筆交易")
                    except Exception as e:
                        logger.error(f"盤中 GPT Agent 決策失敗: {e}")

                    # Gemini Agent 盤中即時決策（與 GPT 同步，獨立帳戶）
                    try:
                        if focus_preds:
                            gemini_results = GEMINI_TRADER.evaluate_all_with_gemini(
                                focus_preds, realtime_prices, recent_accuracy,
                                ta_reports=ta_reports if ta_reports else None
                            )

                            from notifier import send_discord_embed
                            for result in gemini_results:
                                if result['action'] == 'buy':
                                    embed = build_gemini_buy_embed(result)
                                    send_discord_embed(embed, channel=GEMINI_TRADE_CHANNEL)
                                elif result['action'] == 'sell':
                                    embed = build_gemini_sell_embed(result)
                                    send_discord_embed(embed, channel=GEMINI_TRADE_CHANNEL)

                            if gemini_results:
                                logger.info(f"盤中 Gemini Agent 執行 {len(gemini_results)} 筆交易")
                    except Exception as e:
                        logger.error(f"盤中 Gemini Agent 決策失敗: {e}")

                    last_notify_time = now

                if iteration % 10 == 0:
                    logger.info(f"已執行 {iteration} 次，時間: {now.strftime('%H:%M:%S')}")

            except Exception as e:
                logger.error(f"抓取即時資料錯誤: {e}")

            # 隨機等待 10-20 秒
            time.sleep(10 + random.random() * 10)

    logger.info(f"即時監控結束，共執行 {iteration} 次")


def main():
    """主程式"""
    logger.info("=" * 50)
    logger.info("每日股票資料抓取程式啟動")
    logger.info(f"日期: {datetime.date.today()}")
    logger.info("=" * 50)

    now = datetime.datetime.now()

    # 檢查是否為週末
    if now.weekday() >= 5:
        logger.info("今天是週末，不執行")
        return

    # 檢查是否為台股休市日（國定假日）
    MARKET_HOLIDAYS_2026 = {
        '2026-01-01',  # 元旦
        '2026-01-27', '2026-01-28', '2026-01-29', '2026-01-30',  # 春節調整
        '2026-02-14',  # 除夕
        '2026-02-15', '2026-02-16', '2026-02-17', '2026-02-18',  # 春節
        '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22', '2026-02-23',  # 春節連假
        '2026-02-27',  # 228 連假調整
        '2026-02-28',  # 和平紀念日
        '2026-04-03',  # 清明連假
        '2026-04-04',  # 清明節
        '2026-04-05',  # 清明連假
        '2026-05-01',  # 勞動節
        '2026-06-19',  # 端午連假
        '2026-06-20',  # 端午節
        '2026-10-05',  # 中秋節
        '2026-10-09',  # 國慶連假
        '2026-10-10',  # 國慶日
    }
    today_str = datetime.date.today().isoformat()
    if today_str in MARKET_HOLIDAYS_2026:
        logger.info(f"今天 {today_str} 是台股休市日，不執行")
        return

    # 重置 AI 交易日報
    AI_TRADER.reset_daily()
    GEMINI_TRADER.reset_daily()

    # 1. 抓取基本面資料
    try:
        fetch_fundamental_data()
    except Exception as e:
        logger.error(f"基本面資料抓取失敗: {e}")

    # 2. 收集新聞資料（用於 AI 訓練）
    try:
        logger.info("=== 開始收集新聞 ===")
        collect_all_news()
    except Exception as e:
        logger.error(f"新聞收集失敗: {e}")

    # 3. 盤前新聞選股（5 檔焦點）
    try:
        select_focus_stocks()
    except Exception as e:
        logger.error(f"盤前新聞選股失敗: {e}")

    # 4. 盤前分析（粒子模型預測）→ 發送 Discord
    try:
        send_premarket_analysis()
    except Exception as e:
        logger.error(f"盤前分析失敗: {e}")

    # 5. 即時股價監控（等到 9:00 開盤後開始）
    try:
        monitor_realtime_prices()
    except Exception as e:
        logger.error(f"即時監控失敗: {e}")

    # 6. 盤後誤差分析 → 發送 Discord
    try:
        send_postmarket_analysis()
    except Exception as e:
        logger.error(f"盤後分析失敗: {e}")

    # 7. 收盤後發送每日報告（13:30）
    try:
        send_daily_report(
            news_count=0,
            focus_stocks=FOCUS_STOCKS,
            premarket_predictions=PREMARKET_PREDICTIONS,
            channel=DISCORD_CHANNEL
        )
    except Exception as e:
        logger.error(f"發送每日報告失敗: {e}")

    # 8. 每日盤後 GA 優化（rolling window + 穩定性檢查）
    try:
        logger.info("=== 每日 GA 權重優化 ===")
        run_daily_ga_optimization()
    except Exception as e:
        logger.error(f"每日 GA 優化失敗: {e}")

    logger.info("今日任務完成")


def run_daily_ga_optimization():
    """每日盤後 GA 優化（rolling window + 穩定性檢查）"""
    from optimize_weights import run_daily_optimization, load_weights

    logger.info("開始每日 GA 優化...")

    result = run_daily_optimization(
        stock_codes=['2330', '3189', '2454', '2881', '2603'],
        rolling_days=40,
        population_size=30,
        generations=20,
        max_drift=0.25,
        min_improvement=0.005
    )

    # 發送結果到 Discord
    from notifier import send_discord_embed

    status = "✅ 已更新" if result['updated'] else "⚠️ 未更新"
    color = COLOR_INFO if result['updated'] else COLOR_WARNING

    fields = [
        {"name": "狀態", "value": status, "inline": True},
    ]

    if 'new_acc' in result:
        fields.append({"name": "新準確率", "value": f"{result['new_acc']:.1%}", "inline": True})
        fields.append({"name": "舊準確率", "value": f"{result['old_acc']:.1%}", "inline": True})

    if 'drift' in result:
        fields.append({"name": "權重漂移", "value": f"{result['drift']:.1%}", "inline": True})

    fields.append({"name": "原因", "value": result['reason'], "inline": False})

    embed = {
        "title": f"🧬 每日 GA 優化 | {datetime.date.today()}",
        "color": color,
        "fields": fields,
    }
    send_discord_embed(embed, channel=DISCORD_CHANNEL)

    logger.info(f"每日 GA 優化完成: {result['reason']}")


if __name__ == "__main__":
    if '--test' in sys.argv:
        DISCORD_CHANNEL = 'test'
        logger.info("=== 測試模式：通知發送到 test 頻道 ===")
    main()
