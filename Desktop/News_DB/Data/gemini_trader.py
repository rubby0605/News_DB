#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini AI 紙上交易引擎

與 GPT 版 (ai_trader.py) 完全相同的邏輯，
只是改用 Google Gemini 做決策，用獨立帳戶做 PK 比賽。

@author: rubylintu
"""

import os
import json
import datetime
import logging
import re
import urllib.request
import urllib.error

from true_particle_trading_model import generate_distribution_chart, load_pdf_params_from_weights

from config import (
    GEMINI_PORTFOLIO_FILE,
    BROKER_FEE_RATE, SECURITIES_TAX_RATE, LOT_SIZE,
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_WEIGHT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    BUY_CONFIDENCE, BUY_BIAS, SELL_CONFIDENCE, MAX_WARNINGS,
    MIN_HOLD_DAYS, COOLDOWN_HOURS,
    COLOR_BULLISH, COLOR_BEARISH, COLOR_INFO, COLOR_WARNING,
    COLOR_PROFIT, COLOR_LOSS,
)

logger = logging.getLogger(__name__)

# ─── Gemini REST API（不依賴 SDK 版本）───

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _call_gemini(prompt, system_prompt="", image_b64=None):
    """
    直接用 REST API 呼叫 Gemini，不依賴 google-generativeai SDK。
    支援傳入 base64 圖片讓 Gemini vision 分析。
    回傳 Gemini 生成的文字。
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY 環境變數未設定")

    url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"

    # 組裝 parts：文字 + 可選圖片
    parts = [{"text": prompt}]
    if image_b64:
        parts.append({"text": "以下是粒子模擬的報酬率分布圖（肥尾 PDF），請參考分布形狀判斷風險："})
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": image_b64
            }
        })

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    # 解析 Gemini 回覆
    candidates = result.get('candidates', [])
    if not candidates:
        raise ValueError(f"Gemini 無回覆: {result}")
    text = candidates[0]['content']['parts'][0]['text']
    return text


# ─── Gemini 交易績效評分（與 GPT 版相同）───

def build_performance_report(trade_history, positions):
    """
    根據歷史交易生成 Gemini 的「成績單」。
    評分邏輯與 GPT 版完全相同。
    """
    if not trade_history:
        return "", 0

    recent = trade_history[-20:]

    wins = [t for t in recent if t['realized_pnl'] >= 0]
    losses = [t for t in recent if t['realized_pnl'] < 0]
    win_rate = len(wins) / len(recent) if recent else 0

    hold_hours = []
    for t in recent:
        try:
            buy_dt = datetime.datetime.fromisoformat(t['buy_time'])
            sell_dt = datetime.datetime.fromisoformat(t['sell_time'])
            hours = (sell_dt - buy_dt).total_seconds() / 3600
            hold_hours.append(hours)
        except Exception:
            pass

    avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0
    short_trades = sum(1 for h in hold_hours if h < 24)

    total_fees = sum(t.get('total_fees', 0) for t in recent)
    total_volume = sum(abs(t.get('net_proceeds', 0)) + abs(t.get('buy_cost', 0)) for t in recent)
    fee_ratio = total_fees / total_volume * 100 if total_volume > 0 else 0

    total_pnl = sum(t['realized_pnl'] for t in recent)

    score = 50
    score += (win_rate - 0.5) * 40
    if avg_hold_hours >= 72:
        score += 15
    elif avg_hold_hours >= 24:
        score += 5
    else:
        score -= 20

    churn_rate = short_trades / len(recent) if recent else 0
    if churn_rate > 0.5:
        score -= 20

    if fee_ratio > 3:
        score -= 15
    elif fee_ratio > 1:
        score -= 5

    if total_pnl > 0:
        score += 10
    elif total_pnl < -10000:
        score -= 10

    score = max(0, min(100, score))

    grade = "S" if score >= 90 else "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 40 else "D" if score >= 20 else "F"

    from collections import Counter
    traded_codes = [t['stock_code'] for t in recent]
    repeat_stocks = {code: cnt for code, cnt in Counter(traded_codes).items() if cnt >= 3}
    repeat_warning = ""
    if repeat_stocks:
        repeat_str = ", ".join(f"{code}({cnt}次)" for code, cnt in repeat_stocks.items())
        repeat_warning = f"\n⚠️ 重複交易警告: {repeat_str} — 同一支反覆買賣只會燒手續費！"

    report = f"""=== 你的交易成績單（評分 {score}/100 等級 {grade}）===
近{len(recent)}筆: {len(wins)}勝{len(losses)}敗 勝率{win_rate:.0%}
平均持有: {avg_hold_hours:.0f}小時（{avg_hold_hours/24:.1f}天）
短線交易(<24hr): {short_trades}/{len(recent)}筆 ({churn_rate:.0%})
手續費總額: ${total_fees:,.0f}（佔交易額{fee_ratio:.1f}%）
近期淨損益: ${total_pnl:+,.0f}{repeat_warning}
{"🏆 表現優異，繼續保持！" if score >= 75 else ""}{"⚠️ 你交易太頻繁了！持有耐心不足，手續費正在吞噬你的利潤。" if avg_hold_hours < 48 else ""}{"💀 你正在瘋狂洗單！立刻停止頻繁買賣，每次交易成本約0.585%，10次就虧6%！" if churn_rate > 0.5 else ""}{"📈 加分建議：多 HOLD、少交易，讓獲利奔跑。" if score < 60 else ""}"""

    return report, score


# ─── Gemini Agent 決策 ───

def ask_gemini_decision(all_predictions, portfolio_summary, positions, recent_accuracy=None, trade_history=None, ta_reports=None):
    """
    用 Gemini 做交易決策：與 GPT 版完全相同的 prompt，只是改用 Gemini API。

    Returns:
        list[dict]: [{"action": "buy"/"sell"/"hold", "code": "2330", "reason": "..."}]
    """
    try:
        # 先檢查 API key
        if not os.environ.get('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY 未設定")
    except Exception as e:
        logger.warning(f"Gemini 初始化失敗，退回規則模式: {e}")
        return None

    # 整理持倉資訊
    holding_lines = []
    for code, pos in positions.items():
        buy_price = pos['buy_price']
        cur_price = pos.get('current_price', buy_price)
        pnl_pct = (cur_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
        days = '未知'
        try:
            buy_dt = datetime.datetime.fromisoformat(pos['buy_time'])
            days = (datetime.datetime.now() - buy_dt).days
        except Exception:
            pass
        holding_lines.append(
            f"  {code} {pos['name']}: 成本${buy_price:.1f} 現價${cur_price:.1f} "
            f"損益{pnl_pct:+.1f}% 持有{days}天 買入理由:{pos.get('reason','')}"
        )
    holdings_str = '\n'.join(holding_lines) if holding_lines else '  無持倉'

    # 技術分析報告
    if ta_reports:
        ta_section = '\n\n'.join(ta_reports[:10])
    else:
        pred_lines = []
        for p in all_predictions:
            direction = p.get('direction', '')
            if direction in ('觀望', ''):
                continue
            code = p.get('stock_code', '')
            name = p.get('stock_name', '')
            confidence = p.get('confidence', 0)
            bias = p.get('bias', 0)
            warnings = p.get('warnings', [])
            signals = p.get('signals', {})
            signal_str = ' | '.join(f"{k}:{v}" for k, v in signals.items()) if signals else ''
            warn_str = f" ⚠️{','.join(warnings)}" if warnings else ''
            held = '【持有中】' if code in positions else ''
            pred_lines.append(
                f"  {code} {name}: {direction} 信心{confidence:.0%} bias{bias:+.1f} "
                f"{signal_str}{warn_str}{held}"
            )
        ta_section = '\n'.join(pred_lines[:20])

    # 績效報告
    perf_report = ""
    perf_score = 50
    if trade_history:
        perf_report, perf_score = build_performance_report(trade_history, positions)

    accuracy_str = f"近5天預測準確率: {recent_accuracy:.0%}" if recent_accuracy else ""

    prompt = f"""根據以下技術分析數據做出交易決策。
{perf_report}

=== 投資組合 ===
現金: ${portfolio_summary['cash']:,.0f}
持倉數: {portfolio_summary['positions_count']}/{MAX_POSITIONS} 檔
總資產: ${portfolio_summary['total_value']:,.0f}
累計損益: ${portfolio_summary['realized_pnl']:+,.0f}
勝率: {portfolio_summary['win_rate']:.0%} ({portfolio_summary['total_trades']}筆)
{accuracy_str}

=== 當前持倉 ===
{holdings_str}

=== 技術分析報告 ===
{ta_section}

=== 交易規則（硬限制）===
- 最多持有 {MAX_POSITIONS} 檔
- 停損 {STOP_LOSS_PCT}%、停利 {TAKE_PROFIT_PCT}%（已自動執行）
- 每檔倉位約 {POSITION_WEIGHT*100:.0f}% 總資產
- 來回交易成本約 0.585%
- 最低持有 {MIN_HOLD_DAYS} 個交易日

=== 決策方法（你必須遵守）===
買入條件（至少滿足 3 項）：
1. 均線多頭排列或短多
2. MACD 柱狀體轉正 或 金叉
3. KD 黃金交叉 或 K>D 且 K<80
4. RSI 50-70 之間（不超買）
5. 股價在布林中軌之上
6. 成交量放大（量比>1.2）
7. 外資買超

賣出條件（持有>{MIN_HOLD_DAYS}天，至少滿足 2 項）：
1. 均線死叉（EMA5 下穿 EMA10）
2. MACD 死叉 或 柱狀體連續縮小
3. KD 死亡交叉 且 K>80
4. RSI > 75 超買
5. 跌破布林中軌
6. 外資連續賣超
7. 跌破支撐位

不交易的情況：
- 指標矛盾（多空訊號各半）→ HOLD
- 量縮盤整 → HOLD
- 持有不到 {MIN_HOLD_DAYS} 天 → HOLD

請用 JSON 格式回覆：
{{"decisions": [
  {{"action": "buy", "code": "股票代碼", "reason": "30字以內買入理由"}},
  {{"action": "sell", "code": "持倉代碼", "reason": "30字以內賣出理由"}},
  {{"action": "hold", "code": "持倉代碼", "reason": "30字以內持有理由"}}
],
"market_view": "20字以內今日盤勢觀點"}}"""

    try:
        system_prompt = (
            "你是專業台股技術分析師兼波段交易員。"
            "你擅長閱讀 K 線、均線、MACD、KD、RSI、布林通道、量價關係。"
            "你的決策必須基於技術指標的交叉確認，不是直覺。"
            f"你的績效評分: {perf_score}/100。"
            f"{'⚠️ 評分偏低！你之前交易太頻繁，現在要更有耐心。' if perf_score < 60 else ''}"
            "原則：多方確認才進場，趨勢反轉才出場，不確定就不動。只回覆 JSON。"
        )

        # 生成粒子模擬分布圖給 Gemini vision 看
        pdf_params = load_pdf_params_from_weights()
        chart_b64 = generate_distribution_chart(all_predictions, pdf_params=pdf_params, n_particles=500)

        text = _call_gemini(prompt, system_prompt=system_prompt, image_b64=chart_b64)

        # Gemini 有時會用 markdown code block 包 JSON
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        # 清理常見的 JSON 問題（trailing comma 等）
        text = re.sub(r',\s*([}\]])', r'\1', text)

        data = json.loads(text)
        decisions = data.get('decisions', [])
        market_view = data.get('market_view', '')

        logger.info(f"Gemini 交易決策: {len(decisions)} 個指令, 盤勢觀點: {market_view}")
        for d in decisions:
            logger.info(f"  {d['action'].upper()} {d['code']}: {d['reason']}")

        return decisions

    except Exception as e:
        logger.error(f"Gemini 決策失敗: {e}")
        return None


class GeminiTrader:
    """Gemini AI 紙上交易引擎（與 AITrader 邏輯完全相同，獨立帳戶）"""

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.cumulative_stats = {
            'total_realized_pnl': 0.0,
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
        }
        self._loaded_date = None
        self.load_portfolio()

    # ─── 持久化（使用 gemini_portfolio.json）───

    def load_portfolio(self):
        if not os.path.exists(GEMINI_PORTFOLIO_FILE):
            return
        try:
            with open(GEMINI_PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        today = datetime.date.today().isoformat()
        saved_date = data.get('date', '')

        self.cash = data.get('cash', self.initial_capital)
        self.initial_capital = data.get('initial_capital', INITIAL_CAPITAL)
        self.positions = data.get('positions', {})
        self.trade_history = data.get('trade_history', [])
        self.cumulative_stats = data.get('cumulative_stats', self.cumulative_stats)
        self._loaded_date = saved_date

        if saved_date == today:
            self.daily_pnl = data.get('daily_pnl', 0.0)
        else:
            self.daily_pnl = 0.0

    def save_portfolio(self):
        data = {
            'date': datetime.date.today().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': round(self.cash, 2),
            'positions': self.positions,
            'trade_history': self.trade_history[-90:],
            'daily_pnl': round(self.daily_pnl, 2),
            'cumulative_stats': self.cumulative_stats,
        }
        with open(GEMINI_PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ─── 核心交易邏輯（與 AITrader 完全相同）───

    def evaluate_all_with_gemini(self, all_predictions, current_prices, recent_accuracy=None, ta_reports=None):
        """Gemini Agent 主入口"""
        results = []

        # 1. 硬性停損停利
        for code in list(self.positions.keys()):
            price = current_prices.get(code)
            if not price:
                continue
            pos = self.positions[code]
            pnl_pct = (price - pos['buy_price']) / pos['buy_price'] * 100

            if pnl_pct <= STOP_LOSS_PCT:
                result = self.execute_sell(code, price, f'停損 ({pnl_pct:+.1f}%)')
                if result:
                    results.append(result)
            elif pnl_pct >= TAKE_PROFIT_PCT:
                result = self.execute_sell(code, price, f'停利 ({pnl_pct:+.1f}%)')
                if result:
                    results.append(result)

        # 2. 更新持倉現價
        for code, pos in self.positions.items():
            if code in current_prices:
                pos['current_price'] = current_prices[code]

        # 3. 呼叫 Gemini 做決策
        summary = self.get_portfolio_summary(current_prices)
        decisions = ask_gemini_decision(
            all_predictions, summary, self.positions, recent_accuracy,
            trade_history=self.trade_history, ta_reports=ta_reports
        )

        if not decisions:
            logger.warning("Gemini 決策失敗，退回規則模式")
            return self._fallback_rule_based(all_predictions, current_prices) + results

        # 4. 執行決策（有 guardrails）
        gemini_log = []
        for d in decisions:
            action = d.get('action', '')
            code = d.get('code', '')
            reason = d.get('reason', 'Gemini 決策')

            if action == 'buy' and code not in self.positions:
                # Cooldown
                recently_sold = False
                for t in reversed(self.trade_history):
                    if t['stock_code'] == code:
                        try:
                            sell_dt = datetime.datetime.fromisoformat(t['sell_time'])
                            hours_since = (datetime.datetime.now() - sell_dt).total_seconds() / 3600
                            if hours_since < COOLDOWN_HOURS:
                                recently_sold = True
                                gemini_log.append(f"BLOCKED BUY {code}: 賣出後僅{hours_since:.0f}小時，冷卻{COOLDOWN_HOURS}小時")
                        except Exception:
                            pass
                        break
                if recently_sold:
                    continue

                price = current_prices.get(code)
                pred = next((p for p in all_predictions if p.get('stock_code') == code), None)
                name = pred.get('stock_name', code) if pred else code

                if price and price > 0 and len(self.positions) < MAX_POSITIONS:
                    result = self.execute_buy(code, name, price, pred or {}, reason_override=reason)
                    if result:
                        result['gemini_reason'] = reason
                        results.append(result)
                        gemini_log.append(f"BUY {code} {name}: {reason}")

            elif action == 'sell' and code in self.positions:
                pos = self.positions[code]
                try:
                    buy_dt = datetime.datetime.fromisoformat(pos['buy_time'])
                    hold_days = (datetime.datetime.now() - buy_dt).days
                except Exception:
                    hold_days = 0

                if hold_days < MIN_HOLD_DAYS:
                    cur_price = current_prices.get(code, pos['buy_price'])
                    pnl_pct = (cur_price - pos['buy_price']) / pos['buy_price'] * 100
                    if pnl_pct > STOP_LOSS_PCT:
                        gemini_log.append(f"BLOCKED SELL {code}: 持有僅{hold_days}天 < {MIN_HOLD_DAYS}天，繼續持有")
                        continue

                price = current_prices.get(code, self.positions[code]['buy_price'])
                result = self.execute_sell(code, price, f'Gemini: {reason}')
                if result:
                    results.append(result)
                    gemini_log.append(f"SELL {code}: {reason}")

            elif action == 'hold':
                gemini_log.append(f"HOLD {code}: {reason}")

        if gemini_log:
            logger.info(f"Gemini Agent 執行結果:\n  " + '\n  '.join(gemini_log))

        return results

    def _fallback_rule_based(self, all_predictions, current_prices):
        results = []
        for pred in all_predictions:
            code = pred.get('stock_code', '')
            name = pred.get('stock_name', '')
            price = current_prices.get(code)
            if not price or price <= 0:
                continue
            result = self.evaluate_and_trade(code, name, price, pred)
            if result:
                results.append(result)
        return results

    def evaluate_and_trade(self, stock_code, stock_name, current_price, prediction):
        if not prediction or not current_price or current_price <= 0:
            return None
        if stock_code in self.positions:
            sell_result = self._check_sell_conditions(stock_code, current_price, prediction)
            if sell_result:
                return sell_result
        if stock_code not in self.positions:
            buy_result = self._check_buy_conditions(stock_code, stock_name, current_price, prediction)
            if buy_result:
                return buy_result
        return None

    def _check_buy_conditions(self, stock_code, stock_name, current_price, prediction):
        direction = prediction.get('direction', '')
        confidence = prediction.get('confidence', 0)
        bias = prediction.get('bias', 0)
        warnings = prediction.get('warnings', [])

        if direction != '漲':
            return None
        if confidence < BUY_CONFIDENCE:
            return None
        if bias < BUY_BIAS:
            return None
        if len(warnings) > MAX_WARNINGS:
            return None
        if len(self.positions) >= MAX_POSITIONS:
            return None

        return self.execute_buy(stock_code, stock_name, current_price, prediction)

    def _check_sell_conditions(self, stock_code, current_price, prediction):
        pos = self.positions[stock_code]
        buy_price = pos['buy_price']
        pnl_pct = (current_price - buy_price) / buy_price * 100

        if pnl_pct <= STOP_LOSS_PCT:
            return self.execute_sell(stock_code, current_price, f'停損 ({pnl_pct:+.1f}%)')
        if pnl_pct >= TAKE_PROFIT_PCT:
            return self.execute_sell(stock_code, current_price, f'停利 ({pnl_pct:+.1f}%)')

        direction = prediction.get('direction', '')
        confidence = prediction.get('confidence', 0)
        if direction == '跌' and confidence >= SELL_CONFIDENCE:
            return self.execute_sell(stock_code, current_price, f'方向反轉 跌 {confidence:.0%}')

        return None

    def execute_buy(self, stock_code, stock_name, price, prediction, reason_override=None):
        positions_value = sum(p['shares'] * p['buy_price'] for p in self.positions.values())
        total_value = self.cash + positions_value

        position_value = total_value * POSITION_WEIGHT
        position_value = min(position_value, self.cash * 0.95)

        if position_value <= 0:
            return None

        if price * LOT_SIZE <= position_value:
            shares = int(position_value / price / LOT_SIZE) * LOT_SIZE
        else:
            shares = int(position_value / price)

        if shares <= 0:
            return None

        gross_cost = shares * price
        broker_fee = round(gross_cost * BROKER_FEE_RATE, 2)
        total_cost = gross_cost + broker_fee

        if total_cost > self.cash:
            return None

        self.cash -= total_cost

        if reason_override:
            reason = reason_override
        else:
            signals = prediction.get('signals', {})
            reason_parts = []
            for key in ['foreign', 'momentum', 'ema']:
                if key in signals:
                    reason_parts.append(signals[key])
            reason = ' | '.join(reason_parts) if reason_parts else f"信心度 {prediction.get('confidence', 0):.0%}"

        self.positions[stock_code] = {
            'name': stock_name,
            'shares': shares,
            'buy_price': price,
            'buy_cost': round(total_cost, 2),
            'buy_time': datetime.datetime.now().isoformat(),
            'reason': reason,
            'broker_fee': broker_fee,
        }

        self.save_portfolio()
        logger.info(f"Gemini 買入 {stock_name}({stock_code}) {shares}股 @ ${price:.1f} 共${total_cost:,.0f}")

        return {
            'action': 'buy',
            'stock_code': stock_code,
            'stock_name': stock_name,
            'price': price,
            'shares': shares,
            'lots': shares // LOT_SIZE if shares >= LOT_SIZE else 0,
            'odd_shares': shares % LOT_SIZE if shares >= LOT_SIZE else shares,
            'amount': total_cost,
            'broker_fee': broker_fee,
            'reason': reason,
            'portfolio_summary': self.get_portfolio_summary(),
        }

    def execute_sell(self, stock_code, price, reason):
        pos = self.positions.get(stock_code)
        if not pos:
            logger.warning(f"Gemini 嘗試賣出 {stock_code} 但無持倉")
            return None

        shares = pos['shares']
        buy_price = pos['buy_price']
        buy_cost = pos['buy_cost']
        buy_time = pos['buy_time']
        stock_name = pos['name']

        gross_proceeds = shares * price
        sell_broker_fee = round(gross_proceeds * BROKER_FEE_RATE, 2)
        sell_tax = round(gross_proceeds * SECURITIES_TAX_RATE, 2)
        total_sell_cost = sell_broker_fee + sell_tax
        net_proceeds = gross_proceeds - total_sell_cost

        realized_pnl = net_proceeds - buy_cost
        pnl_pct = realized_pnl / buy_cost * 100 if buy_cost > 0 else 0

        self.positions.pop(stock_code)
        self.cash += net_proceeds

        try:
            buy_dt = datetime.datetime.fromisoformat(buy_time)
            hold_duration = datetime.datetime.now() - buy_dt
            hold_str = self._format_duration(hold_duration)
        except Exception:
            hold_str = '未知'

        trade_record = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'buy_price': buy_price,
            'sell_price': price,
            'shares': shares,
            'buy_cost': buy_cost,
            'net_proceeds': round(net_proceeds, 2),
            'realized_pnl': round(realized_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'total_fees': round(pos.get('broker_fee', 0) + sell_broker_fee + sell_tax, 2),
            'buy_time': buy_time,
            'sell_time': datetime.datetime.now().isoformat(),
            'hold_duration': hold_str,
            'reason': reason,
        }
        self.trade_history.append(trade_record)

        self.daily_pnl += realized_pnl
        self.cumulative_stats['total_realized_pnl'] = round(
            self.cumulative_stats['total_realized_pnl'] + realized_pnl, 2)
        self.cumulative_stats['total_trades'] += 1
        if realized_pnl >= 0:
            self.cumulative_stats['win_count'] += 1
        else:
            self.cumulative_stats['loss_count'] += 1

        self.save_portfolio()
        logger.info(f"Gemini 賣出 {stock_name}({stock_code}) {shares}股 @ ${price:.1f} "
                    f"損益 ${realized_pnl:+,.0f} ({pnl_pct:+.1f}%) [{reason}]")

        return {
            'action': 'sell',
            'stock_code': stock_code,
            'stock_name': stock_name,
            'price': price,
            'shares': shares,
            'amount': round(net_proceeds, 2),
            'realized_pnl': round(realized_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'hold_duration': hold_str,
            'sell_broker_fee': sell_broker_fee,
            'sell_tax': sell_tax,
            'reason': reason,
            'portfolio_summary': self.get_portfolio_summary(),
        }

    # ─── Portfolio 查詢 ───

    def get_portfolio_summary(self, current_prices=None):
        positions_value = 0
        unrealized_pnl = 0
        positions_detail = []

        for code, pos in self.positions.items():
            cur_price = current_prices.get(code, pos['buy_price']) if current_prices else pos['buy_price']
            value = pos['shares'] * cur_price
            pnl = value - pos['buy_cost']
            pnl_pct = pnl / pos['buy_cost'] * 100 if pos['buy_cost'] else 0

            positions_value += value
            unrealized_pnl += pnl

            positions_detail.append({
                'code': code,
                'name': pos['name'],
                'shares': pos['shares'],
                'buy_price': pos['buy_price'],
                'current_price': cur_price,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            })

        total_value = self.cash + positions_value
        total_return = total_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100

        stats = self.cumulative_stats
        total_trades = stats['total_trades']
        win_rate = stats['win_count'] / total_trades if total_trades > 0 else 0

        return {
            'cash': round(self.cash, 2),
            'positions_count': len(self.positions),
            'positions_value': round(positions_value, 2),
            'total_value': round(total_value, 2),
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'realized_pnl': stats['total_realized_pnl'],
            'daily_pnl': round(self.daily_pnl, 2),
            'total_trades': total_trades,
            'win_count': stats['win_count'],
            'loss_count': stats['loss_count'],
            'win_rate': round(win_rate, 4),
            'positions_detail': positions_detail,
        }

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.save_portfolio()

    @staticmethod
    def _format_duration(td):
        total_seconds = int(td.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        if days > 0:
            return f'{days}天{hours}時'
        elif hours > 0:
            return f'{hours}時{minutes}分'
        else:
            return f'{minutes}分'


# ─── Discord Embed 建構（標記為 Gemini）───

def build_gemini_buy_embed(trade_result):
    """建構 Gemini 買入通知 Embed"""
    code = trade_result['stock_code']
    name = trade_result['stock_name']
    price = trade_result['price']
    shares = trade_result['shares']
    amount = trade_result['amount']
    fee = trade_result['broker_fee']
    reason = trade_result['reason']
    summary = trade_result['portfolio_summary']

    lots = shares // LOT_SIZE
    odd = shares % LOT_SIZE
    if lots > 0 and odd > 0:
        shares_text = f'{lots} 張 + {odd} 股'
    elif lots > 0:
        shares_text = f'{lots} 張 ({shares:,} 股)'
    else:
        shares_text = f'{shares} 股 (零股)'

    now = datetime.datetime.now()

    embed = {
        "title": f"💎 Gemini 買入 | {code} {name}",
        "color": COLOR_BULLISH,
        "fields": [
            {"name": "買入價", "value": f"**${price:,.1f}**", "inline": True},
            {"name": "股數", "value": shares_text, "inline": True},
            {"name": "金額", "value": f"${amount:,.0f}\n(手續費 ${fee:,.0f})", "inline": True},
            {"name": "買入理由", "value": reason[:200], "inline": False},
            {"name": "剩餘現金", "value": f"${summary['cash']:,.0f}", "inline": True},
            {"name": "持倉", "value": f"{summary['positions_count']}/{MAX_POSITIONS} 檔", "inline": True},
        ],
        "footer": {
            "text": f"Gemini 紙上交易 | 總資產 ${summary['total_value']:,.0f} | {now.strftime('%H:%M')}"
        },
    }
    return embed


def build_gemini_sell_embed(trade_result):
    """建構 Gemini 賣出通知 Embed"""
    code = trade_result['stock_code']
    name = trade_result['stock_name']
    price = trade_result['price']
    shares = trade_result['shares']
    pnl = trade_result['realized_pnl']
    pnl_pct = trade_result['pnl_pct']
    hold = trade_result['hold_duration']
    reason = trade_result['reason']
    fee = trade_result['sell_broker_fee']
    tax = trade_result['sell_tax']
    summary = trade_result['portfolio_summary']

    color = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS
    pnl_emoji = '💰' if pnl >= 0 else '💸'

    now = datetime.datetime.now()

    embed = {
        "title": f"{'🟢' if pnl >= 0 else '🔴'} Gemini 賣出 | {code} {name}",
        "color": color,
        "fields": [
            {"name": "賣出價", "value": f"**${price:,.1f}**", "inline": True},
            {"name": "股數", "value": f"{shares:,} 股", "inline": True},
            {"name": f"{pnl_emoji} 損益", "value": f"**${pnl:+,.0f}** ({pnl_pct:+.1f}%)", "inline": True},
            {"name": "持有時間", "value": hold, "inline": True},
            {"name": "賣出理由", "value": reason, "inline": True},
            {"name": "交易成本", "value": f"手續費 ${fee:,.0f} + 稅 ${tax:,.0f}", "inline": True},
        ],
        "footer": {
            "text": (f"Gemini 紙上交易 | 持倉 {summary['positions_count']}/{MAX_POSITIONS} | "
                     f"總資產 ${summary['total_value']:,.0f} | "
                     f"累計損益 ${summary['realized_pnl']:+,.0f} | {now.strftime('%H:%M')}")
        },
    }
    return embed


def build_gemini_daily_portfolio_embed(trader, current_prices=None):
    """建構 Gemini 每日交易日報 Embed"""
    summary = trader.get_portfolio_summary(current_prices)
    now = datetime.datetime.now()

    if summary['positions_detail']:
        pos_lines = []
        for p in summary['positions_detail']:
            pnl_emoji = '📈' if p['pnl'] >= 0 else '📉'
            pos_lines.append(
                f"{pnl_emoji} {p['name']}({p['code']}): "
                f"{p['shares']}股 @ ${p['buy_price']:,.1f} → ${p['current_price']:,.1f} "
                f"({p['pnl_pct']:+.1f}%)"
            )
        positions_text = '\n'.join(pos_lines)
    else:
        positions_text = '無持倉'

    total_trades = summary['total_trades']
    if total_trades > 0:
        win_text = f"{summary['win_rate']:.0%} ({summary['win_count']}勝{summary['loss_count']}敗 / 共{total_trades}筆)"
    else:
        win_text = '尚無交易'

    total_return = summary['total_return']
    return_emoji = '🚀' if total_return > 0 else '📉' if total_return < 0 else '➡️'

    embed = {
        "title": f"💎 Gemini 交易日報 | {now.strftime('%Y/%m/%d')}",
        "color": COLOR_INFO,
        "fields": [
            {"name": "持倉明細", "value": positions_text, "inline": False},
            {"name": "現金", "value": f"${summary['cash']:,.0f}", "inline": True},
            {"name": "總資產", "value": f"${summary['total_value']:,.0f}", "inline": True},
            {"name": f"{return_emoji} 累計報酬",
             "value": f"${total_return:+,.0f} ({summary['total_return_pct']:+.1f}%)", "inline": True},
            {"name": "今日已實現損益", "value": f"${summary['daily_pnl']:+,.0f}", "inline": True},
            {"name": "未實現損益", "value": f"${summary['unrealized_pnl']:+,.0f}", "inline": True},
            {"name": "勝率", "value": win_text, "inline": True},
        ],
        "footer": {
            "text": f"Gemini 紙上交易 | 初始資金 ${trader.initial_capital:,.0f} | {now.strftime('%H:%M')}"
        },
    }
    return embed


def build_pk_scoreboard_embed(gpt_summary, gemini_summary):
    """建構 GPT vs Gemini PK 計分板 Embed"""
    now = datetime.datetime.now()

    gpt_return = gpt_summary['total_return']
    gemini_return = gemini_summary['total_return']

    if gpt_return > gemini_return:
        winner = "🤖 GPT 領先"
        winner_color = 0x10A37F  # OpenAI green
    elif gemini_return > gpt_return:
        winner = "💎 Gemini 領先"
        winner_color = 0x4285F4  # Google blue
    else:
        winner = "🤝 平手"
        winner_color = COLOR_INFO

    gpt_wr = f"{gpt_summary['win_rate']:.0%}" if gpt_summary['total_trades'] > 0 else "N/A"
    gem_wr = f"{gemini_summary['win_rate']:.0%}" if gemini_summary['total_trades'] > 0 else "N/A"

    embed = {
        "title": f"⚔️ GPT vs Gemini PK | {now.strftime('%Y/%m/%d')}",
        "color": winner_color,
        "fields": [
            {"name": "🤖 GPT 總資產", "value": f"${gpt_summary['total_value']:,.0f}", "inline": True},
            {"name": "💎 Gemini 總資產", "value": f"${gemini_summary['total_value']:,.0f}", "inline": True},
            {"name": "🏆 領先", "value": winner, "inline": True},
            {"name": "🤖 GPT 報酬",
             "value": f"${gpt_return:+,.0f} ({gpt_summary['total_return_pct']:+.1f}%)", "inline": True},
            {"name": "💎 Gemini 報酬",
             "value": f"${gemini_return:+,.0f} ({gemini_summary['total_return_pct']:+.1f}%)", "inline": True},
            {"name": "差距",
             "value": f"${abs(gpt_return - gemini_return):,.0f}", "inline": True},
            {"name": "🤖 GPT 勝率", "value": f"{gpt_wr} ({gpt_summary['total_trades']}筆)", "inline": True},
            {"name": "💎 Gemini 勝率", "value": f"{gem_wr} ({gemini_summary['total_trades']}筆)", "inline": True},
            {"name": "🤖 GPT 持倉", "value": f"{gpt_summary['positions_count']}/{MAX_POSITIONS}", "inline": True},
            {"name": "💎 Gemini 持倉", "value": f"{gemini_summary['positions_count']}/{MAX_POSITIONS}", "inline": True},
        ],
        "footer": {
            "text": f"AI PK 紙上交易 | 初始資金各 $1,000,000 | {now.strftime('%H:%M')}"
        },
    }
    return embed
