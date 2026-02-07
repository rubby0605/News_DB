#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 紙上交易引擎

模擬 100 萬 TWD 虛擬資金自動選股買賣。
基於粒子模型預測（方向 + 信心度 + bias）決定進出場。

@author: rubylintu
"""

import os
import json
import datetime
import logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(SCRIPT_DIR, 'ai_portfolio.json')

# ─── 台股交易成本 ───
BROKER_FEE_RATE = 0.001425    # 0.1425%（買賣都收）
SECURITIES_TAX_RATE = 0.003   # 0.3%（僅賣出）
LOT_SIZE = 1000               # 1 張 = 1000 股

# ─── 交易參數 ───
INITIAL_CAPITAL = 1_000_000   # 100 萬
MAX_POSITIONS = 5             # 最多同時持有
POSITION_WEIGHT = 0.20        # 每檔 20% 總資產
STOP_LOSS_PCT = -3.0          # 停損 %
TAKE_PROFIT_PCT = 5.0         # 停利 %
BUY_CONFIDENCE = 0.70         # 買入最低信心度
BUY_BIAS = 3.0                # 買入最低 bias
SELL_CONFIDENCE = 0.65        # 賣出（反轉）信心度
MAX_WARNINGS = 1              # 買入時最多允許幾個警示

# ─── Discord 顏色（從 notifier.py 一致）───
COLOR_BULLISH = 0xFF4444
COLOR_BEARISH = 0x44FF44
COLOR_INFO = 0x3498DB
COLOR_WARNING = 0xFFAA00
COLOR_PROFIT = 0x2ECC71
COLOR_LOSS = 0xE74C3C


class AITrader:
    """AI 紙上交易引擎"""

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}       # {code: {name, shares, buy_price, buy_cost, buy_time, reason, broker_fee}}
        self.trade_history = []   # list of completed trades
        self.daily_pnl = 0.0
        self.cumulative_stats = {
            'total_realized_pnl': 0.0,
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
        }
        self._loaded_date = None
        self.load_portfolio()

    # ─── 持久化 ───

    def load_portfolio(self):
        """從 ai_portfolio.json 載入"""
        if not os.path.exists(PORTFOLIO_FILE):
            return

        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
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

        # 新的一天 → 重置 daily_pnl
        if saved_date == today:
            self.daily_pnl = data.get('daily_pnl', 0.0)
        else:
            self.daily_pnl = 0.0

    def save_portfolio(self):
        """儲存到 ai_portfolio.json"""
        data = {
            'date': datetime.date.today().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': round(self.cash, 2),
            'positions': self.positions,
            'trade_history': self.trade_history[-90:],  # 只保留 90 筆
            'daily_pnl': round(self.daily_pnl, 2),
            'cumulative_stats': self.cumulative_stats,
        }
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ─── 核心交易邏輯 ───

    def evaluate_and_trade(self, stock_code, stock_name, current_price, prediction):
        """
        主入口：評估預測並決定買賣

        Returns:
            dict or None: 交易結果 {'action': 'buy'/'sell', ...}
        """
        if not prediction or not current_price or current_price <= 0:
            return None

        # 先檢查賣出（持倉股）
        if stock_code in self.positions:
            sell_result = self._check_sell_conditions(stock_code, current_price, prediction)
            if sell_result:
                return sell_result

        # 再檢查買入（未持有股）
        if stock_code not in self.positions:
            buy_result = self._check_buy_conditions(stock_code, stock_name, current_price, prediction)
            if buy_result:
                return buy_result

        return None

    def _check_buy_conditions(self, stock_code, stock_name, current_price, prediction):
        """
        買入條件（全部滿足才買）：
        1. direction == '漲'
        2. confidence >= 0.70
        3. bias >= 3.0
        4. warnings <= MAX_WARNINGS
        5. 未持有此股
        6. 持倉數 < MAX_POSITIONS
        """
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
        """
        賣出條件（任一成立即賣）：
        1. 停損：跌 >= 3%
        2. 停利：漲 >= 5%
        3. 方向反轉：預測跌 + confidence >= 0.65
        """
        pos = self.positions[stock_code]
        buy_price = pos['buy_price']
        pnl_pct = (current_price - buy_price) / buy_price * 100

        # 停損
        if pnl_pct <= STOP_LOSS_PCT:
            return self.execute_sell(stock_code, current_price,
                                    f'停損 ({pnl_pct:+.1f}%)')

        # 停利
        if pnl_pct >= TAKE_PROFIT_PCT:
            return self.execute_sell(stock_code, current_price,
                                    f'停利 ({pnl_pct:+.1f}%)')

        # 方向反轉
        direction = prediction.get('direction', '')
        confidence = prediction.get('confidence', 0)
        if direction == '跌' and confidence >= SELL_CONFIDENCE:
            return self.execute_sell(stock_code, current_price,
                                    f'方向反轉 跌 {confidence:.0%}')

        return None

    def execute_buy(self, stock_code, stock_name, price, prediction):
        """模擬買入"""
        # 計算持倉市值（用買入價近似）
        positions_value = sum(p['shares'] * p['buy_price'] for p in self.positions.values())
        total_value = self.cash + positions_value

        position_value = total_value * POSITION_WEIGHT
        position_value = min(position_value, self.cash * 0.95)  # 留 5% buffer

        if position_value <= 0:
            return None

        # 計算股數（優先整張，不夠就零股）
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

        # 執行買入
        self.cash -= total_cost

        # 從信號提取買入理由
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
        logger.info(f"AI 買入 {stock_name}({stock_code}) {shares}股 @ ${price:.1f} 共${total_cost:,.0f}")

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
        """模擬賣出"""
        pos = self.positions.pop(stock_code)
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
        pnl_pct = realized_pnl / buy_cost * 100

        self.cash += net_proceeds

        # 持有時間
        try:
            buy_dt = datetime.datetime.fromisoformat(buy_time)
            hold_duration = datetime.datetime.now() - buy_dt
            hold_str = self._format_duration(hold_duration)
        except Exception:
            hold_str = '未知'

        # 記錄交易
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

        # 更新統計
        self.daily_pnl += realized_pnl
        self.cumulative_stats['total_realized_pnl'] = round(
            self.cumulative_stats['total_realized_pnl'] + realized_pnl, 2)
        self.cumulative_stats['total_trades'] += 1
        if realized_pnl >= 0:
            self.cumulative_stats['win_count'] += 1
        else:
            self.cumulative_stats['loss_count'] += 1

        self.save_portfolio()
        logger.info(f"AI 賣出 {stock_name}({stock_code}) {shares}股 @ ${price:.1f} "
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

    # ─── 買賣點偵測（不一定成交，但通知用戶）───

    def detect_signals(self, stock_code, stock_name, current_price, prediction):
        """
        偵測買點/賣點訊號（不執行交易，僅回報信號）

        Returns:
            dict or None: {
                'signal': 'buy_signal' / 'sell_signal',
                'stock_code', 'stock_name', 'price',
                'direction', 'confidence', 'bias',
                'reason': str,
                'can_execute': bool,  # 是否可以實際交易
                'block_reason': str,  # 不能交易的原因
            }
        """
        if not prediction or not current_price or current_price <= 0:
            return None

        direction = prediction.get('direction', '')
        confidence = prediction.get('confidence', 0)
        bias = prediction.get('bias', 0)
        warnings = prediction.get('warnings', [])

        # 買點偵測
        if (direction == '漲' and confidence >= BUY_CONFIDENCE and bias >= BUY_BIAS
                and len(warnings) <= MAX_WARNINGS):
            can_execute = True
            block_reason = ''

            if stock_code in self.positions:
                can_execute = False
                block_reason = '已持有'
            elif len(self.positions) >= MAX_POSITIONS:
                can_execute = False
                block_reason = f'持倉已滿 {MAX_POSITIONS} 檔'

            # 從信號提取理由
            signals = prediction.get('signals', {})
            reason_parts = []
            for key in ['foreign', 'momentum', 'ema']:
                if key in signals:
                    reason_parts.append(signals[key])
            reason = ' | '.join(reason_parts) if reason_parts else f'信心度 {confidence:.0%}'

            return {
                'signal': 'buy_signal',
                'stock_code': stock_code,
                'stock_name': stock_name,
                'price': current_price,
                'direction': direction,
                'confidence': confidence,
                'bias': bias,
                'reason': reason,
                'can_execute': can_execute,
                'block_reason': block_reason,
            }

        # 賣點偵測（只對持倉股）
        if stock_code in self.positions:
            pos = self.positions[stock_code]
            buy_price = pos['buy_price']
            pnl_pct = (current_price - buy_price) / buy_price * 100

            sell_reason = None
            if pnl_pct <= STOP_LOSS_PCT:
                sell_reason = f'停損 ({pnl_pct:+.1f}%)'
            elif pnl_pct >= TAKE_PROFIT_PCT:
                sell_reason = f'停利 ({pnl_pct:+.1f}%)'
            elif direction == '跌' and confidence >= SELL_CONFIDENCE:
                sell_reason = f'方向反轉 跌 {confidence:.0%}'

            if sell_reason:
                return {
                    'signal': 'sell_signal',
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'price': current_price,
                    'direction': direction,
                    'confidence': confidence,
                    'bias': bias,
                    'buy_price': buy_price,
                    'pnl_pct': pnl_pct,
                    'reason': sell_reason,
                    'can_execute': True,
                    'block_reason': '',
                }

        return None

    # ─── Portfolio 查詢 ───

    def get_portfolio_summary(self, current_prices=None):
        """取得投資組合摘要"""
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
        """每日重置"""
        self.daily_pnl = 0.0
        self.save_portfolio()

    @staticmethod
    def _format_duration(td):
        """timedelta → 中文可讀"""
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


# ─── Discord Embed 建構 ───

def build_buy_embed(trade_result):
    """建構買入通知 Embed"""
    code = trade_result['stock_code']
    name = trade_result['stock_name']
    price = trade_result['price']
    shares = trade_result['shares']
    amount = trade_result['amount']
    fee = trade_result['broker_fee']
    reason = trade_result['reason']
    summary = trade_result['portfolio_summary']

    # 股數描述
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
        "title": f"🔴 AI 買入 | {code} {name}",
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
            "text": f"紙上交易 | 總資產 ${summary['total_value']:,.0f} | {now.strftime('%H:%M')}"
        },
    }

    return embed


def build_sell_embed(trade_result):
    """建構賣出通知 Embed"""
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

    # 損益顏色
    color = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS
    pnl_emoji = '💰' if pnl >= 0 else '💸'

    now = datetime.datetime.now()

    embed = {
        "title": f"{'🟢' if pnl >= 0 else '🔴'} AI 賣出 | {code} {name}",
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
            "text": (f"紙上交易 | 持倉 {summary['positions_count']}/{MAX_POSITIONS} | "
                     f"總資產 ${summary['total_value']:,.0f} | "
                     f"累計損益 ${summary['realized_pnl']:+,.0f} | {now.strftime('%H:%M')}")
        },
    }

    return embed


def build_buy_signal_embed(signal):
    """建構買點偵測提醒 Embed"""
    code = signal['stock_code']
    name = signal['stock_name']
    price = signal['price']
    confidence = signal['confidence']
    bias = signal['bias']
    reason = signal['reason']
    can_execute = signal['can_execute']
    block_reason = signal.get('block_reason', '')

    now = datetime.datetime.now()

    status = '即將買入' if can_execute else f'無法買入（{block_reason}）'
    status_emoji = '🎯' if can_execute else '⚠️'

    embed = {
        "title": f"📍 買點偵測 | {code} {name}",
        "color": COLOR_BULLISH,
        "fields": [
            {"name": "現價", "value": f"**${price:,.1f}**", "inline": True},
            {"name": "信心度", "value": f"**{confidence:.0%}**", "inline": True},
            {"name": "Bias", "value": f"**{bias:+.1f}**", "inline": True},
            {"name": "訊號依據", "value": reason[:200], "inline": False},
            {"name": f"{status_emoji} 狀態", "value": status, "inline": False},
        ],
        "footer": {"text": f"紙上交易 | 買點提醒 | {now.strftime('%H:%M')}"},
    }

    return embed


def build_sell_signal_embed(signal):
    """建構賣點偵測提醒 Embed"""
    code = signal['stock_code']
    name = signal['stock_name']
    price = signal['price']
    buy_price = signal.get('buy_price', 0)
    pnl_pct = signal.get('pnl_pct', 0)
    reason = signal['reason']

    now = datetime.datetime.now()

    # 停利=綠, 停損=紅
    color = COLOR_PROFIT if pnl_pct >= 0 else COLOR_LOSS
    pnl_emoji = '💰' if pnl_pct >= 0 else '💸'

    embed = {
        "title": f"📍 賣點偵測 | {code} {name}",
        "color": color,
        "fields": [
            {"name": "現價", "value": f"**${price:,.1f}**", "inline": True},
            {"name": "買入價", "value": f"${buy_price:,.1f}", "inline": True},
            {"name": f"{pnl_emoji} 浮動損益", "value": f"**{pnl_pct:+.1f}%**", "inline": True},
            {"name": "賣出理由", "value": reason, "inline": False},
            {"name": "🎯 狀態", "value": "即將賣出", "inline": False},
        ],
        "footer": {"text": f"紙上交易 | 賣點提醒 | {now.strftime('%H:%M')}"},
    }

    return embed


def build_daily_portfolio_embed(trader, current_prices=None):
    """建構每日交易日報 Embed"""
    summary = trader.get_portfolio_summary(current_prices)
    now = datetime.datetime.now()

    # 持倉明細
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

    # 勝率
    total_trades = summary['total_trades']
    if total_trades > 0:
        win_text = f"{summary['win_rate']:.0%} ({summary['win_count']}勝{summary['loss_count']}敗 / 共{total_trades}筆)"
    else:
        win_text = '尚無交易'

    # 總報酬
    total_return = summary['total_return']
    return_emoji = '🚀' if total_return > 0 else '📉' if total_return < 0 else '➡️'

    embed = {
        "title": f"📊 AI 交易日報 | {now.strftime('%Y/%m/%d')}",
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
            "text": f"紙上交易系統 | 初始資金 ${trader.initial_capital:,.0f} | {now.strftime('%H:%M')}"
        },
    }

    return embed


# ─── 測試 ───

if __name__ == "__main__":
    print("=" * 60)
    print("AI 紙上交易引擎 — 測試")
    print("=" * 60)

    trader = AITrader(initial_capital=1_000_000)
    print(f"\n初始資金: ${trader.cash:,.0f}")
    print(f"持倉: {len(trader.positions)} 檔")

    # 模擬買入
    pred_buy = {
        'direction': '漲',
        'confidence': 0.78,
        'bias': 4.2,
        'signals': {
            'foreign': '外資大買 +5200 張',
            'momentum': '5日動量 +2.3%',
            'ema': '多頭排列',
        },
        'warnings': [],
        'predicted_price': 108.0,
    }

    result = trader.evaluate_and_trade('8299', '群聯', 105.5, pred_buy)
    if result:
        print(f"\n{result['action'].upper()}: {result['stock_name']}")
        print(f"  價格: ${result['price']:.1f}")
        print(f"  股數: {result['shares']}")
        print(f"  金額: ${result['amount']:,.0f}")
        print(f"  理由: {result['reason']}")

    # 模擬第二檔買入
    pred_buy2 = {
        'direction': '漲',
        'confidence': 0.72,
        'bias': 3.5,
        'signals': {
            'foreign': '外資買超 +1200 張',
            'momentum': '5日動量 +1.5%',
            'ema': '短多排列',
        },
        'warnings': [],
    }
    result2 = trader.evaluate_and_trade('3189', '景碩', 210.0, pred_buy2)
    if result2:
        print(f"\n{result2['action'].upper()}: {result2['stock_name']}")
        print(f"  價格: ${result2['price']:.1f}")
        print(f"  股數: {result2['shares']}")

    # 顯示組合
    summary = trader.get_portfolio_summary()
    print(f"\n--- 投資組合 ---")
    print(f"現金: ${summary['cash']:,.0f}")
    print(f"持倉: {summary['positions_count']}/{MAX_POSITIONS} 檔")
    print(f"總資產: ${summary['total_value']:,.0f}")

    # 模擬停利
    pred_sell = {
        'direction': '漲',
        'confidence': 0.75,
        'bias': 3.0,
        'signals': {},
        'warnings': [],
    }
    sell_result = trader.evaluate_and_trade('8299', '群聯', 111.0, pred_sell)  # +5.2%
    if sell_result:
        print(f"\n{sell_result['action'].upper()}: {sell_result['stock_name']}")
        print(f"  價格: ${sell_result['price']:.1f}")
        print(f"  損益: ${sell_result['realized_pnl']:+,.0f} ({sell_result['pnl_pct']:+.1f}%)")
        print(f"  理由: {sell_result['reason']}")

    # 最終組合
    summary = trader.get_portfolio_summary()
    print(f"\n--- 最終組合 ---")
    print(f"現金: ${summary['cash']:,.0f}")
    print(f"持倉: {summary['positions_count']} 檔")
    print(f"總資產: ${summary['total_value']:,.0f}")
    print(f"累計損益: ${summary['realized_pnl']:+,.0f}")
    print(f"勝率: {summary['win_rate']:.0%}")

    # 測試發送 Discord Embed
    print("\n--- 測試 Discord Embed ---")
    try:
        from notifier import send_discord_embed

        # 模擬一次買入 embed
        test_trade = {
            'action': 'buy',
            'stock_code': '2330',
            'stock_name': '台積電',
            'price': 1780.0,
            'shares': 112,
            'lots': 0,
            'odd_shares': 112,
            'amount': 199654,
            'broker_fee': 284,
            'reason': '外資大買 +5200 張 | 5日動量 +2.3% | 多頭排列',
            'portfolio_summary': summary,
        }
        embed = build_buy_embed(test_trade)
        send_discord_embed(embed, channel='test')
        print("買入 Embed 已發送到 test channel")

        # 每日日報
        portfolio_embed = build_daily_portfolio_embed(trader)
        send_discord_embed(portfolio_embed, channel='test')
        print("日報 Embed 已發送到 test channel")

    except Exception as e:
        print(f"Discord 發送失敗: {e}")

    # 清理測試資料
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)
        print("\n已清理測試 portfolio 檔案")
