#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通知模組 - 發送股票預測結果到 Discord

支援：
- 基本 markdown 訊息（向後相容）
- 結構化 Embed（fields、顏色、訊號分解）
- 多 Embed 批次發送

@author: rubylintu
"""

import os
import json
import requests
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'notify_config.json')

# ─── 顏色常數（台股慣例：紅漲綠跌）───
COLOR_BULLISH = 0xFF4444   # 紅（漲）
COLOR_BEARISH = 0x44FF44   # 綠（跌）
COLOR_NEUTRAL = 0x808080   # 灰（盤整/觀望）
COLOR_INFO = 0x3498DB      # 藍（資訊）
COLOR_WARNING = 0xFFAA00   # 橘（警告）

DIRECTION_COLOR = {
    '漲': COLOR_BULLISH,
    '跌': COLOR_BEARISH,
    '盤整': COLOR_NEUTRAL,
    '觀望': COLOR_NEUTRAL,
}

DIRECTION_EMOJI = {
    '漲': '🔴',
    '跌': '🟢',
    '盤整': '⚪',
    '觀望': '🔘',
}


def load_config():
    """載入通知設定"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(config):
    """儲存通知設定"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def set_discord_webhook(webhook_url):
    """設定 Discord Webhook URL"""
    config = load_config()
    config['discord_webhook'] = webhook_url
    save_config(config)
    print(f"Discord Webhook 已設定！")


def _get_webhook_url(channel='release'):
    """取得 webhook URL"""
    config = load_config()
    if channel == 'test':
        return config.get('discord_webhook_test') or config.get('discord_webhook')
    return config.get('discord_webhook')


def _truncate(text, max_len=1024):
    """截斷文字到 Discord field 上限"""
    if not text:
        return ''
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + '...'


# ─── 基礎發送函式 ───

def send_discord(message, title=None, channel='release'):
    """
    發送訊息到 Discord（向後相容）

    Args:
        message: 訊息內容（markdown）
        title: 標題（可選）
        channel: 'release'（正式頻道）或 'test'（測試頻道）
    """
    webhook_url = _get_webhook_url(channel)

    if not webhook_url:
        print("錯誤：尚未設定 Discord Webhook")
        print("請執行: python notifier.py --setup YOUR_WEBHOOK_URL")
        return False

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    embed = {
        "title": title or "股票預測通知",
        "description": _truncate(message, 4096),
        "color": COLOR_INFO,
        "footer": {"text": f"News_DB AI 系統 | {now}"}
    }

    payload = {"embeds": [embed]}

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if 200 <= response.status_code < 300:
            print("Discord 通知已發送！")
            return True
        else:
            print(f"發送失敗: {response.status_code}")
            return False

    except Exception as e:
        print(f"發送錯誤: {e}")
        return False


def send_discord_embed(embed_data, channel='release'):
    """
    發送結構化 Embed 到 Discord

    Args:
        embed_data: dict，符合 Discord Embed 格式
                    {title, description, color, fields, footer, timestamp}
        channel: 'release' 或 'test'
    """
    webhook_url = _get_webhook_url(channel)
    if not webhook_url:
        print("錯誤：尚未設定 Discord Webhook")
        return False

    # 確保有 footer
    if 'footer' not in embed_data:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        embed_data['footer'] = {"text": f"News_DB AI 系統 | {now}"}

    # 截斷所有 field values
    for field in embed_data.get('fields', []):
        field['value'] = _truncate(field.get('value', ''), 1024)

    if embed_data.get('description'):
        embed_data['description'] = _truncate(embed_data['description'], 4096)

    payload = {"embeds": [embed_data]}

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        if 200 <= response.status_code < 300:
            return True
        else:
            print(f"Embed 發送失敗: {response.status_code}")
            return False
    except Exception as e:
        print(f"Embed 發送錯誤: {e}")
        return False


def send_multi_embed(embeds_list, channel='release'):
    """
    一次發送多個 Embed（Discord 單則訊息最多 10 個 embed）

    Args:
        embeds_list: list of embed dicts
        channel: 'release' 或 'test'
    """
    webhook_url = _get_webhook_url(channel)
    if not webhook_url:
        print("錯誤：尚未設定 Discord Webhook")
        return False

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # 處理每個 embed
    for embed in embeds_list:
        if 'footer' not in embed:
            embed['footer'] = {"text": f"News_DB AI 系統 | {now}"}
        for field in embed.get('fields', []):
            field['value'] = _truncate(field.get('value', ''), 1024)
        if embed.get('description'):
            embed['description'] = _truncate(embed['description'], 4096)

    # Discord 限制每則最多 10 embed，分批發送
    success = True
    for i in range(0, len(embeds_list), 10):
        batch = embeds_list[i:i+10]
        payload = {"embeds": batch}

        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code != 204:
                print(f"Multi-embed 發送失敗: {response.status_code}")
                success = False
        except Exception as e:
            print(f"Multi-embed 發送錯誤: {e}")
            success = False

    if success:
        print("Discord 通知已發送！")
    return success


# ─── 訊號分解格式化 ───

def format_signal_breakdown(signals):
    """
    把 calc_directional_bias() 回傳的 signals dict 格式化成可讀區塊

    Args:
        signals: dict，例如 {'foreign': '外資大買 +3200 張', 'momentum': '5日動量 +2.3%', ...}

    Returns:
        str: 格式化後的訊號文字
    """
    if not signals:
        return '無訊號資料'

    # 信號名稱對照與排序
    SIGNAL_ORDER = [
        ('foreign', '外資'),
        ('momentum', '動量'),
        ('ema', '均線'),
        ('rsi', 'RSI'),
        ('taiex', '大盤'),
        ('sox', '費半'),
        ('gpt', 'GPT'),
        ('volume', '成量'),
        ('correction', '修正'),
        ('dampening', '抑制'),
    ]

    lines = []
    for key, label in SIGNAL_ORDER:
        if key in signals:
            value = signals[key]
            strength = _signal_strength_bar(key, value)
            lines.append(f"`{label:　<3}` {value} {strength}")

    return '\n'.join(lines) if lines else '無訊號資料'


def _signal_strength_bar(key, value):
    """根據訊號內容產生強度指示"""
    v = value.lower() if isinstance(value, str) else str(value)

    # 正面信號
    if any(w in v for w in ['大買', '多頭排列', '放量上漲', '超賣']):
        return '`[+++]`'
    if any(w in v for w in ['買超', '短多', '偏多', '反彈']):
        return '`[++]`'
    if any(w in v for w in ['+', '漲', '上調']):
        return '`[+]`'

    # 負面信號
    if any(w in v for w in ['大賣', '空頭排列', '放量下跌', '超買']):
        return '`[---]`'
    if any(w in v for w in ['賣超', '短空', '偏空']):
        return '`[--]`'
    if any(w in v for w in ['跌', '下修', '修正', '抑制', '減弱']):
        return '`[-]`'

    # 中性
    return '`[·]`'


# ─── 專業 Embed 建構 ───

def build_prediction_embed(prediction, news_evidence=None,
                           risk_warnings=None, metrics=None):
    """
    建構單檔股票的專業預測 Embed

    Args:
        prediction: dict from DirectionalParticleModel.predict()
            必須包含: stock_code, stock_name, direction, confidence,
                      current_price, predicted_price, expected_change,
                      bias, signals
        news_evidence: list of {'title': str, 'sentiment': str, 'confidence': float}
                       （Top 3 新聞佐證）
        risk_warnings: list of str（風險警示）
        metrics: dict with 'today_hit_rate', 'recent_20_hit_rate',
                         'max_consecutive_loss', 'current_streak'

    Returns:
        dict: Discord embed 物件
    """
    now = datetime.datetime.now()
    code = prediction.get('stock_code', '????')
    name = prediction.get('stock_name', '未知')
    direction = prediction.get('direction', '觀望')
    confidence = prediction.get('confidence', 0)
    current_price = prediction.get('current_price', 0)
    predicted_price = prediction.get('predicted_price', 0)
    expected_change = prediction.get('expected_change', 0)
    signals = prediction.get('signals', {})

    # 顏色與 emoji
    color = DIRECTION_COLOR.get(direction, COLOR_NEUTRAL)
    dir_emoji = DIRECTION_EMOJI.get(direction, '🔘')

    # 信心度視覺化
    conf_pct = int(confidence * 100)
    conf_bar = _confidence_bar(confidence)

    embed = {
        "title": f"{code} {name} | {now.strftime('%H:%M')} 訊號更新",
        "color": color,
        "fields": [],
        "footer": {"text": f"News_DB AI 系統 | {now.strftime('%Y-%m-%d %H:%M')}"},
    }

    # Row 1: 方向 + 信心度 + 預測價格（inline）
    embed['fields'].append({
        "name": "方向",
        "value": f"{dir_emoji} **{direction}**",
        "inline": True,
    })
    embed['fields'].append({
        "name": "信心度",
        "value": f"**{conf_pct}%** {conf_bar}",
        "inline": True,
    })
    if current_price and predicted_price:
        change_emoji = '📈' if expected_change > 0 else '📉' if expected_change < 0 else '➡️'
        embed['fields'].append({
            "name": "預測價格",
            "value": f"${current_price:.1f} → ${predicted_price:.1f} ({expected_change:+.1f}%) {change_emoji}",
            "inline": True,
        })

    # Row 2: 訊號分解
    if signals:
        signal_text = format_signal_breakdown(signals)
        embed['fields'].append({
            "name": "訊號分解",
            "value": signal_text,
            "inline": False,
        })

    # Row 3: 新聞佐證
    if news_evidence:
        news_lines = []
        for i, news in enumerate(news_evidence[:3]):
            title = news.get('title', '無標題')[:50]
            sentiment = news.get('sentiment', '?')
            conf = news.get('confidence', 0)
            s_emoji = DIRECTION_EMOJI.get(sentiment, '⚪')
            news_lines.append(f"{i+1}. {title}\n   {s_emoji} {sentiment} ({conf:.0%})")
        embed['fields'].append({
            "name": "新聞佐證",
            "value": '\n'.join(news_lines),
            "inline": False,
        })

    # Row 4: 風險警示
    if risk_warnings:
        warning_text = '\n'.join(f'⚠️ {w}' for w in risk_warnings)
        embed['fields'].append({
            "name": "風險警示",
            "value": warning_text,
            "inline": False,
        })

    # Row 5: 追蹤指標
    if metrics:
        today_rate = metrics.get('today_hit_rate', 0)
        recent_rate = metrics.get('recent_20_hit_rate', 0)
        max_loss = metrics.get('max_consecutive_loss', 0)
        streak = metrics.get('current_streak', 0)

        today_n = metrics.get('today_predictions', 0)
        today_c = metrics.get('today_correct', 0)

        streak_text = f'🔥 連對 {streak}' if streak > 0 else (
            f'💀 連錯 {abs(streak)}' if streak < 0 else '—')

        metrics_text = (
            f"今日 {today_c}/{today_n} ({today_rate:.0%}) | "
            f"近20筆 {recent_rate:.0%} | "
            f"最大連錯 {max_loss} | {streak_text}"
        )
        embed['fields'].append({
            "name": "追蹤指標",
            "value": metrics_text,
            "inline": False,
        })

    return embed


def _confidence_bar(confidence):
    """產生信心度視覺化 bar"""
    filled = int(confidence * 10)
    return '█' * filled + '░' * (10 - filled)


def build_metrics_embed(today_metrics, advanced_metrics):
    """
    建構每日績效追蹤 Embed

    Args:
        today_metrics: dict from get_tracking_metrics()
        advanced_metrics: dict from calc_advanced_metrics()

    Returns:
        dict: Discord embed 物件
    """
    now = datetime.datetime.now()

    today_n = today_metrics.get('today_predictions', 0)
    today_c = today_metrics.get('today_correct', 0)
    today_rate = today_metrics.get('today_hit_rate', 0)

    coverage = advanced_metrics.get('coverage', 0)
    precision = advanced_metrics.get('precision', 0)
    max_dd = advanced_metrics.get('max_drawdown', 0)
    streak = advanced_metrics.get('current_streak', 0)
    overall = advanced_metrics.get('overall_accuracy', 0)

    streak_text = f'🔥 連對 {streak}' if streak > 0 else (
        f'💀 連錯 {abs(streak)}' if streak < 0 else '—')

    embed = {
        "title": f"📊 每日績效追蹤 | {now.strftime('%Y/%m/%d')}",
        "color": COLOR_INFO,
        "fields": [
            {
                "name": "今日表現",
                "value": f"預測 **{today_n}** 檔\n正確 **{today_c}** 檔\n命中率 **{today_rate:.0%}**",
                "inline": True,
            },
            {
                "name": "出手率 (Coverage)",
                "value": f"**{coverage:.0%}**",
                "inline": True,
            },
            {
                "name": "出手準度 (Precision)",
                "value": f"**{precision:.0%}**",
                "inline": True,
            },
            {
                "name": "整體準確率",
                "value": f"**{overall:.0%}**",
                "inline": True,
            },
            {
                "name": "最大連錯",
                "value": f"**{max_dd}** 次",
                "inline": True,
            },
            {
                "name": "目前狀態",
                "value": streak_text,
                "inline": True,
            },
        ],
        "footer": {"text": f"News_DB AI 系統 | {now.strftime('%Y-%m-%d %H:%M')}"},
    }

    # 方向分佈
    by_dir = advanced_metrics.get('by_direction', {})
    if by_dir:
        dir_lines = []
        for d in ['漲', '跌']:
            info = by_dir.get(d, {})
            cnt = info.get('count', 0)
            correct = info.get('correct', 0)
            acc = info.get('accuracy', 0)
            emoji = DIRECTION_EMOJI.get(d, '')
            dir_lines.append(f"{emoji} {d}: {correct}/{cnt} ({acc:.0%})")
        embed['fields'].append({
            "name": "方向分佈",
            "value": '\n'.join(dir_lines),
            "inline": False,
        })

    return embed


# ─── 既有功能（向後相容）───

def send_daily_report(stock_data=None, news_count=0, predictions=None,
                      focus_stocks=None, premarket_predictions=None,
                      channel='release'):
    """
    發送每日報告（13:30 收盤後）

    Args:
        stock_data: 股票資料 dict
        news_count: 收集的新聞數量
        predictions: 預測結果
        focus_stocks: 今日焦點股 dict {code: {name, reason, news_count, sentiment_score}}
        premarket_predictions: 盤前預測 dict {code: {name, predicted_price, direction, confidence, ...}}
    """
    now = datetime.datetime.now()

    # 建立報告內容
    lines = [
        f"**日期:** {now.strftime('%Y-%m-%d')}",
        f"**執行時間:** {now.strftime('%H:%M:%S')}",
        "",
        f"**收集新聞:** {news_count} 則",
    ]

    if stock_data:
        lines.append(f"**監控股票:** {len(stock_data)} 檔")

    # 今日焦點股摘要
    if focus_stocks:
        lines.append("")
        lines.append(f"**⭐ 今日新聞焦點 {len(focus_stocks)} 檔：**")
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        for i, (code, info) in enumerate(focus_stocks.items()):
            medal = medals[i] if i < len(medals) else f'{i+1}.'
            name = info.get('name', code)
            reason = info.get('reason', '')

            # 盤前預測方向
            pred_info = ''
            if premarket_predictions and code in premarket_predictions:
                pred = premarket_predictions[code]
                pred_info = f" | 預測{pred.get('direction', '?')} {pred.get('confidence', 0):.0%}"

            lines.append(f"{medal} {name}({code}){pred_info}")
            lines.append(f"   └ {reason}")

    # AI 預測摘要
    if premarket_predictions:
        bull_count = sum(1 for p in premarket_predictions.values() if p.get('direction') == '漲')
        bear_count = sum(1 for p in premarket_predictions.values() if p.get('direction') == '跌')
        neutral_count = sum(1 for p in premarket_predictions.values() if p.get('direction') == '盤整')
        lines.extend([
            "",
            "**AI 預測摘要:**",
            f"- 看漲: {bull_count} 檔 | 看跌: {bear_count} 檔 | 盤整: {neutral_count} 檔",
        ])

    if predictions:
        bull_count = sum(1 for p in predictions if p.get('prediction') == '漲')
        bear_count = sum(1 for p in predictions if p.get('prediction') == '跌')
        lines.extend([
            "",
            "**ML 預測摘要:**",
            f"- 看漲: {bull_count} 檔",
            f"- 看跌: {bear_count} 檔",
        ])

        # 列出高信心度預測
        high_conf = [p for p in predictions if p.get('confidence', 0) > 0.7]
        if high_conf:
            lines.append("")
            lines.append("**高信心度預測:**")
            for p in high_conf[:5]:
                name = p.get('stock_name', p.get('name', ''))
                pred = p.get('prediction', '')
                conf = p.get('confidence', 0)
                lines.append(f"- {name}: {pred} ({conf:.0%})")

    message = "\n".join(lines)

    return send_discord(message, title="每日股票報告", channel=channel)


def send_alert(stock_name, alert_type, message):
    """
    發送即時警報

    Args:
        stock_name: 股票名稱
        alert_type: 警報類型（如 "大漲", "大跌", "重大新聞"）
        message: 詳細訊息
    """
    alert_msg = f"**{stock_name}** - {alert_type}\n\n{message}"

    return send_discord(alert_msg, title=f"股票警報: {stock_name}")


def test_notification():
    """測試通知功能"""
    print("測試 Discord 通知...")

    test_msg = """
**測試通知**

這是一則測試訊息，確認 Discord Webhook 設定正確。

- 時間: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
- 狀態: 連線成功
"""

    return send_discord(test_msg, title="News_DB 通知測試", channel='test')


def test_embed_notification():
    """測試結構化 Embed 通知"""
    print("測試結構化 Embed 通知...")

    # 模擬預測結果
    prediction = {
        'stock_code': '2330',
        'stock_name': '台積電',
        'direction': '漲',
        'confidence': 0.72,
        'current_price': 1730.0,
        'predicted_price': 1752.0,
        'expected_change': 1.27,
        'bias': 3.5,
        'signals': {
            'foreign': '外資大買 +5200 張',
            'momentum': '5日動量 +2.3%',
            'ema': '多頭排列',
            'rsi': 'RSI=62 (偏多)',
            'taiex': '加權指數 +0.5%',
            'gpt': 'GPT情緒偏移 +1.2',
            'volume': '放量上漲 (量比 1.8x)',
            'correction': '多頭修正 x0.85 (準確率 45%)',
        },
    }

    news_evidence = [
        {'title': '台積電獲蘋果擴大訂單，Q2營收看增', 'sentiment': '漲', 'confidence': 0.85},
        {'title': '先進製程需求強勁，法人上調目標價', 'sentiment': '漲', 'confidence': 0.75},
        {'title': '半導體庫存調整接近尾聲', 'sentiment': '漲', 'confidence': 0.60},
    ]

    risk_warnings = ['信心度偏低']

    metrics = {
        'today_predictions': 8,
        'today_correct': 5,
        'today_hit_rate': 0.625,
        'recent_20_hit_rate': 0.60,
        'max_consecutive_loss': 3,
        'current_streak': 2,
    }

    embed = build_prediction_embed(prediction, news_evidence, risk_warnings, metrics)
    return send_discord_embed(embed, channel='test')


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup' and len(sys.argv) > 2:
            set_discord_webhook(sys.argv[2])
        elif sys.argv[1] == '--test':
            test_notification()
        elif sys.argv[1] == '--test-embed':
            test_embed_notification()
        else:
            print("用法:")
            print("  設定 Webhook: python notifier.py --setup YOUR_WEBHOOK_URL")
            print("  測試通知:     python notifier.py --test")
            print("  測試 Embed:   python notifier.py --test-embed")
    else:
        print("Discord 通知模組")
        print("")
        print("設定方式:")
        print("  python notifier.py --setup YOUR_DISCORD_WEBHOOK_URL")
        print("")
        print("測試通知:")
        print("  python notifier.py --test")
        print("  python notifier.py --test-embed")
