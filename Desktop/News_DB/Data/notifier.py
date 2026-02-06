#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通知模組 - 發送股票預測結果到 Discord

@author: rubylintu
"""

import os
import json
import requests
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'notify_config.json')


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


def send_discord(message, title=None):
    """
    發送訊息到 Discord

    Args:
        message: 訊息內容
        title: 標題（可選）
    """
    config = load_config()
    webhook_url = config.get('discord_webhook')

    if not webhook_url:
        print("錯誤：尚未設定 Discord Webhook")
        print("請執行: python notifier.py --setup YOUR_WEBHOOK_URL")
        return False

    # 建立 Discord embed 訊息
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    embed = {
        "title": title or "股票預測通知",
        "description": message,
        "color": 3447003,  # 藍色
        "footer": {"text": f"News_DB AI 系統 | {now}"}
    }

    payload = {
        "embeds": [embed]
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 204:
            print("Discord 通知已發送！")
            return True
        else:
            print(f"發送失敗: {response.status_code}")
            return False

    except Exception as e:
        print(f"發送錯誤: {e}")
        return False


def send_daily_report(stock_data=None, news_count=0, predictions=None):
    """
    發送每日報告

    Args:
        stock_data: 股票資料 dict
        news_count: 收集的新聞數量
        predictions: 預測結果
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

    if predictions:
        bull_count = sum(1 for p in predictions if p.get('prediction') == '漲')
        bear_count = sum(1 for p in predictions if p.get('prediction') == '跌')
        lines.extend([
            "",
            "**AI 預測摘要:**",
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

    return send_discord(message, title="每日股票報告")


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

    return send_discord(test_msg, title="News_DB 通知測試")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup' and len(sys.argv) > 2:
            set_discord_webhook(sys.argv[2])
        elif sys.argv[1] == '--test':
            test_notification()
        else:
            print("用法:")
            print("  設定 Webhook: python notifier.py --setup YOUR_WEBHOOK_URL")
            print("  測試通知:     python notifier.py --test")
    else:
        print("Discord 通知模組")
        print("")
        print("設定方式:")
        print("  python notifier.py --setup YOUR_DISCORD_WEBHOOK_URL")
        print("")
        print("測試通知:")
        print("  python notifier.py --test")
