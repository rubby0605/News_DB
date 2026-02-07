#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通知去重 / 冷卻機制

防止同方向、同信心度的通知重複發送。
記錄已發送的通知，支援 min_interval 冷卻與 confidence_delta 過濾。

@author: rubylintu
"""

import os
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUARD_FILE = os.path.join(SCRIPT_DIR, 'notification_guard_state.json')


class NotificationGuard:
    """通知去重 / 冷卻"""

    def __init__(self):
        self._state = self._load_state()
        self._suppressed = 0

    def _load_state(self):
        if os.path.exists(GUARD_FILE):
            try:
                with open(GUARD_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'notifications': {}}

    def _save_state(self):
        with open(GUARD_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def should_notify(self, stock_code, direction, confidence,
                      min_confidence_delta=0.1, min_interval=600):
        """
        判斷是否應該發送通知

        Args:
            stock_code: 股票代號
            direction: 預測方向 ('漲'/'跌'/'盤整'/'觀望')
            confidence: 信心度 (0-1)
            min_confidence_delta: 同方向信心度差距低於此值則不重發
            min_interval: 冷卻秒數（預設 10 分鐘）

        Returns:
            bool: True = 應該發送
        """
        now = time.time()
        key = str(stock_code)

        last = self._state['notifications'].get(key)

        if last is None:
            return True

        elapsed = now - last.get('timestamp', 0)

        # 冷卻期內
        if elapsed < min_interval:
            # 方向相同且信心度差距小 → 不發
            if (last.get('direction') == direction and
                    abs(last.get('confidence', 0) - confidence) < min_confidence_delta):
                self._suppressed += 1
                return False

        # 方向改變 → 一定要發
        if last.get('direction') != direction:
            return True

        # 超過冷卻期 → 發
        if elapsed >= min_interval:
            return True

        # 信心度變化夠大 → 發
        if abs(last.get('confidence', 0) - confidence) >= min_confidence_delta:
            return True

        self._suppressed += 1
        return False

    def record_notification(self, stock_code, direction, confidence):
        """記錄已發送的通知"""
        self._state['notifications'][str(stock_code)] = {
            'direction': direction,
            'confidence': confidence,
            'timestamp': time.time(),
        }
        self._save_state()

    def get_suppressed_count(self):
        """取得本次被壓抑的通知數"""
        return self._suppressed

    def reset_daily(self):
        """每日重置（新的一天清除前日狀態）"""
        self._state = {'notifications': {}}
        self._suppressed = 0
        self._save_state()


if __name__ == "__main__":
    guard = NotificationGuard()
    print("通知去重模組")
    print(f"已記錄通知數: {len(guard._state['notifications'])}")
    print(f"被壓抑通知數: {guard.get_suppressed_count()}")
