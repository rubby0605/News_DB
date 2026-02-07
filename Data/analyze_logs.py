#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日 Log 分析工具

解析 logs/ 目錄下的所有 log 檔案，產出結構化的每日執行報告。

用法:
    python analyze_logs.py              # 分析所有 log
    python analyze_logs.py 2026-02-06   # 分析指定日期
    python analyze_logs.py --latest     # 只分析最新一天

@author: rubylintu
"""

import os
import re
import sys
import glob
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')

# ─── 正規表達式 ───

# 標準 log 行: 2026-02-06 11:40:57,687 - INFO - message
RE_LOG_LINE = re.compile(
    r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ - (\w+) - (.+)$'
)

# 股價抓取: [2/31] 景碩(3189): $263.5000  或  $-
RE_STOCK_PRICE = re.compile(
    r'\[(\d+)/(\d+)\] (.+?)\((\d+)\): \$(.+)'
)

# 抓取失敗: 抓取 群聯 失敗: 'n'
RE_FETCH_ERROR = re.compile(
    r'抓取 (.+?) 失敗: (.+)'
)

# 新聞處理: [1/31] 處理: 群聯 (8299)
RE_NEWS_PROCESS = re.compile(
    r'\[(\d+)/(\d+)\] 處理: (.+?) \((\d+)\)'
)

# 新聞結果: 找到 N 則新新聞
RE_NEWS_FOUND = re.compile(
    r'找到 (\d+) 則新新聞'
)

# 股價漲跌: 股價: 263.5, 漲跌: -2.59%
RE_STOCK_CHANGE = re.compile(
    r'股價: ([\d.]+), 漲跌: ([+-]?[\d.]+)%'
)

# GPT 分析: GPT 盤前分析 群聯: 漲 (70%)  或  GPT 分析 群聯: 中性 (50%)
RE_GPT_ANALYSIS = re.compile(
    r'GPT (?:盤前)?分析 (.+?): (.+?) \((\d+)%\)'
)

# 監控迭代: 已執行 10 次，時間: 11:50:41
RE_MONITOR_ITER = re.compile(
    r'已執行 (\d+) 次，時間: (\d{2}:\d{2}:\d{2})'
)

# 盤後準確率: 盤後分析完成，準確率 31.2%
RE_POSTMARKET_ACC = re.compile(
    r'盤後分析完成，準確率 ([\d.]+)%'
)

# 盤前預測完成: 盤前分析完成，預測 30 檔股票
RE_PREMARKET_DONE = re.compile(
    r'盤前分析完成，預測 (\d+) 檔股票'
)

# 監控結束: 即時監控結束，共執行 N 次
RE_MONITOR_END = re.compile(
    r'即時監控結束，共執行 (\d+) 次'
)

# Discord 通知
RE_DISCORD = re.compile(r'Discord 通知已發送')

# 法人資料 (cron.log 非標準格式)
RE_INSTITUTIONAL = re.compile(
    r'取得 (\d+) 檔股票法人資料 \((\d+)\)'
)

# SyntaxError (非標準格式)
RE_SYNTAX_ERROR = re.compile(r'SyntaxError')


def parse_log_file(filepath):
    """解析單一 log 檔案，回傳結構化資料"""
    results = {
        'file': os.path.basename(filepath),
        'sessions': [],  # 每次程式啟動為一個 session
        'errors': [],
        'warnings': [],
        'raw_lines': 0,
    }

    current_session = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        results['errors'].append(f'無法讀取檔案: {e}')
        return results

    results['raw_lines'] = len(lines)

    if len(lines) == 0:
        return results

    for line in lines:
        line = line.rstrip('\n')

        # 檢查 SyntaxError（非標準 log 格式）
        if RE_SYNTAX_ERROR.search(line) and '- ERROR -' not in line:
            results['errors'].append({
                'time': None,
                'type': 'SyntaxError',
                'message': line.strip(),
            })
            continue

        # 檢查法人資料（非標準 log 格式，出現在 cron.log）
        m = RE_INSTITUTIONAL.search(line)
        if m and '- INFO -' not in line:
            # 這是 cron.log 的 stdout 輸出
            continue

        # 解析標準 log 行
        m = RE_LOG_LINE.match(line)
        if not m:
            continue

        date_str, time_str, level, message = m.groups()

        # 檢查程式啟動（stock_job log 直接從「開始抓取基本面資料」開始）
        if '開始抓取基本面資料' in message and current_session is None:
            current_session = {
                'date': date_str,
                'start_time': time_str,
                'end_time': None,
                'phases': {
                    'fundamental': {
                        'stocks': [], 'errors': [], 'total_stocks': 0,
                        'price_ok': 0, 'price_missing': 0,
                    },
                    'news': {'stocks': [], 'total_news': 0, 'stock_prices': []},
                    'premarket': {'gpt_analyses': [], 'predicted_count': 0},
                    'monitoring': {
                        'iterations': 0, 'start_time': None, 'end_time': None,
                        'discord_notifications': 0, 'gpt_analyses': [],
                    },
                    'postmarket': {'accuracy': None},
                },
            }
            results['sessions'].append(current_session)
            continue

        if '每日股票資料抓取程式啟動' in message:
            current_session = {
                'date': date_str,
                'start_time': time_str,
                'end_time': None,
                'phases': {
                    'fundamental': {
                        'stocks': [],
                        'errors': [],
                        'total_stocks': 0,
                        'price_ok': 0,
                        'price_missing': 0,
                    },
                    'news': {
                        'stocks': [],
                        'total_news': 0,
                        'stock_prices': [],
                    },
                    'premarket': {
                        'gpt_analyses': [],
                        'predicted_count': 0,
                    },
                    'monitoring': {
                        'iterations': 0,
                        'start_time': None,
                        'end_time': None,
                        'discord_notifications': 0,
                        'gpt_analyses': [],
                    },
                    'postmarket': {
                        'accuracy': None,
                    },
                },
            }
            results['sessions'].append(current_session)
            continue

        if current_session is None:
            # 盤前分析（可能在啟動前的獨立執行）
            if '盤前分析' in message:
                if not results['sessions']:
                    current_session = {
                        'date': date_str,
                        'start_time': time_str,
                        'end_time': None,
                        'phases': {
                            'fundamental': {
                                'stocks': [], 'errors': [], 'total_stocks': 0,
                                'price_ok': 0, 'price_missing': 0,
                            },
                            'news': {'stocks': [], 'total_news': 0, 'stock_prices': []},
                            'premarket': {'gpt_analyses': [], 'predicted_count': 0},
                            'monitoring': {
                                'iterations': 0, 'start_time': None, 'end_time': None,
                                'discord_notifications': 0, 'gpt_analyses': [],
                            },
                            'postmarket': {'accuracy': None},
                        },
                    }
                    results['sessions'].append(current_session)
                else:
                    current_session = results['sessions'][-1]
            else:
                continue

        # 記錄 errors 和 warnings
        if level == 'ERROR':
            err = {'time': time_str, 'message': message}
            m_fetch = RE_FETCH_ERROR.match(message)
            if m_fetch:
                err['type'] = 'fetch_error'
                err['stock'] = m_fetch.group(1)
                err['detail'] = m_fetch.group(2)
                current_session['phases']['fundamental']['errors'].append(err)
            else:
                err['type'] = 'other'
            results['errors'].append(err)

        if level == 'WARNING':
            results['warnings'].append({
                'time': time_str,
                'message': message,
            })

        # === Phase: 基本面資料 ===
        m = RE_STOCK_PRICE.match(message)
        if m:
            idx, total, name, code, price = m.groups()
            current_session['phases']['fundamental']['total_stocks'] = int(total)
            has_price = price != '-'
            current_session['phases']['fundamental']['stocks'].append({
                'index': int(idx),
                'name': name,
                'code': code,
                'price': float(price) if has_price else None,
            })
            if has_price:
                current_session['phases']['fundamental']['price_ok'] += 1
            else:
                current_session['phases']['fundamental']['price_missing'] += 1

        # === Phase: 新聞收集 ===
        m = RE_NEWS_PROCESS.match(message)
        if m:
            idx, total, name, code = m.groups()
            current_session['phases']['news']['stocks'].append({
                'name': name,
                'code': code,
                'news_count': 0,
                'price': None,
                'change': None,
            })

        m = RE_NEWS_FOUND.search(message)
        if m:
            count = int(m.group(1))
            if current_session['phases']['news']['stocks']:
                current_session['phases']['news']['stocks'][-1]['news_count'] = count
                current_session['phases']['news']['total_news'] += count

        m = RE_STOCK_CHANGE.search(message)
        if m:
            price, change = float(m.group(1)), float(m.group(2))
            if current_session['phases']['news']['stocks']:
                current_session['phases']['news']['stocks'][-1]['price'] = price
                current_session['phases']['news']['stocks'][-1]['change'] = change
                current_session['phases']['news']['stock_prices'].append({
                    'name': current_session['phases']['news']['stocks'][-1]['name'],
                    'price': price,
                    'change': change,
                })

        # === Phase: 盤前分析 ===
        m = RE_GPT_ANALYSIS.search(message)
        if m:
            stock, sentiment, confidence = m.group(1), m.group(2), int(m.group(3))
            entry = {'stock': stock, 'sentiment': sentiment, 'confidence': confidence}
            if '盤前分析' in message:
                current_session['phases']['premarket']['gpt_analyses'].append(entry)
            else:
                current_session['phases']['monitoring']['gpt_analyses'].append(entry)

        m = RE_PREMARKET_DONE.search(message)
        if m:
            current_session['phases']['premarket']['predicted_count'] = int(m.group(1))

        # === Phase: 監控 ===
        if '開始即時股價監控' in message:
            current_session['phases']['monitoring']['start_time'] = time_str

        m = RE_MONITOR_ITER.search(message)
        if m:
            current_session['phases']['monitoring']['iterations'] = int(m.group(1))

        m = RE_MONITOR_END.search(message)
        if m:
            current_session['phases']['monitoring']['iterations'] = int(m.group(1))
            current_session['phases']['monitoring']['end_time'] = time_str

        if '已過收盤時間' in message:
            current_session['phases']['monitoring']['end_time'] = time_str

        if RE_DISCORD.search(message):
            current_session['phases']['monitoring']['discord_notifications'] += 1

        # === Phase: 盤後 ===
        m = RE_POSTMARKET_ACC.search(message)
        if m:
            current_session['phases']['postmarket']['accuracy'] = float(m.group(1))

        if '今日任務完成' in message:
            current_session['end_time'] = time_str

    return results


def parse_cron_log(filepath):
    """解析 cron.log（混合標準 log + stdout 格式）"""
    # cron.log 的格式和其他 log 相同，只是尾部多了一些非標準輸出
    return parse_log_file(filepath)


def format_report(all_results, target_date=None):
    """產生可讀報告"""
    lines = []
    lines.append('=' * 70)
    lines.append('📊  每日 Log 分析報告')
    lines.append(f'    分析時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 70)

    # 按日期整理所有 session（去重）
    sessions_by_date = defaultdict(list)
    seen_sessions = set()  # (date, start_time) 去重
    all_errors = []
    all_warnings = []
    seen_errors = set()
    seen_warnings = set()

    for result in all_results:
        for e in result['errors']:
            key = str(e)
            if key not in seen_errors:
                seen_errors.add(key)
                all_errors.append(e)
        for w in result['warnings']:
            key = f'{w["time"]}_{w["message"]}'
            if key not in seen_warnings:
                seen_warnings.add(key)
                all_warnings.append(w)
        for session in result['sessions']:
            d = session['date']
            if target_date and d != target_date:
                continue
            # 去重：同日期同啟動時間視為同一 session
            session_key = (d, session['start_time'])
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)
            sessions_by_date[d].append({
                'session': session,
                'source': result['file'],
            })

    if not sessions_by_date:
        lines.append('\n[!] 找不到任何執行記錄')
        return '\n'.join(lines)

    for date in sorted(sessions_by_date.keys()):
        entries = sessions_by_date[date]
        lines.append(f'\n{"─" * 70}')
        lines.append(f'📅  {date}  ({len(entries)} 次執行)')
        lines.append(f'{"─" * 70}')

        for i, entry in enumerate(entries):
            s = entry['session']
            src = entry['source']
            # 判斷是 cron 排程 還是手動測試
            start = s['start_time'] or ''
            if start >= '09:00' and start <= '14:00':
                run_type = '排程'
            elif start >= '00:00' and start < '09:00':
                run_type = '凌晨測試'
            else:
                run_type = '手動測試'
            lines.append(f'\n  ▶ 執行 #{i+1}  [{run_type}]  (來源: {src})')
            lines.append(f'    啟動: {s["start_time"] or "N/A"}  '
                         f'結束: {s["end_time"] or "未完成/持續中"}')

            # --- 基本面 ---
            fund = s['phases']['fundamental']
            if fund['total_stocks'] > 0 or fund['stocks']:
                total = fund['total_stocks'] or len(fund['stocks'])
                ok = fund['price_ok']
                missing = fund['price_missing']
                errs = len(fund['errors'])
                lines.append(f'\n    [基本面抓取]')
                lines.append(f'      股票數: {total}  |  有價格: {ok}  |  '
                             f'無價格($-): {missing}  |  錯誤: {errs}')
                if fund['errors']:
                    for e in fund['errors']:
                        lines.append(f'      ❌ {e["stock"]}: {e["detail"]}')
                if missing > 0:
                    no_price = [st['name'] for st in fund['stocks']
                                if st['price'] is None]
                    if no_price:
                        lines.append(f'      ⚠️  無價格: {", ".join(no_price[:10])}'
                                     + (f' ...等{len(no_price)}檔' if len(no_price) > 10 else ''))

            # --- 新聞收集 ---
            news = s['phases']['news']
            if news['stocks']:
                total_news = news['total_news']
                stocks_with_news = sum(1 for st in news['stocks'] if st['news_count'] > 0)
                lines.append(f'\n    [新聞收集]')
                lines.append(f'      處理股票: {len(news["stocks"])}  |  '
                             f'有新聞: {stocks_with_news}  |  '
                             f'新聞總數: {total_news}')

                # 漲跌統計
                if news['stock_prices']:
                    up = [p for p in news['stock_prices'] if p['change'] > 0]
                    down = [p for p in news['stock_prices'] if p['change'] < 0]
                    flat = [p for p in news['stock_prices'] if p['change'] == 0]
                    lines.append(f'      股價: 上漲 {len(up)}  |  '
                                 f'下跌 {len(down)}  |  平盤 {len(flat)}')

                    if down:
                        worst = sorted(down, key=lambda x: x['change'])[:3]
                        worst_str = ', '.join(
                            f'{w["name"]}({w["change"]:+.2f}%)' for w in worst)
                        lines.append(f'      跌幅前三: {worst_str}')
                    if up:
                        best = sorted(up, key=lambda x: -x['change'])[:3]
                        best_str = ', '.join(
                            f'{b["name"]}({b["change"]:+.2f}%)' for b in best)
                        lines.append(f'      漲幅前三: {best_str}')

            # --- 盤前分析 ---
            pre = s['phases']['premarket']
            if pre['predicted_count'] > 0 or pre['gpt_analyses']:
                lines.append(f'\n    [盤前分析]')
                lines.append(f'      預測股票數: {pre["predicted_count"]}')
                if pre['gpt_analyses']:
                    sentiments = defaultdict(int)
                    for g in pre['gpt_analyses']:
                        sentiments[g['sentiment']] += 1
                    sent_str = ', '.join(f'{k}: {v}' for k, v in sentiments.items())
                    lines.append(f'      GPT 情緒分佈: {sent_str}')
                    # 非中性的焦點分析
                    non_neutral = [g for g in pre['gpt_analyses']
                                   if g['sentiment'] != '中性']
                    if non_neutral:
                        for g in non_neutral:
                            lines.append(
                                f'      🎯 {g["stock"]}: {g["sentiment"]} '
                                f'({g["confidence"]}%)')

            # --- 即時監控 ---
            mon = s['phases']['monitoring']
            if mon['iterations'] > 0 or mon['start_time']:
                lines.append(f'\n    [即時監控]')
                lines.append(f'      時間: {mon["start_time"] or "N/A"} ~ '
                             f'{mon["end_time"] or "N/A"}')
                lines.append(f'      迭代次數: {mon["iterations"]}  |  '
                             f'Discord 通知: {mon["discord_notifications"]}')

                # 監控期間 GPT 分析統計
                if mon['gpt_analyses']:
                    mon_sentiments = defaultdict(int)
                    for g in mon['gpt_analyses']:
                        mon_sentiments[g['sentiment']] += 1
                    non_neutral_mon = [g for g in mon['gpt_analyses']
                                       if g['sentiment'] != '中性']
                    if non_neutral_mon:
                        lines.append(f'      監控期間 GPT 非中性信號:')
                        # 去重（同一股可能多次分析）
                        seen = set()
                        for g in non_neutral_mon:
                            key = f'{g["stock"]}_{g["sentiment"]}'
                            if key not in seen:
                                seen.add(key)
                                lines.append(
                                    f'        🔔 {g["stock"]}: '
                                    f'{g["sentiment"]} ({g["confidence"]}%)')

            # --- 盤後分析 ---
            post = s['phases']['postmarket']
            if post['accuracy'] is not None:
                lines.append(f'\n    [盤後分析]')
                acc = post['accuracy']
                emoji = '✅' if acc >= 50 else '⚠️' if acc >= 40 else '❌'
                lines.append(f'      方向準確率: {acc}% {emoji}')

    # === 錯誤匯總 ===
    if all_errors:
        lines.append(f'\n{"─" * 70}')
        lines.append('⚠️  錯誤匯總')
        lines.append(f'{"─" * 70}')

        # 分類統計
        error_types = defaultdict(list)
        for e in all_errors:
            if isinstance(e, dict):
                etype = e.get('type', 'other')
                error_types[etype].append(e)
            else:
                error_types['raw'].append(e)

        for etype, errors in error_types.items():
            if etype == 'SyntaxError':
                lines.append(f'\n  SyntaxError ({len(errors)} 次):')
                for e in errors:
                    lines.append(f'    {e["message"][:80]}')
            elif etype == 'fetch_error':
                # 統計哪些股票最常失敗
                stock_fails = defaultdict(list)
                for e in errors:
                    stock_fails[e['stock']].append(e['detail'])
                lines.append(f'\n  股價抓取失敗 ({len(errors)} 次):')
                for stock, details in stock_fails.items():
                    unique_reasons = set(details)
                    lines.append(f'    {stock}: {len(details)} 次 '
                                 f'(原因: {", ".join(unique_reasons)})')
            elif etype == 'other':
                lines.append(f'\n  其他錯誤 ({len(errors)} 次):')
                for e in errors[:5]:
                    msg = e['message'] if isinstance(e, dict) else str(e)
                    lines.append(f'    [{e.get("time", "?")}] {msg[:80]}')
            elif etype == 'raw':
                lines.append(f'\n  非標準錯誤 ({len(errors)} 次):')
                for e in errors[:3]:
                    lines.append(f'    {str(e)[:80]}')

    # === 警告匯總 ===
    if all_warnings:
        lines.append(f'\n{"─" * 70}')
        lines.append(f'⚡ 警告匯總 ({len(all_warnings)} 則)')
        lines.append(f'{"─" * 70}')
        for w in all_warnings:
            lines.append(f'  [{w["time"]}] {w["message"][:70]}')

    # === 問題清單 ===
    lines.append(f'\n{"─" * 70}')
    lines.append('🔍  發現的問題')
    lines.append(f'{"─" * 70}')
    issues = []

    # 檢查重複執行
    for date, entries in sessions_by_date.items():
        if len(entries) > 1:
            times = [e['session']['start_time'] for e in entries
                     if e['session']['start_time']]
            issues.append(f'[重複執行] {date} 有 {len(entries)} 次執行 '
                          f'(啟動時間: {", ".join(times)})')

    # 檢查群聯一直失敗
    phunion_fails = sum(
        1 for e in all_errors
        if isinstance(e, dict) and e.get('stock') == '群聯'
    )
    if phunion_fails > 0:
        issues.append(f'[持續錯誤] 群聯(8299) 抓取持續失敗 '
                      f'({phunion_fails} 次)，需檢查 GoodInfo 頁面格式')

    # 檢查大量無價格
    for date, entries in sessions_by_date.items():
        for entry in entries:
            fund = entry['session']['phases']['fundamental']
            if fund['price_missing'] > 20:
                issues.append(f'[資料缺失] {date} {entry["session"]["start_time"]} '
                              f'執行時 {fund["price_missing"]} 檔無價格'
                              f'（可能在非交易時間執行）')

    # 檢查新聞 0 則
    for date, entries in sessions_by_date.items():
        for entry in entries:
            news = entry['session']['phases']['news']
            if news['stocks'] and news['total_news'] == 0:
                issues.append(f'[無新聞] {date} {entry["session"]["start_time"]} '
                              f'所有股票都沒有抓到新聞')

    # 檢查準確率低（每日只報一次最低的）
    for date, entries in sessions_by_date.items():
        low_accs = []
        for entry in entries:
            acc = entry['session']['phases']['postmarket']['accuracy']
            if acc is not None and acc < 40:
                low_accs.append(acc)
        if low_accs:
            worst = min(low_accs)
            issues.append(f'[低準確率] {date} 盤後方向準確率最低只有 {worst}%')

    # 檢查 2/5 NoneType 全面失敗
    nonetype_fails = sum(
        1 for e in all_errors
        if isinstance(e, dict) and e.get('type') == 'fetch_error'
        and 'NoneType' in e.get('detail', '')
    )
    if nonetype_fails > 5:
        issues.append(f'[API 異常] GoodInfo API 全面失敗 '
                      f'({nonetype_fails} 次)，可能被封鎖或網站改版')

    if issues:
        for issue in issues:
            lines.append(f'  • {issue}')
    else:
        lines.append('  ✅ 未發現明顯問題')

    lines.append(f'\n{"=" * 70}')
    lines.append('報告結束')
    lines.append(f'{"=" * 70}')

    return '\n'.join(lines)


def main():
    target_date = None
    latest_only = False

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--latest':
            latest_only = True
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', arg):
            target_date = arg
        else:
            print(f'用法: python {sys.argv[0]} [YYYY-MM-DD | --latest]')
            sys.exit(1)

    # 收集所有 log 檔案
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))

    if not log_files:
        print(f'[!] 在 {LOG_DIR} 找不到任何 log 檔案')
        sys.exit(1)

    if latest_only:
        # 只取最新有內容的 log
        log_files = [f for f in log_files if os.path.getsize(f) > 0]
        if log_files:
            log_files = [log_files[-1]]

    # 解析所有 log
    all_results = []
    for filepath in log_files:
        if os.path.getsize(filepath) == 0:
            continue
        result = parse_log_file(filepath)
        all_results.append(result)

    # 產生報告
    report = format_report(all_results, target_date)
    print(report)

    # 儲存報告
    report_file = os.path.join(LOG_DIR, 'analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n(報告已儲存至 {report_file})')


if __name__ == '__main__':
    main()
