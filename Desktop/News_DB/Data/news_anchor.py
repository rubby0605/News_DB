#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 虛擬主播新聞播報 Pipeline

流程：新聞 CSV → Gemini 生成播報稿 → OpenAI TTS 語音 → Hedra 說話影片 → Discord 推播

支援模式：
  - video: 完整影片播報（需 Hedra API key）
  - audio: 純語音播報（僅需 OpenAI key）
  - text:  純文字播報稿（免費）

用法：
  python news_anchor.py                 # 今天新聞，自動偵測最佳模式
  python news_anchor.py --date 20260317 # 指定日期
  python news_anchor.py --mode audio    # 強制語音模式
  python news_anchor.py --test          # 測試用短稿

@author: rubylintu
"""

import os
import sys
import csv
import json
import time
import logging
import datetime
import tempfile
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logger = logging.getLogger(__name__)

# ─── 設定 ───

ANCHOR_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'anchor_config.json')
ANCHOR_IMAGE_FILE = os.path.join(SCRIPT_DIR, 'anchor_avatar.png')
NEWS_DATA_DIR = os.path.join(SCRIPT_DIR, 'news_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'anchor_output')

# 播報稿長度（字數）— 影響影片時長
SCRIPT_MAX_CHARS = 450    # ~1.5 分鐘語音
SCRIPT_MIN_CHARS = 200    # ~45 秒語音

# TTS 設定
TTS_MODEL = 'tts-1'       # tts-1（快）或 tts-1-hd（品質好）
TTS_VOICE = 'nova'        # alloy/echo/fable/onyx/nova/shimmer — nova 最適合新聞
TTS_SPEED = 1.0

# Gemini 設定
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def load_anchor_config():
    """載入主播設定"""
    if os.path.exists(ANCHOR_CONFIG_FILE):
        with open(ANCHOR_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_anchor_config(config):
    """儲存主播設定"""
    with open(ANCHOR_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def detect_mode():
    """自動偵測可用的最佳模式"""
    config = load_anchor_config()

    # 檢查 Hedra API key
    hedra_key = config.get('hedra_api_key') or os.environ.get('HEDRA_API_KEY')
    if hedra_key and os.path.exists(ANCHOR_IMAGE_FILE):
        return 'video'

    # 檢查 OpenAI key（TTS 用）
    try:
        from gpt_sentiment import get_client
        get_client()
        return 'audio'
    except Exception:
        return 'text'


# ═══════════════════════════════════════════
# Step 1: 讀取新聞 → 整理素材
# ═══════════════════════════════════════════

def load_news(date_str=None):
    """
    讀取當天新聞 CSV

    Returns:
        list of dict: [{stock_name, title, source, price_change, change_pct}, ...]
    """
    if not date_str:
        date_str = datetime.datetime.now().strftime('%Y%m%d')

    csv_path = os.path.join(NEWS_DATA_DIR, f'news_{date_str}.csv')
    if not os.path.exists(csv_path):
        logger.error(f"找不到新聞檔: {csv_path}")
        return []

    news = []
    seen_titles = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('title', '').strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            news.append({
                'stock_code': row.get('stock_code', ''),
                'stock_name': row.get('stock_name', row.get('keyword', '')),
                'title': title,
                'source': row.get('source', ''),
                'price_change': row.get('price_change', ''),
                'change_pct': row.get('change_pct', ''),
            })

    logger.info(f"讀取 {len(news)} 則新聞（{date_str}）")
    return news


def load_predictions():
    """讀取今日預測結果（如果有的話）"""
    from config import PREDICTIONS_FILE
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ═══════════════════════════════════════════
# Step 2: Gemini 生成播報稿
# ═══════════════════════════════════════════

def generate_script(news_items, predictions=None, style='professional'):
    """
    用 Gemini 將新聞列表轉成主播播報稿

    Args:
        news_items: load_news() 回傳的新聞列表
        predictions: 預測結果（可選）
        style: 播報風格 — professional / casual / energetic

    Returns:
        str: 播報稿文字
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY 環境變數未設定")

    # 整理新聞摘要：按股票分組，每股只取最重要的 1-2 則標題
    by_stock = {}
    for item in news_items:
        name = item['stock_name']
        if name not in by_stock:
            by_stock[name] = {
                'titles': [],
                'change_pct': item.get('change_pct', ''),
            }
        if len(by_stock[name]['titles']) < 2:
            by_stock[name]['titles'].append(item['title'])

    news_summary = []
    for name, info in list(by_stock.items())[:15]:  # 最多 15 檔
        titles = '；'.join(info['titles'])
        line = f"- {name}：{titles}"
        if info['change_pct']:
            line += f"（{info['change_pct']}%）"
        news_summary.append(line)

    # 加入預測摘要
    pred_summary = ""
    if predictions:
        pred_lines = []
        for code, pred in predictions.items():
            if isinstance(pred, dict):
                name = pred.get('stock_name', code)
                direction = pred.get('direction', '?')
                confidence = pred.get('confidence', 0)
                if isinstance(confidence, (int, float)) and confidence > 0.6:
                    pred_lines.append(f"- {name}: AI 預測{direction}（信心 {confidence:.0%}）")
        if pred_lines:
            pred_summary = "\n\nAI 預測摘要：\n" + '\n'.join(pred_lines[:5])

    style_guide = {
        'professional': '專業穩重、條理清晰，像財經新聞主播',
        'casual': '親切自然、口語化，像 YouTube 財經 KOL',
        'energetic': '活潑有精神、適度使用驚嘆語氣，像早間節目主持人',
    }

    today = datetime.datetime.now().strftime('%Y年%m月%d日')

    prompt = f"""你是台股虛擬主播。請根據以下新聞和資料，撰寫一段播報稿。

風格：{style_guide.get(style, style_guide['professional'])}

要求：
1. 開場問候（含日期 {today}）
2. 今日大盤概況（從新聞推斷）
3. 重點個股新聞（挑 3-5 則最重要的）
4. AI 預測提示（如有）
5. 結尾總結 + 提醒投資風險

限制：
- 繁體中文
- 總字數嚴格控制在 {SCRIPT_MIN_CHARS}-{SCRIPT_MAX_CHARS} 字，不能超過
- 精簡扼要，每檔股票一句話帶過
- 不要用 markdown 格式，純文字即可
- 講稿要適合「唸出來」，不要有表格或符號
- 回傳純播報稿文字，不要加任何額外說明

今日新聞：
{chr(10).join(news_summary)}
{pred_summary}"""

    url = GEMINI_API_URL + f"?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        },
    }

    logger.info("正在生成播報稿...")
    t0 = time.time()

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    script = result['candidates'][0]['content']['parts'][0]['text']
    script = script.strip()

    elapsed = time.time() - t0
    logger.info(f"播報稿生成完成（{len(script)} 字，{elapsed:.1f}s）")

    return script


# ═══════════════════════════════════════════
# Step 3: OpenAI TTS 語音合成
# ═══════════════════════════════════════════

def generate_tts(script_text, output_path=None):
    """
    用 OpenAI TTS 將播報稿轉成語音

    Args:
        script_text: 播報稿文字
        output_path: 輸出 mp3 路徑（預設 anchor_output/）

    Returns:
        str: 音檔路徑
    """
    from gpt_sentiment import get_client

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not output_path:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_path = os.path.join(OUTPUT_DIR, f'anchor_{ts}.mp3')

    client = get_client()

    logger.info(f"正在合成語音（{len(script_text)} 字）...")
    t0 = time.time()

    # OpenAI TTS — 一次最多 4096 字元
    # 中文 800 字 ≈ 2400 字元，通常不用分割
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        speed=TTS_SPEED,
        input=script_text,
    ) as response:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    elapsed = time.time() - t0
    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"語音合成完成: {output_path} ({size_kb:.0f} KB, {elapsed:.1f}s)")

    return output_path


# ═══════════════════════════════════════════
# Step 4: Hedra 影片生成（圖片 + 音檔 → 說話影片）
# ═══════════════════════════════════════════

def generate_video_hedra(audio_path, image_path=None, output_path=None):
    """
    用 Hedra API 生成虛擬主播影片

    Args:
        audio_path: TTS 音檔路徑
        image_path: 主播形象圖（預設 anchor_avatar.png）
        output_path: 輸出影片路徑

    Returns:
        str: 影片路徑，或 None（失敗）
    """
    config = load_anchor_config()
    api_key = config.get('hedra_api_key') or os.environ.get('HEDRA_API_KEY')

    if not api_key:
        logger.error("Hedra API key 未設定。請執行: python news_anchor.py --setup-hedra YOUR_KEY")
        return None

    image_path = image_path or ANCHOR_IMAGE_FILE
    if not os.path.exists(image_path):
        logger.error(f"找不到主播形象圖: {image_path}")
        logger.error("請準備一張正面人像圖片放在 anchor_avatar.png")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_path:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_path = os.path.join(OUTPUT_DIR, f'anchor_{ts}.mp4')

    base_url = "https://api.hedra.com/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        # 1) 上傳圖片
        logger.info("上傳主播形象圖到 Hedra...")
        import mimetypes
        img_mime = mimetypes.guess_type(image_path)[0] or 'image/png'

        # multipart upload for image
        boundary = '----HedraUpload'
        with open(image_path, 'rb') as img_f:
            img_data = img_f.read()

        body_parts = []
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(image_path)}"'.encode())
        body_parts.append(f'Content-Type: {img_mime}'.encode())
        body_parts.append(b'')
        body_parts.append(img_data)
        body_parts.append(f'--{boundary}--'.encode())
        body_bytes = b'\r\n'.join(body_parts)

        req = urllib.request.Request(
            f"{base_url}/portrait",
            data=body_bytes,
            headers={
                **headers,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            portrait_result = json.loads(resp.read().decode())
        portrait_id = portrait_result.get('id') or portrait_result.get('portrait_id')
        logger.info(f"圖片上傳成功: {portrait_id}")

        # 2) 上傳音檔
        logger.info("上傳音檔到 Hedra...")
        with open(audio_path, 'rb') as audio_f:
            audio_data = audio_f.read()

        body_parts2 = []
        body_parts2.append(f'--{boundary}'.encode())
        body_parts2.append(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"'.encode())
        body_parts2.append(b'Content-Type: audio/mpeg')
        body_parts2.append(b'')
        body_parts2.append(audio_data)
        body_parts2.append(f'--{boundary}--'.encode())
        body_bytes2 = b'\r\n'.join(body_parts2)

        req2 = urllib.request.Request(
            f"{base_url}/audio",
            data=body_bytes2,
            headers={
                **headers,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=60) as resp:
            audio_result = json.loads(resp.read().decode())
        audio_id = audio_result.get('id') or audio_result.get('audio_id')
        logger.info(f"音檔上傳成功: {audio_id}")

        # 3) 建立影片任務
        logger.info("建立影片生成任務...")
        job_body = json.dumps({
            "portrait_id": portrait_id,
            "audio_id": audio_id,
        }).encode()

        req3 = urllib.request.Request(
            f"{base_url}/characters",
            data=job_body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req3, timeout=30) as resp:
            job_result = json.loads(resp.read().decode())
        job_id = job_result.get('id') or job_result.get('job_id')
        logger.info(f"影片任務建立: {job_id}")

        # 4) 輪詢等待完成
        logger.info("等待影片生成...")
        t0 = time.time()
        max_wait = 600  # 最多等 10 分鐘

        while time.time() - t0 < max_wait:
            req_status = urllib.request.Request(
                f"{base_url}/characters/{job_id}",
                headers=headers,
            )
            with urllib.request.urlopen(req_status, timeout=30) as resp:
                status = json.loads(resp.read().decode())

            state = status.get('status', status.get('state', ''))
            if state in ('completed', 'done', 'ready'):
                video_url = status.get('video_url') or status.get('url')
                logger.info(f"影片生成完成！({time.time()-t0:.0f}s)")

                # 下載影片
                urllib.request.urlretrieve(video_url, output_path)
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"影片已下載: {output_path} ({size_mb:.1f} MB)")
                return output_path

            elif state in ('failed', 'error'):
                logger.error(f"影片生成失敗: {status}")
                return None

            elapsed = time.time() - t0
            logger.info(f"  生成中... ({elapsed:.0f}s, status={state})")
            time.sleep(10)

        logger.error(f"影片生成超時（>{max_wait}s）")
        return None

    except Exception as e:
        logger.error(f"Hedra 影片生成錯誤: {e}")
        return None


# ═══════════════════════════════════════════
# Step 5: 推播到 Discord
# ═══════════════════════════════════════════

def broadcast_to_discord(script_text, media_path=None, channel='release'):
    """
    將播報結果推到 Discord

    Args:
        script_text: 播報稿（作為文字附帶）
        media_path: 音檔或影片路徑（可選）
        channel: Discord 頻道

    Returns:
        bool: 成功/失敗
    """
    from notifier import send_discord_embed, send_discord_file, _get_webhook_url

    now = datetime.datetime.now()

    # 先發 embed 摘要
    embed = {
        "title": f"📺 AI 主播播報 | {now.strftime('%Y/%m/%d %H:%M')}",
        "description": script_text[:500] + ('...' if len(script_text) > 500 else ''),
        "color": 0x9B59B6,  # 紫色
        "footer": {"text": f"AI News Anchor | {now.strftime('%H:%M')}"},
    }
    send_discord_embed(embed, channel=channel)

    # 再發媒體檔
    if media_path and os.path.exists(media_path):
        ext = os.path.splitext(media_path)[1].lower()
        if ext == '.mp4':
            msg = "🎬 虛擬主播影片播報"
        elif ext == '.mp3':
            msg = "🎙️ AI 語音播報"
        else:
            msg = "📎 播報附件"

        return send_discord_file(media_path, message=msg, channel=channel)

    return True


# ═══════════════════════════════════════════
# 主 Pipeline
# ═══════════════════════════════════════════

def run_pipeline(date_str=None, mode=None, channel='release',
                 style='professional', test=False):
    """
    執行完整播報 pipeline

    Args:
        date_str: 新聞日期（預設今天）
        mode: 'video' / 'audio' / 'text'（預設自動偵測）
        channel: Discord 頻道
        style: 播報風格
        test: 測試模式（短稿）

    Returns:
        dict: {script, audio_path, video_path, elapsed, mode}
    """
    t_start = time.time()
    result = {'script': None, 'audio_path': None, 'video_path': None}

    if not mode:
        mode = detect_mode()
    result['mode'] = mode
    logger.info(f"=== AI 主播 Pipeline 啟動（mode={mode}）===")

    # Step 1: 讀取新聞
    t1 = time.time()
    news = load_news(date_str)
    if not news:
        logger.error("沒有新聞資料，中止")
        return result
    logger.info(f"[Step 1] 讀取新聞: {time.time()-t1:.1f}s")

    # 載入預測
    predictions = load_predictions()

    # Step 2: 生成播報稿
    t2 = time.time()
    if test:
        script = f"大家好，這是測試播報。今天有 {len(news)} 則新聞。最重要的是：{news[0]['title']}。以上是今天的股市速報，我們明天見！"
    else:
        script = generate_script(news, predictions, style=style)
    result['script'] = script
    logger.info(f"[Step 2] 播報稿: {time.time()-t2:.1f}s（{len(script)} 字）")

    # 存播報稿
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    script_path = os.path.join(OUTPUT_DIR, f'script_{ts}.txt')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

    if mode == 'text':
        broadcast_to_discord(script, channel=channel)
        result['elapsed'] = time.time() - t_start
        logger.info(f"[完成] 純文字播報 ({result['elapsed']:.1f}s)")
        return result

    # Step 3: TTS 語音
    t3 = time.time()
    audio_path = generate_tts(script)
    result['audio_path'] = audio_path
    logger.info(f"[Step 3] TTS: {time.time()-t3:.1f}s")

    if mode == 'audio':
        broadcast_to_discord(script, media_path=audio_path, channel=channel)
        result['elapsed'] = time.time() - t_start
        logger.info(f"[完成] 語音播報 ({result['elapsed']:.1f}s)")
        return result

    # Step 4: 影片生成
    t4 = time.time()
    video_path = generate_video_hedra(audio_path)
    result['video_path'] = video_path
    logger.info(f"[Step 4] 影片: {time.time()-t4:.1f}s")

    # Step 5: 推播
    media = video_path or audio_path
    broadcast_to_discord(script, media_path=media, channel=channel)

    result['elapsed'] = time.time() - t_start
    logger.info(f"[完成] 影片播報 ({result['elapsed']:.1f}s)")

    return result


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description='AI 虛擬主播新聞播報')
    parser.add_argument('--date', help='新聞日期 YYYYMMDD')
    parser.add_argument('--mode', choices=['video', 'audio', 'text'],
                        help='播報模式')
    parser.add_argument('--style', default='professional',
                        choices=['professional', 'casual', 'energetic'],
                        help='播報風格')
    parser.add_argument('--channel', default='release',
                        choices=['release', 'test'],
                        help='Discord 頻道')
    parser.add_argument('--test', action='store_true',
                        help='測試模式（短稿）')
    parser.add_argument('--no-discord', action='store_true',
                        help='不推播到 Discord（只生成檔案）')
    parser.add_argument('--setup-hedra', metavar='API_KEY',
                        help='設定 Hedra API key')

    args = parser.parse_args()

    if args.setup_hedra:
        config = load_anchor_config()
        config['hedra_api_key'] = args.setup_hedra
        save_anchor_config(config)
        print("Hedra API key 已設定！")
        sys.exit(0)

    result = run_pipeline(
        date_str=args.date,
        mode=args.mode,
        channel=args.channel,
        style=args.style,
        test=args.test,
    )

    print(f"\n{'='*50}")
    print(f"模式: {result['mode']}")
    if result['script']:
        print(f"播報稿: {len(result['script'])} 字")
    if result['audio_path']:
        print(f"音檔: {result['audio_path']}")
    if result['video_path']:
        print(f"影片: {result['video_path']}")
    if result.get('elapsed'):
        print(f"總耗時: {result['elapsed']:.1f} 秒")
    print(f"{'='*50}")
