import streamlit as st
import os
import whisper
import yt_dlp
import pandas as pd
import cv2
import math
from datetime import timedelta
import base64
import pytesseract
from PIL import Image
import Levenshtein

# --- SecretsからCookieを読み込む ---
# Hugging FaceのSecretsに 'YOUTUBE_COOKIE' という名前で設定した値を読み込む
youtube_cookie_content = os.environ.get('YOUTUBE_COOKIE')
cookie_file_path = 'cookies.txt'

# アプリ起動時に一度だけCookieファイルを作成する
if youtube_cookie_content and not os.path.exists(cookie_file_path):
    with open(cookie_file_path, 'w') as f:
        f.write(youtube_cookie_content)

# --- バックエンド処理関数の定義 ---
@st.cache_resource
def load_whisper_model():
    # mediumモデルはメモリを多く消費するため、最初は "base" や "small" で試すのがおすすめ
    return whisper.load_model("medium", device="cpu")

@st.cache_data
def process_video(url):
    progress_bar = st.progress(0, text="処理を開始します...")
    video_path = None

    # Cookieファイルが存在するかチェック
    use_cookie = os.path.exists(cookie_file_path) and os.path.getsize(cookie_file_path) > 0
    if not use_cookie:
        st.info("Cookie情報が設定されていません。公開動画のみ解析可能です。")

    try:
        progress_bar.progress(5, text="動画をダウンロード中...")
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]',
            'outtmpl': 'temp_video.%(ext)s',
        }
        # Cookieファイルが存在すればオプションに追加
        if use_cookie:
            ydl_opts['cookies'] = cookie_file_path
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)

        progress_bar.progress(25, text="音声を文字起こし中...")
        whisper_model = load_whisper_model()
        audio_result = whisper_model.transcribe(video_path, verbose=False)
        voice_data = [{'timestamp': segment['start'], 'text': segment['text'].strip()} for segment in audio_result['segments']]

        progress_bar.progress(65, text="テロップの出現を検知・撮影中...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        telop_screenshots = []
        last_text_block = ""
        frame_count = 0
        custom_config = r'-l jpn+eng --psm 3'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % math.ceil(fps) == 0:
                timestamp = frame_count / fps
                height, _, _ = frame.shape
                roi = frame[height // 2:, :]

                ocr_data = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)
                current_text_parts = []
                for i in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][i]) > 60:
                        text = ocr_data['text'][i].strip()
                        if text:
                            current_text_parts.append(text)

                current_text_block = "".join(current_text_parts)

                is_new_subtitle = False
                if current_text_block:
                    if not last_text_block:
                        is_new_subtitle = True
                    else:
                        max_len = max(len(current_text_block), len(last_text_block))
                        if max_len > 0:
                            similarity = 1.0 - (Levenshtein.distance(current_text_block, last_text_block) / max_len)
                            if similarity < 0.9:
                                is_new_subtitle = True

                if is_new_subtitle:
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_str = base64.b64encode(buffer).decode()
                    telop_screenshots.append({'timestamp': timestamp, 'image_str': img_str, 'text': current_text_block})
                    last_text_block = current_text_block
                
                progress_value = 65 + int(30 * (frame_count / total_frames)) if total_frames > 0 else 65
                progress_text = f"テロップ検知中... {int(100 * frame_count/total_frames)}%" if total_frames > 0 else "テロップ検知中..."
                progress_bar.progress(progress_value, text=progress_text)
            frame_count += 1
        cap.release()
    finally:
        if video_path and os.path.exists(video_path): os.remove(video_path)

    progress_bar.progress(100, text="処理完了！")
    return voice_data, telop_screenshots

# --- Streamlit Webアプリケーション本体 ---
st.set_page_config(page_title="AI動画レビュー支援ツール", layout="wide")
st.title("🤖 AI動画レビュー支援ツール")

youtube_url = st.text_input("YouTube動画のURLを入力してください", "")

if st.button("解析開始", type="primary"):
    if youtube_url:
        with st.spinner('解析中です...完了まで数分かかることがあります。'):
            # セッションステートに結果を保存
            st.session_state.results = process_video(youtube_url)
    else:
        st.warning("URLを入力してください。")


# --- 表示ロジック ---
if 'results' in st.session_state and st.session_state.results:
    voice_data, telop_screenshots = st.session_state.results
    
    all_events = []
    if voice_data:
        for item in voice_data: all_events.append({'timestamp': item['timestamp'], 'type': 'voice', 'data': item['text']})
    if telop_screenshots:
        for item in telop_screenshots: all_events.append({'timestamp': item['timestamp'], 'type': 'telop', 'data': item['image_str'], 'text': item['text']})

    grouped_events = {}
    for event in all_events:
        time_key = int(event['timestamp'])
        if time_key not in grouped_events:
            grouped_events[time_key] = {'voice': [], 'telop': []}

        if event['type'] == 'voice':
            grouped_events[time_key]['voice'].append(event)
        else:
            grouped_events[time_key]['telop'].append(event)

    st.subheader("解析結果")
    sorted_timestamps = sorted(grouped_events.keys())

    for ts in sorted_timestamps:
        group = grouped_events[ts]
        g_col1, g_col2 = st.columns(2)
        time_str = str(timedelta(seconds=ts))
        jump_url = f"{youtube_url.split('&')[0]}&t={ts}s"

        with g_col1:
            if group['voice']:
                st.markdown(f'##### <a href="{jump_url}" target="_blank">🗣️ [{time_str}]</a>', unsafe_allow_html=True)
                for voice_event in group['voice']:
                    st.markdown(f"> {voice_event['data']}")

        with g_col2:
            if group['telop']:
                if not group['voice']:
                    st.markdown(f'##### <a href="{jump_url}" target="_blank">🖼️ [{time_str}]</a>', unsafe_allow_html=True)

                for telop_event in group['telop']:
                    st.image(f"data:image/jpeg;base64,{telop_event['data']}", use_container_width=True)
                    st.caption(f"認識テキスト: {telop_event['text']}")
        st.markdown("---")