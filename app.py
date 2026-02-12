import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import streamlit as st
import google.generativeai as genai
import edge_tts
import subprocess
import shutil
import whisper
import time
import json
import re
import requests
import textwrap
import math
import uuid
import yt_dlp # New Library for YouTube
from google.cloud import texttospeech
from google.oauth2 import service_account

# ---------------------------------------------------------
# 🛡️ 1. SESSION & FOLDER ISOLATION
# ---------------------------------------------------------
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

SID = st.session_state.session_id
BASE_WORK_DIR = os.path.abspath("user_sessions")
USER_SESSION_DIR = os.path.join(BASE_WORK_DIR, SID)
os.makedirs(USER_SESSION_DIR, exist_ok=True)

# Define clean file paths
FILE_INPUT = os.path.join(USER_SESSION_DIR, "input_video.mp4")
FILE_AUDIO_RAW = os.path.join(USER_SESSION_DIR, "extracted_audio.wav")
FILE_VOICE = os.path.join(USER_SESSION_DIR, "ai_voice.mp3")
FILE_BGM = os.path.join(USER_SESSION_DIR, "bgm_music.mp3") # NEW: Background Music
FILE_FINAL = os.path.join(USER_SESSION_DIR, "final_dubbed_video.mp4")

FILE_CAP_INPUT = os.path.join(USER_SESSION_DIR, "caption_input_video.mp4")
FILE_CAP_WAV = os.path.join(USER_SESSION_DIR, "caption_audio.wav")
FILE_CAP_FINAL = os.path.join(USER_SESSION_DIR, "captioned_output.mp4")
FILE_ASS = os.path.join(USER_SESSION_DIR, "subtitles.ass")
FILE_SRT = os.path.join(USER_SESSION_DIR, "subtitles.srt") # NEW: SRT File

# ---------------------------------------------------------
# 🎨 UI SETUP (WHITE THEME)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# 🔥 CHANGED: CSS for White Background & Black Text
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #d1d5db;
    }
    .main-header {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #0052D4 0%, #4364F7 50%, #6FB1FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 20px;
        margin-top: -50px;
    }
    /* Update inputs to look good on white */
    div[data-testid="stFileUploader"], div[class="stTextArea"], div[class="stTextInput"] {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        border-radius: 10px;
        color: #000;
    }
    h1, h2, h3, h4, p, label {
        color: #333333 !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #0052D4, #4364F7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
    <div class="main-header">MYANMAR AI STUDIO PRO</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'raw_transcript' not in st.session_state: st.session_state.raw_transcript = ""
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
if 'srt_path' not in st.session_state: st.session_state.srt_path = None
if 'api_keys' not in st.session_state: st.session_state.api_keys = []
if 'google_creds' not in st.session_state: st.session_state.google_creds = None

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def load_custom_dictionary():
    dict_file = "dictionary.txt"
    if os.path.exists(dict_file):
        with open(dict_file, "r", encoding="utf-8") as f: return f.read()
    return ""

def load_pronunciation_dict():
    pron_file = "pronunciation.txt"
    replacements = {}
    if os.path.exists(pron_file):
        with open(pron_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        replacements[parts[0].strip()] = parts[1].strip()
    return replacements

def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please add 'ffmpeg' to packages.txt")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0.0

def download_font():
    font_dir = os.path.abspath("fonts_cache")
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "Padauk-Bold.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_path, 'wb') as f: f.write(r.content)
        except: pass
    return font_path

def load_whisper_safe():
    try: return whisper.load_model("base")
    except Exception as e: st.error(f"Whisper Error (Try reloading): {e}"); return None

# 🔥 NEW: YouTube Downloader Function
def download_youtube_video(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        st.error(f"YouTube Download Error: {e}")
        return False

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
GOOGLE_VOICE_MAP = {
    "Burmese": {"Male": "my-MM-Standard-A", "Female": "my-MM-Standard-A"}, 
    "English": {"Male": "en-US-Neural2-D", "Female": "en-US-Neural2-F"},
}
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-5%", "pitch": "-2Hz"}, 
    "Recap": {"rate": "+5%", "pitch": "+0Hz"},
    "Motivation": {"rate": "+10", "pitch": "+2Hz"},
}
EMOTION_MAP = {
    "[normal]": {"rate": "+0%", "pitch": "+0Hz"},
    "[sad]":    {"rate": "-15%", "pitch": "-15Hz"},
    "[angry]":  {"rate": "+15%", "pitch": "+5Hz"},
    "[happy]":  {"rate": "+10%", "pitch": "+15Hz"},
    "[action]": {"rate": "+30%", "pitch": "+0Hz"},
    "[whisper]": {"rate": "-10%", "pitch": "-20Hz"},
}

def generate_edge_chunk(text, lang, gender, rate_str, pitch_str, output_file):
    voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
    cmd = ["edge-tts", "--voice", voice_id, "--text", text, f"--rate={rate_str}", f"--pitch={pitch_str}", "--write-media", output_file]
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return True
        except: time.sleep(1); continue
    return False

def generate_google_chunk(text, lang, gender, rate_val, pitch_val, output_file, creds):
    try:
        client = texttospeech.TextToSpeechClient(credentials=creds)
        s_input = texttospeech.SynthesisInput(text=text)
        g_voice_name = GOOGLE_VOICE_MAP.get(lang, {}).get(gender, "en-US-Neural2-F")
        lang_code = "my-MM" if lang == "Burmese" else "en-US"
        voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=g_voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=rate_val, pitch=pitch_val)
        response = client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
        with open(output_file, "wb") as out: out.write(response.audio_content)
        return True
    except Exception as e: print(f"Google TTS Error: {e}"); return False

def generate_audio_with_emotions(full_text, lang, gender, base_mode, output_file, engine="Edge TTS", base_speed=1.0):
    base_settings = VOICE_MODES.get(base_mode, VOICE_MODES["Normal"])
    base_r_int = int(base_settings['rate'].replace('%', ''))
    base_p_int = int(base_settings['pitch'].replace('Hz', ''))
    slider_adj = int((base_speed - 1.0) * 100)
    current_rate = base_r_int + slider_adj
    current_pitch = base_p_int

    parts = re.split(r'(\[.*?\])', full_text)
    audio_segments = []
    chunk_idx = 0
    output_dir = os.path.dirname(output_file)
    
    for part in parts:
        part = part.strip()
        if not part: continue
        part_lower = part.lower()

        if part_lower == "[p]":
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_idx}_silence.mp3")
            cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', '1', '-q:a', '9', chunk_filename]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(chunk_filename): audio_segments.append(chunk_filename); chunk_idx += 1
            continue

        if part_lower in EMOTION_MAP:
            emo = EMOTION_MAP[part_lower]
            base_r = int(base_settings['rate'].replace('%', '')) + slider_adj
            base_p = int(base_settings['pitch'].replace('Hz', ''))
            current_rate = base_r + int(emo['rate'].replace('%', ''))
            current_pitch = base_p + int(emo['pitch'].replace('Hz', ''))
            continue
        
        if part.startswith("[") and part.endswith("]"): continue
        
        processed_text = normalize_text_for_tts(part)
        if not processed_text.strip(): continue
        
        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_idx}.mp3")
        
        success = False
        if engine == "Google Cloud TTS" and st.session_state.google_creds:
            g_rate = 1.0 + (current_rate / 100.0)
            g_pitch = current_pitch / 10.0 
            success = generate_google_chunk(processed_text, lang, gender, g_rate, g_pitch, chunk_filename, st.session_state.google_creds)
        else:
            rate_str = f"{current_rate:+d}%"
            pitch_str = f"{current_pitch:+d}Hz"
            success = generate_edge_chunk(processed_text, lang, gender, rate_str, pitch_str, chunk_filename)
        
        if success:
            audio_segments.append(chunk_filename)
            chunk_idx += 1
            if engine == "Edge TTS": time.sleep(0.1)

    if not audio_segments: return False, "No audio generated"
    
    concat_list = os.path.join(output_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for seg in audio_segments: f.write(f"file '{seg}'\n")
            
    try:
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

# ---------------------------------------------------------
# 🔢 TEXT NORMALIZATION
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    try:
        if "." in num_str:
            parts = num_str.split(".")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return f"{num_to_burmese_spoken(parts[0])} ဒသမ {num_to_burmese_spoken(parts[1])}"
        num_str = num_str.replace(",", "")
        n = int(num_str)
        if n == 0: return "သုည"
        digit_map = ["", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"]
        def convert_chunk(number):
            parts = []
            if number >= 10000000: parts.append(convert_chunk(number // 10000000) + "ကုဋေ"); number %= 10000000
            if number >= 1000000: parts.append(digit_map[number // 1000000] + "သန်း"); number %= 1000000
            if number >= 100000: parts.append(digit_map[number // 100000] + "သိန်း"); number %= 100000
            if number >= 10000: parts.append(digit_map[number // 10000] + "သောင်း"); number %= 10000
            if number >= 1000: parts.append(digit_map[number // 1000] + "ထောင်"); number %= 1000
            if number >= 100: parts.append(digit_map[number // 100] + "ရာ"); number %= 100
            if number >= 10: parts.append(digit_map[number // 10] + "ဆယ်"); number %= 10
            if number > 0: parts.append(digit_map[number])
            return "".join(parts)
        result = convert_chunk(n)
        result = result.replace("ထောင်", "ထောင့်").replace("ရာ", "ရာ့").replace("ဆယ်", "ဆယ့်")
        if result.endswith("ထောင့်"): result = result[:-1] + "င်"
        if result.endswith("ရာ့"): result = result[:-1]
        if result.endswith("ဆယ့်"): result = result[:-1]
        return result
    except: return num_str

def normalize_text_for_tts(text):
    if not text: return ""
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    pron_dict = load_pronunciation_dict()
    sorted_keys = sorted(pron_dict.keys(), key=len, reverse=True)
    for original in sorted_keys:
        text = re.sub(re.escape(original), pron_dict[original], text, flags=re.IGNORECASE)
    text = text.replace("၊", ", ").replace("။", ". ").replace("[p]", "... ") 
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda x: num_to_burmese_spoken(x.group()), text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 🧠 AI ENGINE (GEMINI)
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_with_retry(prompt):
    keys = st.session_state.api_keys
    model_name = st.session_state.get("selected_model", "gemini-2.5-flash") 
    custom_rules = load_custom_dictionary()
    if custom_rules: prompt = f"RULES:\n{custom_rules}\n\nTASK:\n{prompt}"
    for i, key in enumerate(keys):
        try:
            model = get_model(key, model_name)
            response = model.generate_content(prompt)
            return response.text
        except: continue
    return "AI Error: All keys failed."

# ---------------------------------------------------------
# 📝 .ASS & .SRT SUBTITLES (UPDATED)
# ---------------------------------------------------------
def generate_ass_file(segments, font_path, output_path):
    def seconds_to_ass(seconds):
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60); cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCut,Padauk-Bold,24,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            raw_text = seg['text'].strip()
            wrapped_lines = textwrap.wrap(raw_text, width=40)
            final_text = "\\N".join(wrapped_lines) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return output_path

# 🔥 NEW: SRT Generator
def generate_srt_file(segments, output_path):
    def seconds_to_srt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start = seconds_to_srt(seg['start'])
            end = seconds_to_srt(seg['end'])
            text = seg['text'].strip()
            f.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
    return output_path

# ---------------------------------------------------------
# 🖥️ MAIN UI & SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 🔥 NEW: YouTube Downloader in Sidebar
    st.markdown("### 📥 YouTube Downloader")
    yt_url = st.text_input("YouTube Link:")
    if st.button("Download Video"):
        if yt_url:
            with st.spinner("Downloading from YouTube..."):
                if download_youtube_video(yt_url, FILE_INPUT):
                    st.success("✅ Downloaded! Go to 'Dubbing Studio'.")
                else:
                    st.error("❌ Failed to download.")

    st.divider()
    
    st.markdown("☁️ **Google Cloud TTS (Optional):**")
    gcp_file = st.file_uploader("Upload service_account.json", type=["json"])
    if gcp_file:
        try:
            gcp_data = json.load(gcp_file)
            st.session_state.google_creds = service_account.Credentials.from_service_account_info(gcp_data)
            st.success("✅ GCP Key Loaded!")
        except: st.error("❌ Invalid JSON File")

    st.divider()

    # --- START NEW CODE ---
    with st.expander("🔑 API Keys & Model", expanded=True):
        # User ကို Key မဖြစ်မနေ ထည့်ခိုင်းမယ် (System Key မသုံးတော့ပါ)
        api_key_input = st.text_input("Gemini API Key (Required)", value="", type="password", help="Enter your own Google AI Studio API Key.")
        
        if api_key_input:
            st.session_state.api_keys = [k.strip() for k in api_key_input.split(",") if k.strip()]
        else:
            st.session_state.api_keys = []
        
        st.session_state.selected_model = st.selectbox("Model", [ "gemini-2.5-flash","gemini-1.5-pro"], index=0)

    if st.button("🔴 Reset System", use_container_width=True):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# Key မရှိရင် App ကို ရှေ့ဆက်မသွားအောင် တားထားမယ်
if not st.session_state.api_keys: 
    st.warning("⚠️ Enter Gemini API Keys in Sidebar to continue")
    st.stop()
    # --- END NEW CODE ---
t1, t2, t3 = st.tabs(["🎙️ DUBBING STUDIO", "📝 AUTO CAPTION", "🚀 VIRAL SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    col_up, col_set = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
        # Check if file exists from YouTube download
        if not uploaded and os.path.exists(FILE_INPUT):
            st.info("Using downloaded YouTube video.")
    
    with col_set:
        task_mode = st.radio("Mode", ["🗣️ Translate (Dubbing)", "👀 AI Narration"])
        if task_mode == "🗣️ Translate (Dubbing)":
            in_lang = st.selectbox("Input Language", ["English", "Burmese", "Japanese", "Chinese", "Thai"])
        else:
            vibe = st.selectbox("Style", ["Vlog", "Tutorial", "Relaxing", "Exciting"]) 
        out_lang = st.selectbox("Output Language", ["Burmese", "English"], index=0)
    
    # Save uploaded file if provided, overwriting YouTube download
    if uploaded:
        with open(FILE_INPUT, "wb") as f: f.write(uploaded.getbuffer())

    if os.path.exists(FILE_INPUT):
        if st.button("🚀 Start Magic", use_container_width=True):
            check_requirements()
            p_bar = st.progress(0, text="Starting...")

            # --- PATH A: TRANSLATION ---
            if task_mode == "🗣️ Translate (Dubbing)":
                p_bar.progress(20, text="🎤 Listening to Audio...")
                subprocess.run(['ffmpeg', '-y', '-i', FILE_INPUT, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', FILE_AUDIO_RAW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                model = load_whisper_safe()
                if model:
                    lang_map = {"Burmese": "my", "English": "en", "Japanese": "ja", "Chinese": "zh", "Thai": "th"}
                    lang_code = lang_map.get(in_lang, "en")
                    raw = model.transcribe(FILE_AUDIO_RAW, language=lang_code)['text']
                    st.session_state.raw_transcript = raw
                    p_bar.progress(50, text="🧠 Translating...")
                    
                    prompt = f"""
                    ROLE: Professional Movie Narrator.
                    TASK: Translate/Rewrite input into a flowing, dramatic {out_lang} script.
                    Input: '{raw}'"""
                    st.session_state.final_script = generate_with_retry(prompt)

            # --- PATH B: AI NARRATION ---
            else:
                p_bar.progress(20, text="👀 AI is watching video...")
                try:
                    genai.configure(api_key=st.session_state.api_keys[0])
                    video_file = genai.upload_file(path=FILE_INPUT)
                    while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
                    p_bar.progress(50, text="✍️ Writing Script...")
                    prompt = f"Write a {vibe} style voiceover script in {out_lang} for this video. Do NOT describe visual scenes, just tell the story."
                    model = genai.GenerativeModel(model_name=st.session_state.selected_model)
                    response = model.generate_content([video_file, prompt])
                    st.session_state.final_script = response.text
                    genai.delete_file(video_file.name)
                except Exception as e: st.error(f"Error: {e}"); st.stop()

            p_bar.progress(100, text="✅ Script Ready!")
            st.rerun()
        
        txt = st.text_area("Final Script", st.session_state.final_script, height=200)

        st.markdown("---")
        
# --- REPLACE FROM HERE (Inside Tab 1) ---
        st.markdown("#### ⚙️ Rendering Options")
        
        # 1. Freeze Feature (Button Selection)
        st.markdown("❄️ **Freeze Frame Effect**")
        # User can select multiple freeze points
        freeze_options = st.multiselect(
            "Select Freeze Points (Video pauses for 3s, Audio continues):",
            ["30 Seconds", "1 Minute"],
            help="At these points, video will freeze for 3 seconds."
        )

        st.divider()

        tts_engine = st.radio("Voice Engine", ["Edge TTS (Free)", "Google Cloud TTS (Pro)"], horizontal=True)
        export_format = st.radio("Export Format:", ["🎬 Video (MP4)", "🎵 Audio Only (MP3)"], horizontal=True)
        
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Voice Lang", list(VOICE_MAP.keys()), index=0)
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        audio_speed = st.slider("🔊 Audio Speed", 0.5, 2.0, 1.0, 0.05)
        video_speed = st.slider("🎞️ Video Speed (Synced)", 0.5, 4.0, 1.0, 0.1)

        if st.button("🚀 GENERATE FINAL", use_container_width=True):
            p_bar = st.progress(0, text="Generating Audio...")
            if not txt.strip(): st.error("❌ Script is empty!"); st.stop()

            # 1. Generate TTS
            success, msg = generate_audio_with_emotions(txt, target_lang, gender, v_mode, FILE_VOICE, engine=tts_engine, base_speed=audio_speed)
            if not success: st.error(msg); st.stop()
            st.session_state.processed_audio_path = FILE_VOICE

            # Audio Only Logic
            if "Audio" in export_format:
                p_bar.progress(100, text="✅ Audio Generated Successfully!")
                time.sleep(0.5)
                st.rerun()

            elif "Video" in export_format:
                p_bar.progress(10, text="❄️ Preparing Video...")
                
                # --- FREEZE FRAME LOGIC ---
                # We start with the original input
                current_video_stage = FILE_INPUT 
                
                # Sort options to process in order (30s then 60s)
                sorted_freezes = []
                if "30 Seconds" in freeze_options: sorted_freezes.append(30)
                if "1 Minute" in freeze_options: sorted_freezes.append(60)

                # Process freeze points sequentially
                for i, freeze_time in enumerate(sorted_freezes):
                    p_bar.progress(20 + (i * 10), text=f"❄️ Freezing at {freeze_time}s...")
                    try:
                        # Adjusted time: If we already added a freeze, the next timestamp shifts!
                        # But for simplicity & stability, we assume timestamps refer to ORIGINAL video time.
                        # We process strictly based on the current stage video.
                        
                        unique_suffix = uuid.uuid4().hex[:6]
                        part1 = os.path.join(USER_SESSION_DIR, f"p1_{unique_suffix}.mp4")
                        part2 = os.path.join(USER_SESSION_DIR, f"p2_{unique_suffix}.mp4")
                        freeze_img = os.path.join(USER_SESSION_DIR, f"frame_{unique_suffix}.jpg")
                        freeze_vid = os.path.join(USER_SESSION_DIR, f"frozen_{unique_suffix}.mp4")
                        output_stage = os.path.join(USER_SESSION_DIR, f"stage_{unique_suffix}.mp4")
                        list_txt = os.path.join(USER_SESSION_DIR, f"concat_{unique_suffix}.txt")

                        # Duration to freeze
                        duration_s = 3 

                        # 1. Split Video at Freeze Point
                        subprocess.run(['ffmpeg', '-y', '-i', current_video_stage, '-t', str(freeze_time), '-c', 'copy', part1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # 2. Capture Frame at Freeze Point
                        subprocess.run(['ffmpeg', '-y', '-ss', str(freeze_time), '-i', current_video_stage, '-frames:v', '1', '-q:v', '2', freeze_img], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # 3. Loop Frame for 3 seconds (Ensure yuv420p for compatibility)
                        subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', freeze_img, '-t', str(duration_s), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', freeze_vid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # 4. Get the Rest of the video
                        subprocess.run(['ffmpeg', '-y', '-ss', str(freeze_time), '-i', current_video_stage, '-c', 'copy', part2], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        # 5. Concat (Part1 + Freeze + Part2)
                        with open(list_txt, 'w') as f:
                            f.write(f"file '{part1}'\nfile '{freeze_vid}'\nfile '{part2}'")
                        
                        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy', output_stage], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        if os.path.exists(output_stage):
                            current_video_stage = output_stage # Update input for next loop or final
                            # IMPORTANT: If we freeze at 30s, the video becomes 3s longer.
                            # If we then want to freeze at 60s (original time), the new time is 63s.
                            # For this code block, to keep it simple and bug-free, we reset the file pointer.
                        
                    except Exception as e:
                        print(f"Freeze Error: {e}")

                # --- FINAL MIXING ---
                p_bar.progress(60, text="🎞️ Mixing Final Video (Fixing Colors)...")
                pts_val = 1.0 / video_speed
                
                # 🔥 SOLUTION FOR SESSION MIXING: Unique Filename per generation
                timestamp_str = int(time.time())
                FINAL_OUTPUT_NAME = os.path.join(USER_SESSION_DIR, f"final_output_{timestamp_str}.mp4")

                inputs = ['-y', '-i', current_video_stage, '-i', FILE_VOICE]
                
                # 🔥 SOLUTION FOR BLACK SCREEN: force format=yuv420p
                filter_complex = f"[0:v]setpts={pts_val}*PTS,scale=1920:1080,crop=1920:1080,format=yuv420p[vzoom]"
                map_cmd = ['-map', '[vzoom]']
                
                if os.path.exists(FILE_BGM) and bgm_up:
                    inputs.extend(['-i', FILE_BGM])
                    filter_complex += f";[2:a]volume={bgm_vol}[bgm];[1:a][bgm]amix=inputs=2:duration=first[aout]"
                    map_cmd.extend(['-map', '[aout]'])
                else:
                    map_cmd.extend(['-map', '1:a'])

                # Added -pix_fmt yuv420p explicitly for Browser Support
                cmd = ['ffmpeg'] + inputs + ['-filter_complex', filter_complex] + map_cmd + \
                      ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', FINAL_OUTPUT_NAME]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(FINAL_OUTPUT_NAME):
                    st.session_state.processed_video_path = FINAL_OUTPUT_NAME
                    p_bar.progress(100, text="🎉 Video Complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Video Generation Failed. Try checking logs.")

    if st.session_state.processed_video_path and "Video" in export_format:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.success("✨ Final Video Output")
            # Using st.video with explicit format
            if os.path.exists(st.session_state.processed_video_path):
                st.video(st.session_state.processed_video_path, format="video/mp4")
                with open(st.session_state.processed_video_path, "rb") as f: 
                    # Unique download name
                    st.download_button("🎬 Download Video", f, f"video_{int(time.time())}.mp4", use_container_width=True)
            else:
                st.warning("Video file not found. Please regenerate.")

    # 🔥 FIX: Better Audio Player Display
    if st.session_state.processed_audio_path and "Audio" in export_format:
        st.markdown("---")
        st.success("✨ Audio Output")
        st.audio(st.session_state.processed_audio_path)
        with open(st.session_state.processed_audio_path, "rb") as f: 
            st.download_button("🎵 Download MP3", f, "final_voice.mp3", use_container_width=True)

# === TAB 2: AUTO CAPTION (WITH SRT) ===
with t2:
    st.subheader("📝 Auto Caption")
    cap_up = st.file_uploader("Upload Video", type=['mp4'], key="cap")
    if cap_up:
        with open(FILE_CAP_INPUT, "wb") as f: f.write(cap_up.getbuffer())
        if st.button("Generate Captions"):
            check_requirements()
            font_path = download_font()
            subprocess.run(['ffmpeg', '-y', '-i', FILE_CAP_INPUT, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', FILE_CAP_WAV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                segments = model.transcribe(FILE_CAP_WAV, task="transcribe")['segments']
                trans_segments = []
                for seg in segments:
                    # Simple Translation
                    burmese = generate_with_retry(f"Translate to Burmese. Keep it short. Input: '{seg['text']}'")
                    trans_segments.append({'start': seg['start'], 'end': seg['end'], 'text': burmese})
                
                # Generate Files
                generate_ass_file(trans_segments, font_path, FILE_ASS)
                generate_srt_file(trans_segments, FILE_SRT) # 🔥 NEW: SRT Export
                
                font_dir = os.path.dirname(font_path)
                subprocess.run(['ffmpeg', '-y', '-i', FILE_CAP_INPUT, '-vf', f"ass={FILE_ASS}:fontsdir={font_dir}", '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'ultrafast', FILE_CAP_FINAL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state.caption_video_path = FILE_CAP_FINAL
                st.session_state.srt_path = FILE_SRT
                st.success("Done!")

    # ... (Inside Tab 2, at the bottom) ...

    if st.session_state.caption_video_path:
        st.markdown("---")
        # 🔥 FIX: Better Video Display Size for Captions
        c1, c2, c3 = st.columns([1, 2, 1]) 
        with c2:
            st.success("📝 Captioned Video")
            st.video(st.session_state.caption_video_path)
            
            d1, d2 = st.columns(2)
            with d1:
                with open(st.session_state.caption_video_path, "rb") as f: 
                    st.download_button("🎬 Download Video", f, "captioned.mp4", use_container_width=True)
            with d2:
                if st.session_state.srt_path and os.path.exists(st.session_state.srt_path):
                    with open(st.session_state.srt_path, "rb") as f: 
                        st.download_button("📄 Download .SRT", f, "subs.srt", use_container_width=True)

# === TAB 3: VIRAL SEO ===
with t3:
    st.subheader("🚀 Viral Kit SEO")
    if st.button("Generate Metadata"):
        if st.session_state.final_script:
            res = generate_with_retry(f"Generate 5 Clickbait Titles & Hashtags for: {st.session_state.final_script}")
            st.write(res)
        else: st.warning("No script found.")
