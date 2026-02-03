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
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FF8C00 100%);
        color: black; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 12px; font-size: 16px;
    }
    textarea, input { 
        background-color: #1a1a1a !important; color: #fff !important; 
        border: 1px solid #333 !important; border-radius: 8px !important;
        font-family: 'Padauk', sans-serif !important;
    }
    .viral-box { background: #0f0f0f; padding: 20px; border-left: 5px solid #00ff88; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'raw_transcript' not in st.session_state: st.session_state.raw_transcript = ""
if 'burmese_draft' not in st.session_state: st.session_state.burmese_draft = ""
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
if 'api_keys' not in st.session_state: st.session_state.api_keys = []

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS (MISSING FUNCTIONS RESTORED HERE)
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please add 'ffmpeg' to packages.txt")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

def download_font():
    font_filename = "Padauk-Bold.ttf"
    if not os.path.exists(font_filename):
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_filename, 'wb') as f: f.write(r.content)
        except: pass
    return os.path.abspath(font_filename)

def load_whisper_safe():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Whisper Load Error: {e}")
        return None

# 🔥 RESTORED: Auto Freeze Function
def apply_auto_freeze(input_video, output_video, interval_sec, freeze_duration=4.0):
    try:
        duration = get_duration(input_video)
        if duration == 0: return False
        
        concat_list = "freeze_list.txt"
        with open(concat_list, "w") as f:
            curr = 0
            idx = 0
            while curr < duration:
                # Normal video segment
                nxt = min(curr + interval_sec, duration)
                seg_dur = nxt - curr
                p_name = f"p_{idx}.mp4"
                subprocess.run(['ffmpeg', '-y', '-ss', str(curr), '-t', str(seg_dur), '-i', input_video, '-c', 'copy', p_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{p_name}'\n")
                
                # Freeze segment (if not end of video)
                if nxt < duration:
                    f_name = f"f_{idx}.mp4"
                    # Capture last frame
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', p_name, '-update', '1', '-q:v', '1', 'frame.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Loop frame for freeze_duration (4s default)
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'frame.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f.write(f"file '{f_name}'\n")
                curr = nxt
                idx += 1
        
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

# 🔥 RESTORED: Manual Freeze Command
def process_freeze_command(command, input_video, output_video):
    try:
        match = re.search(r'freeze\s*[:=]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command, re.IGNORECASE)
        if match:
            t_pt = float(match.group(1))
            dur = float(match.group(2))
            # Split video: Part A + Freeze + Part B
            subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(t_pt), '-c', 'copy', 'a.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(t_pt), '-i', input_video, '-vframes', '1', 'f.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'f.jpg', '-t', str(dur), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'fr.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(t_pt), '-i', input_video, '-c', 'copy', 'b.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open("list.txt", "w") as f: f.write("file 'a.mp4'\nfile 'fr.mp4'\nfile 'b.mp4'")
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        return False
    except: return False

# ---------------------------------------------------------
# 🔢 NUMBER NORMALIZATION (23000 -> နှစ်သောင်းသုံးထောင်)
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    try:
        num_str = num_str.replace(",", "")
        n = int(num_str)
        if n == 0: return "သုည"
        
        digit_map = ["", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"]
        
        def convert_chunk(number):
            parts = []
            if number >= 10000000:
                chunk = number // 10000000
                parts.append(convert_chunk(chunk) + "ကုဋေ")
                number %= 10000000
            if number >= 1000000:
                chunk = number // 1000000
                parts.append(digit_map[chunk] + "သန်း")
                number %= 1000000
            if number >= 100000:
                chunk = number // 100000
                parts.append(digit_map[chunk] + "သိန်း")
                number %= 100000
            if number >= 10000:
                chunk = number // 10000
                parts.append(digit_map[chunk] + "သောင်း")
                number %= 10000
            if number >= 1000:
                chunk = number // 1000
                parts.append(digit_map[chunk] + "ထောင်")
                number %= 1000
            if number >= 100:
                chunk = number // 100
                parts.append(digit_map[chunk] + "ရာ")
                number %= 100
            if number >= 10:
                chunk = number // 10
                parts.append(digit_map[chunk] + "ဆယ်")
                number %= 10
            if number > 0:
                parts.append(digit_map[number])
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
    text = text.replace("*", "").replace("#", "").replace("- ", "")
    text = re.sub(r'\b\d+\b', lambda x: num_to_burmese_spoken(x.group()), text)
    text = text.replace("\n", " ") # Prevent pausing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE GENERATOR (CAPCUT STYLE)
# ---------------------------------------------------------
def generate_ass_file(segments, font_path):
    filename = "captions.ass"
    def seconds_to_ass(seconds):
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60); cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCut,Padauk-Bold,24,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            raw_text = seg['text'].strip()
            wrapped_lines = textwrap.wrap(raw_text, width=40)
            final_text = "\\N".join(wrapped_lines) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return filename

# ---------------------------------------------------------
# 🧠 AI ENGINE (API ROTATION)
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_with_retry(prompt):
    keys = st.session_state.api_keys
    model_name = st.session_state.get("selected_model", "gemini-1.5-flash")
    errors = []
    for i, key in enumerate(keys):
        try:
            model = get_model(key, model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            errors.append(f"Key {i+1}: {str(e)}")
            continue
    return f"AI Error: All keys failed. {errors}"

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (VOICE SELECTION)
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-5%", "pitch": "-5Hz"},
    "Documentary": {"rate": "-2%", "pitch": "-8Hz"},
    "Recap": {"rate": "+5%", "pitch": "+0Hz"},
    "Motivation": {"rate": "-8%", "pitch": "-12Hz"},
    "Animation": {"rate": "+10%", "pitch": "+15Hz"}
}

def generate_audio_cli(text, lang, gender, mode_name, output_file):
    if not text.strip(): return False, "Empty text"
    processed_text = normalize_text_for_tts(text)
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        cmd = ["edge-tts", "--voice", voice_id, "--text", processed_text, f"--rate={settings['rate']}", f"--pitch={settings['pitch']}", "--write-media", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

# ---------------------------------------------------------
# 🖥️ MAIN UI & SIDEBAR
# ---------------------------------------------------------
st.markdown("""<div class="main-header"><h1>🎬 Myanmar AI Studio Pro</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    try:
        if "GOOGLE_API_KEYS" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEYS"]
        elif "GOOGLE_API_KEY" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEY"]
        else: default_keys = ""
    except: default_keys = ""
    
    api_key_input = st.text_area("🔑 API Keys (Comma separated)", value=default_keys, height=100)
    
    if api_key_input:
        key_list = [k.strip() for k in api_key_input.split(",") if k.strip()]
        st.session_state.api_keys = key_list
        st.success(f"Active: {len(key_list)} Keys")
    else:
        st.session_state.api_keys = []
    
    st.divider()
    if "selected_model" not in st.session_state: st.session_state.selected_model = "gemini-1.5-flash"
    st.session_state.selected_model = st.selectbox("🤖 AI Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    
    if st.button("🔴 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_keys: st.warning("⚠️ Enter Keys"); st.stop()

t1, t2 = st.tabs(["🎙️ Dubbing", "📝 Auto Caption"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    st.subheader("Dubbing Studio")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    source_lang = st.selectbox("Original Lang", ["English", "Japanese", "Chinese", "Thai"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        if st.button("📝 Extract & Translate"):
            check_requirements()
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                raw = model.transcribe("temp.wav")['text']
                st.session_state.raw_transcript = raw
                prompt = f"Translate {source_lang} to Burmese. Input: '{raw}'. Rules: Keep Proper Nouns in English."
                st.session_state.final_script = generate_with_retry(prompt)
                st.rerun()

    if st.session_state.final_script:
        st.subheader("Script & Production")
        if st.button("✨ Refine with H.V.C (Host Voice Only)"):
            with st.spinner("Refining..."):
                prompt = f"Refine into H-V-C. Output ONLY host spoken words. Input: {st.session_state.final_script}"
                st.session_state.final_script = generate_with_retry(prompt)
                st.rerun()

        txt = st.text_area("Script", st.session_state.final_script, height=200)
        
        # 🔥 VOICE SELECTION ADDED
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        # 🔥 ZOOM SLIDER ADDED
        zoom_val = st.slider("🔍 Video Zoom (Avoid Copyright)", 1.0, 1.2, 1.05, 0.01)
        
        # 🔥 FREEZE SETTINGS ADDED
        c1, c2 = st.columns(2)
        with c1:
            ft1, ft2 = st.tabs(["Auto Freeze", "Manual Command"])
            auto_freeze = None; manual_freeze = None
            with ft1:
                # User asked for 30s, 1m checkboxes
                if st.checkbox("Every 30s (Freeze 4s)"): auto_freeze = 30
                if st.checkbox("Every 1min (Freeze 4s)"): auto_freeze = 60
            with ft2: manual_freeze = st.text_input("Command", placeholder="freeze 10,5")
        
        if st.button("🚀 Render Dubbed Video"):
            with st.spinner("Rendering..."):
                # 1. TTS with Mode
                generate_audio_cli(txt, target_lang, gender, v_mode, "voice.mp3")
                
                # 2. Freeze Logic
                input_vid = "input.mp4"
                if auto_freeze: apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze); input_vid = "frozen.mp4"
                elif manual_freeze: process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"); input_vid = "frozen.mp4"
                
                # 3. Zoom & Merge
                w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                if w_s % 2 != 0: w_s += 1
                if h_s % 2 != 0: h_s += 1
                
                # Complex filter: Zoom -> Crop -> Merge Audio
                subprocess.run([
                    'ffmpeg', '-y', '-i', input_vid, '-i', "voice.mp3",
                    '-filter_complex', f"[0:v]scale={w_s}:{h_s},crop=1920:1080[vzoom]",
                    '-map', '[vzoom]', '-map', '1:a',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', "dubbed_final.mp4"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                st.session_state.processed_video_path = "dubbed_final.mp4"
                st.success("Dubbing Complete!")

    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f: st.download_button("Download", f, "dubbed.mp4")

# === TAB 2: AUTO CAPTION ===
with t2:
    st.subheader("📝 CapCut Style Captions")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        if st.button("Generate Captions"):
            check_requirements(); font_path = download_font()
            with st.spinner("Processing..."):
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                model = load_whisper_safe()
                if model:
                    segments = model.transcribe("cap.wav", task="transcribe")['segments']
                    trans_segments = []
                    for seg in segments:
                        txt = seg['text'].strip()
                        if txt:
                            # Rotation & Translation
                            burmese = generate_with_retry(f"Translate to Burmese. Short. Input: '{txt}'")
                            trans_segments.append({'start': seg['start'], 'end': seg['end'], 'text': burmese})
                            time.sleep(0.3)
                    
                    ass_file = generate_ass_file(trans_segments, font_path)
                    font_dir = os.path.dirname(font_path)
                    subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vf', f"ass={ass_file}:fontsdir={font_dir}", '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', "captioned_final.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    st.session_state.caption_video_path = "captioned_final.mp4"
                    st.success("Done!")

    if st.session_state.caption_video_path:
        st.video(st.session_state.caption_video_path)
