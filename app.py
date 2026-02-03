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
# 🛠️ HELPER FUNCTIONS
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
    """Safely loads Whisper model"""
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Whisper Load Error: {e}")
        return None

# ---------------------------------------------------------
# 🔢 NUMBER NORMALIZATION (FIXED LOGIC)
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    """
    Converts numbers to Burmese spoken words accurately.
    23000 -> နှစ်သောင်းသုံးထောင်
    600000 -> ခြောက်သိန်း
    """
    try:
        num_str = num_str.replace(",", "")
        n = int(num_str)
        if n == 0: return "သုည"
        
        digit_map = ["", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"]
        
        def convert_chunk(number):
            parts = []
            # 1. ကုဋေ (10,000,000+)
            if number >= 10000000:
                chunk = number // 10000000
                parts.append(convert_chunk(chunk) + "ကုဋေ")
                number %= 10000000
            
            # 2. သန်း (1,000,000+)
            if number >= 1000000:
                chunk = number // 1000000
                parts.append(digit_map[chunk] + "သန်း")
                number %= 1000000
            
            # 3. သိန်း (100,000+) - Fix for 600,000 -> 6 Thein
            if number >= 100000:
                chunk = number // 100000
                parts.append(digit_map[chunk] + "သိန်း")
                number %= 100000
                
            # 4. သောင်း (10,000+) - Fix for 23,000 -> 2 Thaung 3 Htaung
            if number >= 10000:
                chunk = number // 10000
                parts.append(digit_map[chunk] + "သောင်း")
                number %= 10000
                
            # 5. ထောင် (1,000+)
            if number >= 1000:
                chunk = number // 1000
                parts.append(digit_map[chunk] + "ထောင်")
                number %= 1000
                
            # 6. ရာ (100+)
            if number >= 100:
                chunk = number // 100
                parts.append(digit_map[chunk] + "ရာ")
                number %= 100
                
            # 7. ဆယ် (10+)
            if number >= 10:
                chunk = number // 10
                parts.append(digit_map[chunk] + "ဆယ်")
                number %= 10
                
            # 8. Unit
            if number > 0:
                parts.append(digit_map[number])
                
            return "".join(parts)

        result = convert_chunk(n)
        
        # Tone adjustments
        result = result.replace("ထောင်", "ထောင့်").replace("ရာ", "ရာ့").replace("ဆယ်", "ဆယ့်")
        
        # Fix ending tones
        if result.endswith("ထောင့်"): result = result[:-1] + "င်"
        if result.endswith("ရာ့"): result = result[:-1]
        if result.endswith("ဆယ့်"): result = result[:-1]
        
        return result
    except: return num_str

def normalize_text_for_tts(text):
    if not text: return ""
    # 1. Clean Markdown
    text = text.replace("*", "").replace("#", "").replace("- ", "")
    # 2. Convert Numbers
    text = re.sub(r'\b\d+\b', lambda x: num_to_burmese_spoken(x.group()), text)
    # 3. FIX PAUSING: Replace newlines with space to prevent TTS stutter
    text = text.replace("\n", " ")
    # 4. Remove extra spaces
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

    # CapCut Style: Yellow Text, Black Box, Bottom Center
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
            # Wrap text to prevent full screen cover
            wrapped_lines = textwrap.wrap(raw_text, width=40)
            final_text = "\\N".join(wrapped_lines) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return filename

# ---------------------------------------------------------
# 🧠 AI ENGINE (API ROTATION SYSTEM)
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_with_retry(prompt):
    """
    Auto-switches API keys if one fails (Quota Limit).
    """
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
            continue # Try next key
    
    return f"AI Error: All keys failed. {errors}"

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
VOICE_MODES = {"Normal": {"rate": "+0%", "pitch": "+0Hz"}}

def generate_audio_cli(text, lang, gender, mode_name, output_file):
    if not text.strip(): return False, "Empty text"
    
    # 🔥 Fix: Normalize numbers & Remove newlines before TTS
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
    
    # Load Multiple Keys from Secrets
    try:
        default_keys = st.secrets.get("GOOGLE_API_KEYS", "")
    except:
        default_keys = ""
    
    api_key_input = st.text_area("🔑 API Keys (Comma separated)", value=default_keys, height=100)
    
    if api_key_input:
        key_list = [k.strip() for k in api_key_input.split(",") if k.strip()]
        st.session_state.api_keys = key_list
        st.success(f"Active: {len(key_list)} Keys")
    else:
        st.session_state.api_keys = []
    
    st.divider()
    st.session_state.selected_model = st.selectbox("🤖 AI Model", ["gemini-2.5-flash", "gemini-2.5-flashlite-exp"])
    
    if st.button("🔴 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_keys: 
    st.warning("⚠️ Please enter API Keys.")
    st.stop()

t1, t2 = st.tabs(["🎙️ Dubbing", "📝 Auto Caption (CapCut Style)"])

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
                
                # API Rotation Used Here
                prompt = f"Translate {source_lang} to Burmese. Input: '{raw}'. Rules: Keep Proper Nouns in English."
                draft = generate_with_retry(prompt)
                st.session_state.final_script = draft
                st.rerun()

    if st.session_state.final_script:
        st.subheader("Script & Production")
        
        # 🔥 H.V.C Fixed Prompt
        if st.button("✨ Refine with H.V.C (Host Voice Only)"):
            with st.spinner("Refining..."):
                prompt = f"""
                Act as a Video Scriptwriter. Refine this Burmese text into H-V-C structure.
                Input: "{st.session_state.final_script}"
                
                **STRICT RULES:**
                1. Output ONLY the spoken words for the host.
                2. NO labels like "Hook:", "Intro:", "Body:".
                3. Keep technical English words as English.
                4. Maintain natural Burmese spoken flow.
                """
                refined = generate_with_retry(prompt)
                st.session_state.final_script = refined
                st.rerun()

        txt = st.text_area("Script", st.session_state.final_script, height=200)
        
        c1, c2 = st.columns(2)
        with c1:
            ft1, ft2 = st.tabs(["Auto Freeze", "Manual Command"])
            auto_freeze = None; manual_freeze = None
            with ft1:
                if st.checkbox("Every 30s"): auto_freeze = 30
            with ft2: manual_freeze = st.text_input("Command", placeholder="freeze 10,5")
        
        if st.button("🚀 Render Dubbed Video"):
            with st.spinner("Rendering..."):
                # Audio Generation with Number Fix & No Pause Fix
                generate_audio_cli(txt, "Burmese", "Female", "Normal", "voice.mp3")
                
                input_vid = "input.mp4"
                if auto_freeze: apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze); input_vid = "frozen.mp4"
                elif manual_freeze: process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"); input_vid = "frozen.mp4"
                
                subprocess.run(['ffmpeg', '-y', '-i', input_vid, '-i', "voice.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', "dubbed_final.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state.processed_video_path = "dubbed_final.mp4"
                st.success("Dubbing Complete!")

    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)

# === TAB 2: AUTO CAPTION (CAPCUT STYLE FIXED) ===
with t2:
    st.subheader("📝 CapCut Style Captions")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    
    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        
        if st.button("Generate Captions"):
            check_requirements()
            font_path = download_font()
            
            with st.spinner("1. Transcribing (Whisper)..."):
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                model = load_whisper_safe()
                if model:
                    segments = model.transcribe("cap.wav", task="transcribe")['segments']
            
                    with st.spinner("2. Translating (Rotation AI)..."):
                        trans_segments = []
                        for seg in segments:
                            text = seg['text'].strip()
                            if not text: continue
                            prompt = f"Translate to Burmese. Keep it short (max 10 words). Input: '{text}'"
                            burmese_text = generate_with_retry(prompt)
                            trans_segments.append({'start': seg['start'], 'end': seg['end'], 'text': burmese_text})
                            time.sleep(0.3)
                    
                    with st.spinner("3. Applying CapCut Style (.ASS)..."):
                        ass_file = generate_ass_file(trans_segments, font_path)
                        font_dir = os.path.dirname(font_path)
                        subprocess.run([
                            'ffmpeg', '-y', '-i', "cap_input.mp4",
                            '-vf', f"ass={ass_file}:fontsdir={font_dir}",
                            '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', 
                            "captioned_final.mp4"
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        st.session_state.caption_video_path = "captioned_final.mp4"
                        st.success("✅ Success!")

    if st.session_state.caption_video_path:
        st.video(st.session_state.caption_video_path)
        with open(st.session_state.caption_video_path, "rb") as f:
            st.download_button("Download Video", f, "capcut_style.mp4")
