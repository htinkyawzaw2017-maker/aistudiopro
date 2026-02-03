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
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'raw_transcript' not in st.session_state: st.session_state.raw_transcript = ""
if 'burmese_draft' not in st.session_state: st.session_state.burmese_draft = ""
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None
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
    except: return 0.0

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

def apply_auto_freeze(input_video, output_video, interval_sec, freeze_duration=4.0):
    try:
        duration = get_duration(input_video)
        if duration == 0: return False
        concat_list = "freeze_list.txt"
        with open(concat_list, "w") as f:
            curr = 0; idx = 0
            while curr < duration:
                nxt = min(curr + interval_sec, duration)
                p_name = f"p_{idx}.mp4"
                subprocess.run(['ffmpeg', '-y', '-ss', str(curr), '-t', str(nxt-curr), '-i', input_video, '-c', 'copy', p_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{p_name}'\n")
                if nxt < duration:
                    f_name = f"f_{idx}.mp4"
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', p_name, '-update', '1', '-q:v', '1', 'frame.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'frame.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f.write(f"file '{f_name}'\n")
                curr = nxt; idx += 1
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

def process_freeze_command(command, input_video, output_video):
    try:
        match = re.search(r'freeze\s*[:=]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command, re.IGNORECASE)
        if match:
            t_pt = float(match.group(1)); dur = float(match.group(2))
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
# 🔢 NUMBER NORMALIZATION
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    try:
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
    # 1. Cleaning
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    # 2. Number Conversion
    text = re.sub(r'\b\d+\b', lambda x: num_to_burmese_spoken(x.group()), text)
    # 3. Flow Fix (No Newlines, No Ellipsis)
    text = text.replace("\n", " ")
    text = text.replace("...", " ")
    text = text.replace("၊", " ") 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 🧠 AI ENGINE (API ROTATION)
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_with_retry(prompt):
    keys = st.session_state.api_keys
    model_name = st.session_state.get("selected_model", "gemini-1.5-flash")
    for i, key in enumerate(keys):
        try:
            model = get_model(key, model_name)
            response = model.generate_content(prompt)
            return response.text
        except: continue
    return "AI Error: All keys failed."

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (SPEED CONTROL & STYLES)
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-5%", "pitch": "-2Hz"}, 
    "Recap": {"rate": "+5%", "pitch": "+0Hz"},
    "Horror/Thriller": {"rate": "-10%", "pitch": "-5Hz"},
}

def generate_audio_cli(text, lang, gender, mode_name, output_file, speed_multiplier=1.0):
    if not text.strip(): return False, "Empty text"
    
    # Text Normalization
    processed_text = normalize_text_for_tts(text)
    
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        
        # 🔥 Calculate Rate based on Mode + Slider
        # Parse base rate (e.g., "-5%")
        base_rate_str = settings['rate'].replace('%', '')
        base_rate = int(base_rate_str)
        
        # Add slider effect (1.0 = 0%, 1.2 = +20%)
        slider_rate = int((speed_multiplier - 1.0) * 100)
        
        final_rate_val = base_rate + slider_rate
        final_rate_str = f"{final_rate_val:+d}%"
        
        cmd = ["edge-tts", "--voice", voice_id, "--text", processed_text, f"--rate={final_rate_str}", f"--pitch={settings['pitch']}", "--write-media", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE
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
# 🖥️ MAIN UI
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
    else: st.session_state.api_keys = []
    
    st.divider()
    if "selected_model" not in st.session_state: st.session_state.selected_model = "gemini-1.5-flash"
    st.session_state.selected_model = st.selectbox("🤖 AI Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    if st.button("🔴 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_keys: st.warning("⚠️ Enter API Keys"); st.stop()

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
                # Rotation AI
                prompt = f"Translate {source_lang} to Burmese. Input: '{raw}'. Rules: Keep Proper Nouns in English."
                st.session_state.final_script = generate_with_retry(prompt)
                st.rerun()

    if st.session_state.final_script:
        st.subheader("Script & Production")
        
        # 🔥 UPDATED PROMPT: NO 'ဗျ/ရှင့်/သူ၏', STORYTELLING STYLE
        if st.button("✨ Refine: Movie Recap Style (No 'ဗျ/ရှင့်')"):
            with st.spinner("Refining..."):
                prompt = f"""
                Act as a Myanmar Movie Recap Narrator. Rewrite this text.
                Input: "{st.session_state.final_script}"
                
                **STRICT RULES:**
                1. Style: Engaging, Storytelling, Flowing (ဇာတ်လမ်းပြောပုံစံ).
                2. **FORBIDDEN:** Do NOT use 'ဗျ', 'ရှင့်', 'ခင်ဗျာ', 'သူ၏', '၎င်း', 'သည်', '၍'.
                3. **ALLOWED:** Use 'သူမ' (for female), 'သူ' (for male/general). End sentences with 'တယ်', 'မယ်', 'ပါ'.
                4. Flow: Continuous, no robotic pauses.
                """
                st.session_state.final_script = generate_with_retry(prompt)
                st.rerun()

        txt = st.text_area("Script", st.session_state.final_script, height=200)
        
        # 🔥 VOICE SETTINGS
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        # 🔥 AUDIO SPEED SLIDER
        audio_speed = st.slider("🔊 Audio Speed (Multiplier)", 0.8, 1.5, 1.0, 0.05, help="1.0 is Normal. 1.2 is 20% Faster.")
        
        # 🔥 ZOOM SLIDER
        zoom_val = st.slider("🔍 Video Zoom (Avoid Copyright)", 1.0, 1.2, 1.0, 0.01)
        
        # 🔥 FREEZE
        c1, c2 = st.columns(2)
        with c1:
            ft1, ft2 = st.tabs(["Auto Freeze", "Manual"])
            auto_freeze = None; manual_freeze = None
            with ft1:
                if st.checkbox("Every 30s (Freeze 4s)"): auto_freeze = 30
                if st.checkbox("Every 1min (Freeze 4s)"): auto_freeze = 60
            with ft2: manual_freeze = st.text_input("Command", placeholder="freeze 10,5")
        
        if st.button("🚀 Render Dubbed Video"):
            with st.spinner("Rendering..."):
                # 1. TTS with Speed Slider
                generate_audio_cli(txt, target_lang, gender, v_mode, "voice.mp3", speed_multiplier=audio_speed)
                st.session_state.processed_audio_path = "voice.mp3" # Save for download
                
                # 2. Freeze Logic
                input_vid = "input.mp4"
                if auto_freeze: apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze); input_vid = "frozen.mp4"
                elif manual_freeze: process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"); input_vid = "frozen.mp4"
                
                # 3. Zoom & Merge
                w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                if w_s % 2 != 0: w_s += 1
                if h_s % 2 != 0: h_s += 1
                
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
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            with open(st.session_state.processed_video_path, "rb") as f: st.download_button("🎬 Download Video", f, "dubbed.mp4")
        with c_d2:
            if st.session_state.processed_audio_path and os.path.exists(st.session_state.processed_audio_path):
                with open(st.session_state.processed_audio_path, "rb") as f: st.download_button("🎵 Download Audio", f, "voice.mp3")

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
        with open(st.session_state.caption_video_path, "rb") as f: st.download_button("Download", f, "captioned.mp4")
