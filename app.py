import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import shutil
import whisper
from pydub import AudioSegment
import time
import json
import re
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI & CSS SETUP
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
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
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None
if 'seo_result' not in st.session_state: st.session_state.seo_result = ""

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

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (FIXED COMMAND FORMAT)
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
    "Japanese": {"Male": "ja-JP-KeitaNeural", "Female": "ja-JP-NanamiNeural"},
    "Chinese": {"Male": "zh-CN-YunxiNeural", "Female": "zh-CN-XiaoxiaoNeural"},
    "Thai": {"Male": "th-TH-NiwatNeural", "Female": "th-TH-PremwadeeNeural"},
    "Hindi": {"Male": "hi-IN-MadhurNeural", "Female": "hi-IN-SwaraNeural"}
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
    # 1. Validation
    if not text or not text.strip():
        return False, "Input text is empty (စာသားမရှိပါ)"
    
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        
        # 🔥 CRITICAL FIX: Use --flag=value format to handle negative numbers safely
        rate_arg = f"--rate={settings['rate']}"
        pitch_arg = f"--pitch={settings['pitch']}"
        
        cmd = [
            "edge-tts",
            "--voice", voice_id,
            "--text", text,
            rate_arg,   # Combined argument (Safer)
            pitch_arg,  # Combined argument (Safer)
            "--write-media", output_file
        ]
        
        # 2. Run with Error Capture
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"TTS Error: {result.stderr}"
            
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            return False, "File created but empty (0 bytes)."
            
        return True, "Success"
        
    except Exception as e:
        return False, f"System Error: {str(e)}"

# ---------------------------------------------------------
# 🧠 AI ENGINE
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name, safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ])

def transcribe_video(video_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e: return f"Error: {e}"

def translate_to_burmese_draft(model, text, source_lang):
    prompt = f"""
    Translate {source_lang} to Burmese.
    Input: "{text}"
    Rules: Keep Proper Nouns in English. Translate accurately.
    """
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def refine_script_hvc(model, text, title, custom_prompt):
    prompt = f"""
    Refine this Burmese draft for video '{title}'.
    Input: "{text}"
    Structure: H-V-C (Hook, Value, Call).
    Constraint: Keep content length roughly same as draft. Do not summarize too much.
    Output: Only Burmese spoken text.
    Extra: {custom_prompt}
    """
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def generate_viral_metadata(model, title, keywords, lang):
    prompt = f"""
    Write SEO Description for '{title}'.
    Language: {lang}
    Keywords: {keywords}
    Include: Hook, Bullet points, Timestamps, Tags.
    """
    try: return model.generate_content(prompt).text
    except: return "AI Error"

# ---------------------------------------------------------
# ❄️ FREEZE & VIDEO ENGINE
# ---------------------------------------------------------
def apply_auto_freeze(input_video, output_video, interval_sec, freeze_duration=4.0):
    try:
        duration = get_duration(input_video)
        if duration == 0: return False
        
        concat_list = "freeze_list.txt"
        with open(concat_list, "w") as f:
            curr = 0
            idx = 0
            while curr < duration:
                nxt = min(curr + interval_sec, duration)
                seg_dur = nxt - curr
                p_name = f"p_{idx}.mp4"
                subprocess.run(['ffmpeg', '-y', '-ss', str(curr), '-t', str(seg_dur), '-i', input_video, '-c', 'copy', p_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{p_name}'\n")
                
                if nxt < duration:
                    f_name = f"f_{idx}.mp4"
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', p_name, '-update', '1', '-q:v', '1', 'frame.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'frame.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f.write(f"file '{f_name}'\n")
                curr = nxt
                idx += 1
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

def process_freeze_command(command, input_video, output_video):
    try:
        match = re.search(r'freeze\s*[:=]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command, re.IGNORECASE)
        if match:
            t_pt = float(match.group(1))
            dur = float(match.group(2))
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
# 🖥️ MAIN UI
# ---------------------------------------------------------
st.markdown("""<div class="main-header"><h1>🎬 Myanmar AI Studio Pro</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 Google API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    model_name = st.text_input("Model ID", value="gemini-2.5-flash")
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2 = st.tabs(["🎬 Production", "🚀 Viral SEO"])

with t1:
    st.subheader("Step 1: Upload & Translate")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'])
    source_lang = st.selectbox("Original Language
