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
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None

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
        # Use Bold font for better visibility
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_filename, 'wb') as f: f.write(r.content)
        except: pass
    return os.path.abspath(font_filename)

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE GENERATOR (CAPCUT STYLE ENGINE)
# ---------------------------------------------------------
def generate_ass_file(segments, font_path):
    """
    Generates an Advanced Substation Alpha (.ass) file.
    This allows precise control over Yellow Text and Black Box Background.
    """
    filename = "captions.ass"
    font_dir = os.path.dirname(font_path)
    font_name = "Padauk" # Family name

    # convert time to ASS format (H:MM:SS.cs)
    def seconds_to_ass(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCut,Padauk-Bold,60,&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Color Codes in ASS are BGR (Blue Green Red). 
    # Yellow = &H0000FFFF (00 Blue, FF Green, FF Red)
    # BackColour = &H80000000 (80 is alpha/transparency, 000000 is black)
    # BorderStyle = 3 (Opaque Box)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            text = seg['text'].strip()
            # Force line wrap if too long
            text = "\\N".join(textwrap.wrap(text, width=30)) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{text}\n")
            
    return filename

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
VOICE_MODES = {"Normal": {"rate": "+0%", "pitch": "+0Hz"}, "Story": {"rate": "-10%", "pitch": "-5Hz"}}

def generate_audio_cli(text, lang, gender, mode_name, output_file):
    if not text.strip(): return False, "Empty text"
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        cmd = ["edge-tts", "--voice", voice_id, "--text", text, f"--rate={settings['rate']}", f"--pitch={settings['pitch']}", "--write-media", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

# ---------------------------------------------------------
# 🧠 AI ENGINE
# ---------------------------------------------------------
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

def transcribe_for_captions(video_path):
    model = whisper.load_model("base") # Use small or base for speed
    result = model.transcribe(video_path, task="transcribe")
    return result['segments']

def translate_captions(model, segments):
    translated = []
    for seg in segments:
        text = seg['text'].strip()
        if not text: continue
        try:
            # Enforce shortness
            prompt = f"Translate to Burmese. Keep it very short (max 8 words). Input: '{text}'"
            res = model.generate_content(prompt)
            burmese_text = res.text.strip()
        except: burmese_text = text
        
        translated.append({'start': seg['start'], 'end': seg['end'], 'text': burmese_text})
        time.sleep(0.2) # Rate limit
    return translated

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
st.markdown("""<div class="main-header"><h1>🎬 Myanmar AI Studio Pro</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    api_key = st.text_input("🔑 Google API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2 = st.tabs(["🎙️ Dubbing", "📝 Auto Caption (CapCut Style)"])

# === TAB 1: DUBBING (Simplified for brevity, fully functional) ===
with t1:
    st.subheader("Dubbing Studio")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        if st.button("Start Dubbing Workflow"):
            check_requirements()
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raw = transcribe_video("temp.wav")
            model = get_model(st.session_state.api_key)
            # Direct translation for now
            draft = model.generate_content(f"Translate to Burmese: {raw}").text
            st.session_state.final_script = draft
            st.rerun()
            
    if st.session_state.final_script:
        txt = st.text_area("Script", st.session_state.final_script)
        if st.button("Render Video"):
            generate_audio_cli(txt, "Burmese", "Female", "Normal", "voice.mp3")
            # Simple Merge
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-i', "voice.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-shortest', "final.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.session_state.processed_video_path = "final.mp4"
            st.success("Done")
            
    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)

# === TAB 2: AUTO CAPTION (FIXED) ===
with t2:
    st.subheader("📝 CapCut Style Captions (Black Box + Yellow Text)")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    
    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        
        if st.button("Generate Captions"):
            check_requirements()
            font_path = download_font() # Ensure font exists
            
            with st.spinner("1. Transcribing (Whisper)..."):
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                segments = transcribe_for_captions("cap.wav")
            
            with st.spinner("2. Translating (Gemini)..."):
                model = get_model(st.session_state.api_key)
                trans_segments = translate_captions(model, segments)
            
            with st.spinner("3. Applying CapCut Style..."):
                # Generate .ass file
                ass_file = generate_ass_file(trans_segments, font_path)
                
                # Burn subtitles using ASS filter
                # We map fonts directory to current folder so ffmpeg finds Padauk-Bold.ttf
                # font_path variable is absolute, so we can use that directory.
                font_dir = os.path.dirname(font_path)
                
                subprocess.run([
                    'ffmpeg', '-y', '-i', "cap_input.mp4",
                    '-vf', f"ass={ass_file}:fontsdir={font_dir}",
                    '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', 
                    "captioned_final.mp4"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                st.session_state.caption_video_path = "captioned_final.mp4"
                st.success("✅ Success! Check the styling below.")

    if st.session_state.caption_video_path:
        st.video(st.session_state.caption_video_path)
        with open(st.session_state.caption_video_path, "rb") as f:
            st.download_button("Download Video", f, "capcut_style.mp4")
