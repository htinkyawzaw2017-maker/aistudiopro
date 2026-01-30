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
# 🎨 UI & CSS SETUP (MODERN DASHBOARD DESIGN)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp { background-color: #050505; color: #e0e0e0; }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; font-weight: 700; }
    .main-header p { color: #f0f0f0; margin: 5px 0 0; font-size: 1.1rem; }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 12px; font-size: 16px;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); }

    /* Input Fields */
    textarea, input { 
        background-color: #1a1a1a !important; 
        color: #fff !important; 
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: 'Padauk', sans-serif !important;
    }

    /* Cards/Containers */
    .css-1r6slb0 { background-color: #111; border: 1px solid #333; border-radius: 10px; padding: 20px; }
    
    /* Viral Box */
    .viral-box { background: #0f0f0f; padding: 20px; border-left: 5px solid #00ff88; border-radius: 5px; margin-top: 15px; }
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
        st.error("❌ FFmpeg is missing. System cannot process video.")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE
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
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        cmd = ["edge-tts", "--voice", voice_id, "--text", text, "--rate", settings["rate"], "--pitch", settings["pitch"], "--write-media", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 100
    except: return False

# ---------------------------------------------------------
# 🧠 AI ENGINE (PROMPT ENGINEERED)
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
    except: return "Error during transcription"

def translate_to_burmese_draft(model, text, source_lang):
    prompt = f"""
    Translate the following {source_lang} transcript to Burmese.
    Input: "{text}"
    Rules:
    1. Translate sentence by sentence.
    2. DO NOT Summarize. Keep all details.
    3. Keep Proper Nouns (Names, Places) in English.
    """
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def refine_script_hvc(model, text, title, custom_prompt):
    prompt = f"""
    Act as a Professional Video Scriptwriter.
    Refine the following Burmese draft into a final script for a video titled '{title}'.
    
    Input Draft: "{text}"
    
    **CRITICAL INSTRUCTION FOR LENGTH:** - The input text corresponds to a specific video length. 
    - **DO NOT SUMMARIZE OR SHORTEN.** - You must rewrite it to be smoother and more engaging using the H-V-C structure, BUT keep the content length roughly the same as the draft so it matches the video timing.
    
    Structure (H-V-C):
    1. Hook: Make the opening strong.
    2. Value: Deliver the story/content clearly.
    3. Call: End with a CTA.
    
    **OUTPUT FORMAT:**
    - Output ONLY the Burmese spoken text (Host Voice). 
    - No labels like "Hook:", "Intro:".
    - Keep English names in English.
    
    Extra Style: {custom_prompt}
    """
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def generate_viral_metadata(model, title, keywords, output_lang):
    prompt = f"""
    Write a video description for '{title}' optimized for SEO.
    Target Language: {output_lang}
    Keywords: {keywords}
    
    Include:
    1. Engaging Hook Paragraph.
    2. Bullet points of what viewers learn.
    3. 5 Timestamps.
    4. 15 High-traffic Tags.
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
        concat_list_path = "freeze_list.txt"
        with open(concat_list_path, "w") as f:
            current_time = 0
            part_idx = 0
            while current_time < duration:
                next_time = min(current_time + interval_sec, duration)
                seg_duration = next_time - current_time
                part_name = f"part_{part_idx}.mp4"
                subprocess.run(['ffmpeg', '-y', '-ss', str(current_time), '-t', str(seg_duration), '-i', input_video, '-c', 'copy', part_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{part_name}'\n")
                if next_time < duration:
                    freeze_name = f"freeze_{part_idx}.mp4"
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', part_name, '-update', '1', '-q:v', '1', 'freeze_frame.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'freeze_frame.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', freeze_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f.write(f"file '{freeze_name}'\n")
                current_time = next_time
                part_idx += 1
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

def process_freeze_command(command, input_video, output_video):
    try:
        match = re.search(r'freeze\s*[:=]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command, re.IGNORECASE)
        if match:
            time_point = float(match.group(1))
            duration = float(match.group(2))
            subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(time_point), '-c', 'copy', 'part_a.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-vframes', '1', 'freeze.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'freeze.jpg', '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'part_freeze.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-c', 'copy', 'part_b.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open("list.txt", "w") as f:
                f.write("file 'part_a.mp4'\nfile 'part_freeze.mp4'\nfile 'part_b.mp4'")
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        return False
    except: return False

# ---------------------------------------------------------
# 🖥️ MAIN UI LAYOUT
# ---------------------------------------------------------
# HERO HEADER
st.markdown("""
<div class="main-header">
    <h1>🎬 Myanmar AI Studio Pro</h1>
    <p>Professional Video Dubbing, Scripting & Viral SEO Suite</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 Google API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    model_name = st.selectbox("🤖 AI Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    if st.button("🔴 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: 
    st.warning("⚠️ Please enter your API Key in the sidebar to start.")
    st.stop()

# TABS
t1, t2 = st.tabs(["🎙️ Production Studio", "🚀 Viral SEO Kit"])

# === TAB 1: PRODUCTION ===
with t1:
    # STEP 1
    st.markdown("### 1️⃣ Upload & Translate")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        # Limit text removed as requested, supports large files if server allows
        uploaded = st.file_uploader("Upload Video File (MP4/MOV)", type=['mp4','mov'])
    with col_u2:
        source_lang = st.selectbox("Original Language", ["English", "Japanese", "Chinese", "Thai", "Indian (Hindi)"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        if st.button("📝 Extract & Translate (Draft)"):
            with st.spinner("Processing Audio & Translating..."):
                check_requirements()
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raw_text = transcribe_video("temp.wav")
                st.session_state.raw_transcript = raw_text
                model = get_model(st.session_state.api_key, model_name)
                draft = translate_to_burmese_draft(model, raw_text, source_lang)
                st.session_state.burmese_draft = draft
                st.rerun()

    # STEP 2
    if st.session_state.burmese_draft:
        st.divider()
        st.markdown("### 2️⃣ Script Editing (H-V-C)")
        c1, c2 = st.columns(2)
        with c1: st.text_area("Burmese Draft", st.session_state.burmese_draft, height=200)
        with c2: 
            custom_prompt = st.text_input("Style Instructions", "Make it engaging and emotional")
            if st.button("✨ Refine Script (H-V-C Structure)"):
                with st.spinner("Refining..."):
                    model = get_model(st.session_state.api_key, model_name)
                    final = refine_script_hvc(model, st.session_state.burmese_draft, "Video", custom_prompt)
                    st.session_state.final_script = final
                    st.rerun()

    # STEP 3
    if st.session_state.final_script:
        st.divider()
        st.markdown("### 3️⃣ Audio & Video Production")
        
        # SCRIPT EDITOR
        final_script_edit = st.text_area("Final Script (Editable - Host Voice Only)", st.session_state.final_script, height=150)
        
        # SETTINGS ROW
        col1, col2, col3, col4 = st.columns(4)
        with col1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with col2: gender = st.selectbox("Gender", ["Male", "Female"])
        with col3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        with col4: zoom_val = st.slider("Zoom", 1.0, 1.1, 1.05, 0.01)
        
        # FREEZE CONTROLS
        st.markdown("#### ❄️ Freeze Controls")
        ft1, ft2 = st.tabs(["Auto Interval", "Manual Command"])
        input_vid = "input.mp4"
        processed_vid = "processed_visuals.mp4"
        auto_freeze_int = None
        manual_freeze_cmd = None
        
        with ft1:
            fc1, fc2, fc3, fc4 = st.columns(4)
            if fc1.checkbox("Every 30s"): auto_freeze_int = 30
            if fc2.checkbox("Every 1m"): auto_freeze_int = 60
            if fc3.checkbox("Every 3m"): auto_freeze_int = 180
            if fc4.checkbox("Every 6m"): auto_freeze_int = 360
        with ft2:
            manual_freeze_cmd = st.text_input("Command (e.g., freeze 10,5)", key="mfreeze")

        if st.button("🚀 Render Final Video (1080p)"):
            with st.spinner("Rendering Video & Syncing Audio..."):
                # 1. GENERATE AUDIO
                clean_text = final_script_edit.replace("*", "").strip()
                if generate_audio_cli(clean_text, target_lang, gender, v_mode, "base_voice.mp3"):
                    
                    # 2. VIDEO PROCESSING (FREEZE)
                    if auto_freeze_int:
                        apply_auto_freeze("input.mp4", "frozen_temp.mp4", auto_freeze_int)
                        input_vid = "frozen_temp.mp4"
                    elif manual_freeze_cmd:
                        if process_freeze_command(manual_freeze_cmd, "input.mp4", "frozen_temp.mp4"):
                            input_vid = "frozen_temp.mp4"
                    
                    # 3. ZOOM & UPSCALING TO 1080p
                    zoom_filter = f"scale=1920*{zoom_val}:1080*{zoom_val},crop=1920:1080"
                    subprocess.run(['ffmpeg', '-y', '-i', input_vid, '-vf', f"scale=1920:1080,{zoom_filter}", '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'copy', processed_vid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 4. SYNC (STRETCH AUDIO TO MATCH VIDEO EXACTLY)
                    vid_dur = get_duration(processed_vid)
                    aud_dur = get_duration("base_voice.mp3")
                    
                    # Calculate stretch factor. 
                    # If Audio < Video, we stretch Audio (slow down slightly or add pauses - here we use speed).
                    # If Audio > Video, we speed up Audio.
                    # We prioritize Video length to ensure full 10min video is shown.
                    speed_factor = aud_dur / vid_dur if vid_dur > 0 else 1.0
                    
                    # Safety Clamp (Don't turn into chipmunk or monster)
                    speed_factor = max(0.7, min(speed_factor, 1.5))
                    
                    subprocess.run(['ffmpeg', '-y', '-i', "base_voice.mp3", '-filter:a', f"atempo={speed_factor}", "final_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 5. MERGE
                    outfile = f"final_1080p_{int(time.time())}.mp4"
                    cmd = ['ffmpeg', '-y', '-i', processed_vid, '-i', "final_audio.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    st.session_state.processed_video_path = outfile
                    st.session_state.processed_audio_path = "final_audio.mp3"
                    st.success("✅ Production Complete!")

    # DOWNLOAD SECTION
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        st.divider()
        st.markdown("### 📥 Downloads")
        d1, d2 = st.columns(2)
        with d1:
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button("🎬 Download Video (MP4)", f, "final_video.mp4")
        with d2:
            if st.session_state.processed_audio_path and os.path.exists(st.session_state.processed_audio_path):
                with open(st.session_state.processed_audio_path, "rb") as f:
                    st.download_button("🎵 Download Audio Only (MP3)", f, "final_audio.mp3")
        
        st.video(st.session_state.processed_video_path)

# === TAB 2: VIRAL KIT ===
with t2:
    st.markdown("### 🚀 Viral SEO Generator")
    v_title = st.text_input("Video Title")
    v_keywords = st.text_input("Keywords (comma separated)")
    seo_lang = st.selectbox("Output Language", ["Myanmar", "English", "Thai", "Chinese", "Indian"])
    
    if st.button("✨ Generate SEO Data"):
        if not v_title: st.error("Please add a title")
        else:
            with st.spinner("Analyzing Trends..."):
                model = get_model(st.session_state.api_key, model_name)
                res = generate_viral_metadata(model, v_title, v_keywords, seo_lang)
                st.session_state.seo_result = res
                st.rerun()
    
    if st.session_state.seo_result:
        st.markdown("<div class='viral-box'>", unsafe_allow_html=True)
        st.markdown(st.session_state.seo_result)
        st.markdown("</div>", unsafe_allow_html=True)
