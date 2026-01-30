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
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B, #FF416C); 
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    textarea { font-size: 1.1rem !important; font-family: 'Padauk', sans-serif !important; }
    .viral-box { background: #111; padding: 15px; border-left: 4px solid #00ff00; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
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
    "Story": {"rate": "-10%", "pitch": "-5Hz"},
    "Documentary": {"rate": "-5%", "pitch": "-10Hz"},
    "Recap": {"rate": "+10%", "pitch": "+0Hz"},
    "Motivation": {"rate": "-10%", "pitch": "-15Hz"},
    "Animation": {"rate": "+5%", "pitch": "+20Hz"}
}

def generate_audio_cli(text, lang, gender, mode_name, output_file):
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        
        cmd = [
            "edge-tts",
            "--voice", voice_id,
            "--text", text,
            "--rate", settings["rate"],
            "--pitch", settings["pitch"],
            "--write-media", output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 100
    except: return False

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
    Constraint: Keep length same as draft. Do not summarize.
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
# ❄️ FREEZE & VIDEO ENGINE (FIXED INTEGERS)
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
            time_point = float(match.group(1))
            dur = float(match.group(2))
            
            # Use filter_complex for safer freeze
            # But stick to split/concat for speed
            subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(time_point), '-c', 'copy', 'a.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-vframes', '1', 'f.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'f.jpg', '-t', str(dur), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'fr.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-c', 'copy', 'b.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            with open("list.txt", "w") as f:
                f.write("file 'a.mp4'\nfile 'fr.mp4'\nfile 'b.mp4'")
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        return False
    except: return False

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    model_name = st.text_input("Model ID", value="gemini-2.5-flash")
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2 = st.tabs(["🎬 Production", "🚀 Viral SEO"])

with t1:
    # STEP 1
    st.subheader("Step 1: Upload & Initial Translate")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'])
    source_lang = st.selectbox("Original Language", ["English", "Japanese", "Chinese", "Thai", "Hindi"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        if st.button("📝 Extract & Translate"):
            with st.spinner("Processing..."):
                check_requirements()
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raw = transcribe_video("temp.wav")
                st.session_state.raw_transcript = raw
                model = get_model(st.session_state.api_key, model_name)
                draft = translate_to_burmese_draft(model, raw, source_lang)
                st.session_state.burmese_draft = draft
                st.rerun()

    # STEP 2
    if st.session_state.burmese_draft:
        st.subheader("Step 2: Script Refinement")
        draft_text = st.text_area("Burmese Draft", st.session_state.burmese_draft, height=150)
        prompt = st.text_input("Instructions", "H-V-C structure")
        if st.button("✨ Refine Script"):
            with st.spinner("Refining..."):
                model = get_model(st.session_state.api_key, model_name)
                final = refine_script_hvc(model, draft_text, "Video", prompt)
                st.session_state.final_script = final
                st.rerun()

    # STEP 3
    if st.session_state.final_script:
        st.subheader("Step 3: Production")
        final_text = st.text_area("Final Script", st.session_state.final_script, height=200)
        
        c1, c2, c3 = st.columns(3)
        with c1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        st.markdown("**Video Controls**")
        zoom_val = st.slider("Zoom", 1.0, 1.1, 1.05, 0.01)
        
        ft1, ft2 = st.tabs(["Auto Freeze", "Manual Command"])
        input_vid = "input.mp4"
        processed_vid = "proc_visuals.mp4"
        auto_freeze = None
        manual_freeze = None
        
        with ft1:
            if st.checkbox("Every 30s"): auto_freeze = 30
            if st.checkbox("Every 1m"): auto_freeze = 60
            if st.checkbox("Every 3m"): auto_freeze = 180
        with ft2:
            manual_freeze = st.text_input("Command", placeholder="freeze 10,5")

        if st.button("🚀 Render Final Video"):
            with st.spinner("Rendering..."):
                # 1. AUDIO
                clean_txt = final_text.replace("*", "").strip()
                if generate_audio_cli(clean_text, target_lang, gender, v_mode, "final_audio.mp3"):
                    st.session_state.processed_audio_path = "final_audio.mp3"
                    
                    # 2. FREEZE
                    if auto_freeze:
                        if apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze):
                            input_vid = "frozen.mp4"
                    elif manual_freeze:
                        if process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"):
                            input_vid = "frozen.mp4"
                    
                    # 3. ZOOM & UPSCALE (CRITICAL INT FIX)
                    # Convert float zoom to integer width/height
                    # 1920 * 1.05 = 2016.0 -> int(2016)
                    w_scaled = int(1920 * zoom_val)
                    h_scaled = int(1080 * zoom_val)
                    # Ensure divisible by 2
                    if w_scaled % 2 != 0: w_scaled += 1
                    if h_scaled % 2 != 0: h_scaled += 1
                    
                    zoom_filter = f"scale={w_scaled}:{h_scaled},crop=1920:1080"
                    
                    subprocess.run(['ffmpeg', '-y', '-i', input_vid, '-vf', zoom_filter, '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'copy', processed_vid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 4. SYNC
                    vid_dur = get_duration(processed_vid)
                    aud_dur = get_duration("final_audio.mp3")
                    speed = 1.0
                    if vid_dur > 0 and aud_dur > vid_dur:
                        speed = min(aud_dur / vid_dur, 1.5)
                    
                    subprocess.run(['ffmpeg', '-y', '-i', "final_audio.mp3", '-filter:a', f"atempo={speed}", "sync_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 5. MERGE
                    outfile = f"final_{int(time.time())}.mp4"
                    cmd = ['ffmpeg', '-y', '-i', processed_vid, '-i', "sync_audio.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    st.session_state.processed_video_path = outfile
                    st.success("✅ Done!")
                else:
                    st.error("❌ Audio Generation Failed! (Empty text or API error)")

    # DOWNLOADS
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button("🎬 Download Video", f, "final_video.mp4")
        with c2:
            if st.session_state.processed_audio_path:
                with open(st.session_state.processed_audio_path, "rb") as f:
                    st.download_button("🎵 Download Audio", f, "final_audio.mp3")
        st.video(st.session_state.processed_video_path)

# === TAB 2: SEO ===
with t2:
    st.subheader("Viral SEO")
    title = st.text_input("Title")
    keys = st.text_input("Keywords")
    lang = st.selectbox("SEO Language", ["Myanmar", "English", "Thai", "Chinese"])
    if st.button("Generate SEO"):
        model = get_model(st.session_state.api_key, model_name)
        res = generate_viral_metadata(model, title, keys, lang)
        st.session_state.seo_result = res
        st.rerun()
    if st.session_state.seo_result:
        st.markdown(f"<div class='viral-box'>{st.session_state.seo_result}</div>", unsafe_allow_html=True)
