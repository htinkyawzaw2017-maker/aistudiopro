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
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
if 'srt_content' not in st.session_state: st.session_state.srt_content = ""

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
    # Download Padauk font for Burmese subtitles
    font_path = "Padauk-Regular.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Regular.ttf"
        r = requests.get(url)
        with open(font_path, 'wb') as f:
            f.write(r.content)
    return font_path

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
    if not text or not text.strip(): return False, "Empty text"
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        cmd = ["edge-tts", "--voice", voice_id, "--text", text, f"--rate={settings['rate']}", f"--pitch={settings['pitch']}", "--write-media", output_file]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: return False, res.stderr
        return True, "Success"
    except Exception as e: return False, str(e)

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

def transcribe_for_captions(video_path):
    try:
        model = whisper.load_model("base")
        # Returns segments with start/end times
        result = model.transcribe(video_path)
        return result['segments']
    except Exception as e: return []

def translate_to_burmese_draft(model, text, source_lang):
    prompt = f"Translate {source_lang} to Burmese. Input: '{text}'. Rules: Keep Proper Nouns in English. Translate accurately sentence by sentence."
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

def translate_segments(model, segments):
    # Translates subtitle segments while keeping structure
    translated_srt = ""
    for i, seg in enumerate(segments):
        start = time.strftime('%H:%M:%S,000', time.gmtime(seg['start']))
        end = time.strftime('%H:%M:%S,000', time.gmtime(seg['end']))
        text = seg['text']
        
        # Translate small chunk
        try:
            res = model.generate_content(f"Translate to Burmese (Short subtitle style): '{text}'")
            trans_text = res.text.strip()
        except:
            trans_text = text # Fallback
            
        translated_srt += f"{i+1}\n{start} --> {end}\n{trans_text}\n\n"
        time.sleep(1) # Rate limit safety
    return translated_srt

# ---------------------------------------------------------
# ❄️ FREEZE & VIDEO ENGINE
# ---------------------------------------------------------
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
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', p_name, '-update', '1', '-q:v', '1', 'f.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'f.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
# 🖥️ MAIN UI
# ---------------------------------------------------------
st.markdown("""<div class="main-header"><h1>🎬 Myanmar AI Studio Pro</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 Google API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

# --- TABS ---
t1, t2, t3 = st.tabs(["🎙️ Dubbing Studio", "📝 Auto Caption (Subtitle)", "🚀 Viral SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    st.subheader("1. Upload & Translate")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub_up")
    source_lang = st.selectbox("Original Lang", ["English", "Japanese", "Chinese", "Thai"], key="sl1")
    
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

    if st.session_state.burmese_draft:
        st.subheader("2. Scripting Mode")
        # TOGGLE FOR HVC vs DIRECT
        script_mode = st.radio("Choose Script Style:", ["Direct Translation (Best for 10min Videos)", "H-V-C Rewrite (Creative/Short)"])
        
        draft_text = st.text_area("Current Draft", st.session_state.burmese_draft, height=200)
        
        if st.button("Confirm Script"):
            if "H-V-C" in script_mode:
                with st.spinner("Refining with H-V-C..."):
                    model = get_model(st.session_state.api_key, model_name)
                    final = refine_script_hvc(model, draft_text, "Video", "Engaging")
                    st.session_state.final_script = final
            else:
                # Direct Mode - Use the draft as is
                st.session_state.final_script = draft_text
            st.rerun()

    if st.session_state.final_script:
        st.subheader("3. Production")
        final_text = st.text_area("Final Script", st.session_state.final_script, height=200)
        
        c1, c2, c3 = st.columns(3)
        with c1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        zoom_val = st.slider("Zoom", 1.0, 1.1, 1.05, 0.01)
        
        ft1, ft2 = st.tabs(["Auto Freeze", "Manual Command"])
        input_vid = "input.mp4"
        processed_vid = "proc_visuals.mp4"
        auto_freeze = None; manual_freeze = None
        
        with ft1:
            if st.checkbox("Every 30s"): auto_freeze = 30
            if st.checkbox("Every 1m"): auto_freeze = 60
            if st.checkbox("Every 3m"): auto_freeze = 180
        with ft2: manual_freeze = st.text_input("Command", placeholder="freeze 10,5")

        if st.button("🚀 Render Final Video"):
            with st.spinner("Rendering..."):
                clean_text = final_text.replace("*", "").strip()
                success, msg = generate_audio_cli(clean_text, target_lang, gender, v_mode, "final_audio.mp3")
                
                if success:
                    if auto_freeze: apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze); input_vid = "frozen.mp4"
                    elif manual_freeze: process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"); input_vid = "frozen.mp4"
                    
                    w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                    if w_s % 2 != 0: w_s += 1
                    if h_s % 2 != 0: h_s += 1
                    
                    subprocess.run(['ffmpeg', '-y', '-i', input_vid, '-vf', f"scale={w_s}:{h_s},crop=1920:1080", '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'copy', processed_vid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    v_dur = get_duration(processed_vid)
                    a_dur = get_duration("final_audio.mp3")
                    speed = min(a_dur / v_dur, 1.5) if v_dur > 0 else 1.0
                    
                    subprocess.run(['ffmpeg', '-y', '-i', "final_audio.mp3", '-filter:a', f"atempo={speed}", "sync_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    outfile = f"final_{int(time.time())}.mp4"
                    subprocess.run(['ffmpeg', '-y', '-i', processed_vid, '-i', "sync_audio.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    st.session_state.processed_video_path = outfile
                    st.success("✅ Done!")
                else: st.error(f"Audio Error: {msg}")

    if st.session_state.processed_video_path:
        with open(st.session_state.processed_video_path, "rb") as f: st.download_button("Download Video", f, "video.mp4")

# === TAB 2: AUTO CAPTION (NEW FEATURE) ===
with t2:
    st.subheader("📝 Auto Subtitle Generator (Timeline Exact)")
    cap_up = st.file_uploader("Upload Video for Captioning", type=['mp4','mov'], key="cap_up")
    
    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        
        if st.button("Generate Burmese Captions"):
            with st.spinner("1. Analyzing Timeline (Whisper)..."):
                check_requirements()
                # Extract Audio
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap_temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Get Segments with Time
                segments = transcribe_for_captions("cap_temp.wav")
                
            with st.spinner("2. Translating Segments (Gemini)..."):
                model = get_model(st.session_state.api_key, model_name)
                # Translate & Build SRT
                srt_content = translate_segments(model, segments)
                st.session_state.srt_content = srt_content
                with open("subs.srt", "w", encoding="utf-8") as f: f.write(srt_content)
                
            with st.spinner("3. Burning Subtitles..."):
                font_path = download_font() # Get Padauk Font
                # FFmpeg Burn
                out_cap = f"captioned_{int(time.time())}.mp4"
                # Use force_style to set font
                # Note: This requires the font file to be in the same dir or fontconfig to see it. 
                # We point directly to it if possible or use default.
                # Simplified burn command:
                subprocess.run([
                    'ffmpeg', '-y', '-i', "cap_input.mp4", 
                    '-vf', f"subtitles=subs.srt:fontsdir=.:force_style='FontName=Padauk-Regular,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1'",
                    '-c:a', 'copy', out_cap
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                st.session_state.caption_video_path = out_cap
                st.success("✅ Captions Added!")

    if st.session_state.caption_video_path:
        st.video(st.session_state.caption_video_path)
        c1, c2 = st.columns(2)
        with c1:
            with open(st.session_state.caption_video_path, "rb") as f: st.download_button("Download Video", f, "captioned.mp4")
        with c2:
            st.download_button("Download SRT File", st.session_state.srt_content, "subs.srt")

# === TAB 3: SEO ===
with t3:
    st.subheader("Viral SEO")
    # (Same SEO Code as before)
    title = st.text_input("Title"); keys = st.text_input("Keywords")
    if st.button("Generate"):
        model = get_model(st.session_state.api_key, model_name)
        res = generate_viral_metadata(model, title, keys, "Myanmar")
        st.markdown(f"<div class='viral-box'>{res}</div>", unsafe_allow_html=True)
