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
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None # Added for Audio Download
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
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
    except RuntimeError as e:
        if "checksum" in str(e).lower() or "mismatch" in str(e).lower():
            st.warning("⚠️ Fixing Model Corruption... Please wait.")
            cache_dir = os.path.expanduser("~/.cache/whisper")
            if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
            return whisper.load_model("base")
        else: raise e

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE ENGINE (CAPCUT OVERLAY)
# ---------------------------------------------------------
def generate_ass_file(segments, font_path, font_size=20, margin_v=50):
    filename = "captions.ass"
    font_dir = os.path.dirname(font_path)
    
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
Style: CapCut,Padauk-Bold,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            raw_text = seg['text'].strip()
            wrapped_lines = textwrap.wrap(raw_text, width=35)
            final_text = "\\N".join(wrapped_lines)
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return filename

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (RESTORED FULL MODES)
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}

# ✅ FULL VOICE MODES RESTORED
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-10%", "pitch": "-5Hz"},
    "Documentary": {"rate": "-5%", "pitch": "-10Hz"},
    "Recap": {"rate": "+10%", "pitch": "+0Hz"},
    "Motivation": {"rate": "-15%", "pitch": "-10Hz"},
    "Animation": {"rate": "+10%", "pitch": "+15Hz"}
}

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
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def transcribe_video(video_path):
    try:
        model = load_whisper_safe()
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e: return f"Error: {e}"

def transcribe_for_captions(video_path):
    try:
        model = load_whisper_safe()
        result = model.transcribe(video_path, task="transcribe")
        return result['segments']
    except Exception as e: 
        st.error(f"Whisper Error: {e}")
        return []

def translate_captions(model, segments):
    translated = []
    for seg in segments:
        text = seg['text'].strip()
        if not text: continue
        try:
            prompt = f"Translate to Burmese. Keep it short (max 10 words). Input: '{text}'"
            res = model.generate_content(prompt)
            burmese_text = res.text.strip()
        except: burmese_text = text
        translated.append({'start': seg['start'], 'end': seg['end'], 'text': burmese_text})
        time.sleep(0.3)
    return translated

def refine_script_hvc(model, text, title, custom_prompt):
    prompt = f"Refine this Burmese draft for video '{title}'. Input: '{text}'. Structure: H-V-C. Constraint: Keep content length roughly same. Output: Only Burmese spoken text. Extra: {custom_prompt}"
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def translate_to_burmese_draft(model, text, source_lang):
    prompt = f"Translate {source_lang} to Burmese. Input: '{text}'. Rules: Keep Proper Nouns in English. Translate accurately."
    try: return model.generate_content(prompt).text
    except: return "AI Error"

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
    model_name = st.selectbox("AI Model", ["gemini-2.5-flash", "gemini-2.0-flash-exp"])
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2, t3 = st.tabs(["🎙️ Dubbing Studio", "📝 Auto Caption (Overlay)", "🚀 Viral SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    st.subheader("1. Upload & Translate")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    source_lang = st.selectbox("Original Lang", ["English", "Japanese", "Chinese", "Thai"], key="sl1")
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        if st.button("📝 Extract & Translate"):
            check_requirements()
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raw = transcribe_video("temp.wav")
            st.session_state.raw_transcript = raw
            model = get_model(st.session_state.api_key, model_name)
            draft = translate_to_burmese_draft(model, raw, source_lang)
            st.session_state.burmese_draft = draft
            st.rerun()

    if st.session_state.burmese_draft:
        st.subheader("2. Scripting")
        script_mode = st.radio("Style:", ["Direct Translation (Best for 10min)", "H-V-C Rewrite (Short/Creative)"])
        draft_text = st.text_area("Draft", st.session_state.burmese_draft, height=200)
        if st.button("Confirm Script"):
            if "H-V-C" in script_mode:
                with st.spinner("Refining..."):
                    model = get_model(st.session_state.api_key, model_name)
                    final = refine_script_hvc(model, draft_text, "Video", "Engaging")
                    st.session_state.final_script = final
            else: st.session_state.final_script = draft_text
            st.rerun()

    if st.session_state.final_script:
        st.subheader("3. Production")
        final_text = st.text_area("Final Script", st.session_state.final_script, height=200)
        c1, c2, c3 = st.columns(3)
        with c1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c2: gender = st.selectbox("Gender", ["Male", "Female"])
        # ✅ VOICE MODES FULLY RESTORED IN UI
        with c3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        zoom_val = st.slider("Zoom", 1.0, 1.1, 1.05, 0.01)
        
        ft1, ft2 = st.tabs(["Auto Freeze", "Manual Command"])
        auto_freeze = None; manual_freeze = None
        with ft1:
            if st.checkbox("Every 30s"): auto_freeze = 30
            if st.checkbox("Every 1m"): auto_freeze = 60
        with ft2: manual_freeze = st.text_input("Command", placeholder="freeze 10,5")

        if st.button("🚀 Render Final Video"):
            with st.spinner("Rendering..."):
                clean_text = final_text.replace("*", "").strip()
                success, msg = generate_audio_cli(clean_text, target_lang, gender, v_mode, "final_audio.mp3")
                if success:
                    st.session_state.processed_audio_path = "final_audio.mp3" # Save audio path
                    input_vid = "input.mp4"
                    if auto_freeze: apply_auto_freeze("input.mp4", "frozen.mp4", auto_freeze); input_vid = "frozen.mp4"
                    elif manual_freeze: process_freeze_command(manual_freeze, "input.mp4", "frozen.mp4"); input_vid = "frozen.mp4"
                    
                    w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                    if w_s % 2 != 0: w_s += 1
                    if h_s % 2 != 0: h_s += 1
                    subprocess.run(['ffmpeg', '-y', '-i', input_vid, '-vf', f"scale={w_s}:{h_s},crop=1920:1080", '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'copy', "proc.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    v_dur = get_duration("proc.mp4"); a_dur = get_duration("final_audio.mp3")
                    speed = min(a_dur / v_dur, 1.5) if v_dur > 0 else 1.0
                    subprocess.run(['ffmpeg', '-y', '-i', "final_audio.mp3", '-filter:a', f"atempo={speed}", "sync.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    outfile = f"final_{int(time.time())}.mp4"
                    subprocess.run(['ffmpeg', '-y', '-i', "proc.mp4", '-i', "sync.mp3", '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    st.session_state.processed_video_path = outfile; st.success("Done!")
                else: st.error(msg)
    
    # ✅ RESTORED AUDIO DOWNLOAD BUTTON
    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)
        d1, d2 = st.columns(2)
        with d1:
            with open(st.session_state.processed_video_path, "rb") as f: st.download_button("🎬 Download Video", f, "video.mp4")
        with d2:
            if st.session_state.processed_audio_path:
                with open(st.session_state.processed_audio_path, "rb") as f: st.download_button("🎵 Download Audio Only", f, "audio.mp3")

# === TAB 2: AUTO CAPTION (FIXED OVERLAY STYLE) ===
with t2:
    st.subheader("📝 Auto Subtitle (Cover Original Text)")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    
    c_style1, c_style2 = st.columns(2)
    with c_style1: font_size = st.slider("Font Size", 15, 60, 24)
    with c_style2: margin_v = st.slider("Position", 10, 300, 50)

    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        
        if st.button("Generate Overlay Captions"):
            check_requirements()
            font_path = download_font()
            
            with st.spinner("1. Transcribing..."):
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                segments = transcribe_for_captions("cap.wav")
            
            with st.spinner("2. Translating..."):
                model = get_model(st.session_state.api_key, model_name)
                trans_segments = translate_captions(model, segments)
            
            with st.spinner("3. Applying Overlay..."):
                ass_file = generate_ass_file(trans_segments, font_path, font_size, margin_v)
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
            st.download_button("Download Video", f, "capcut_overlay.mp4")

# === TAB 3: VIRAL SEO ===
with t3:
    st.subheader("Viral SEO Kit")
    title = st.text_input("Title")
    keys = st.text_input("Keywords")
    lang = st.selectbox("SEO Lang", ["Myanmar", "English", "Thai", "Chinese"])
    
    if st.button("Generate SEO Data"):
        if not title: st.error("Please add a title")
        else:
            with st.spinner("Analyzing..."):
                model = get_model(st.session_state.api_key, model_name)
                prompt = f"Write viral SEO description for '{title}'. Language: {lang}. Keywords: {keys}. Include Hook, Bullet points, Tags."
                res = model.generate_content(prompt).text
                st.markdown(f"<div class='viral-box'>{res}</div>", unsafe_allow_html=True)
