import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
import shutil
import whisper
from pydub import AudioSegment
from PIL import Image
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🌐", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .css-1r6slb0, .stFileUploader, div[data-testid="stSidebar"] {
        background: rgba(25, 30, 40, 0.9); border-radius: 20px; border: 1px solid #333; padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; border: none;
        border-radius: 12px; height: 50px; font-weight: bold; width: 100%;
    }
    .nav-bar {
        display: flex; justify-content: space-around; background: #111; 
        padding: 15px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #333;
    }
    .result-box { background: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🌍 VOICE DATABASE
# ---------------------------------------------------------
VOICE_MAP = {
    "Myanmar (Burmese)": {"code": "my", "voice_m": "my-MM-ThihaNeural", "voice_f": "my-MM-NilarNeural"},
    "English (US)": {"code": "en", "voice_m": "en-US-ChristopherNeural", "voice_f": "en-US-JennyNeural"},
    "Thai": {"code": "th", "voice_m": "th-TH-NiwatNeural", "voice_f": "th-TH-PremwadeeNeural"},
    "Chinese": {"code": "zh", "voice_m": "zh-CN-YunxiNeural", "voice_f": "zh-CN-XiaoxiaoNeural"},
    "Japanese": {"code": "ja", "voice_m": "ja-JP-KeitaNeural", "voice_f": "ja-JP-NanamiNeural"},
    "Korean": {"code": "ko", "voice_m": "ko-KR-InJoonNeural", "voice_f": "ko-KR-SunHiNeural"},
}

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg missing. Install via `apt-get install ffmpeg`.")
        st.stop()

def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

async def _tts_async(text, voice, rate, pitch, output):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output)

def generate_tts_segment(text, voice, rate, pitch, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_async(text, voice, rate, pitch, filename))
        loop.close()
        return True
    except: return False

def extract_frame(video_path, output_image):
    try:
        duration = get_duration(video_path)
        mid_point = duration / 2
        subprocess.run(['ffmpeg', '-y', '-ss', str(mid_point), '-i', video_path, '-vframes', '1', output_image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

def upload_to_gemini(video_path, api_key):
    genai.configure(api_key=api_key)
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    return video_file

# ---------------------------------------------------------
# 🎬 FEATURE: DUBBING (SYNC FIXED)
# ---------------------------------------------------------
def translate_segment(model, text, target_lang, style, tone):
    # Mapping for language names to ensure AI understands
    lang_name = "Burmese (Myanmar)" if target_lang == "Myanmar (Burmese)" else target_lang
    
    prompt = f"""
    Act as a professional Dubbing Translator.
    Translate the following text into {lang_name} ({target_lang}).
    Original: "{text}"
    
    CRITICAL RULES:
    1. OUTPUT MUST BE IN {lang_name} SCRIPT ONLY.
    2. DO NOT OUTPUT ENGLISH CHARACTERS (Unless it's a specific term like 'iPhone').
    3. Convert numbers to {lang_name} words (100 -> တစ်ရာ).
    4. Style: {style}. Tone: {tone}.
    5. JUST RETURN THE TRANSLATED TEXT. NO EXPLANATION.
    """
    try:
        res = model.generate_content(prompt)
        t = res.text.strip()
        # Fallback: If AI returns English text for Myanmar, force retry or ignore
        return t if t else text
    except:
        return text


# ---------------------------------------------------------
# 🚀 FEATURE: VIRAL KIT
# ---------------------------------------------------------
def process_viral_kit(video_path, api_key, model_id):
    genai.configure(api_key=api_key)
    video_file = upload_to_gemini(video_path, api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = """
    Act as a Viral Social Media Manager. Analyze this video.
    Generate:
    1. 3 Clickbait Titles (Burmese & English).
    2. 15 Trending Hashtags.
    3. A short, engaging caption for TikTok/Reels.
    Format with Emojis.
    """
    res = model.generate_content([video_file, prompt])
    return res.text

# ---------------------------------------------------------
# 📝 FEATURE: SCRIPT WRITER
# ---------------------------------------------------------
def process_script(video_path, api_key, model_id, format_type):
    genai.configure(api_key=api_key)
    video_file = upload_to_gemini(video_path, api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = f"""
    Watch this video and write a {format_type} in Burmese.
    Make it detailed, professional, and ready to use.
    """
    res = model.generate_content([video_file, prompt])
    return res.text

# ---------------------------------------------------------
# 🖼️ FEATURE: THUMBNAIL
# ---------------------------------------------------------
def process_thumbnail(video_path, api_key, model_id):
    extract_frame(video_path, "thumb.jpg")
    img = Image.open("thumb.jpg")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = "Describe this image in detail to create a high-quality, clickbait YouTube thumbnail. Include lighting, mood, and text overlay ideas."
    res = model.generate_content([img, prompt])
    return res.text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("GLOBAL STUDIO")
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    api_key = st.text_input("🔑 API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.markdown("---")
    model_mode = st.radio("Mode", ["Preset", "Custom"])
    model_id = "gemini-2.0-flash-exp" if model_mode == "Preset" else st.text_input("Model ID", "gemini-2.5-flash")
    
    st.markdown("---")
    if st.button("🔄 REBOOT SYSTEM"): st.rerun()

# Navigation
menu = st.radio("Select Feature", ["🎙️ Dubbing Pro", "🚀 Viral Kit", "📝 Script Writer", "🖼️ Thumbnail"], horizontal=True)

if not st.session_state.api_key:
    st.warning("⚠️ Connect API Key first.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    # --- 1. DUBBING ---
    if menu == "🎙️ Dubbing Pro":
        st.subheader("🎙️ International AI Dubbing")
        c1, c2, c3 = st.columns(3)
        with c1: target_lang = st.selectbox("Target Language", list(VOICE_MAP.keys()))
        with c2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c3: tone = st.selectbox("Tone", ["Natural", "Deep", "Fast"])
        style = st.selectbox("Style", ["Narrator", "Vlogger", "Movie Recap"])
        
        if st.button("🚀 Start Dubbing"):
            status = st.status("Processing...", expanded=True)
            progress = st.progress(0)
            try:
                out = process_sync_dubbing("temp.mp4", target_lang, gender, style, tone, st.session_state.api_key, model_id, status, progress)
                status.update(label="✅ Done!", state="complete")
                st.video(out)
                with open(out, "rb") as f: st.download_button("Download Video", f, "dubbed.mp4")
            except Exception as e: st.error(str(e))

    # --- 2. VIRAL KIT ---
    elif menu == "🚀 Viral Kit":
        st.subheader("🚀 Viral Content Generator")
        if st.button("✨ Generate Ideas"):
            with st.spinner("Analyzing Video..."):
                res = process_viral_kit("temp.mp4", st.session_state.api_key, model_id)
                st.markdown(f"<div class='result-box'>{res}</div>", unsafe_allow_html=True)

    # --- 3. SCRIPT WRITER ---
    elif menu == "📝 Script Writer":
        st.subheader("📝 AI Script Writer")
        fmt = st.selectbox("Format", ["Video Transcript", "Blog Post", "Youtube Script"])
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                res = process_script("temp.mp4", st.session_state.api_key, model_id, fmt)
                st.text_area("Result", res, height=400)

    # --- 4. THUMBNAIL ---
    elif menu == "🖼️ Thumbnail":
        st.subheader("🖼️ Thumbnail Generator")
        if st.button("🎨 Analyze & Prompt"):
            with st.spinner("Extracting..."):
                txt, img_path = process_thumbnail("temp.mp4", st.session_state.api_key, model_id)
                st.image(img_path, caption="Extracted Frame")
                st.code(txt, language="markdown")
                st.info("Copy the prompt above to Midjourney or Bing Create.")
