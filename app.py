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
import re
from pydub import AudioSegment
from PIL import Image

# ---------------------------------------------------------
# 🎨 UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    .box { background: #111; border: 1px solid #333; padding: 20px; border-radius: 15px; margin-bottom: 20px; }
    .stButton>button {
        background: linear-gradient(90deg, #ff00cc, #333399); color: white; border: none; height: 50px; font-weight: bold; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ SYSTEM SETUP
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. System cannot run.")
        st.stop()

async def tts_save(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_id, rate, pitch, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_save(text, voice_id, rate, pitch, filename))
        loop.close()
        return True
    except: return False

def get_video_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🛑 ENGLISH KILLER LOGIC (CRITICAL FIX)
# ---------------------------------------------------------
def clean_text_for_burmese(text):
    # This removes any English characters (A-Z, a-z) to force Burmese only
    cleaned = re.sub(r'[A-Za-z]', '', text)
    # If text is empty after cleaning, it means it was all English
    if not cleaned.strip():
        return "ဘာသာပြန်ခြင်း အဆင်မပြေပါ။" # Fallback Burmese text
    return cleaned

def ai_translate_strict(model, text, style, tone):
    prompt = f"""
    Translate the following English text to Burmese (Myanmar).
    Input: "{text}"
    
    RULES:
    1. OUTPUT BURMESE ONLY. NO ENGLISH.
    2. Convert numbers to Burmese words (100 -> တစ်ရာ).
    3. Style: {style}. Tone: {tone}.
    4. Do not explain. Just translate.
    """
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        # Apply the English Killer Filter
        return clean_text_for_burmese(raw_text)
    except:
        return text

def process_dubbing(video_path, voice_config, style, tone, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Setup
    voice_id = voice_config["id"]
    pitch = voice_config["pitch"]
    rate = voice_config["rate"]
    
    # 2. Extract Audio
    status.update(label="🎧 Extracting Audio...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Whisper
    status.update(label="🧠 Speech Recognition...", state="running")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 4. AI Process
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name)
    except:
        st.error(f"❌ Model {model_name} not found. Falling back to gemini-1.5-flash")
        model = genai.GenerativeModel("gemini-1.5-flash")

    final_audio = AudioSegment.silent(duration=get_video_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        text = seg['text']
        
        # Translate with strict filter
        burmese_text = ai_translate_strict(model, text, style, tone)
        
        # Generate Audio
        fname = f"seg_{i}.mp3"
        generate_audio(burmese_text, voice_id, rate, pitch, fname)
        
        # Sync
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            seg_dur = end - start
            curr_dur = len(seg_audio) / 1000.0
            
            if curr_dur > 0:
                speed = max(0.6, min(curr_dur / seg_dur, 1.5))
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(f"s_{i}.mp3"):
                    final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
        
        progress.progress((i + 1) / total)
        try: os.remove(fname); os.remove(f"s_{i}.mp3")
        except: pass

    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. Merge
    status.update(label="🎬 Finalizing...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', "output.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "output.mp4"

# ---------------------------------------------------------
# 🚀 FEATURES (FIXED)
# ---------------------------------------------------------
def run_viral_kit(video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    f = genai.upload_file(video_path)
    while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
    model = genai.GenerativeModel(model_name)
    return model.generate_content([f, "Generate 3 Viral Titles (Burmese) and 10 Hashtags."]).text

def run_script(video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    f = genai.upload_file(video_path)
    while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
    model = genai.GenerativeModel(model_name)
    return model.generate_content([f, "Write a detailed YouTube script in Burmese based on this video."]).text

def run_thumbnail(video_path, api_key, model_name):
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vframes', '1', 'thumb.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists("thumb.jpg"): return "Error capturing frame", None
    
    genai.configure(api_key=api_key)
    img = Image.open("thumb.jpg")
    model = genai.GenerativeModel(model_name)
    res = model.generate_content([img, "Describe this image for a YouTube thumbnail prompt."])
    return res.text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎙️ AI STUDIO PRO")
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    # 🔥 DEFAULT MODEL SET TO GEMINI-2.5-FLASH AS REQUESTED
    model_name = st.text_input("AI Model", value="gemini-2.5-flash")
    if st.button("🔴 REBOOT SYSTEM"): st.rerun()

st.title("🔥 All-in-One AI Video Tool")

if not st.session_state.api_key:
    st.warning("Please enter API Key.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    # TABS FOR FEATURES
    t1, t2, t3, t4 = st.tabs(["🎙️ DUBBING", "🚀 VIRAL KIT", "📝 SCRIPT", "🖼️ THUMBNAIL"])

    # 1. DUBBING
    with t1:
        st.markdown("### 🔊 Auto Dubbing (Burmese Only)")
        c1, c2 = st.columns(2)
        with c1:
            narrator = st.selectbox("Narrator", [
                "Male (Thiha - News)", "Male (Thiha - Deep)", 
                "Female (Nilar - Story)", "Female (Nilar - Sweet)"
            ])
        with c2:
            style = st.selectbox("Style", ["Natural", "News", "Dramatic", "Vlog"])
        
        # Voice Config Map
        v_map = {
            "Male (Thiha - News)": {"id": "my-MM-ThihaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Male (Thiha - Deep)": {"id": "my-MM-ThihaNeural", "rate": "-5%", "pitch": "-10Hz"},
            "Female (Nilar - Story)": {"id": "my-MM-NilarNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Female (Nilar - Sweet)": {"id": "my-MM-NilarNeural", "rate": "+10%", "pitch": "+10Hz"}
        }

        if st.button("🚀 START DUBBING"):
            status = st.status("Processing...", expanded=True)
            prog = st.progress(0)
            try:
                out = process_dubbing("temp.mp4", v_map[narrator], style, "Natural", st.session_state.api_key, model_name, status, prog)
                status.update(label="✅ Done!", state="complete")
                st.video(out)
                with open(out, "rb") as f: st.download_button("Download", f, "dubbed.mp4")
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    # 2. VIRAL KIT
    with t2:
        if st.button("✨ Generate Viral Data"):
            with st.spinner("Analyzing..."):
                try:
                    res = run_viral_kit("temp.mp4", st.session_state.api_key, model_name)
                    st.markdown(f"<div class='box'>{res}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(str(e))

    # 3. SCRIPT
    with t3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                try:
                    res = run_script("temp.mp4", st.session_state.api_key, model_name)
                    st.text_area("Script", res, height=400)
                except Exception as e: st.error(str(e))

    # 4. THUMBNAIL
    with t4:
        if st.button("🎨 Create Thumbnail Prompt"):
            with st.spinner("Analyzing..."):
                try:
                    txt, img = run_thumbnail("temp.mp4", st.session_state.api_key, model_name)
                    if img: st.image(img)
                    st.code(txt)
                except Exception as e: st.error(str(e))
