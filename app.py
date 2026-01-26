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
import json
import time
import shutil
import whisper
import re
from pydub import AudioSegment
from PIL import Image
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B, #FF416C); 
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444; background: #222; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing."); st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

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

# ---------------------------------------------------------
# 🛡️ STRICT TRANSLATION LOGIC (ENGLISH KILLER)
# ---------------------------------------------------------
def contains_burmese(text):
    # Check if text contains Myanmar Unicode range (U+1000 to U+109F)
    return bool(re.search(r'[\u1000-\u109F]', text))

def translate_strict(model, text, style):
    prompt = f"""
    Translate the following English text to Burmese (Myanmar).
    Input: "{text}"
    
    CRITICAL RULES:
    1. **OUTPUT BURMESE ONLY**. Do NOT output the English text.
    2. If you cannot translate, output a generic phrase like "ဆက်လက်ကြည့်ရှုပေးပါ".
    3. Convert Numbers -> Burmese Words (100 -> တစ်ရာ).
    4. Style: {style}.
    5. NO EXPLANATIONS.
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # 🔥 ENGLISH KILLER FILTER 🔥
            if contains_burmese(translated):
                # Clean accidental English chars (A-Z)
                cleaned = re.sub(r'[A-Za-z]', '', translated).strip()
                return cleaned
            else:
                # If AI returns English, treat as failure and retry
                continue
                
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except:
            continue
            
    # Final Fallback: NEVER return original English text.
    # Return empty string to silence audio, or a generic Burmese filler.
    return "" 

# ---------------------------------------------------------
# 🎬 PROCESSING WORKFLOW
# ---------------------------------------------------------
def process_video_workflow(video_path, voice_data, style, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract Audio
    status.info("🎧 Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper Transcription
    status.info("🧠 Listening (Whisper)...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Translate & Dub
    status.info(f"🎙️ Dubbing {len(segments)} segments (Strict Mode)...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        
        # Rate Limit Safety
        time.sleep(2) 
        
        # Translate (Strict)
        burmese_text = translate_strict(model, seg['text'], style)
        
        # Skip if empty (English Killer worked)
        if not burmese_text:
            progress.progress((i + 1) / total)
            continue
            
        # TTS
        fname = f"seg_{i}.mp3"
        generate_audio(burmese_text, voice_data["id"], voice_data["rate"], voice_data["pitch"], fname)
        
        # Sync
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            curr_dur = len(seg_audio) / 1000.0
            target_dur = end - start
            
            if curr_dur > 0 and target_dur > 0:
                speed = max(0.6, min(curr_dur / target_dur, 1.5))
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(f"s_{i}.mp3"):
                    final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
                    try: os.remove(f"s_{i}.mp3")
                    except: pass
            try: os.remove(fname)
            except: pass
            
        progress.progress((i + 1) / total)

    status.info("🔊 Mixing Audio...")
    final_audio.export("final_track.mp3", format="mp3")
    
    status.info("🎬 Finalizing...")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    # Mix with original background audio lowered (optional, here we replace fully for clarity)
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_filename

# ---------------------------------------------------------
# 🚀 FEATURES
# ---------------------------------------------------------
def run_genai_wrapper(prompt, video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    try:
        f = genai.upload_file(video_path)
        while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
        model = genai.GenerativeModel(model_name)
        return model.generate_content([f, prompt]).text
    except Exception as e: return str(e)

def run_thumbnail(video_path, api_key, model_name):
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vframes', '1', 'thumb.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    genai.configure(api_key=api_key)
    img = Image.open("thumb.jpg")
    model = genai.GenerativeModel(model_name)
    return model.generate_content([img, "Describe this for a Thumbnail."]).text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    model_choice = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro"])
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key."); st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    t1, t2, t3, t4 = st.tabs(["🎙️ DUBBING", "🚀 VIRAL KIT", "📝 SCRIPT", "🖼️ THUMBNAIL"])

    # --- DUBBING ---
    with t1:
        st.subheader("🔊 Auto Dubbing")
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.success("✅ Video Ready!")
            st.video(st.session_state.processed_video)
            with open(st.session_state.processed_video, "rb") as f:
                st.download_button("💾 Download", f, "dubbed.mp4")

        c1, c2 = st.columns(2)
        with c1: narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
        with c2: style = st.selectbox("Style", ["Natural", "News", "Dramatic"])
            
        v_data = {
            "Male (Thiha)": {"id": "my-MM-ThihaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Female (Nilar)": {"id": "my-MM-NilarNeural", "rate": "+10%", "pitch": "+10Hz"}
        }

        if st.button("🚀 Start Dubbing"):
            status = st.empty()
            prog = st.progress(0)
            try:
                # Correct function name used here
                out = process_video_workflow("temp.mp4", v_data[narrator], style, st.session_state.api_key, model_choice, status, prog)
                st.session_state.processed_video = out
                status.success("Done!")
                st.rerun()
            except Exception as e:
                status.error(f"Error: {e}")

    # --- VIRAL KIT ---
    with t2:
        if st.button("✨ Generate Viral Info"):
            with st.spinner("Processing..."):
                res = run_genai_wrapper("Generate 3 Viral Titles in Burmese.", "temp.mp4", st.session_state.api_key, model_choice)
                st.info(res)

    # --- SCRIPT ---
    with t3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                res = run_genai_wrapper("Write a full Burmese script.", "temp.mp4", st.session_state.api_key, model_choice)
                st.text_area("Script", res, height=400)

    # --- THUMBNAIL ---
    with t4:
        if st.button("🎨 Thumbnail Idea"):
            with st.spinner("Analyzing..."):
                txt, img = run_thumbnail("temp.mp4", st.session_state.api_key, model_choice)
                st.image(img)
                st.code(txt)
