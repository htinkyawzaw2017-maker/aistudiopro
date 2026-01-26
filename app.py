import warnings
warnings.filterwarnings("ignore")  # Warning အနီစာတန်းများ ပိတ်ခြင်း
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
# 🎨 UI SETUP (Dark Theme & Styling)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D); 
        color: black; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #333; }
    .success { background-color: #051a05; border-color: #00ff00; color: #00ff00; }
    .info { background-color: #001a33; border-color: #0099ff; color: #0099ff; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🛠️ CORE FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please install FFmpeg."); st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# Async TTS Wrapper
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
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------------------------------------------------
# 🧠 SMART TRANSLATION (Rate Limit Safe)
# ---------------------------------------------------------
def translate_text_safe(model, text, style):
    prompt = f"""
    Translate the following English text to spoken Burmese (Myanmar).
    
    Input: "{text}"
    
    RULES:
    1. Output **ONLY Burmese**. No English.
    2. **Numbers**: Convert to words (e.g., 50 -> ငါးဆယ်).
    3. **Units**: Convert to words (e.g., km -> ကီလိုမီတာ).
    4. **Tone**: {style}. Natural speaking style.
    5. Do NOT include explanations.
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            # Clean up potential English chars
            cleaned = re.sub(r'[A-Za-z]', '', translated).strip()
            if not cleaned: return text # Fallback to original if empty
            return cleaned
        except exceptions.ResourceExhausted:
            time.sleep(5) # Wait 5s on 429 error
            continue
        except Exception:
            return text # Fallback
            
    return text

# ---------------------------------------------------------
# 🎬 MAIN PROCESS ENGINE
# ---------------------------------------------------------
def process_video(video_path, voice_data, style, api_key, model_name, status_container, progress_bar):
    check_requirements()
    
    # 1. Audio Extraction
    status_container.info("🎧 Step 1/5: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper Transcription
    status_container.info("🧠 Step 2/5: Recognizing Speech (Whisper)...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Translation & TTS Loop
    status_container.info(f"🎙️ Step 3/5: Dubbing ({len(segments)} segments)...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash") # Safe fallback

    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total_segments = len(segments)
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text']
        
        # --- CRITICAL: RATE LIMIT HANDLING ---
        # Free Tier allows ~15 requests per minute. 
        # We enforce a 4-second delay to be safe and avoid "ResourceExhausted".
        time.sleep(4) 
        
        # Translate
        burmese_text = translate_text_safe(model, text, style)
        
        # Generate Audio
        fname = f"seg_{i}.mp3"
        success = generate_audio(burmese_text, voice_data["id"], voice_data["rate"], voice_data["pitch"], fname)
        
        # Sync & Overlay
        if success and os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            curr_dur = len(seg_audio) / 1000.0
            target_dur = end - start
            
            if curr_dur > 0 and target_dur > 0:
                # Time stretch (Limit 0.6x - 1.5x to avoid robot voice)
                speed = max(0.6, min(curr_dur / target_dur, 1.5))
                stretched_name = f"s_{i}.mp3"
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", stretched_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(stretched_name):
                    final_seg = AudioSegment.from_file(stretched_name)
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
                    
                    # Cleanup immediately
                    try: os.remove(stretched_name)
                    except: pass
            
            try: os.remove(fname)
            except: pass

        # Update Progress
        progress_bar.progress((i + 1) / total_segments)

    # 4. Export Final Audio
    status_container.info("🔊 Step 4/5: Mixing Audio...")
    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. Merge with Video
    status_container.info("🎬 Step 5/5: Finalizing...")
    output_file = f"dubbed_{int(time.time())}.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_file

# ---------------------------------------------------------
# 🚀 FEATURE WRAPPERS (VIRAL/SCRIPT/THUMBNAIL)
# ---------------------------------------------------------
def run_genai_feature(prompt, video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    try:
        f = genai.upload_file(video_path)
        while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
        model = genai.GenerativeModel(model_name)
        return model.generate_content([f, prompt]).text
    except Exception as e: return f"Error: {e}"

def run_thumbnail(video_path, api_key, model_name):
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vframes', '1', 'thumb.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    genai.configure(api_key=api_key)
    img = Image.open("thumb.jpg")
    model = genai.GenerativeModel(model_name)
    return model.generate_content([img, "Generate a viral YouTube thumbnail description."]).text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    
    # API Key Input
    api_key_input = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key_input: st.session_state.api_key = api_key_input
    
    st.divider()
    
    # Model Selection
    model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro"])
    st.info(f"Using: {model_choice}")
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

# Main Area
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video (MP4/MOV)", type=['mp4', 'mov'])

if uploaded_file:
    # Save uploaded file
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "🚀 Viral Kit", "📝 Script", "🖼️ Thumbnail"])

    # --- TAB 1: DUBBING ---
    with t1:
        st.subheader("🔊 Auto Dubbing")
        
        # Result Display
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.markdown('<div class="status-box success">✅ Video Successfully Dubbed!</div>', unsafe_allow_html=True)
            st.video(st.session_state.processed_video)
            with open(st.session_state.processed_video, "rb") as f:
                st.download_button("💾 Download Video", f, "myanmar_dub.mp4")

        # Controls
        c1, c2 = st.columns(2)
        with c1: narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
        with c2: style = st.selectbox("Style", ["Natural", "News", "Dramatic"])
            
        v_data = {
            "Male (Thiha)": {"id": "my-MM-ThihaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Female (Nilar)": {"id": "my-MM-NilarNeural", "rate": "+10%", "pitch": "+10Hz"}
        }

        if st.button("🚀 Start Dubbing"):
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Correct Function Call
                output = process_video("temp.mp4", v_data[narrator], style, st.session_state.api_key, model_choice, status_container, progress_bar)
                
                st.session_state.processed_video = output
                status_container.markdown('<div class="status-box success">✅ Done!</div>', unsafe_allow_html=True)
                st.rerun()
                
            except Exception as e:
                status_container.error(f"Error: {str(e)}")

    # --- TAB 2: VIRAL KIT ---
    with t2:
        if st.button("✨ Generate Viral Info"):
            with st.spinner("Analyzing Video..."):
                res = run_genai_feature("Generate 3 Viral Titles in Burmese & 10 Hashtags.", "temp.mp4", st.session_state.api_key, model_choice)
                st.info(res)

    # --- TAB 3: SCRIPT ---
    with t3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing Script..."):
                res = run_genai_feature("Write a full video script in Burmese.", "temp.mp4", st.session_state.api_key, model_choice)
                st.text_area("Script", res, height=400)

    # --- TAB 4: THUMBNAIL ---
    with t4:
        if st.button("🎨 Thumbnail Prompt"):
            with st.spinner("Analyzing Image..."):
                txt, img = run_thumbnail("temp.mp4", st.session_state.api_key, model_choice)
                st.image(img)
                st.code(txt)
