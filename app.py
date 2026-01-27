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
    .error-box { padding: 10px; background: #330000; border: 1px solid #ff0000; border-radius: 5px; color: #ffcccc; margin-bottom: 10px; }
    .success-box { padding: 10px; background: #003300; border: 1px solid #00ff00; border-radius: 5px; color: #ccffcc; margin-bottom: 10px; }
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
        st.error("❌ FFmpeg is missing. System cannot process audio.")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (Native Library - Most Stable)
# ---------------------------------------------------------
async def tts_gen(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_config, filename):
    try:
        # Create new loop to avoid Cloud concurrency errors
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_gen(text, voice_config['id'], voice_config['rate'], voice_config['pitch'], filename))
        loop.close()
        
        # Verify file
        if os.path.exists(filename) and os.path.getsize(filename) > 100:
            return True
        return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ TEXT CLEANER
# ---------------------------------------------------------
def clean_burmese(text):
    # Fix Units
    replacements = {
        "No.": "နံပါတ် ", "kg": " ကီလို ", "cm": " စင်တီမီတာ ", 
        "mm": " မီလီမီတာ ", "%": " ရာခိုင်နှုန်း ", "$": " ဒေါ်လာ "
    }
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    # Remove English characters (A-Z)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_content(model, text, style):
    prompt = f"""
    Translate English to spoken Burmese (Myanmar).
    Input: "{text}"
    Rules: 
    1. Burmese ONLY. No English words.
    2. Convert numbers to words (e.g. 100 -> တစ်ရာ).
    3. Tone: {style}.
    """
    for _ in range(2):
        try:
            res = model.generate_content(prompt)
            clean = clean_burmese(res.text.strip())
            if clean: return clean
        except Exception as e:
            print(f"Translation Error: {e}")
            time.sleep(2)
    return ""

# ---------------------------------------------------------
# 🎬 MAIN PROCESS
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style, mix_bg, bg_vol, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. API Test
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name)
        model.generate_content("Test") # Quick test
    except Exception as e:
        status.markdown(f"<div class='error-box'>❌ API Error: {str(e)}<br>Please check your API Key or Model Name.</div>", unsafe_allow_html=True)
        return None

    # 2. Extract
    status.info("🎧 Step 1: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Whisper
    status.info("🧠 Step 2: Speech Recognition...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 4. Dubbing Loop
    status.info(f"🎙️ Step 3: Dubbing {len(segments)} segments...")
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    
    # Translation Log Container
    log_box = st.expander("📝 Translation Logs (Live)", expanded=True)
    
    success_count = 0
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Calculate Duration
        if i < len(segments) - 1:
            max_dur = segments[i+1]['start'] - start
        else:
            max_dur = end - start + 2.0
            
        # Translate
        text = translate_content(model, seg['text'], style)
        
        # LOGGING (Crucial for debugging)
        if text:
            with log_box: st.write(f"✅ **Seg {i+1}:** {text}")
        else:
            with log_box: st.markdown(f":red[❌ **Seg {i+1} Failed:** AI returned empty.]")
        
        if text:
            raw = f"raw_{i}.mp3"
            if generate_audio(text, voice_config, raw):
                # Simple stretch to fit time (Prevents overlap)
                curr_len = get_duration(raw)
                if curr_len > 0:
                    speed = max(0.5, min(curr_len / max_dur, 1.8)) # Limit speed
                    proc = f"proc_{i}.mp3"
                    
                    subprocess.run(['ffmpeg', '-y', '-i', raw, '-filter:a', f"atempo={speed}", proc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(proc):
                        seg_audio = AudioSegment.from_file(proc)
                        final_audio = final_audio.overlay(seg_audio, position=start * 1000)
                        success_count += 1
                        try: os.remove(proc)
                        except: pass
                try: os.remove(raw)
                except: pass
        
        time.sleep(1) # Rate limit protection
        progress.progress((i + 1) / len(segments))

    if success_count == 0:
        status.error("❌ No audio was generated. Please use 'gemini-1.5-flash' model.")
        return None

    # 5. Final Mix
    status.info("🔊 Step 4: Mixing...")
    final_audio.export("voice.mp3", format="mp3")
    output_file = f"dubbed_{int(time.time())}.mp4"
    
    if mix_bg:
        vol = bg_vol / 100.0
        # Ducking: Original audio lowered, Voice boosted
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-filter_complex', f'[0:a]volume={vol}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
    else:
        # Replace Audio
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
        
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

# ---------------------------------------------------------
# 🖥️ UI MAIN
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    # 🔥 FORCED MODEL SELECTION (To prevent errors)
    st.markdown("### 🤖 Model Settings")
    model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"])
    st.caption("⚠️ Use 'gemini-1.5-flash' for best results.")
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("🔑 Please enter your API Key first.")
    st.stop()

# TABS
t1, t2 = st.tabs(["🎙️ Video Dubbing", "📝 Tools"])

with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    if uploaded:
        with open("temp.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
    c1, c2, c3 = st.columns(3)
    with c1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Style", ["Normal", "Movie Recap", "News", "Deep"])
    with c3: 
        mix_bg = st.checkbox("Mix Original?", value=True)
        bg_vol = st.slider("BG Vol", 0, 50, 10) if mix_bg else 0

    # Config
    base = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
    if tone == "Movie Recap": conf = {"id": base, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": conf = {"id": base, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": conf = {"id": base, "rate": "-5%", "pitch": "-20Hz"}
    else: conf = {"id": base, "rate": "+0%", "pitch": "+0Hz"}

    if st.button("🚀 Start Dubbing") and uploaded:
        st.session_state.processed_video = None
        status_msg = st.empty()
        pg = st.progress(0)
        try:
            out = process_video_dubbing("temp.mp4", conf, tone, mix_bg, bg_vol, st.session_state.api_key, model_name, status_msg, pg)
            if out:
                st.session_state.processed_video = out
                status_msg.success("✅ Dubbing Complete!")
                st.rerun()
        except Exception as e:
            status_msg.error(f"System Error: {e}")

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("💾 Download Video", f, "dubbed.mp4")

with t2:
    st.info("Additional tools available in next update.")
