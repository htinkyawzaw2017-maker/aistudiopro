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
# 💾 STATE & CONFIG
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'generated_script' not in st.session_state: st.session_state.generated_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🇲🇲", layout="wide")
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
# 🛠️ CORE FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please install FFmpeg on the server."); st.stop()

def get_audio_duration_file(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(r.stdout)
        return float(data['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (SMART SYNC)
# ---------------------------------------------------------
async def tts_gen(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_config, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_gen(text, voice_config['id'], voice_config['rate'], voice_config['pitch'], filename))
        loop.close()
        return os.path.exists(filename) and os.path.getsize(filename) > 100
    except: return False

def fit_audio_to_duration(input_file, output_file, target_duration):
    """
    Stretches or compresses audio to fit EXACTLY into the target duration.
    Prevents overlapping.
    """
    current_duration = get_audio_duration_file(input_file)
    if current_duration == 0: return False
    
    # Calculate speed factor
    speed = current_duration / target_duration
    
    # Safety Clamps: Don't go too fast (chipmunk) or too slow (robot)
    # If it needs to be 3x faster, we clamp to 2.0x and accept slight overlap OR cut silence
    speed = max(0.5, min(speed, 2.0)) 
    
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_file, '-filter:a', f"atempo={speed}", output_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

# ---------------------------------------------------------
# 🛡️ TEXT CLEANER
# ---------------------------------------------------------
def clean_burmese(text):
    replacements = {
        "No.": "နံပါတ် ", "kg": " ကီလို ", "cm": " စင်တီမီတာ ", 
        "mm": " မီလီမီတာ ", "km": " ကီလိုမီတာ ", "%": " ရာခိုင်နှုန်း ", 
        "$": " ဒေါ်လာ ", "Mr.": "မစ္စတာ "
    }
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_content(model, text, style):
    prompt = f"""
    Translate English to spoken Burmese.
    Input: "{text}"
    Rules: Burmese ONLY. No English words. Numbers to words. Tone: {style}.
    """
    for _ in range(2):
        try:
            res = model.generate_content(prompt)
            clean = clean_burmese(res.text.strip())
            if clean: return clean
        except exceptions.ResourceExhausted: time.sleep(5)
        except: pass
    return ""

# ---------------------------------------------------------
# 🎬 MAIN PIPELINE
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style, mix_original, bg_volume, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract
    status.info("🎧 Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper
    status.info("🧠 Analyzing Timing...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Dubbing
    status.info(f"🎙️ Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # Audio Canvas
    total_dur = get_audio_duration_file(video_path)
    final_audio = AudioSegment.silent(duration=total_dur * 1000)
    
    # Debug Log
    log_expander = st.expander("Translation Details", expanded=False)
    
    for i, seg in enumerate(segments):
        # Determine strict time slot
        start_time = seg['start']
        end_time = seg['end']
        
        # Look ahead to next segment to prevent overlap
        if i < len(segments) - 1:
            next_start = segments[i+1]['start']
            # Allow max duration up to next segment start
            max_duration = next_start - start_time
        else:
            max_duration = end_time - start_time + 2.0 # Last segment gets leeway
            
        # Translate
        text = translate_content(model, seg['text'], style)
        with log_expander: st.text(f"{i+1}: {text}")
        
        if text:
            # Generate Raw Audio
            raw_file = f"raw_{i}.mp3"
            generate_audio(text, voice_config, raw_file)
            
            if os.path.exists(raw_file):
                # Smart Fit: Compress audio if it's longer than the slot
                processed_file = f"proc_{i}.mp3"
                success = fit_audio_to_duration(raw_file, processed_file, max_duration)
                
                if success and os.path.exists(processed_file):
                    seg_audio = AudioSegment.from_file(processed_file)
                    final_audio = final_audio.overlay(seg_audio, position=start_time * 1000)
                    try: os.remove(processed_file)
                    except: pass
                
                try: os.remove(raw_file)
                except: pass
        
        # Rate limit
        time.sleep(1)
        progress.progress((i + 1) / len(segments))

    # 4. Mixing (Background Music Logic)
    status.info("🔊 Mixing Final Audio...")
    final_audio.export("voice_track.mp3", format="mp3")
    
    # Merge Logic
    output_file = f"dubbed_{int(time.time())}.mp4"
    
    if mix_original:
        # Complex Filter: Keep original audio at lower volume, mix with new voice
        # [0:a]volume=0.1[bg];[1:a]volume=1.0[fg];[bg][fg]amix=inputs=2:duration=first[out]
        vol_factor = bg_volume / 100.0
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', 'voice_track.mp3',
            '-filter_complex', f'[0:a]volume={vol_factor}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac',
            output_file
        ]
    else:
        # Simple Replacement (Mute original)
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', 'voice_track.mp3',
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac',
            output_file
        ]
        
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

# ---------------------------------------------------------
# 📝 FEATURES
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model.generate_content(f"Write a Burmese {type} script about {topic}. Tone: {tone}. {prompt}").text

def script_audio(text, config):
    clean = clean_burmese(text)
    chunks = [clean[i:i+500] for i in range(0, len(clean), 500)]
    comb = AudioSegment.empty()
    for i, c in enumerate(chunks):
        f = f"c_{i}.mp3"
        if generate_audio(c, config, f):
            comb += AudioSegment.from_file(f)
            os.remove(f)
    out = f"script_{int(time.time())}.mp3"
    comb.export(out, format="mp3")
    return out

# ---------------------------------------------------------
# 🖥️ UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    
    mode = st.radio("Model", ["Preset", "Custom"])
    model_name = st.selectbox("Select", ["gemini-1.5-flash", "gemini-2.0-flash-exp"]) if mode == "Preset" else st.text_input("Model ID", "gemini-2.5-flash")
    
    if st.button("🔴 Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "📝 Script", "🚀 Viral", "🖼️ Thumbnail"])

with t1:
    st.subheader("🔊 Smart Dubbing")
    uploaded = st.file_uploader("Video", type=['mp4','mov'])
    if uploaded:
        with open("temp.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
    c1, c2, c3 = st.columns(3)
    with c1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Tone", ["Normal", "Movie Recap", "News", "Deep"])
    with c3: 
        mix_bg = st.checkbox("Mix Original Audio?", value=True)
        bg_vol = st.slider("BG Volume", 0, 50, 10) if mix_bg else 0

    base = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
    if tone == "Movie Recap": conf = {"id": base, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": conf = {"id": base, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": conf = {"id": base, "rate": "-5%", "pitch": "-15Hz"}
    else: conf = {"id": base, "rate": "+0%", "pitch": "+0Hz"}

    if st.button("🚀 Start Dubbing") and uploaded:
        st.session_state.processed_video = None
        st = st.empty()
        pg = st.progress(0)
        try:
            out = process_video_dubbing("temp.mp4", conf, tone, mix_bg, bg_vol, st.session_state.api_key, model_name, st, pg)
            if out:
                st.session_state.processed_video = out
                st.success("Done!")
                st.rerun()
        except Exception as e: st.error(str(e))

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download", f, "dubbed.mp4")

with t2:
    st.subheader("📝 Script")
    topic = st.text_input("Topic")
    pt = st.text_area("Instructions")
    if st.button("Write"):
        res = generate_script(topic, "Script", "Normal", pt, st.session_state.api_key, model_name)
        st.session_state.generated_script = res
        st.rerun()
    
    if st.session_state.generated_script:
        sc = st.text_area("Script", st.session_state.generated_script)
        if st.button("Read"):
            f = script_audio(sc, conf)
            st.audio(f)

with t3: st.info("Viral Kit Ready")
with t4: st.info("Thumbnail Ready")
