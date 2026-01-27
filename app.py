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
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'generated_script' not in st.session_state: st.session_state.generated_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

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
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444; background: #222; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please install FFmpeg."); st.stop()

def get_audio_duration_file(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 ROBUST AUDIO ENGINE (LIBRARY METHOD)
# ---------------------------------------------------------
async def tts_generation_async(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_config, filename):
    try:
        # Create a FRESH loop for every call to avoid Cloud concurrency issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_generation_async(text, voice_config['id'], voice_config['rate'], voice_config['pitch'], filename))
        loop.close()
        
        # Verify file
        if os.path.exists(filename) and os.path.getsize(filename) > 100:
            return True
        return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def fit_audio_to_time(input_file, output_file, target_duration):
    # Smart Sync: Stretch audio to fit the segment time exactly
    curr = get_audio_duration_file(input_file)
    if curr == 0: return False
    
    speed = curr / target_duration
    # Clamp speed to avoid crazy robot sounds (0.5x to 2.0x)
    speed = max(0.5, min(speed, 2.0))
    
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_file, '-filter:a', f"atempo={speed}", output_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

# ---------------------------------------------------------
# 🛡️ TRANSLATION ENGINE
# ---------------------------------------------------------
def clean_burmese(text):
    # Fix Units
    replacements = {"No.": "နံပါတ် ", "kg": " ကီလို ", "cm": " စင်တီမီတာ ", "mm": " မီလီမီတာ ", "%": " ရာခိုင်နှုန်း ", "$": " ဒေါ်လာ "}
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    # Remove English
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_content(model, text, style):
    prompt = f"""
    Translate English to Spoken Burmese.
    Input: "{text}"
    Rules: Burmese ONLY. No English. Numbers to words. Tone: {style}.
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
# 🎬 VIDEO PIPELINE
# ---------------------------------------------------------
def process_video_pipeline(video_path, voice_config, style, mix_bg, bg_vol, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract
    status.info("🎧 Step 1: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper
    status.info("🧠 Step 2: Speech Recognition...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Dubbing Loop
    status.info(f"🎙️ Step 3: Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    total_dur = get_audio_duration_file(video_path)
    final_audio = AudioSegment.silent(duration=total_dur * 1000)
    
    # Translation Log
    log_box = st.expander("Show Translation Log", expanded=False)
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Calculate allowed duration
        if i < len(segments) - 1:
            max_dur = segments[i+1]['start'] - start
        else:
            max_dur = end - start + 2.0
            
        # Translate
        text = translate_content(model, seg['text'], style)
        with log_box: st.text(f"{i+1}: {text}")
        
        if text:
            raw = f"raw_{i}.mp3"
            if generate_audio(text, voice_config, raw):
                proc = f"proc_{i}.mp3"
                # Time Stretch to prevent overlapping
                if fit_audio_to_time(raw, proc, max_dur) and os.path.exists(proc):
                    seg_audio = AudioSegment.from_file(proc)
                    final_audio = final_audio.overlay(seg_audio, position=start * 1000)
                    try: os.remove(proc)
                    except: pass
                else:
                    # Fallback to raw if ffmpeg failed
                    seg_audio = AudioSegment.from_file(raw)
                    final_audio = final_audio.overlay(seg_audio, position=start * 1000)
                try: os.remove(raw)
                except: pass
        
        time.sleep(1)
        progress.progress((i + 1) / len(segments))

    # 4. Mixing
    status.info("🔊 Step 4: Mixing...")
    final_audio.export("voice.mp3", format="mp3")
    output_file = f"dubbed_{int(time.time())}.mp4"
    
    # Mix Logic
    if mix_bg:
        vol = bg_vol / 100.0
        # Complex filter to mix original audio (bg) with new voice (fg)
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-filter_complex', f'[0:a]volume={vol}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
    else:
        # Replace audio completely
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', output_file
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

def script_audio_gen(text, conf):
    clean = clean_burmese(text)
    chunks = [clean[i:i+500] for i in range(0, len(clean), 500)]
    comb = AudioSegment.empty()
    for i, c in enumerate(chunks):
        f = f"c_{i}.mp3"
        if generate_audio(c, conf, f):
            comb += AudioSegment.from_file(f)
            os.remove(f)
    out = f"script_{int(time.time())}.mp3"
    comb.export(out, format="mp3")
    return out

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    
    # Custom Model Selection
    mode = st.radio("Model", ["Preset", "Custom Input"])
    if mode == "Preset":
        model_name = st.selectbox("Select", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    else:
        model_name = st.text_input("Model ID", "gemini-2.5-flash")
    
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
        bg_vol = st.slider("BG Vol", 0, 50, 10) if mix_bg else 0

    base = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
    if tone == "Movie Recap": conf = {"id": base, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": conf = {"id": base, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": conf = {"id": base, "rate": "-5%", "pitch": "-20Hz"}
    else: conf = {"id": base, "rate": "+0%", "pitch": "+0Hz"}

    if st.button("🚀 Start Dubbing") and uploaded:
        st.session_state.processed_video = None
        status_msg = st.empty() # Fixed Variable Name
        pg = st.progress(0)
        try:
            out = process_video_pipeline("temp.mp4", conf, tone, mix_bg, bg_vol, st.session_state.api_key, model_name, status_msg, pg)
            if out:
                st.session_state.processed_video = out
                status_msg.success("Done!")
                st.rerun()
        except Exception as e: status_msg.error(str(e))

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
        sc = st.text_area("Result", st.session_state.generated_script, height=300)
        if st.button("Read"):
            f = script_audio_gen(sc, conf)
            st.audio(f)

with t3: st.info("Viral Kit Ready")
with t4: st.info("Thumbnail Ready")
