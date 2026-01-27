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
        background: linear-gradient(90deg, #00C9FF, #92FE9D); 
        color: black; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .eng-text { color: #888; font-size: 12px; font-style: italic; }
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
        st.error("❌ FFmpeg is missing.")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (CLI)
# ---------------------------------------------------------
def generate_audio_cli(text, voice, rate, pitch, output_file):
    try:
        cmd = [
            "edge-tts",
            "--voice", voice,
            "--text", text,
            "--rate", rate,
            "--pitch", pitch,
            "--write-media", output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 100
    except: return False

# ---------------------------------------------------------
# 🛡️ EDUCATIONAL TRANSLATION ENGINE
# ---------------------------------------------------------
def clean_burmese(text):
    replacements = {"No.": "နံပါတ် ", "kg": " ကီလို ", "cm": " စင်တီမီတာ ", "$": " ဒေါ်လာ "}
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_educational(model, text, style):
    # Skip noise
    if len(text.strip()) < 2 or text.strip() in ["[Music]", "[Applause]", "[Silence]"]:
        return "", "Skipped Noise"

    # STRATEGY: EDUCATIONAL CONTEXT (Better for Motivation)
    prompt = f"""
    Role: Professional Linguistic Translator.
    Task: Translate this motivational/educational text to Burmese (Myanmar) for a database.
    
    Source Text: "{text}"
    
    Instruction: 
    1. Translate the meaning accurately. 
    2. If there are metaphors (e.g., 'kill the boy'), translate the *meaning* (e.g., 'destroy the weakness'), do not be literal if it sounds violent.
    3. Tone: {style}.
    4. Output: Burmese ONLY.
    """
    
    safety = [
        { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" },
    ]

    for attempt in range(3):
        try:
            res = model.generate_content(prompt, safety_settings=safety)
            if res.text:
                clean = clean_burmese(res.text.strip())
                if clean: return clean, "Success"
        except Exception:
            time.sleep(1)
            
    return "ဆက်လက်ကြိုးစားပါ။", "Fallback" # Generic motivational filler

# ---------------------------------------------------------
# 🎬 MAIN PROCESS
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style, mix_bg, bg_vol, api_key, model_name, status, progress):
    check_requirements()
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
    except:
        model = genai.GenerativeModel("gemini-1.5-flash")

    # Extract
    status.info("🎧 Extracting...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Whisper
    status.info("🧠 Transcribing...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # Dubbing
    status.info(f"🎙️ Dubbing {len(segments)} segments...")
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    
    log_box = st.expander("📝 Live Translation Logs", expanded=True)
    
    success_cnt = 0
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        eng_text = seg['text']
        
        # Calculate strict time slot
        if i < len(segments) - 1:
            max_dur = segments[i+1]['start'] - start
        else:
            max_dur = end - start + 2.0
            
        # Translate
        burmese, status_msg = translate_educational(model, eng_text, style)
        
        # LOGGING (With English Source)
        if status_msg == "Success":
            with log_box: 
                st.markdown(f"**Seg {i+1}:** <span class='eng-text'>{eng_text}</span><br>✅ {burmese}", unsafe_allow_html=True)
                st.divider()
        elif status_msg == "Skipped Noise":
            with log_box: st.write(f"🔇 Seg {i+1}: Noise Skipped")
        else:
            with log_box: 
                st.markdown(f"**Seg {i+1}:** <span class='eng-text'>{eng_text}</span><br>⚠️ Blocked -> Using Filler", unsafe_allow_html=True)
        
        if burmese:
            raw = f"raw_{i}.mp3"
            if generate_audio_cli(burmese, voice_config['id'], voice_config['rate'], voice_config['pitch'], raw):
                curr_len = get_duration(raw)
                if curr_len > 0:
                    speed = max(0.5, min(curr_len / max_dur, 1.8))
                    proc = f"proc_{i}.mp3"
                    subprocess.run(['ffmpeg', '-y', '-i', raw, '-filter:a', f"atempo={speed}", proc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(proc):
                        seg_audio = AudioSegment.from_file(proc)
                        final_audio = final_audio.overlay(seg_audio, position=start * 1000)
                        success_cnt += 1
                        try: os.remove(proc)
                        except: pass
                try: os.remove(raw)
                except: pass
        
        time.sleep(0.5)
        progress.progress((i + 1) / len(segments))

    # Mix
    status.info("🔊 Mixing...")
    final_audio.export("voice.mp3", format="mp3")
    output_file = f"dubbed_{int(time.time())}.mp4"
    
    if mix_bg:
        vol = bg_vol / 100.0
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-filter_complex', f'[0:a]volume={vol}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice.mp3',
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
        
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

# ---------------------------------------------------------
# 🖥️ UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    
    st.subheader("🤖 Model")
    model_name = st.text_input("Model ID", value="gemini-1.5-flash") # 1.5 is safer for motivation
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

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
                status_msg.success("✅ Done!")
                st.rerun()
        except Exception as e:
            status_msg.error(f"Error: {e}")

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("💾 Download", f, "dubbed.mp4")

with t2: st.info("Tools coming soon")
