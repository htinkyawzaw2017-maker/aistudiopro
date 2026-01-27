import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import google.generativeai as genai
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
    .success-box { padding: 10px; background: #004400; border: 1px solid #00ff00; border-radius: 5px; margin-bottom: 10px; }
    .error-box { padding: 10px; background: #440000; border: 1px solid #ff0000; border-radius: 5px; margin-bottom: 10px; }
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
        st.error("❌ FFmpeg is missing. Please install FFmpeg.")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (CLI - 100% STABLE)
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
        # Run process
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check if file exists and has size
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            return True
        else:
            return False
    except:
        return False

# ---------------------------------------------------------
# 🛡️ SMART TRANSLATION
# ---------------------------------------------------------
def clean_burmese(text):
    # Fix Units
    replacements = {
        "No.": "နံပါတ် ", "kg": " ကီလို ", "cm": " စင်တီမီတာ ", 
        "mm": " မီလီမီတာ ", "%": " ရာခိုင်နှုန်း ", "$": " ဒေါ်လာ "
    }
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    # Remove English (A-Z)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_safe(model, text, style):
    prompt = f"""
    Translate English to spoken Burmese.
    Input: "{text}"
    Rules: Burmese ONLY. No English. Numbers to words. Tone: {style}.
    """
    for _ in range(3):
        try:
            res = model.generate_content(prompt)
            clean = clean_burmese(res.text.strip())
            if clean: return clean
        except exceptions.ResourceExhausted:
            time.sleep(5) # Wait if limit hit
            continue
        except Exception:
            time.sleep(1)
            continue
    return ""

# ---------------------------------------------------------
# 🧪 API TESTER
# ---------------------------------------------------------
def test_api_connection(api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Test connection")
        return True, "Connection Successful!"
    except exceptions.NotFound:
        return False, f"Model '{model_name}' not found (404). Check spelling."
    except exceptions.ResourceExhausted:
        return False, f"Quota Exceeded (429). Try 'gemini-1.5-flash'."
    except Exception as e:
        return False, f"Error: {str(e)}"

# ---------------------------------------------------------
# 🎬 MAIN PROCESS
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style, mix_bg, bg_vol, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. API Check
    status.info(f"🔌 Connecting to {model_name}...")
    is_valid, msg = test_api_connection(api_key, model_name)
    if not is_valid:
        status.markdown(f"<div class='error-box'>❌ API Error: {msg}</div>", unsafe_allow_html=True)
        return None

    # 2. Extract
    status.info("🎧 Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Whisper
    status.info("🧠 Transcribing...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 4. Dubbing Loop
    status.info(f"🎙️ Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    log_box = st.expander("📝 Translation Logs", expanded=True)
    
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
        text = translate_safe(model, seg['text'], style)
        
        if text:
            with log_box: st.write(f"✅ {i+1}: {text}")
            raw = f"raw_{i}.mp3"
            
            # Generate Audio
            if generate_audio_cli(text, voice_config['id'], voice_config['rate'], voice_config['pitch'], raw):
                curr_len = get_duration(raw)
                if curr_len > 0:
                    # Smart Speed (Fit to slot)
                    speed = max(0.5, min(curr_len / max_dur, 1.8))
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
        else:
            with log_box: st.write(f"⚠️ {i+1}: Failed to translate")
        
        time.sleep(1) # Safety delay
        progress.progress((i + 1) / len(segments))

    if success_count == 0:
        status.error("❌ Audio generation failed completely.")
        return None

    # 5. Mix
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
# 🖥️ UI MAIN
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    
    # 🔥 MODEL SELECTION (Research-Based)
    st.markdown("### 🤖 Model Settings")
    model_mode = st.radio("Select Type", ["Free Tier (Recommended)", "Custom Input"])
    
    if model_mode == "Free Tier (Recommended)":
        model_name = st.selectbox(
            "Choose Model", 
            ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"]
        )
        st.caption("✅ 'gemini-1.5-flash' is the most stable.")
    else:
        model_name = st.text_input("Enter Model ID", value="gemini-2.5-flash")
        st.caption("⚠️ Warning: 'gemini-2.5' does not exist yet. Use at own risk.")

    # 🔌 API TESTER BUTTON
    if st.button("🔌 Test API Connection"):
        if not api_key:
            st.error("Please enter API Key!")
        else:
            with st.spinner("Testing..."):
                valid, msg = test_api_connection(api_key, model_name)
                if valid:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("🔑 Please enter API Key")
    st.stop()

# MAIN TABS
t1, t2 = st.tabs(["🎙️ Video Dubbing", "📝 Tools"])

with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    if uploaded:
        with open("temp.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
    c1, c2, c3 = st.columns(3)
    with c1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Tone", ["Normal", "Movie Recap", "News", "Deep"])
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
        
        # FINAL API CHECK
        valid, msg = test_api_connection(st.session_state.api_key, model_name)
        if not valid:
            status_msg.error(msg)
        else:
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

with t2:
    st.info("Additional tools available in full version.")
