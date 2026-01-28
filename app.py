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
import shutil
import whisper
from pydub import AudioSegment
import time
import json

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
    .info-box { padding: 15px; background: #222; border: 1px solid #444; border-radius: 8px; margin: 10px 0; }
    .sync-badge { background-color: #007bff; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'transcript' not in st.session_state: st.session_state.transcript = ""
if 'burmese_script' not in st.session_state: st.session_state.burmese_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'audio_path' not in st.session_state: st.session_state.audio_path = ""

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
# 🧠 AI ENGINE (SAFETY BYPASS)
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    return genai.GenerativeModel(model_name, safety_settings=safety_settings)

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

def generate_rewrite(model, text, style, custom_prompt):
    prompt = f"""
    Role: Professional Movie Recap Scriptwriter.
    Task: Rewrite/Translate the English transcript into Burmese.
    Input: "{text}"
    Style: {style}
    User Instruction: {custom_prompt}
    Rules: Burmese ONLY. No Censorship (Fictional Context).
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except: return "Error: AI blocked the response. Please write manually."

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    model_name = st.text_input("Model ID", value="gemini-1.5-flash")
    
    if st.button("🔴 Reset App"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

# --- TABS ---
t1, t2 = st.tabs(["🛠️ Manual Studio (Best)", "🚀 Auto Mode"])

with t1:
    st.subheader("Step 1: Upload & Extract")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("input.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        
        if st.button("📝 Extract Transcript"):
            with st.spinner("Processing..."):
                check_requirements()
                # Audio extraction for Whisper
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                text = transcribe_video("temp.wav")
                st.session_state.transcript = text
                st.success("Extracted!")
                st.rerun()

    if st.session_state.transcript:
        st.subheader("Step 2: Translate & Edit")
        en_text = st.text_area("English", st.session_state.transcript, height=100)
        
        c1, c2 = st.columns(2)
        with c1: style = st.selectbox("Style", ["Movie Recap", "News", "Funny"])
        with c2: custom = st.text_input("Instructions", "Exciting tone")
        
        if st.button("🤖 Generate Burmese Script"):
            with st.spinner("Writing..."):
                model = get_model(st.session_state.api_key, model_name)
                res = generate_rewrite(model, en_text, style, custom)
                st.session_state.burmese_script = res
                st.rerun()

    if st.session_state.burmese_script:
        st.subheader("Step 3: Audio Generation")
        final_script = st.text_area("Burmese Script (Edit here)", st.session_state.burmese_script, height=250)
        
        vc1, vc2 = st.columns(2)
        with vc1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
        
        if st.button("🔊 Generate Base Audio"):
            with st.spinner("Creating Audio..."):
                v_id = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
                clean_text = final_script.replace("\n", " ").strip()
                if generate_audio_cli(clean_text, v_id, "+0%", "+0Hz", "base_voice.mp3"):
                    st.session_state.audio_path = "base_voice.mp3"
                    st.success("Audio Generated!")
                    st.rerun()
                else:
                    st.error("TTS Failed")

    # --- STEP 4: SYNC MASTER (NEW FEATURE) ---
    if st.session_state.audio_path and os.path.exists("base_voice.mp3") and os.path.exists("input.mp4"):
        st.subheader("Step 4: Sync & Merge (Video/Audio Matching)")
        
        # Calculate Durations
        vid_dur = get_duration("input.mp4")
        aud_dur = get_duration("base_voice.mp3")
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("🎥 Video Duration", f"{vid_dur:.2f}s")
        col2.metric("🎙️ Audio Duration", f"{aud_dur:.2f}s")
        diff = aud_dur - vid_dur
        col3.metric("Difference", f"{diff:.2f}s", delta_color="inverse" if diff > 0 else "normal")

        st.markdown("---")
        st.markdown("#### 🎛️ Speed Controls")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Video Speed** (Change Video Length)")
            # Slider: 0.5x (Slow) to 2.0x (Fast)
            # If Audio is longer (e.g. 120s) and Video is (100s), we need to SLOW video (0.8x) to stretch it to 120s.
            suggested_vid_speed = 1.0
            if aud_dur > vid_dur:
                suggested_vid_speed = vid_dur / aud_dur # e.g. 100/120 = 0.83
            
            vid_speed = st.slider("Video Speed Multiplier", 0.5, 2.0, float(round(suggested_vid_speed, 2)), 0.05)
            st.caption(f"Less than 1.0 = Slow Motion (Stretches Video). Greater than 1.0 = Fast Forward.")

        with sc2:
            st.markdown("**Audio Speed** (Change Audio Length)")
            # Slider
            aud_speed = st.slider("Audio Speed Multiplier", 0.5, 2.0, 1.0, 0.05)
            st.caption(f"Greater than 1.0 = Faster Speaking. Less than 1.0 = Slower Speaking.")

        st.markdown("---")
        st.markdown("#### 🎚️ Mixing")
        mix_orig = st.checkbox("Keep Original Background Audio?", value=True)
        bg_vol = st.slider("Background Volume", 0, 100, 10)

        if st.button("🎬 Render Final Video"):
            with st.spinner("Rendering..."):
                # 1. Process Audio Speed
                final_audio_file = "final_audio.mp3"
                subprocess.run(['ffmpeg', '-y', '-i', "base_voice.mp3", '-filter:a', f"atempo={aud_speed}", final_audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 2. Process Video Speed
                # PTS filter: setpts=(1/SPEED)*PTS
                final_video_input = "input.mp4"
                
                # 3. Final Command Construction
                outfile = f"final_render_{int(time.time())}.mp4"
                
                # Video Filter Chain (Speed)
                # setpts=PTS/vid_speed
                v_filter = f"setpts=PTS/{vid_speed}"
                
                if mix_orig:
                    # Complex Filter for Mixing + Video Speed
                    vol_factor = bg_vol / 100.0
                    # [0:v]setpts...[v]; [0:a]volume..[bg]; [1:a]...[fg]; [bg][fg]amix...
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', "input.mp4", 
                        '-i', final_audio_file,
                        '-filter_complex', f'[0:v]{v_filter}[v];[0:a]volume={vol_factor}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=longest[aout]',
                        '-map', '[v]', '-map', '[aout]',
                        '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac',
                        outfile
                    ]
                else:
                    # Video Speed + Replace Audio
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', "input.mp4",
                        '-i', final_audio_file,
                        '-filter_complex', f'[0:v]{v_filter}[v]',
                        '-map', '[v]', '-map', '1:a',
                        '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac',
                        outfile
                    ]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(outfile):
                    st.success("✅ Rendering Complete!")
                    st.video(outfile)
                    with open(outfile, "rb") as f:
                        st.download_button("Download Video", f, "myanmar_dub.mp4")
                else:
                    st.error("Rendering Failed.")

with t2:
    st.info("Please use the Manual Studio (Tab 1) for full control.")
