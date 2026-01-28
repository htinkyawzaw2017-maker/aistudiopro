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
    textarea { font-size: 1.2rem !important; line-height: 1.5 !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'transcript' not in st.session_state: st.session_state.transcript = ""
if 'burmese_script' not in st.session_state: st.session_state.burmese_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing.")
        st.stop()

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE (CLI - STABLE)
# ---------------------------------------------------------
def generate_audio_cli(text, voice, rate, pitch, output_file):
    try:
        # Edge-TTS often fails with very long text chunks, so we split by newlines if needed
        # But for CLI, we pass the whole text file usually. Here we pass string.
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
    # 🔥 DISABLE SAFETY FILTERS COMPLETELY
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
    Role: Professional Movie Recap Scriptwriter (Burmese).
    Task: Rewrite/Translate the following English transcript into an engaging Burmese story.
    
    Input Transcript: "{text}"
    
    Style: {style}
    User Instruction: {custom_prompt}
    
    CRITICAL RULES:
    1. Output Burmese ONLY.
    2. Even if the story contains violence (fighting, survival, death), DO NOT CENSOR it. This is a fictional movie recap.
    3. Use natural, storytelling Burmese (e.g., "ဒီလိုနဲ့...", "နောက်ဆုံးမှာတော့...").
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    model_name = st.text_input("Model ID", value="gemini-1.5-flash")
    st.caption("Recommended: gemini-1.5-flash (Highest Rate Limit)")
    
    if st.button("🔴 Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

if not st.session_state.api_key:
    st.warning("Enter API Key")
    st.stop()

# --- TABS ---
t1, t2 = st.tabs(["📝 Script & Voice (Manual)", "🚀 Auto Dubbing (Classic)"])

# === TAB 1: NEW MANUAL WORKFLOW (BEST FOR BLOCKED CONTENT) ===
with t1:
    st.subheader("🛠️ Video to Script to Audio (Safe Mode)")
    st.info("ဒီအကွက်က အကောင်းဆုံးပါ။ Video ကို စာအရင်ထုတ်၊ ပြီးမှ AI ကို ဘာသာပြန်ခိုင်း၊ ပြီးမှ ကိုယ်တိုင်စစ်ပြီး အသံထုတ်မှာမို့ Error ကင်းပါတယ်။")
    
    uploaded_file = st.file_uploader("1. Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("input.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        
        # Step 1: Transcribe
        if st.button("2. Extract English Text"):
            with st.spinner("Listening to video..."):
                check_requirements()
                # Extract Audio first
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Whisper
                text = transcribe_video("temp.wav")
                st.session_state.transcript = text
                st.success("Extraction Complete!")
                st.rerun()

    if st.session_state.transcript:
        st.subheader("3. English Transcript")
        en_text = st.text_area("Original Text", st.session_state.transcript, height=150)
        
        st.subheader("4. Translate to Burmese")
        c1, c2 = st.columns(2)
        with c1: style = st.selectbox("Style", ["Movie Recap (Dramatic)", "News (Formal)", "Funny", "Horror"])
        with c2: custom_prompt = st.text_input("Custom Instructions", "Make it exciting, use informal Burmese.")
        
        if st.button("Translate / Rewrite"):
            with st.spinner("AI Writing..."):
                model = get_model(st.session_state.api_key, model_name)
                res = generate_rewrite(model, en_text, style, custom_prompt)
                st.session_state.burmese_script = res
                st.rerun()

    if st.session_state.burmese_script:
        st.subheader("5. Final Burmese Script (Editable)")
        # USER CAN EDIT THIS if AI fails or makes mistakes
        final_script = st.text_area("Edit Script Here (AI Blocked? Write manually here!)", st.session_state.burmese_script, height=300)
        
        st.subheader("6. Generate Audio")
        vc1, vc2 = st.columns(2)
        with vc1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
        with vc2: 
            mix_orig = st.checkbox("Mix with Original Background?", value=True)
            bg_vol = st.slider("BG Volume", 0, 50, 10)

        if st.button("🚀 Create Final Video"):
            with st.spinner("Generating Audio & Mixing..."):
                # Config
                v_id = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
                rate = "+0%" if "Male" in voice else "+10%"
                pitch = "-5Hz" if "Male" in voice else "+5Hz"
                
                # Generate Full Audio from Script
                # Cleaning text for TTS
                clean_text = final_script.replace("\n", " ").strip()
                if generate_audio_cli(clean_text, v_id, rate, pitch, "voice_full.mp3"):
                    
                    # Get Durations
                    video_dur = 0
                    try:
                        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', "input.mp4"]
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        video_dur = float(json.loads(r.stdout)['format']['duration'])
                    except: pass
                    
                    audio_dur = 0
                    try:
                        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', "voice_full.mp3"]
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        audio_dur = float(json.loads(r.stdout)['format']['duration'])
                    except: pass
                    
                    # Auto-Speed Adjust: If audio is longer than video, speed it up
                    speed_factor = 1.0
                    if video_dur > 0 and audio_dur > video_dur:
                        speed_factor = audio_dur / video_dur
                        # Cap at 1.5x to prevent gibberish
                        speed_factor = min(speed_factor, 1.5)
                        
                    # Process Audio Speed
                    subprocess.run(['ffmpeg', '-y', '-i', "voice_full.mp3", '-filter:a', f"atempo={speed_factor}", "voice_sync.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Final Mix
                    outfile = f"final_{int(time.time())}.mp4"
                    if mix_orig:
                        vol = bg_vol / 100.0
                        cmd = [
                            'ffmpeg', '-y', '-i', "input.mp4", '-i', "voice_sync.mp3",
                            '-filter_complex', f'[0:a]volume={vol}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=shortest[aout]',
                            '-map', '0:v', '-map', '[aout]',
                            '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile
                        ]
                    else:
                        cmd = [
                            'ffmpeg', '-y', '-i', "input.mp4", '-i', "voice_sync.mp3",
                            '-map', '0:v', '-map', '1:a',
                            '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile
                        ]
                    
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    st.success("✅ Video Created!")
                    st.video(outfile)
                    with open(outfile, "rb") as f:
                        st.download_button("Download Video", f, "myanmar_dub.mp4")
                else:
                    st.error("Audio Generation Failed.")

# === TAB 2: OLD AUTO DUBBING (Simpler but prone to blocks) ===
with t2:
    st.warning("This mode translates segment-by-segment. If one segment is blocked, it might skip. Use 'Script & Voice' tab for 100% control.")
    # (Leaving this blank or simple as user prefers the new manual control)
    st.info("Please use the first tab for complex/violent movie recaps.")
