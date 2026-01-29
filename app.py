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
import re
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
    textarea { font-size: 1.1rem !important; }
    .viral-box { background: #111; padding: 15px; border-left: 4px solid #00ff00; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'transcript' not in st.session_state: st.session_state.transcript = ""
if 'burmese_script' not in st.session_state: st.session_state.burmese_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'audio_path' not in st.session_state: st.session_state.audio_path = ""
if 'seo_result' not in st.session_state: st.session_state.seo_result = ""

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
# 🧠 AI ENGINE (H-V-C & Viral Kit)
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
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e:
        return f"Error during transcription: {e}"

def generate_hvc_script(model, title, text, style, custom_prompt):
    prompt = f"""
    Write a script outline for a video titled '{title}'.
    Context: {text}
    Style: {style}
    
    Use the 'H-V-C' structure:
    Hook (0-30s): A visual or verbal pattern interrupt to grab attention immediately.
    Value (Body): 3 distinct value points, with a 're-engagement' moment every 60 seconds.
    Call (End): A specific call to action related to this topic.
    
    Additional Instructions: {custom_prompt}
    
    CRITICAL: Translate the final output into spoken Burmese (Myanmar).
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"AI Error: {e}"

def generate_viral_metadata(model, title, keywords):
    prompt = f"""
    Write a video description for '{title}' optimized for SEO.
    Include:
    1. A natural first paragraph incorporating keywords: {keywords}.
    2. A bulleted list of what the viewer will learn.
    3. Timestamps for 5 key chapters.
    4. A list of 15 relevant, high-traffic tags separated by commas.
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"AI Error: {e}"

# ---------------------------------------------------------
# ❄️ FREEZE FRAME ENGINE (FIXED)
# ---------------------------------------------------------
def process_freeze_command(command, input_video, output_video):
    """
    Parses [freeze 10,5] -> Freezes at 10s for 5s duration.
    """
    try:
        match = re.search(r'freeze\s*(\d+),(\d+)', command, re.IGNORECASE)
        if match:
            time_point = match.group(1)
            duration = match.group(2)
            
            # 1. Cut Part A
            subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(time_point), '-c', 'copy', 'part_a.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 2. Freeze Frame
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-vframes', '1', 'freeze.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'freeze.jpg', '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'part_freeze.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 3. Cut Part B
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-c', 'copy', 'part_b.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 4. Concat
            with open("list.txt", "w") as f:
                f.write("file 'part_a.mp4'\nfile 'part_freeze.mp4'\nfile 'part_b.mp4'")
            
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Cleanup
            for f in ['part_a.mp4', 'part_b.mp4', 'part_freeze.mp4', 'freeze.jpg', 'list.txt']:
                if os.path.exists(f): os.remove(f)
                
            return True
        return False
    except Exception as e:
        print(f"Freeze Error: {e}")
        return False

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    model_name = st.text_input("Model ID", value="gemini-1.5-flash")
    if st.button("🔴 Reset"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2 = st.tabs(["🎬 Production Studio", "🚀 Viral Kit (SEO)"])

# === TAB 1: PRODUCTION ===
with t1:
    st.subheader("Step 1: Upload & Script")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("📝 Extract Transcript"):
            with st.spinner("Processing..."):
                check_requirements()
                # Extract Audio
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Transcribe
                text = transcribe_video("temp.wav")
                st.session_state.transcript = text
                st.rerun()

    if st.session_state.transcript:
        en_text = st.text_area("English Transcript", st.session_state.transcript, height=100)
        c1, c2 = st.columns(2)
        with c1: style = st.selectbox("Style", ["Movie Recap", "News", "Funny"])
        with c2: custom = st.text_input("Extra Instructions", "Make it exciting")
        
        if st.button("🤖 Generate H-V-C Script"):
            with st.spinner("Writing Script..."):
                model = get_model(st.session_state.api_key, model_name)
                # Using Video Filename as Title approximation or Generic
                res = generate_hvc_script(model, "My Video", en_text, style, custom)
                st.session_state.burmese_script = res
                st.rerun()

    if st.session_state.burmese_script:
        st.subheader("Step 2: Edit & Freeze")
        final_script = st.text_area("Burmese Script (H-V-C)", st.session_state.burmese_script, height=250)
        
        # 🔥 FREEZE CONTROL
        st.markdown("#### ❄️ Freeze Control")
        st.caption("Example: `freeze 10,5` (Freeze at 10s for 5s)")
        freeze_cmd = st.text_input("Freeze Command (Optional)")
        
        vc1, vc2 = st.columns(2)
        with vc1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
        
        if st.button("🔊 Generate Audio & Sync"):
            with st.spinner("Processing..."):
                v_id = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
                
                # 1. Handle Freeze
                input_video_file = "input.mp4"
                if freeze_cmd:
                    if process_freeze_command(freeze_cmd, "input.mp4", "frozen_input.mp4"):
                        input_video_file = "frozen_input.mp4"
                        st.success(f"Video Frozen with: {freeze_cmd}")
                
                # 2. Generate Audio
                clean_text = final_script.replace("*", "").replace("#", "").strip() # Remove MD formatting
                generate_audio_cli(clean_text, v_id, "+0%", "+0Hz", "base_voice.mp3")
                st.session_state.audio_path = "base_voice.mp3"
                
                # 3. Sync Logic
                vid_dur = get_duration(input_video_file)
                aud_dur = get_duration("base_voice.mp3")
                
                # Auto-Speed logic to fit video
                speed_factor = 1.0
                if vid_dur > 0 and aud_dur > vid_dur:
                    speed_factor = aud_dur / vid_dur
                    speed_factor = min(speed_factor, 1.5) 
                
                subprocess.run(['ffmpeg', '-y', '-i', "base_voice.mp3", '-filter:a', f"atempo={speed_factor}", "final_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Final Render
                outfile = f"final_{int(time.time())}.mp4"
                cmd = [
                    'ffmpeg', '-y', '-i', input_video_file, '-i', "final_audio.mp3",
                    '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                st.success("✅ Video Ready!")
                st.video(outfile)
                with open(outfile, "rb") as f:
                    st.download_button("Download", f, "final.mp4")

# === TAB 2: VIRAL KIT ===
with t2:
    st.subheader("🚀 Viral Kit & SEO")
    v_title = st.text_input("Video Title")
    v_keywords = st.text_input("Keywords (comma separated)")
    
    if st.button("✨ Generate Viral Metadata"):
        if not v_title: st.error("Add Title")
        else:
            with st.spinner("Analyzing SEO..."):
                model = get_model(st.session_state.api_key, model_name)
                res = generate_viral_metadata(model, v_title, v_keywords)
                st.session_state.seo_result = res
                st.rerun()
    
    if st.session_state.seo_result:
        st.markdown("<div class='viral-box'>", unsafe_allow_html=True)
        st.markdown(st.session_state.seo_result)
        st.markdown("</div>", unsafe_allow_html=True)
