import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
from PIL import Image

# ---------------------------------------------------------
# 🎨 UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Studio Pro", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    header[data-testid="stHeader"] { visibility: hidden; }
    header[data-testid="stHeader"]:hover { visibility: visible; }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white; border-radius: 8px; height: 45px; border: none; font-weight: bold; width: 100%;
    }
    .stButton>button:hover { transform: scale(1.02); }
    .progress-box {
        padding: 20px; border-radius: 10px; background-color: #111; border: 1px solid #00e5ff;
        text-align: center; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 SESSION STATE
# ---------------------------------------------------------
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'current_task' not in st.session_state: st.session_state.current_task = ""
if 'api_key_storage' not in st.session_state: st.session_state.api_key_storage = ""
if 'last_error' not in st.session_state: st.session_state.last_error = None # 🔥 ERROR သိမ်းမည့်နေရာ

# Results
if 'dub_result' not in st.session_state: st.session_state.dub_result = None
if 'viral_result' not in st.session_state: st.session_state.viral_result = None
if 'script_result' not in st.session_state: st.session_state.script_result = None
if 'thumb_img' not in st.session_state: st.session_state.thumb_img = None
if 'thumb_prompt' not in st.session_state: st.session_state.thumb_prompt = None

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        return 0

def extract_frame(video_path, output_image):
    try:
        duration = get_duration(video_path)
        mid_point = duration / 2
        cmd = ['ffmpeg', '-y', '-ss', str(mid_point), '-i', video_path, '-vframes', '1', output_image]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

# ---------------------------------------------------------
# 🧠 ASYNC PROCESSING FUNCTIONS
# ---------------------------------------------------------
async def process_dubbing(video_path, gender, style, tone, progress_bar, status_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 

        status_text.write("📤 Uploading Video...")
        progress_bar.progress(10)
        video_file = genai.upload_file(video_path)
        
        start_time = time.time()
        while video_file.state.name == "PROCESSING":
            if time.time() - start_time > 300: raise Exception("Timeout: Video processing took too long.")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED": raise Exception("Gemini API failed to process the video.")

        status_text.write("🧠 Translating...")
        progress_bar.progress(40)
        prompt = f"Translate dialogue to Burmese. Style: {style}. STRICT: Spoken words only, NO timestamps."
        response = model.generate_content([video_file, prompt])
        burmese_text = response.text.strip()
        
        status_text.write(f"🎙️ Generating Audio ({tone})...")
        progress_bar.progress(70)
        voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
        pitch, rate = "+0Hz", "+0%"
        if tone == "Deep": pitch = "-15Hz"
        elif tone == "Fast": rate = "+15%"
        
        audio_path = "temp_audio.mp3"
        communicate = edge_tts.Communicate(burmese_text, voice, rate=rate, pitch=pitch)
        await communicate.save(audio_path)

        status_text.write("🎬 Rendering Final Video...")
        progress_bar.progress(90)
        output_video = "final_output.mp4"
        final_audio = "temp_sync.mp3"
        
        # Check FFmpeg here
        if not os.path.exists(audio_path): raise Exception("Audio generation failed.")

        vid_dur = get_duration(video_path)
        aud_dur = get_duration(audio_path)
        
        if vid_dur > 0 and aud_dur > 0:
            speed = max(0.6, min(aud_dur / vid_dur, 1.5))
            subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-filter:a', f"atempo={speed}", final_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            final_audio = audio_path

        cmd = ['ffmpeg', '-y', '-i', video_path, '-i', final_audio, '-vf', 'scale=-2:720', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_video]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        progress_bar.progress(100)
        return output_video, final_audio

    except Exception as e:
        raise e

def generate_viral_content(video_path, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING": time.sleep(1); video_file = genai.get_file(video_file.name)
    return model.generate_content([video_file, "Generate 3 Viral Captions & 15 Hashtags in Burmese."]).text

def generate_human_script(video_path, format_type, api_key, progress_bar, status_text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    status_text.write("📤 Uploading...")
    progress_bar.progress(20)
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
    status_text.write("✍️ Writing...")
    progress_bar.progress(70)
    res = model.generate_content([video_file, f"Write a {format_type} in Burmese."])
    progress_bar.progress(100)
    return res.text

def generate_thumbnail_idea(video_path, api_key):
    extract_frame(video_path, "thumb.jpg")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    img = Image.open("thumb.jpg")
    return model.generate_content([img, "Describe a thumbnail prompt."]).text, "thumb.jpg"

# ---------------------------------------------------------
# ⚙️ SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    if not st.session_state.api_key_storage:
        key_input = st.text_input("🔑 Enter Gemini API Key", type="password")
        if key_input: st.session_state.api_key_storage = key_input; st.rerun()
    else:
        st.text_input("✅ API Key Active", value="******", disabled=True)
        if st.button("🔄 Reset Key"): st.session_state.api_key_storage = ""; st.rerun()
    st.markdown("---")
    if st.button("⚠️ Force Unlock"): st.session_state.is_processing = False
