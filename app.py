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
    textarea { font-size: 1.1rem !important; font-family: 'Padauk', sans-serif !important; }
    .viral-box { background: #111; padding: 15px; border-left: 4px solid #00ff00; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'raw_transcript' not in st.session_state: st.session_state.raw_transcript = ""
if 'burmese_draft' not in st.session_state: st.session_state.burmese_draft = ""
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'audio_path' not in st.session_state: st.session_state.audio_path = ""
if 'seo_result' not in st.session_state: st.session_state.seo_result = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None

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
# 🔊 AUDIO ENGINE (MULTI-LANG & MODES)
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
    "Japanese": {"Male": "ja-JP-KeitaNeural", "Female": "ja-JP-NanamiNeural"},
    "Chinese": {"Male": "zh-CN-YunxiNeural", "Female": "zh-CN-XiaoxiaoNeural"},
    "Thai": {"Male": "th-TH-NiwatNeural", "Female": "th-TH-PremwadeeNeural"},
    "Hindi": {"Male": "hi-IN-MadhurNeural", "Female": "hi-IN-SwaraNeural"}
}

VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-10%", "pitch": "-5Hz"},
    "Documentary": {"rate": "-5%", "pitch": "-10Hz"},
    "Recap": {"rate": "+10%", "pitch": "+0Hz"},
    "Motivation": {"rate": "-10%", "pitch": "-15Hz"},
    "Animation": {"rate": "+5%", "pitch": "+20Hz"}
}

def generate_audio_cli(text, lang, gender, mode_name, output_file):
    try:
        # Get Voice ID
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        
        # Get Mode Settings
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        rate = settings["rate"]
        pitch = settings["pitch"]

        cmd = [
            "edge-tts",
            "--voice", voice_id,
            "--text", text,
            "--rate", rate,
            "--pitch", pitch,
            "--write-media", output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 100
    except: return False

# ---------------------------------------------------------
# 🧠 AI ENGINE
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
        return f"Error: {e}"

def translate_to_burmese_draft(model, text, source_lang):
    prompt = f"""
    Task: Translate the following {source_lang} transcript to Burmese.
    Input Text: "{text}"
    
    CRITICAL RULES:
    1. Translate to natural Burmese.
    2. **DO NOT TRANSLATE PROPER NOUNS OR TECHNICAL TERMS.** Keep Names, Places, and specific Objects in English (e.g., "iPhone", "New York", "John").
    3. Output ONLY the translated text.
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"AI Error: {e}"

def refine_script_hvc(model, text, title, custom_prompt):
    prompt = f"""
    Act as a Professional Video Scriptwriter.
    Refine the following Burmese draft into a final script for a video titled '{title}'.
    
    Input Draft: "{text}"
    
    Structure Constraint: Use the **'H-V-C' (Hook-Value-Call)** structure.
    
    OUTPUT FORMAT RULE: 
    - **Output ONLY the spoken words (Host Voice) in Burmese.** - Do NOT include labels like "Hook:", "Scene 1:", etc. 
    - Just the raw text to be spoken.
    - Keep English nouns/names in English.
    
    Additional Instructions: {custom_prompt}
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
# ❄️ AUTO FREEZE LOOP ENGINE
# ---------------------------------------------------------
def apply_auto_freeze(input_video, output_video, interval_sec, freeze_duration=4.0):
    try:
        duration = get_duration(input_video)
        if duration == 0: return False

        # Create a list file for concat
        concat_list_path = "freeze_list.txt"
        
        with open(concat_list_path, "w") as f:
            current_time = 0
            part_idx = 0
            
            while current_time < duration:
                next_time = min(current_time + interval_sec, duration)
                seg_duration = next_time - current_time
                
                # Cut Segment
                part_name = f"part_{part_idx}.mp4"
                subprocess.run(['ffmpeg', '-y', '-ss', str(current_time), '-t', str(seg_duration), '-i', input_video, '-c', 'copy', part_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                f.write(f"file '{part_name}'\n")
                
                # Create Freeze Frame (only if not at the very end)
                if next_time < duration:
                    freeze_name = f"freeze_{part_idx}.mp4"
                    # Capture last frame of segment
                    subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', part_name, '-update', '1', '-q:v', '1', 'freeze_frame.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Loop it
                    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'freeze_frame.jpg', '-t', str(freeze_duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', freeze_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    f.write(f"file '{freeze_name}'\n")
                
                current_time = next_time
                part_idx += 1
                
        # Concat all parts
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        if os.path.exists("freeze_frame.jpg"): os.remove("freeze_frame.jpg")
        # Cleanup parts is messy in loop, can use glob in production, skipping for speed
        return True
    except Exception as e:
        print(f"Auto Freeze Error: {e}")
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
    # --- STEP 1: UPLOAD & TRANSLATE ---
    st.subheader("Step 1: Upload & Initial Translate")
    uploaded = st.file_uploader("Upload Video", type=['mp4','mov'])
    
    # Source Language Selection
    source_lang = st.selectbox("Original Video Language", ["English", "Japanese", "Chinese", "Thai", "Indian (Hindi)"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("📝 Extract & Translate to Burmese"):
            with st.spinner("Listening & Translating..."):
                check_requirements()
                # Extract Audio
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Transcribe
                raw_text = transcribe_video("temp.wav")
                st.session_state.raw_transcript = raw_text
                
                # Translate
                model = get_model(st.session_state.api_key, model_name)
                draft = translate_to_burmese_draft(model, raw_text, source_lang)
                st.session_state.burmese_draft = draft
                st.rerun()

    # --- STEP 2: REFINE SCRIPT ---
    if st.session_state.burmese_draft:
        st.subheader("Step 2: Script Refinement")
        draft_text = st.text_area("Burmese Draft", st.session_state.burmese_draft, height=150)
        
        custom_prompt = st.text_input("Script Instructions", "Make it exciting, H-V-C style")
        
        if st.button("✨ Convert to H-V-C Final Script"):
            with st.spinner("Applying Magic..."):
                model = get_model(st.session_state.api_key, model_name)
                final = refine_script_hvc(model, draft_text, "My Video", custom_prompt)
                st.session_state.final_script = final
                st.rerun()

    # --- STEP 3: AUDIO & VIDEO PROCESSING ---
    if st.session_state.final_script:
        st.subheader("Step 3: Final Production")
        final_script_edit = st.text_area("Final Script (Host Voice Only)", st.session_state.final_script, height=200)
        
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            target_lang = st.selectbox("Output Language", list(VOICE_MAP.keys()))
        with col_v2: 
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col_v3:
            v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
            
        st.caption(f"Pitch: {VOICE_MODES[v_mode]['pitch']} | Rate: {VOICE_MODES[v_mode]['rate']}")

        # VIDEO EDITING TOOLS
        st.markdown("#### 🛠️ Video Tools")
        
        # 1. ZOOM SLIDER
        zoom_val = st.slider("Video Zoom (Scale)", 1.0, 1.1, 1.05, 0.01)
        st.caption("Zoom 1.05 = 105% (Safe for Copyright)")
        
        # 2. AUTO FREEZE BUTTONS
        st.markdown("Auto Freeze Interval (Duration: 4s)")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        freeze_interval = None
        
        if col_f1.checkbox("30s Interval"): freeze_interval = 30
        if col_f2.checkbox("1 Minute"): freeze_interval = 60
        if col_f3.checkbox("3 Minutes"): freeze_interval = 180
        if col_f4.checkbox("6 Minutes"): freeze_interval = 360

        if st.button("🚀 Render 1080p Video"):
            with st.spinner("Rendering... (This may take a moment)"):
                
                # A. AUDIO GENERATION
                clean_text = final_script_edit.replace("*", "").strip()
                if generate_audio_cli(clean_text, target_lang, gender, v_mode, "base_voice.mp3"):
                    
                    # B. VIDEO PROCESSING (FREEZE & ZOOM)
                    input_vid = "input.mp4"
                    processed_vid = "processed_visuals.mp4"
                    
                    # 1. Apply Freeze if selected
                    if freeze_interval:
                        apply_auto_freeze("input.mp4", "frozen_temp.mp4", freeze_interval)
                        input_vid = "frozen_temp.mp4"
                    
                    # 2. Apply Zoom & 1080p Upscale
                    # Scale to 1080p first, then apply Zoom
                    # Complex Filter: Scale -> Zoom (Scale+Crop)
                    # Simple Zoom: scale=iw*Z:ih*Z, crop=iw/Z:ih/Z -> This keeps resolution same as input
                    # We want FORCE 1080p.
                    
                    zoom_filter = f"scale=1920* {zoom_val}:1080*{zoom_val},crop=1920:1080"
                    # If input is not 1080p, first scale to 1920:-1, then zoom/crop
                    
                    subprocess.run([
                        'ffmpeg', '-y', '-i', input_vid, 
                        '-vf', f"scale=1920:1080,{zoom_filter}", 
                        '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'copy', 
                        processed_vid
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # C. SYNC & MERGE
                    vid_dur = get_duration(processed_vid)
                    aud_dur = get_duration("base_voice.mp3")
                    
                    speed_factor = 1.0
                    if vid_dur > 0 and aud_dur > vid_dur:
                        speed_factor = aud_dur / vid_dur
                        speed_factor = min(speed_factor, 1.5) 
                    
                    subprocess.run(['ffmpeg', '-y', '-i', "base_voice.mp3", '-filter:a', f"atempo={speed_factor}", "final_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    outfile = f"final_1080p_{int(time.time())}.mp4"
                    cmd = [
                        'ffmpeg', '-y', '-i', processed_vid, '-i', "final_audio.mp3",
                        '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    st.session_state.processed_video_path = outfile
                    st.success("✅ Video Created Successfully!")

    # SHOW VIDEO (PERSISTENT)
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button("Download 1080p Video", f, "final_dubbed.mp4")

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
