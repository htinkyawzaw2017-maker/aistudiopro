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

def transcribe_video(video_path, language_code=None):
    try:
        model = whisper.load_model("base")
        # Whisper auto-detects, but we can hint if needed, though 'base' model is good at auto
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
    1. Hook (0-30s): Grab attention.
    2. Value (Body): Deliver the core message/story.
    3. Call (End): Call to action.
    
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
# ❄️ FREEZE FRAME ENGINE
# ---------------------------------------------------------
def process_freeze_command(command, input_video, output_video):
    """
    Parses [freeze 10,5] or [duration 10,5]
    Format: command time_point, duration
    """
    try:
        # Regex to find "freeze" or "duration" followed by two numbers
        match = re.search(r'(freeze|duration)\s*[:=]?\s*(-?\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command, re.IGNORECASE)
        
        if match:
            time_point = float(match.group(2))
            duration = float(match.group(3))
            
            # Handle negative time (relative to end) - basic implementation assumes positive
            if time_point < 0: time_point = 0 # Fallback for simplicity
            
            # FFmpeg: Cut A, Freeze Frame, Cut B, Concat
            # 1. Part A
            subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(time_point), '-c', 'copy', 'part_a.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 2. Freeze
            subprocess.run(['ffmpeg', '-y', '-ss', str(time_point), '-i', input_video, '-vframes', '1', 'freeze.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'freeze.jpg', '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'part_freeze.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 3. Part B
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
        print(f"Freeze Logic Error: {e}")
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
    
    # Language Selection
    source_lang = st.selectbox("Original Video Language", ["English", "Japanese", "Chinese", "Thai", "Indian (Hindi)"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("📝 Extract & Translate to Burmese"):
            with st.spinner("Listening & Translating..."):
                check_requirements()
                # Extract Audio
                subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Transcribe (Get Original Text)
                raw_text = transcribe_video("temp.wav")
                st.session_state.raw_transcript = raw_text
                
                # Translate immediately to Burmese (Draft)
                model = get_model(st.session_state.api_key, model_name)
                draft = translate_to_burmese_draft(model, raw_text, source_lang)
                st.session_state.burmese_draft = draft
                st.rerun()

    # --- STEP 2: REFINE SCRIPT ---
    if st.session_state.burmese_draft:
        st.subheader("Step 2: Script Refinement")
        # Show Burmese Draft directly (User wanted to see Burmese immediately)
        draft_text = st.text_area("Burmese Draft (Translated)", st.session_state.burmese_draft, height=150)
        
        custom_prompt = st.text_input("Script Instructions", "Make it exciting, H-V-C style")
        
        if st.button("✨ Convert to H-V-C Final Script"):
            with st.spinner("Applying Magic..."):
                model = get_model(st.session_state.api_key, model_name)
                # Use "My Video" as generic title or ask user
                final = refine_script_hvc(model, draft_text, "My Video", custom_prompt)
                st.session_state.final_script = final
                st.rerun()

    # --- STEP 3: AUDIO & FREEZE ---
    if st.session_state.final_script:
        st.subheader("Step 3: Audio, Sync & Freeze")
        final_script_edit = st.text_area("Final Script (Host Voice Only)", st.session_state.final_script, height=200)
        
        c1, c2 = st.columns(2)
        with c1: 
            voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
        with c2:
            # 🔥 MOVED FREEZE COMMAND HERE
            freeze_cmd = st.text_input("Freeze/Duration Command", placeholder="e.g. freeze 10,5 or duration 10,5")
            st.caption("Freeze video at 10s for 5s duration.")

        if st.button("🚀 Generate Final Video"):
            with st.spinner("Rendering..."):
                v_id = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
                
                # 1. Handle Freeze First (Modify Input Video)
                working_video = "input.mp4"
                if freeze_cmd:
                    if process_freeze_command(freeze_cmd, "input.mp4", "frozen_output.mp4"):
                        working_video = "frozen_output.mp4"
                        st.success(f"Video modified with: {freeze_cmd}")
                    else:
                        st.warning("Freeze command ignored (Format error). Using original video.")

                # 2. Generate Audio
                clean_text = final_script_edit.replace("*", "").strip()
                generate_audio_cli(clean_text, v_id, "+0%", "+0Hz", "base_voice.mp3")
                
                # 3. Sync Logic (Auto Speed)
                vid_dur = get_duration(working_video)
                aud_dur = get_duration("base_voice.mp3")
                
                speed_factor = 1.0
                if vid_dur > 0 and aud_dur > vid_dur:
                    speed_factor = aud_dur / vid_dur
                    speed_factor = min(speed_factor, 1.5) # Cap speed
                
                subprocess.run(['ffmpeg', '-y', '-i', "base_voice.mp3", '-filter:a', f"atempo={speed_factor}", "final_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 4. Final Merge
                outfile = f"final_{int(time.time())}.mp4"
                cmd = [
                    'ffmpeg', '-y', '-i', working_video, '-i', "final_audio.mp3",
                    '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', outfile
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                st.success("✅ Video Created Successfully!")
                st.video(outfile)
                with open(outfile, "rb") as f:
                    st.download_button("Download", f, "final_dubbed.mp4")

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
