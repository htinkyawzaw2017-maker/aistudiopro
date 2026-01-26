import warnings
# 1. Warning များကို ပိတ်ခြင်း (User စိတ်မညစ်ရအောင်)
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
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D); 
        color: black; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .success-box { padding: 15px; background: #051a05; border: 1px solid #00ff00; border-radius: 10px; margin-bottom: 20px;}
    .wait-box { padding: 10px; background: #332b00; border: 1px solid #ffcc00; color: #ffcc00; border-radius: 5px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing."); st.stop()

async def tts_save(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_id, rate, pitch, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_save(text, voice_id, rate, pitch, filename))
        loop.close()
        return True
    except: return False

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🛡️ API RETRY & SMART TRANSLATION LOGIC
# ---------------------------------------------------------
def generate_content_with_retry(model, content, is_translation=False, style="Natural"):
    """
    Handles API calls with Auto-Retry for Free Tier Limits.
    Includes Smart Prompting for Translation.
    """
    retries = 3
    
    # If it's a translation task, we wrap the content in a powerful prompt
    final_content = content
    if is_translation and isinstance(content, str):
        final_content = f"""
        Act as a professional Burmese Translator and Dubbing Artist.
        Translate the following English text to Burmese (Myanmar).
        
        INPUT: "{content}"
        
        STRICT RULES FOR HIGHEST QUALITY:
        1. **Numbers**: Convert ALL numbers to Burmese spoken words. 
           (e.g., "10000" -> "တစ်သောင်း", "1995" -> "တစ်ထောင့်ကိုးရာကိုးဆယ့်ငါး", "5" -> "ငါး").
        2. **Units**: Convert units to full Burmese words.
           (e.g., "mm" -> "မီလီမီတာ", "km" -> "ကီလိုမီတာ", "%" -> "ရာခိုင်နှုန်း").
        3. **Style/Tone**: {style}. Make it flow naturally like a real human speaking.
        4. **Output**: ONLY the Burmese translation. No explanations. No English characters.
        """

    for attempt in range(retries):
        try:
            # If content is a list (for images/video), pass as list. If string, pass as string or prompt.
            if isinstance(final_content, str):
                response = model.generate_content(final_content)
            else:
                response = model.generate_content(final_content)
                
            return response.text.strip()
            
        except exceptions.ResourceExhausted:
            wait_time = 20 # Free tier limit hit
            st.markdown(f"""
                <div class="wait-box">
                    ⚠️ Free Tier Limit Hit. Waiting {wait_time}s... ({attempt+1}/{retries})
                </div>
            """, unsafe_allow_html=True)
            time.sleep(wait_time)
            continue
        except Exception as e:
            return f"Error: {str(e)}"
            
    return "Translation Failed"

def clean_burmese_text(text):
    # Remove English characters that might sneak in (A-Z, a-z)
    # Keep Burmese characters, numbers, and basic punctuation
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

# ---------------------------------------------------------
# 🎬 MAIN WORKFLOW
# ---------------------------------------------------------
def process_full_workflow(video_path, voice_data, style, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract Audio
    status.update(label="🎧 Extracting Audio...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper Transcription
    status.update(label="🧠 Listening (Whisper AI)...", state="running")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Smart Translation & Dubbing Loop
    status.update(label="🎙️ Translating & Dubbing (Smart Mode)...", state="running")
    genai.configure(api_key=api_key)
    
    # 🔥 Force Free Tier Model if not specified
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        orig_text = seg['text']
        
        # Translate with Retry Logic & Smart Prompt
        translated_text = generate_content_with_retry(model, orig_text, is_translation=True, style=style)
        
        # Cleanup
        burmese_text = clean_burmese_text(translated_text)
        
        if not burmese_text: continue
        
        # TTS
        fname = f"seg_{i}.mp3"
        generate_audio(burmese_text, voice_data["id"], voice_data["rate"], voice_data["pitch"], fname)
        
        # Sync Logic (Time Stretching)
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            curr_dur = len(seg_audio) / 1000.0
            target_dur = end - start
            
            if curr_dur > 0 and target_dur > 0:
                # Limit speed change to avoid robotic sound (0.6x to 1.6x)
                speed = max(0.6, min(curr_dur / target_dur, 1.6))
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(f"s_{i}.mp3"):
                    final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
        
        progress.progress((i + 1) / total)
        # Cleanup temp files
        try: os.remove(fname); os.remove(f"s_{i}.mp3")
        except: pass

    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. Merge
    status.update(label="🎬 Finalizing Video...", state="running")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_filename

# ---------------------------------------------------------
# 🚀 FEATURES WRAPPERS
# ---------------------------------------------------------
def run_genai_features(prompt, video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    try:
        f = genai.upload_file(video_path)
        while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
    except exceptions.ResourceExhausted: return "⚠️ Limit Reached. Wait 1 min."
    except Exception as e: return f"Upload Error: {e}"
    
    model = genai.GenerativeModel(model_name)
    return generate_content_with_retry(model, [f, prompt])

def run_thumbnail(video_path, api_key, model_name):
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vframes', '1', 'thumb.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    genai.configure(api_key=api_key)
    img = Image.open("thumb.jpg")
    model = genai.GenerativeModel(model_name)
    return generate_content_with_retry(model, [img, "Describe this for a Thumbnail prompt."]), "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    # 🔥 DEFAULT TO 1.5-FLASH FOR FREE TIER (BILL SAVER)
    model_name = st.text_input("Model", value="gemini-1.5-flash")
    st.caption("✅ Recommended: gemini-1.5-flash for Free Tier")
    
    if st.button("🔴 Force Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key."); st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    t1, t2, t3, t4 = st.tabs(["🎙️ DUBBING", "🚀 VIRAL KIT", "📝 SCRIPT", "🖼️ THUMBNAIL"])

    # 1. DUBBING
    with t1:
        st.subheader("🔊 Auto Dubbing")
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.markdown('<div class="success-box"><h3>✅ Video Ready!</h3></div>', unsafe_allow_html=True)
            st.video(st.session_state.processed_video)
            with open(st.session_state.processed_video, "rb") as f:
                st.download_button("💾 Download", f, "final_dub.mp4")

        c1, c2 = st.columns(2)
        with c1: narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
        with c2: style = st.selectbox("Style", ["Natural", "News", "Dramatic", "Vlog"])
            
        v_data = {
            "Male (Thiha)": {"id": "my-MM-ThihaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Female (Nilar)": {"id": "my-MM-NilarNeural", "rate": "+10%", "pitch": "+10Hz"}
        }

        if st.button("🚀 START DUBBING"):
            status = st.status("Processing...", expanded=True)
            prog = st.progress(0)
            try:
                out = process_full_workflow("temp.mp4", v_data[narrator], style, st.session_state.api_key, model_name, status, prog)
                st.session_state.processed_video = out
                status.update(label="✅ Done!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    # 2. VIRAL KIT
    with t2:
        if st.button("✨ Generate Viral Info"):
            with st.spinner("Processing..."):
                res = run_genai_features("Generate 3 Viral Titles (Burmese) & Hashtags.", "temp.mp4", st.session_state.api_key, model_name)
                st.info(res)

    # 3. SCRIPT
    with t3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                res = run_genai_features("Write a full Burmese script.", "temp.mp4", st.session_state.api_key, model_name)
                st.text_area("Script", res, height=400)

    # 4. THUMBNAIL
    with t4:
        if st.button("🎨 Thumbnail Idea"):
            with st.spinner("Analyzing..."):
                txt, img = run_thumbnail("temp.mp4", st.session_state.api_key, model_name)
                st.image(img)
                st.code(txt)
