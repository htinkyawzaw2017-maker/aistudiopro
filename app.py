import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
import shutil
import traceback
from google.api_core import exceptions

# ---------------------------------------------------------
# ⚙️ UI CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Studio Pro", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .error-box { padding: 20px; background-color: #330000; border: 1px solid red; border-radius: 10px; margin-top: 10px; }
    .retry-box { padding: 10px; background-color: #332200; border: 1px solid #ffcc00; border-radius: 10px; color: #ffcc00; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ SYSTEM CHECKS
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ CRITICAL ERROR: FFmpeg is not installed!")
        st.stop()

# ---------------------------------------------------------
# 🚀 SAFE ASYNC RUNNER
# ---------------------------------------------------------
def safe_tts_generate(text, voice, rate, pitch, output_file):
    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_file)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()
    except Exception as e:
        raise e

# ---------------------------------------------------------
# 🤖 AI GENERATION (SMART PROMPT)
# ---------------------------------------------------------
def generate_content_with_retry(model, content, retries=3):
    for attempt in range(retries):
        try:
            return model.generate_content(content)
        except exceptions.ResourceExhausted:
            wait_time = 20
            st.markdown(f"<div class='retry-box'>⚠️ Quota Exceeded (429). Waiting {wait_time}s...</div>", unsafe_allow_html=True)
            time.sleep(wait_time)
            continue
        except Exception as e:
            raise e
    raise Exception("❌ Quota exceeded. Please try again later.")

# ---------------------------------------------------------
# 🎬 PROCESSING WORKFLOW
# ---------------------------------------------------------
# ---------------------------------------------------------
# 🎬 PROCESSING WORKFLOW (Improved Sync for Long Videos)
# ---------------------------------------------------------
def process_video_workflow(video_path, gender, style, tone, api_key, model_id):
    check_ffmpeg()
    
    # 1. Get Duration EARLY to tell AI
    duration_sec = get_duration(video_path)
    duration_min = round(duration_sec / 60, 2)
    print(f"DEBUG: Video Duration: {duration_min} minutes")

    # --- 2. UPLOAD ---
    st.info(f"🔹 Step 1: Uploading Video ({duration_min} mins)...")
    genai.configure(api_key=api_key)
    video_file = genai.upload_file(video_path)
    
    start = time.time()
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
        if time.time() - start > 600: raise Exception("Timeout error.")
    if video_file.state.name == "FAILED": raise Exception("Upload failed.")

    # --- 3. GENERATE SCRIPT (TIME-AWARE PROMPT) ---
    st.info(f"🔹 Step 2: Translating & Syncing...")
    
    model = genai.GenerativeModel(model_id)
    
    # 🔥 PROMPT UPDATE: Time-Aware Injection 🔥
    prompt = f"""
    Act as a professional Movie Dubbing Artist.
    The video is exactly {duration_min} minutes long.
    
    Your Task: Translate the dialogue into Burmese (Myanmar) to MATCH this duration.
    
    CRITICAL SYNC RULES:
    1. **Expand or Condense**: If the video is long, use detailed descriptions and slightly longer sentences to fill the time. If short, be concise.
    2. **Pacing**: The generated text must flow naturally for {duration_min} minutes.
    3. **Style**: {style} | **Tone**: {tone}
    4. **Output**: ONLY the spoken Burmese words. NO timestamps.
    """
    
    response = generate_content_with_retry(model, [video_file, prompt])
    text = response.text.strip()
    if not text: raise Exception("AI returned empty text.")

    # --- 4. TTS ---
    st.info("🔹 Step 3: Generating Audio...")
    voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
    
    pitch_val, rate_val = "+0Hz", "+0%"
    if tone == "Deep": pitch_val = "-12Hz"
    elif tone == "Fast": rate_val = "+10%" 
    elif tone == "Motivation": pitch_val = "+5Hz"; rate_val = "+5%"
    elif tone == "Calm": pitch_val = "-5Hz"; rate_val = "-5%"
    
    safe_tts_generate(text, voice, rate_val, pitch_val, "temp_audio.mp3")
    if not os.path.exists("temp_audio.mp3"): raise Exception("Audio failed.")

    # --- 5. SMART SYNC & MERGE ---
    st.info("🔹 Step 4: Finalizing & Syncing...")
    final_video = "final_dubbed.mp4"
    
    aud_len = get_duration("temp_audio.mp3")
    
    # 🔥 SMART SYNC LOGIC 🔥
    if duration_sec > 0 and aud_len > 0:
        # Calculate ratio
        ratio = aud_len / duration_sec
        
        # If Audio is too short (e.g., 5 mins audio for 8 mins video), slow it down slightly
        # If Audio is too long, speed it up
        # We clamp the speed between 0.75x (slower) and 1.35x (faster) to keep it natural
        speed = max(0.75, min(ratio, 1.35))
        
        print(f"DEBUG: Syncing Speed Factor: {speed}")
        subprocess.run(['ffmpeg', '-y', '-i', "temp_audio.mp3", '-filter:a', f"atempo={speed}", "temp_sync.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copy("temp_audio.mp3", "temp_sync.mp3")

    # Merge Command (Removed -shortest to prevent cutting video if audio is slightly short)
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-i', "temp_sync.mp3",
        '-c:v', 'copy',       # Keep original video quality (Fast)
        '-c:a', 'aac', 
        '-map', '0:v:0', 
        '-map', '1:a:0', 
        # '-shortest',        # Removed this so 8 min video stays 8 mins
        final_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(final_video): raise Exception("FFmpeg failed.")
    return final_video


# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
st.title("⚡ AI Video Studio Pro (Smart Voice)")

if 'api_key' not in st.session_state: st.session_state.api_key = ""

with st.sidebar:
    st.header("🔑 Settings")
    key_input = st.text_input("API Key", type="password")
    if key_input: st.session_state.api_key = key_input
    
    st.markdown("---")
    st.header("🤖 Model")
    model_mode = st.radio("Mode:", ["Custom", "Preset"])
    if model_mode == "Custom":
        model_id = st.text_input("Model Name:", value="gemini-2.5-flash") # User favorite
    else:
        model_id = st.selectbox("Select:", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    
    st.markdown("---")
    if st.button("🔄 Reset App"): st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
    st.video("temp.mp4")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Voice", ["Male", "Female"])
        # Added specific styles requested
        style = st.selectbox("Style", ["Narrator", "Movie Recap", "Vlogger", "Documentary"])
    with c2:
        # Added specific tones requested
        tone = st.selectbox("Tone", ["Natural", "Deep", "Motivation", "Calm"])

    if st.button("🔴 Start Dubbing", type="primary"):
        status_box = st.status("⚙️ AI is thinking...", expanded=True)
        try:
            output = process_video_workflow("temp.mp4", gender, style, tone, st.session_state.api_key, model_id)
            status_box.update(label="✅ Success!", state="complete", expanded=False)
            st.success("Translation Complete!")
            st.video(output)
            with open(output, "rb") as f: st.download_button("💾 Download Video", f, "dubbed.mp4")
        except Exception as e:
            status_box.update(label="❌ Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
