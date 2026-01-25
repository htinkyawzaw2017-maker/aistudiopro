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
def process_video_workflow(video_path, gender, style, tone, api_key, model_id):
    check_ffmpeg()
    
    # --- 1. UPLOAD ---
    st.info(f"🔹 Step 1: Uploading Video...")
    genai.configure(api_key=api_key)
    video_file = genai.upload_file(video_path)
    
    start = time.time()
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
        if time.time() - start > 600: raise Exception("Timeout error.")
    if video_file.state.name == "FAILED": raise Exception("Upload failed.")

    # --- 2. GENERATE SCRIPT (UPGRADED PROMPT) ---
    st.info(f"🔹 Step 2: Translating with Smart Context ({model_id})...")
    
    model = genai.GenerativeModel(model_id)
    
    # 🔥 ဒီနေရာက မိတ်ဆွေလိုချင်တဲ့ အဓိက ပြောင်းလဲမှုပါ 🔥
    prompt = f"""
    Act as a professional Burmese Translator & Voiceover Scriptwriter.
    Your Task: Translate the spoken dialogue into natural, spoken Burmese (Myanmar).

    CONTEXT SETTINGS:
    - Style: {style}
    - Tone: {tone}

    CRITICAL RULES FOR TRANSLATION (MUST FOLLOW):
    1. **NUMBERS & UNITS**: Convert ALL numbers and symbols into full spoken Burmese words.
       - "10000" -> "တစ်သောင်း" (NOT "တစ် သုည သုည").
       - "1990" -> "တစ်ထောင့် ကိုးရာ ကိုးဆယ်".
       - "50%" -> "ငါးဆယ် ရာခိုင်နှုန်း".
       - "mm" -> "မီလီမီတာ", "kg" -> "ကီလိုဂရမ်", "$" -> "ဒေါ်လာ".
    
    2. **STYLE ADAPTATION**:
       - If Style is 'Vlogger': Use casual words like "ဘော်ဒါတို့", "ကျွန်တော်/ကျွန်မ" (Not "ကျုပ်"). Use active, high-energy sentence structures.
       - If Style is 'Documentary/Narrator': Use formal, polished Burmese (စာဆန်ဆန်).
       - If Style is 'Movie Recap': Use dramatic, storytelling flow.

    3. **TIMELINE SYNC**:
       - Keep sentences CONCISE. Do not explain things unnecessarily.
       - The translation length MUST match the original speech length to avoid audio speed issues.
    
    4. **CLEAN OUTPUT**:
       - Return ONLY the Burmese text to be spoken.
       - NO timestamps (00:01), NO notes, NO English words.
    """
    
    response = generate_content_with_retry(model, [video_file, prompt])
    text = response.text.strip()
    if not text: raise Exception("AI returned empty text.")

    # --- 3. TTS ---
    st.info("🔹 Step 3: Generating Human-like Audio...")
    voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
    
    # Fine-tuning Pitch/Rate for Tones
    pitch_val, rate_val = "+0Hz", "+0%"
    if tone == "Deep": pitch_val = "-12Hz"
    elif tone == "Fast": rate_val = "+15%"
    elif tone == "Motivation": pitch_val = "+5Hz"; rate_val = "+10%"
    elif tone == "Calm": pitch_val = "-5Hz"; rate_val = "-5%"
    
    safe_tts_generate(text, voice, rate_val, pitch_val, "temp_audio.mp3")
    if not os.path.exists("temp_audio.mp3"): raise Exception("Audio failed.")

    # --- 4. MERGE ---
    st.info("🔹 Step 4: Finalizing Video...")
    final_video = "final_dubbed.mp4"
    
    # Calculate duration to handle sync
    def get_len(f):
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', f], capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])

    try:
        vid_len = get_len(video_path)
        aud_len = get_len("temp_audio.mp3")
        # Smart Speed Adjustment (0.8x to 1.3x limit to keep it natural)
        speed = max(0.8, min(aud_len / vid_len, 1.3))
    except:
        speed = 1.0

    subprocess.run(['ffmpeg', '-y', '-i', "temp_audio.mp3", '-filter:a', f"atempo={speed}", "temp_sync.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = [
        'ffmpeg', '-y', '-i', video_path, '-i', "temp_sync.mp3",
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', 
        '-shortest', final_video
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