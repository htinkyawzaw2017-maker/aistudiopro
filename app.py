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
# 🎨 3D UI & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Studio Pro", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp { 
        background-color: #0d1117; 
        color: #e6edf3; 
    }
    
    /* 3D Container Box */
    .css-1r6slb0, .stFileUploader {
        background: #161b22;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #30363d;
        padding: 20px;
    }

    /* 3D Neon Buttons */
    .stButton>button {
        background: linear-gradient(145deg, #238636, #2ea043);
        box-shadow: 5px 5px 10px #0f3d0f, -5px -5px 10px #2f9e4f;
        color: white;
        border: none;
        border-radius: 12px;
        height: 55px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.8);
    }
    .stButton>button:active {
        transform: translateY(2px);
        box-shadow: inset 2px 2px 5px #1b6629;
    }

    /* Status Boxes */
    .error-box { 
        padding: 15px; background: linear-gradient(145deg, #5a1e1e, #3d1414); 
        border-left: 5px solid #ff4b4b; border-radius: 10px; margin-top: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    }
    .success-box {
        padding: 15px; background: linear-gradient(145deg, #1e5a2c, #143d1e);
        border-left: 5px solid #00ff00; border-radius: 10px; margin-top: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS (Missing Functions Added)
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ CRITICAL ERROR: FFmpeg is not installed!")
        st.stop()

def get_duration(file_path):
    """Get duration of video/audio in seconds"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0

def safe_tts_generate(text, voice, rate, pitch, output_file):
    """Run TTS safely"""
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

def generate_content_with_retry(model, content, retries=3):
    """AI Generation with Retry Logic for 429 Errors"""
    for attempt in range(retries):
        try:
            return model.generate_content(content)
        except exceptions.ResourceExhausted:
            wait_time = 20
            st.warning(f"⚠️ Quota Exceeded (429). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            raise e
    raise Exception("❌ Quota exceeded. Please try again later.")

# ---------------------------------------------------------
# 🎬 PROCESSING WORKFLOW (SYNC FIXED)
# ---------------------------------------------------------
def process_video_workflow(video_path, gender, style, tone, api_key, model_id):
    check_ffmpeg()
    
    # 1. Get Duration EARLY
    duration_sec = get_duration(video_path)
    duration_min = round(duration_sec / 60, 2)
    st.info(f"⏱️ Video Duration: {duration_min} minutes ({int(duration_sec)}s)")

    # --- 2. UPLOAD ---
    st.info(f"🔹 Step 1: Uploading to Google AI...")
    genai.configure(api_key=api_key)
    video_file = genai.upload_file(video_path)
    
    start = time.time()
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
        if time.time() - start > 600: raise Exception("Timeout error.")
    if video_file.state.name == "FAILED": raise Exception("Upload failed.")

    # --- 3. GENERATE SCRIPT (SMART PROMPT) ---
    st.info(f"🔹 Step 2: Translating & Writing Script...")
    
    model = genai.GenerativeModel(model_id)
    
    # 🔥 PROMPT: Numbers, Units, and Time Sync 🔥
    prompt = f"""
    Act as a professional Myanmar Dubbing Artist.
    The video is exactly {int(duration_sec)} seconds long.
    
    Your Task: Translate the dialogue into natural spoken Burmese.

    CRITICAL RULES:
    1. **NUMBERS**: Write numbers as words.
       - "10000" -> "တစ်သောင်း"
       - "1995" -> "တစ်ထောင့် ကိုးရာ ကိုးဆယ့်ငါး"
       - "50%" -> "ငါးဆယ် ရာခိုင်နှုန်း"
    2. **UNITS**: Expand units.
       - "mm" -> "မီလီမီတာ", "kg" -> "ကီလိုဂရမ်", "$" -> "ဒေါ်လာ"
    3. **SYNC & TIMING**: 
       - The translation length MUST match the video length. 
       - If the video is long ({duration_min} mins), elaborate slightly to fill time.
    4. **STYLE**: {style} | **TONE**: {tone}
    5. **OUTPUT**: Spoken Burmese text ONLY. No timestamps.
    """
    
    response = generate_content_with_retry(model, [video_file, prompt])
    text = response.text.strip()
    if not text: raise Exception("AI returned empty text.")

    # --- 4. TTS ---
    st.info("🔹 Step 3: Generating Audio...")
    voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
    
    # Tone Settings
    pitch_val, rate_val = "+0Hz", "+0%"
    if tone == "Deep": pitch_val = "-10Hz"
    elif tone == "Fast": rate_val = "+10%" 
    elif tone == "Motivation": pitch_val = "+5Hz"; rate_val = "+10%"
    elif tone == "Calm": pitch_val = "-5Hz"; rate_val = "-5%"
    
    safe_tts_generate(text, voice, rate_val, pitch_val, "temp_audio.mp3")
    if not os.path.exists("temp_audio.mp3"): raise Exception("Audio generation failed.")

    # --- 5. SMART SYNC (The Fix for 8min Videos) ---
    st.info("🔹 Step 4: Syncing Audio to Video Timeline...")
    final_video = "final_dubbed.mp4"
    
    aud_len = get_duration("temp_audio.mp3")
    
    # Calculate Speed Factor to match Video Length
    if duration_sec > 0 and aud_len > 0:
        ratio = aud_len / duration_sec
        # Limit speed change to avoid robotic sound (0.7x to 1.3x)
        speed = max(0.7, min(ratio, 1.3))
        
        # Apply Time Stretch
        subprocess.run(['ffmpeg', '-y', '-i', "temp_audio.mp3", '-filter:a', f"atempo={speed}", "temp_sync.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copy("temp_audio.mp3", "temp_sync.mp3")

    # Merge
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-i', "temp_sync.mp3",
        '-c:v', 'copy', '-c:a', 'aac', 
        '-map', '0:v:0', '-map', '1:a:0', 
        final_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(final_video): raise Exception("FFmpeg failed.")
    return final_video

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
# Sidebar Menu
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("AI Studio Pro")
    
    # 3D Feature Menu
    menu = st.radio("Feature Menu", ["🎙️ Auto Dubbing", "🚀 Viral Kit", "📝 Script Writer", "🖼️ Thumbnail"])
    
    st.markdown("---")
    st.header("🔑 Settings")
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    key_input = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if key_input: st.session_state.api_key = key_input
    
    st.markdown("---")
    st.header("🤖 Model")
    model_mode = st.radio("Mode:", ["Custom", "Preset"])
    if model_mode == "Custom":
        model_id = st.text_input("Model Name:", value="gemini-2.5-flash") 
    else:
        model_id = st.selectbox("Select:", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
        
    st.markdown("---")
    if st.button("🔄 Reboot System"): st.rerun()

# Main Content
if menu == "🎙️ Auto Dubbing":
    st.title("🎙️ AI Auto Dubbing (Pro Sync)")
    
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter API Key in Sidebar.")
        st.stop()

    uploaded_file = st.file_uploader("📂 Upload Video (MP4)", type=['mp4', 'mov'])

    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        
        # 3D Control Panel
        st.markdown("### 🎛️ Control Panel")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("👤 Voice Gender", ["Male", "Female"])
            style = st.selectbox("🎭 Speaking Style", ["Narrator", "Movie Recap", "Vlogger", "Documentary"])
        with col2:
            tone = st.selectbox("🎚️ Voice Tone", ["Natural", "Deep", "Motivation", "Calm"])
        
        st.write("")
        if st.button("🚀 START PRODUCTION", type="primary"):
            status_box = st.status("⚙️ AI Processing Started...", expanded=True)
            try:
                output = process_video_workflow("temp.mp4", gender, style, tone, st.session_state.api_key, model_id)
                status_box.update(label="✅ Dubbing Complete!", state="complete", expanded=False)
                
                # Result Display
                st.markdown("<div class='success-box'><h3>✨ Production Successful!</h3></div>", unsafe_allow_html=True)
                st.video(output)
                
                with open(output, "rb") as f:
                    st.download_button("💾 Download 4K Video", f, "dubbed_movie.mp4")
            except Exception as e:
                status_box.update(label="❌ Process Failed", state="error")
                st.markdown(f"<div class='error-box'><h3>⚠️ Error Log</h3><pre>{str(e)}</pre></div>", unsafe_allow_html=True)
                st.code(traceback.format_exc())

# Placeholder for other menus
elif menu == "🚀 Viral Kit":
    st.title("🚀 Viral Content Kit")
    st.info("Coming soon in Pro Version!")
elif menu == "📝 Script Writer":
    st.title("📝 AI Script Writer")
    st.info("Coming soon in Pro Version!")
elif menu == "🖼️ Thumbnail":
    st.title("🖼️ AI Thumbnail Gen")
    st.info("Coming soon in Pro Version!")

