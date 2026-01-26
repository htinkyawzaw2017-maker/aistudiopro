import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
import shutil
import whisper
from pydub import AudioSegment
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 3D UI CONFIGURATION (UNCHANGED)
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Studio Pro", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .css-1r6slb0, .stFileUploader {
        background: #161b22; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #30363d; padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(145deg, #238636, #2ea043);
        box-shadow: 5px 5px 10px #0f3d0f, -5px -5px 10px #2f9e4f;
        color: white; border: none; border-radius: 12px; height: 55px;
        font-weight: bold; font-size: 16px; width: 100%;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 0 15px rgba(46, 160, 67, 0.8); }
    .success-box { padding: 15px; background: #143d1e; border-left: 5px solid #00ff00; border-radius: 10px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ CRITICAL: FFmpeg missing. Please install it.")
        st.stop()

def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# Safe TTS Generation
async def _tts_async(text, voice, rate, pitch, output):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output)

def generate_tts_segment(text, voice, rate, pitch, filename):
    try:
        asyncio.run(_tts_async(text, voice, rate, pitch, filename))
        return True
    except:
        # Fallback loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_async(text, voice, rate, pitch, filename))
        return True

# AI Translation with Retry
def translate_segment(model, text, style, tone):
    prompt = f"""
    Translate this specific subtitle segment into spoken Burmese.
    Original: "{text}"
    
    RULES:
    1. **NUMBERS**: "10000" -> "တစ်သောင်း", "1990" -> "တစ်ထောင့်ကိုးရာ ကိုးဆယ်".
    2. **UNITS**: "mm" -> "မီလီမီတာ", "kg" -> "ကီလိုဂရမ်".
    3. **STYLE**: {style} (Use "ကျွန်တော်/ကျွန်မ" for Vlogger, Formal for Narrator).
    4. **TONE**: {tone}.
    5. **LENGTH**: Keep it short to match original timing.
    6. **OUTPUT**: Burmese text ONLY.
    """
    for _ in range(3):
        try:
            res = model.generate_content(prompt)
            return res.text.strip()
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except:
            return text # Fallback to original if fail
    return text

# ---------------------------------------------------------
# 🎬 ADVANCED SYNC ENGINE (SEGMENT BASED)
# ---------------------------------------------------------
def process_precise_sync(video_path, gender, style, tone, api_key, model_id, status_box, progress_bar):
    check_ffmpeg()
    
    # 1. EXTRACT AUDIO
    status_box.update(label="🎧 Step 1: Analyzing Audio Structure...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp_audio.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. WHISPER TRANSCRIPTION (TIMESTAMPS)
    status_box.update(label="🧠 Step 2: Detecting Speech Segments (Whisper AI)...", state="running")
    model = whisper.load_model("base") # 'base' is good balance of speed/accuracy
    result = model.transcribe("temp_audio.wav")
    segments = result['segments']
    
    total_segments = len(segments)
    st.info(f"✨ Detected {total_segments} speech segments to dub.")
    
    # 3. PROCESS SEGMENTS
    status_box.update(label="🎙️ Step 3: Dubbing Each Segment...", state="running")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_id)
    
    # Create silent canvas
    video_dur = get_duration(video_path)
    final_audio = AudioSegment.silent(duration=video_dur * 1000) # milliseconds
    
    voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
    
    # Tone settings
    rate_str, pitch_str = "+0%", "+0Hz"
    if tone == "Fast": rate_str = "+15%"
    elif tone == "Deep": pitch_str = "-10Hz"
    
    for i, seg in enumerate(segments):
        start_time = seg['start']
        end_time = seg['end']
        original_text = seg['text']
        duration_needed = end_time - start_time
        
        # Translate
        translated_text = translate_segment(gemini_model, original_text, style, tone)
        
        # TTS Generation
        seg_filename = f"seg_{i}.mp3"
        generate_tts_segment(translated_text, voice, rate_str, pitch_str, seg_filename)
        
        # Time Stretching (Fit to slot)
        seg_audio = AudioSegment.from_file(seg_filename)
        current_dur = len(seg_audio) / 1000.0
        
        # Calculate speed change needed
        if duration_needed > 0 and current_dur > 0:
            speed_factor = current_dur / duration_needed
            # Clamp speed (0.7x to 1.5x) to prevent distortion
            speed_factor = max(0.7, min(speed_factor, 1.5))
            
            # Apply FFmpeg for high quality stretching
            stretched_filename = f"seg_{i}_final.mp3"
            subprocess.run(['ffmpeg', '-y', '-i', seg_filename, '-filter:a', f"atempo={speed_factor}", stretched_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(stretched_filename):
                seg_final = AudioSegment.from_file(stretched_filename)
                # Overlay onto main track at exact start time
                final_audio = final_audio.overlay(seg_final, position=start_time * 1000)
        
        # Update Progress
        prog = int((i / total_segments) * 80) + 10
        progress_bar.progress(prog)
        
        # Cleanup temp
        try: 
            os.remove(seg_filename)
            if os.path.exists(f"seg_{i}_final.mp3"): os.remove(f"seg_{i}_final.mp3")
        except: pass

    # Export Full Audio
    final_audio.export("final_track.mp3", format="mp3")
    
    # 4. MERGE WITH VIDEO
    status_box.update(label="🎬 Step 4: Final Merging...", state="running")
    progress_bar.progress(95)
    
    final_video = "final_dubbed.mp4"
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-i', "final_track.mp3",
        '-c:v', 'copy', # Keep video quality
        '-c:a', 'aac', 
        '-map', '0:v:0', '-map', '1:a:0', 
        final_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    progress_bar.progress(100)
    
    return final_video

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("AI Studio Pro")
    st.caption("v3.0 | Precise Sync Engine")
    
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    key_input = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if key_input: st.session_state.api_key = key_input
    
    st.markdown("---")
    model_mode = st.radio("Model Mode", ["Custom", "Preset"])
    if model_mode == "Custom":
        model_id = st.text_input("Model Name", value="gemini-2.5-flash")
    else:
        model_id = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    
    st.markdown("---")
    if st.button("🔄 Reboot System"): st.rerun()

st.title("🎙️ AI Precise Dubbing")
st.markdown("**Core Feature:** Lip-sync timing matching using `Whisper AI` + `Segment Stretching`.")

if not st.session_state.api_key:
    st.warning("⚠️ Please enter API Key in Sidebar.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
    
    st.markdown("### 🎛️ Control Panel")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        style = st.selectbox("🎭 Style", ["Narrator", "Movie Recap", "Vlogger", "Documentary"])
    with col2:
        tone = st.selectbox("🎚️ Tone", ["Natural", "Deep", "Motivation", "Calm"])
    
    st.write("")
    if st.button("🚀 START PRECISE DUBBING", type="primary"):
        status_box = st.status("⚙️ Initializing Whisper AI...", expanded=True)
        progress = st.progress(0)
        
        try:
            output = process_precise_sync("temp.mp4", gender, style, tone, st.session_state.api_key, model_id, status_box, progress)
            
            status_box.update(label="✅ Dubbing Complete!", state="complete", expanded=False)
            st.markdown("<div class='success-box'><h3>✨ Result Ready!</h3></div>", unsafe_allow_html=True)
            st.video(output)
            with open(output, "rb") as f:
                st.download_button("💾 Download Video", f, "precise_dub.mp4")
                
        except Exception as e:
            status_box.update(label="❌ Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
