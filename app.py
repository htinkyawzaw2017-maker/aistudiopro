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
# 🎨 3D PRO UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🌐", layout="wide")

st.markdown("""
    <style>
    /* Global Deep Dark Theme */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    
    /* 3D Glassmorphism Containers */
    .css-1r6slb0, .stFileUploader, div[data-testid="stSidebar"] {
        background: rgba(25, 30, 40, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
        padding: 20px;
    }

    /* 3D Neon Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white; border: none; border-radius: 12px; height: 50px;
        font-weight: 700; letter-spacing: 1px;
        box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4), inset 0 2px 5px rgba(255,255,255,0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%; text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px rgba(0, 114, 255, 0.6);
    }
    .stButton>button:active { transform: translateY(2px); }

    /* Custom Navigation Bar */
    .nav-bar {
        display: flex; justify-content: space-around; background: #111; 
        padding: 15px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #333;
    }
    .nav-item { color: #888; font-weight: bold; font-size: 14px; }
    .nav-item.active { color: #00c6ff; text-shadow: 0 0 10px #00c6ff; }

    /* Success/Error Boxes */
    .success-box { padding: 15px; background: #0f3d0f; border-left: 5px solid #00ff00; border-radius: 10px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🌍 MULTI-LANGUAGE VOICE DATABASE
# ---------------------------------------------------------
VOICE_MAP = {
    "Myanmar (Burmese)": {"code": "my", "voice_m": "my-MM-ThihaNeural", "voice_f": "my-MM-NilarNeural"},
    "English (US)": {"code": "en", "voice_m": "en-US-ChristopherNeural", "voice_f": "en-US-JennyNeural"},
    "Thai": {"code": "th", "voice_m": "th-TH-NiwatNeural", "voice_f": "th-TH-PremwadeeNeural"},
    "Chinese (Mandarin)": {"code": "zh", "voice_m": "zh-CN-YunxiNeural", "voice_f": "zh-CN-XiaoxiaoNeural"},
    "Japanese": {"code": "ja", "voice_m": "ja-JP-KeitaNeural", "voice_f": "ja-JP-NanamiNeural"},
    "Korean": {"code": "ko", "voice_m": "ko-KR-InJoonNeural", "voice_f": "ko-KR-SunHiNeural"},
    "Spanish": {"code": "es", "voice_m": "es-ES-AlvaroNeural", "voice_f": "es-ES-ElviraNeural"},
    "French": {"code": "fr", "voice_m": "fr-FR-HenriNeural", "voice_f": "fr-FR-DeniseNeural"},
    "German": {"code": "de", "voice_m": "de-DE-ConradNeural", "voice_f": "de-DE-KatjaNeural"},
    "Hindi": {"code": "hi", "voice_m": "hi-IN-MadhurNeural", "voice_f": "hi-IN-SwaraNeural"}
}

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ CRITICAL: FFmpeg missing. Install it via `apt-get install ffmpeg` or `brew install ffmpeg`.")
        st.stop()

def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

async def _tts_async(text, voice, rate, pitch, output):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output)

def generate_tts_segment(text, voice, rate, pitch, filename):
    try:
        # Create a new loop for thread safety
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_async(text, voice, rate, pitch, filename))
        loop.close()
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def translate_segment(model, text, target_lang, style, tone):
    # Strict prompt to force Number conversion and Language accuracy
    lang_name = target_lang
    
    prompt = f"""
    Act as a professional Dubbing Translator.
    Translate the following subtitle text into {lang_name}.
    Original Text: "{text}"
    
    CRITICAL RULES:
    1. **LANGUAGE**: Output MUST be in {lang_name}. Do NOT output English unless the target is English.
    2. **NUMBERS**: Convert ALL numbers to spoken words in {lang_name}.
       - Example (if Burmese): "10000" -> "တစ်သောင်း", "50mm" -> "ငါးဆယ် မီလီမီတာ".
       - Example (if English): "10000" -> "Ten thousand".
    3. **STYLE**: {style}. (If Burmese Vlogger: Use "ကျွန်တော်/ကျွန်မ" and casual particles like "ဗျ/ရှင်").
    4. **TONE**: {tone}.
    5. **LENGTH**: Keep translation concise to match original duration.
    
    OUTPUT: Just the translated spoken text. No notes.
    """
    try:
        res = model.generate_content(prompt)
        t = res.text.strip()
        return t if t else text
    except:
        return text

# ---------------------------------------------------------
# 🎬 SYNC ENGINE (WHISPER + TIME STRETCH)
# ---------------------------------------------------------
def process_sync_dubbing(video_path, target_lang_key, gender, style, tone, api_key, model_id, status, progress):
    check_ffmpeg()
    
    # 1. SETUP LANGUAGE & VOICE
    lang_data = VOICE_MAP[target_lang_key]
    voice = lang_data["voice_m"] if gender == "Male" else lang_data["voice_f"]
    
    # Tone Adjustment
    rate_str, pitch_str = "+0%", "+0Hz"
    if tone == "Fast": rate_str = "+15%"
    elif tone == "Deep": pitch_str = "-10Hz"
    elif tone == "Calm": pitch_str = "-5Hz"; rate_str = "-5%"
    
    # 2. EXTRACT AUDIO
    status.update(label="🎧 Step 1: Extracting Audio Track...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp_audio.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. WHISPER DETECTION
    status.update(label=f"🧠 Step 2: Detecting Speech (Whisper AI)...", state="running")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp_audio.wav")
    segments = result['segments']
    
    total_segs = len(segments)
    st.info(f"✨ Detected {total_segs} speech segments.")
    
    # 4. SEGMENT PROCESSING
    status.update(label=f"🎙️ Step 3: Translating & Dubbing to {target_lang_key}...", state="running")
    
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model_id)
    
    # Master Silent Track
    video_dur = get_duration(video_path)
    final_audio = AudioSegment.silent(duration=video_dur * 1000)
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        orig_text = seg['text']
        seg_dur = end - start
        
        # Translate
        trans_text = translate_segment(gemini, orig_text, target_lang_key, style, tone)
        
        # TTS
        fname = f"seg_{i}.mp3"
        generate_tts_segment(trans_text, voice, rate_str, pitch_str, fname)
        
        # Time Stretch (Sync)
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            curr_dur = len(seg_audio) / 1000.0
            
            if curr_dur > 0 and seg_dur > 0:
                # Clamp speed to avoid robotic sounds (0.6x to 1.5x)
                speed = max(0.6, min(curr_dur / seg_dur, 1.5))
                
                stretched_name = f"seg_{i}_final.mp3"
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", stretched_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(stretched_name):
                    final_seg = AudioSegment.from_file(stretched_name)
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
        
        # Progress Update
        p = int((i / total_segs) * 80) + 10
        progress.progress(p)
        
        # Cleanup
        try:
            os.remove(fname)
            if os.path.exists(f"seg_{i}_final.mp3"): os.remove(f"seg_{i}_final.mp3")
        except: pass

    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. MERGE
    status.update(label="🎬 Step 4: Final Rendering...", state="running")
    progress.progress(95)
    
    final_vid = "final_output.mp4"
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-i', "final_track.mp3",
        '-c:v', 'copy', 
        '-c:a', 'aac', 
        '-map', '0:v:0', '-map', '1:a:0', 
        final_vid
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    progress.progress(100)
    
    return final_vid

# ---------------------------------------------------------
# 🖥️ MAIN UI LAYOUT
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("GLOBAL STUDIO")
    st.caption("Pro AI Dubbing v4.0")
    
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    api_key = st.text_input("🔑 API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.markdown("---")
    st.header("🤖 Model Engine")
    mode = st.radio("Mode", ["Preset", "Custom"], horizontal=True)
    if mode == "Custom":
        model_id = st.text_input("Model ID", "gemini-2.5-flash")
    else:
        model_id = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    
    st.markdown("---")
    if st.button("🔄 REBOOT SYSTEM"): st.rerun()

# Feature Navigation Bar
st.markdown("""
<div class="nav-bar">
    <span class="nav-item active">🎙️ DUBBING PRO</span>
    <span class="nav-item">🚀 VIRAL KIT</span>
    <span class="nav-item">📝 SCRIPT</span>
    <span class="nav-item">🖼️ THUMBNAIL</span>
</div>
""", unsafe_allow_html=True)

st.title("🎙️ International AI Dubbing")
st.markdown("Precision Lip-Syncing • Multi-Language • 3D Audio Engine")

if not st.session_state.api_key:
    st.warning("⚠️ Please connect API Key in Sidebar to unlock Pro features.")
    st.stop()

# 3D Upload Container
uploaded_file = st.file_uploader("📂 Upload Source Video (Any Language)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
    
    # 3D Settings Panel
    st.markdown("### 🎛️ Production Settings")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        target_lang = st.selectbox("🌐 Target Language", list(VOICE_MAP.keys()))
    with c2:
        gender = st.selectbox("👤 Voice Gender", ["Male", "Female"])
    with c3:
        tone = st.selectbox("🎚️ Audio Tone", ["Natural", "Deep", "Fast", "Calm"])
        
    style = st.selectbox("🎭 Content Style", ["Narrator (Formal)", "Vlogger (Casual)", "Movie Recap (Dramatic)", "Documentary (Serious)"])
    
    st.write("")
    if st.button("🚀 START PRODUCTION RENDER", type="primary"):
        status = st.status("⚙️ Initializing AI Engine...", expanded=True)
        progress = st.progress(0)
        
        try:
            output = process_sync_dubbing(
                "temp.mp4", target_lang, gender, style, tone, 
                st.session_state.api_key, model_id, status, progress
            )
            
            status.update(label="✅ Rendering Complete!", state="complete", expanded=False)
            
            st.markdown(f"""
            <div class="success-box">
                <h3>✨ {target_lang} Dubbing Ready!</h3>
                <p>Synced to timeline with <b>{style}</b> style.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.video(output)
            with open(output, "rb") as f:
                st.download_button(f"💾 Download {target_lang} Video", f, f"dubbed_{target_lang}.mp4")
                
        except Exception as e:
            status.update(label="❌ Critical Error", state="error")
            st.error(f"System Log: {str(e)}")
            st.code(str(e))
