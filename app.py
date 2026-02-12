import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import streamlit as st
import google.generativeai as genai
import edge_tts
import subprocess
import shutil
import whisper
import time
import json
import re
import requests
import textwrap
import math
import uuid
import asyncio
from google.cloud import texttospeech
from google.oauth2 import service_account

# ---------------------------------------------------------
# 🛡️ 1. SESSION & FOLDER ISOLATION
# ---------------------------------------------------------
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

SID = st.session_state.session_id
BASE_WORK_DIR = os.path.abspath("user_sessions")
USER_SESSION_DIR = os.path.join(BASE_WORK_DIR, SID)
os.makedirs(USER_SESSION_DIR, exist_ok=True)

# File Paths
FILE_INPUT = os.path.join(USER_SESSION_DIR, "input.mp4")
FILE_AUDIO_RAW = os.path.join(USER_SESSION_DIR, "extracted.wav")
FILE_VOICE = os.path.join(USER_SESSION_DIR, "voice.mp3")
FILE_VIDEO_FREEZE = os.path.join(USER_SESSION_DIR, "video_frozen.mp4")
FILE_FINAL = os.path.join(USER_SESSION_DIR, "final_output.mp4")
FILE_TEMP_DIR = os.path.join(USER_SESSION_DIR, "temp_chunks")

FILE_CAP_INPUT = os.path.join(USER_SESSION_DIR, "cap_in.mp4")
FILE_CAP_WAV = os.path.join(USER_SESSION_DIR, "cap_audio.wav")
FILE_CAP_FINAL = os.path.join(USER_SESSION_DIR, "cap_out.mp4")
FILE_ASS = os.path.join(USER_SESSION_DIR, "subs.ass")

# ---------------------------------------------------------
# 🎨 UI SETUP (FIXED: Sidebar Button Visible + Neon White)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# Keep Screen Awake
keep_awake_js = """
<script>
async function requestWakeLock() {
  try {
    const wakeLock = await navigator.wakeLock.request('screen');
    console.log('Wake Lock is active!');
  } catch (err) {
    console.log(`${err.name}, ${err.message}`);
  }
}
requestWakeLock();
</script>
"""
st.components.v1.html(keep_awake_js, height=0, width=0)

st.markdown("""
    <style>
    /* 1. Pure White Background & Black Text */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* 2. HEADER FIX: Keep header visible so Sidebar button shows, but hide decoration */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #eeeeee;
        z-index: 999;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
    }
    /* Force Sidebar Toggle Button to be Black */
    button[kind="header"] {
        color: #000000 !important;
    }
    
    /* 3. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 2px solid #FF0000;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* 4. 🔴 3D RED BUTTONS */
    .stButton > button {
        background: linear-gradient(145deg, #ff0000, #cc0000) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px;
        box-shadow: 0px 5px 0px #990000; /* 3D Bottom Shadow */
        font-weight: bold;
        transition: all 0.1s;
    }
    .stButton > button:active {
        transform: translateY(4px);
        box-shadow: 0px 1px 0px #990000;
    }

    /* 5. NEON LIGHT 3D TEXT */
    .neon-text {
        font-family: 'Arial Black', sans-serif;
        font-size: 3.5rem;
        text-align: center;
        color: #fff;
        text-transform: uppercase;
        background-color: #111;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FF0000;
        /* Neon Glow Effect */
        text-shadow: 
            0 0 5px #FF0000,
            0 0 10px #FF0000,
            0 0 20px #FF0000,
            0 0 40px #FF0000;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.6);
        margin-bottom: 20px;
    }
    
    /* Mobile Fix */
    @media only screen and (max-width: 600px) {
        .neon-text { font-size: 1.8rem; padding: 15px; }
    }
    </style>
""", unsafe_allow_html=True)

# 3D Neon Header
st.markdown('<div class="neon-text">MYANMAR AI STUDIO</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'user_api_key' not in st.session_state: st.session_state.user_api_key = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
if 'google_creds' not in st.session_state: st.session_state.google_creds = None

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing.")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0.0

def load_custom_dictionary():
    dict_file = "dictionary.txt"
    if os.path.exists(dict_file):
        with open(dict_file, "r", encoding="utf-8") as f: return f.read()
    return ""

def load_pronunciation_dict():
    pron_file = "pronunciation.txt"
    replacements = {}
    if os.path.exists(pron_file):
        with open(pron_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        replacements[parts[0].strip()] = parts[1].strip()
    return replacements

def download_font():
    font_dir = os.path.abspath("fonts_cache")
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "Padauk-Bold.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_path, 'wb') as f: f.write(r.content)
        except: pass
    return font_path

def load_whisper_safe():
    try: return whisper.load_model("base")
    except Exception as e: st.error(f"Whisper Error: {e}"); return None

# ---------------------------------------------------------
# ❄️ FREEZE LOGIC (Robust)
# ---------------------------------------------------------
def process_video_with_freeze(input_path, output_path, interval_sec, freeze_duration=3.0):
    """
    Splits video at intervals, inserts a 3s freeze frame, creates a longer video.
    """
    if interval_sec <= 0:
        shutil.copy(input_path, output_path)
        return True

    try:
        # Clean temp directory
        if os.path.exists(FILE_TEMP_DIR): shutil.rmtree(FILE_TEMP_DIR)
        os.makedirs(FILE_TEMP_DIR, exist_ok=True)
        
        total_duration = get_duration(input_path)
        current_time = 0.0
        segment_idx = 0
        concat_list_path = os.path.join(FILE_TEMP_DIR, "concat.txt")
        
        with open(concat_list_path, "w") as f:
            while current_time < total_duration:
                # 1. Cut Normal Segment
                duration = min(interval_sec, total_duration - current_time)
                seg_file = os.path.join(FILE_TEMP_DIR, f"seg_{segment_idx}.mp4")
                
                subprocess.run([
                    'ffmpeg', '-y', '-ss', str(current_time), '-t', str(duration),
                    '-i', input_path, '-c:v', 'libx264', '-c:a', 'aac', seg_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                f.write(f"file '{seg_file}'\n")
                
                # 2. Add Freeze Frame (if not end)
                if current_time + duration < total_duration:
                    freeze_file = os.path.join(FILE_TEMP_DIR, f"freeze_{segment_idx}.mp4")
                    
                    # Extract last frame
                    subprocess.run([
                        'ffmpeg', '-y', '-sseof', '-0.1', '-i', seg_file,
                        '-update', '1', '-q:v', '2', os.path.join(FILE_TEMP_DIR, "last.jpg")
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Make freeze video (Silent)
                    subprocess.run([
                        'ffmpeg', '-y', '-loop', '1', '-i', os.path.join(FILE_TEMP_DIR, "last.jpg"),
                        '-f', 'lavfi', '-i', 'anullsrc', 
                        '-t', str(freeze_duration), '-c:v', 'libx264', '-c:a', 'aac', 
                        '-pix_fmt', 'yuv420p', freeze_file
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    f.write(f"file '{freeze_file}'\n")

                current_time += duration
                segment_idx += 1

        # 3. Concatenate
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path,
            '-c', 'copy', output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return True
    except Exception as e:
        st.error(f"Freeze Error: {e}")
        return False

# ---------------------------------------------------------
# 🔊 AUDIO ENGINE
# ---------------------------------------------------------
VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
GOOGLE_VOICE_MAP = {
    "Burmese": {"Male": "my-MM-Standard-A", "Female": "my-MM-Standard-A"}, 
    "English": {"Male": "en-US-Neural2-D", "Female": "en-US-Neural2-F"},
}
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-5%", "pitch": "-2Hz"}, 
    "Recap": {"rate": "+5%", "pitch": "+0Hz"},
    "Motivation": {"rate": "+10", "pitch": "+2Hz"},
}
EMOTION_MAP = {
    "[normal]": {"rate": "+0%", "pitch": "+0Hz"},
    "[sad]":    {"rate": "-15%", "pitch": "-15Hz"},
    "[angry]":  {"rate": "+15%", "pitch": "+5Hz"},
    "[happy]":  {"rate": "+10%", "pitch": "+15Hz"},
    "[action]": {"rate": "+30%", "pitch": "+0Hz"},
    "[whisper]": {"rate": "-10%", "pitch": "-20Hz"},
}

def generate_edge_chunk(text, lang, gender, rate_str, pitch_str, output_file):
    voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
    cmd = ["edge-tts", "--voice", voice_id, "--text", text, f"--rate={rate_str}", f"--pitch={pitch_str}", "--write-media", output_file]
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return True
        except: time.sleep(1); continue
    return False

def generate_google_chunk(text, lang, gender, rate_val, pitch_val, output_file, creds):
    try:
        client = texttospeech.TextToSpeechClient(credentials=creds)
        s_input = texttospeech.SynthesisInput(text=text)
        g_voice_name = GOOGLE_VOICE_MAP.get(lang, {}).get(gender, "en-US-Neural2-F")
        lang_code = "my-MM" if lang == "Burmese" else "en-US"
        voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=g_voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=rate_val, pitch=pitch_val)
        response = client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
        with open(output_file, "wb") as out: out.write(response.audio_content)
        return True
    except Exception as e: print(f"Google TTS Error: {e}"); return False

def num_to_burmese_spoken(num_str):
    try:
        num_str = num_str.replace(",", "")
        n = int(num_str)
        if n == 0: return "သုည"
        digit_map = ["", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"]
        def convert_chunk(number):
            parts = []
            if number >= 10000000: parts.append(convert_chunk(number // 10000000) + "ကုဋေ"); number %= 10000000
            if number >= 1000000: parts.append(digit_map[number // 1000000] + "သန်း"); number %= 1000000
            if number >= 100000: parts.append(digit_map[number // 100000] + "သိန်း"); number %= 100000
            if number >= 10000: parts.append(digit_map[number // 10000] + "သောင်း"); number %= 10000
            if number >= 1000: parts.append(digit_map[number // 1000] + "ထောင်"); number %= 1000
            if number >= 100: parts.append(digit_map[number // 100] + "ရာ"); number %= 100
            if number >= 10: parts.append(digit_map[number // 10] + "ဆယ်"); number %= 10
            if number > 0: parts.append(digit_map[number])
            return "".join(parts)
        result = convert_chunk(n)
        result = result.replace("ထောင်", "ထောင့်").replace("ရာ", "ရာ့").replace("ဆယ်", "ဆယ့်")
        if result.endswith("ထောင့်"): result = result[:-1] + "င်"
        if result.endswith("ရာ့"): result = result[:-1]
        if result.endswith("ဆယ့်"): result = result[:-1]
        return result
    except: return num_str

def normalize_text_for_tts(text):
    if not text: return ""
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    pron_dict = load_pronunciation_dict()
    sorted_keys = sorted(pron_dict.keys(), key=len, reverse=True)
    for original in sorted_keys:
        text = re.compile(re.escape(original), re.IGNORECASE).sub(pron_dict[original], text)
    text = text.replace("၊", ", ").replace("။", ". ").replace("[p]", "... ") 
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda x: num_to_burmese_spoken(x.group()), text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_audio_with_emotions(full_text, lang, gender, base_mode, output_file, engine="Edge TTS", base_speed=1.0):
    base_settings = VOICE_MODES.get(base_mode, VOICE_MODES["Normal"])
    base_r_int = int(base_settings['rate'].replace('%', ''))
    base_p_int = int(base_settings['pitch'].replace('Hz', ''))
    slider_adj = int((base_speed - 1.0) * 100)
    current_rate = base_r_int + slider_adj
    current_pitch = base_p_int

    parts = re.split(r'(\[.*?\])', full_text)
    audio_segments = []
    chunk_idx = 0
    output_dir = os.path.dirname(output_file)
    
    for part in parts:
        part = part.strip()
        if not part: continue
        part_lower = part.lower()

        if part_lower == "[p]":
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_idx}_silence.mp3")
            cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', '1', '-q:a', '9', chunk_filename]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(chunk_filename): audio_segments.append(chunk_filename); chunk_idx += 1
            continue

        if part_lower in EMOTION_MAP:
            emo = EMOTION_MAP[part_lower]
            base_r = int(base_settings['rate'].replace('%', '')) + slider_adj
            base_p = int(base_settings['pitch'].replace('Hz', ''))
            current_rate = base_r + int(emo['rate'].replace('%', ''))
            current_pitch = base_p + int(emo['pitch'].replace('Hz', ''))
            continue
        
        if part.startswith("[") and part.endswith("]"): continue
        
        processed_text = normalize_text_for_tts(part)
        if not processed_text.strip(): continue
        
        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_idx}.mp3")
        success = False
        if engine == "Google Cloud TTS" and st.session_state.google_creds:
            g_rate = 1.0 + (current_rate / 100.0)
            g_pitch = current_pitch / 10.0 
            success = generate_google_chunk(processed_text, lang, gender, g_rate, g_pitch, chunk_filename, st.session_state.google_creds)
        else:
            rate_str = f"{current_rate:+d}%"
            pitch_str = f"{current_pitch:+d}Hz"
            success = generate_edge_chunk(processed_text, lang, gender, rate_str, pitch_str, chunk_filename)
        
        if success:
            audio_segments.append(chunk_filename)
            chunk_idx += 1
            if engine == "Edge TTS": time.sleep(0.1)

    if not audio_segments: return False, "No audio generated"
    
    concat_list = os.path.join(output_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for seg in audio_segments: f.write(f"file '{seg}'\n")
            
    try:
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

# ---------------------------------------------------------
# 🧠 AI ENGINE (VISION & TEXT)
# ---------------------------------------------------------
def generate_content(prompt, image_input=None):
    api_key = st.session_state.user_api_key
    if not api_key: return "⚠️ Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    # Corrected Model List
    model_name = st.session_state.get("selected_model", "gemini-2.5-flash")
    
    try:
        model = genai.GenerativeModel(model_name)
        custom_rules = load_custom_dictionary()
        full_prompt = f"RULES:\n{custom_rules}\n\nTASK:\n{prompt}" if custom_rules else prompt

        if image_input:
            # 🔥 AI VISION LOGIC
            response = model.generate_content([image_input, full_prompt])
        else:
            response = model.generate_content(full_prompt)
        return response.text
    except Exception as e: return f"AI Error: {str(e)}"

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE
# ---------------------------------------------------------
def generate_ass_file(segments, font_path, output_path):
    def seconds_to_ass(seconds):
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60); cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCut,Padauk-Bold,24,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            raw_text = seg['text'].strip()
            wrapped_lines = textwrap.wrap(raw_text, width=40)
            final_text = "\\N".join(wrapped_lines) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return output_path

# ---------------------------------------------------------
# 🖥️ SIDEBAR (NOW VISIBLE)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ SETTINGS")
    
    # API KEY INPUT
    st.markdown("### 🔑 API Key")
    user_key = st.text_input("Paste Gemini API Key", type="password", help="Visible now!")
    if user_key:
        st.session_state.user_api_key = user_key.strip()
        st.success("✅ Key Active")
    else:
        st.warning("⚠️ Key Missing")

    st.divider()
    
    # FREEZE SETTINGS
    st.markdown("### ❄️ FREEZE EFFECT")
    freeze_option = st.selectbox(
        "Freeze Interval (3s Pause)", 
        ["None", "Every 30 Seconds", "Every 1 Minute", "Every 2 Minutes"]
    )
    
    freeze_interval = 0
    if freeze_option == "Every 30 Seconds": freeze_interval = 30
    elif freeze_option == "Every 1 Minute": freeze_interval = 60
    elif freeze_option == "Every 2 Minutes": freeze_interval = 120
    
    st.divider()
    
    st.markdown("☁️ **Google Cloud TTS:**")
    gcp_file = st.file_uploader("Upload service_account.json", type=["json"])
    if gcp_file:
        try:
            gcp_data = json.load(gcp_file)
            st.session_state.google_creds = service_account.Credentials.from_service_account_info(gcp_data)
            st.success("✅ GCP Active")
        except: st.error("❌ Invalid JSON")

    st.divider()
    # Model Selection
    st.session_state.selected_model = st.selectbox(
        "AI Model", 
        ["", "", "gemini-2.5-flash"], 
        index=0
    )

    if st.button("🗑️ Clear Cache"):
        if os.path.exists(USER_SESSION_DIR): shutil.rmtree(USER_SESSION_DIR)
        st.rerun()

# ---------------------------------------------------------
# 🎞️ MAIN APP
# ---------------------------------------------------------
if not st.session_state.user_api_key:
    st.info("👈 Please enter API Key in sidebar (Slider Button is now visible at top left).")
    st.stop()

t1, t2, t3 = st.tabs(["🎬 STUDIO", "📝 CAPTION", "🚀 VIRAL SEO"])

with t1:
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    with col2:
        mode = st.radio("Mode", ["🗣️ Translate", "👀 Narration"])
        out_lang = st.selectbox("Output Language", ["Burmese", "English"])

    if uploaded:
        with open(FILE_INPUT, "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("🚀 START PROCESS", use_container_width=True):
            p_bar = st.progress(0, "Initializing...")
            check_requirements()
            
            # 1. SCRIPT
            if mode == "🗣️ Translate":
                p_bar.progress(20, "🎧 Extracting Audio...")
                subprocess.run(['ffmpeg', '-y', '-i', FILE_INPUT, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', FILE_AUDIO_RAW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                model = load_whisper_safe()
                if model:
                    raw_text = model.transcribe(FILE_AUDIO_RAW)['text']
                    prompt = f"Rewrite as dramatic {out_lang} recap: '{raw_text}'"
                    st.session_state.final_script = generate_content(prompt)
            else:
                p_bar.progress(20, "👀 Watching Video...")
                genai.configure(api_key=st.session_state.user_api_key)
                video_file = genai.upload_file(path=FILE_INPUT)
                while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
                
                prompt = f"Write a dramatic {out_lang} narration script for this video."
                st.session_state.final_script = generate_content(prompt, image_input=video_file)
            
            # 2. AUDIO
            p_bar.progress(60, "🔊 Generating Voice...")
            gender = "Male"
            v_mode = "Recap"
            tts_engine = "Edge TTS" 
            
            success, msg = generate_audio_with_emotions(
                st.session_state.final_script, 
                out_lang, 
                gender, 
                v_mode, 
                FILE_VOICE, 
                engine=tts_engine
            )
            
            if not success:
                st.error(f"Audio Failed: {msg}")
                st.stop()

            # 3. VIDEO PROCESSING (FREEZE)
            p_bar.progress(80, "🎞️ Processing Video...")
            video_source = FILE_INPUT
            
            if freeze_interval > 0:
                success = process_video_with_freeze(FILE_INPUT, FILE_VIDEO_FREEZE, freeze_interval, freeze_duration=3.0)
                if success: video_source = FILE_VIDEO_FREEZE

            # 4. MERGE
            cmd = ['ffmpeg', '-y', '-i', video_source, '-i', FILE_VOICE, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', FILE_FINAL]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(FILE_FINAL):
                st.session_state.processed_video_path = FILE_FINAL
                p_bar.progress(100, "🎉 DONE!")
            else:
                st.error("Render Failed")

    # RESULT
    if st.session_state.final_script:
        st.text_area("Script", st.session_state.final_script)
        
    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button("📥 Download Video", f, "output.mp4", use_container_width=True)

# === TAB 2: AUTO CAPTION ===
with t2:
    st.subheader("📝 Auto Caption")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    if cap_up:
        with open(FILE_CAP_INPUT, "wb") as f: f.write(cap_up.getbuffer())
        if st.button("Generate Captions", use_container_width=True):
            check_requirements(); font_path = download_font()
            p_bar = st.progress(0, text="Processing...")
            subprocess.run(['ffmpeg', '-y', '-i', FILE_CAP_INPUT, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', FILE_CAP_WAV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                segments = model.transcribe(FILE_CAP_WAV, task="transcribe")['segments']
                trans_segments = []
                for i, seg in enumerate(segments):
                    p_bar.progress(int((i/len(segments))*50), text=f"🧠 Translating...")
                    txt = seg['text'].strip()
                    if txt:
                        burmese = generate_content(f"Translate to Burmese. Short. Input: '{txt}'")
                        trans_segments.append({'start': seg['start'], 'end': seg['end'], 'text': burmese})
                        time.sleep(0.3)
                p_bar.progress(90, text="✍️ Burning Subtitles...")
                generate_ass_file(trans_segments, font_path, FILE_ASS)
                font_dir = os.path.dirname(font_path)
                subprocess.run(['ffmpeg', '-y', '-i', FILE_CAP_INPUT, '-vf', f"ass={FILE_ASS}:fontsdir={font_dir}", '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'ultrafast', FILE_CAP_FINAL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(FILE_CAP_FINAL):
                    st.session_state.caption_video_path = FILE_CAP_FINAL
                    p_bar.progress(100, text="Done!")

    if st.session_state.caption_video_path:
        st.video(st.session_state.caption_video_path)
        with open(st.session_state.caption_video_path, "rb") as f: st.download_button("Download", f, "captioned.mp4", use_container_width=True)

# === TAB 3: VIRAL SEO ===
with t3:
    st.subheader("🚀 Viral Kit SEO")
    if st.session_state.final_script:
        if st.button("Generate Metadata", use_container_width=True):
            with st.spinner("Analyzing..."):
                prompt = f"""Based on: {st.session_state.final_script}\nGenerate:\n1. 5 Clickbait Titles (Burmese)\n2. 10 Hashtags\n3. Description"""
                seo_result = generate_content(prompt)
                st.success("SEO Generated!")
                st.code(seo_result, language="markdown")
    else:
        st.info("Please generate a script in Tab 1 first.")
