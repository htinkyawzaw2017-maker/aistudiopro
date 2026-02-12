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

FILE_INPUT = os.path.join(USER_SESSION_DIR, "input_video.mp4")
FILE_AUDIO_RAW = os.path.join(USER_SESSION_DIR, "extracted_audio.wav")
FILE_VOICE = os.path.join(USER_SESSION_DIR, "ai_voice.mp3")
FILE_FINAL = os.path.join(USER_SESSION_DIR, "final_dubbed_video.mp4")
FILE_CAP_INPUT = os.path.join(USER_SESSION_DIR, "caption_input_video.mp4")
FILE_CAP_WAV = os.path.join(USER_SESSION_DIR, "caption_audio.wav")
FILE_CAP_FINAL = os.path.join(USER_SESSION_DIR, "captioned_output.mp4")
FILE_ASS = os.path.join(USER_SESSION_DIR, "subtitles.ass")

# ---------------------------------------------------------
# 🎨 UI SETUP (Responsive Header & Red Buttons)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2072&auto=format&fit=crop");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 25, 47, 0.95);
        border-right: 1px solid #FF0000;
    }
    
    /* 🔴 RED BUTTONS STYLE (Bright Red as requested) */
    .stButton > button {
        background: linear-gradient(45deg, #FF0000, #B30000) !important;
        color: white !important;
        border: 2px solid #FF4444 !important;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(255, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 20px rgba(255, 0, 0, 0.7);
    }

    /* 📱 RESPONSIVE HEADER FOR MOBILE */
    .header-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 15px;
        padding: 20px;
        background: rgba(0,0,0,0.4);
        border-radius: 15px;
        margin-bottom: 20px;
    }
    
    .header-icon {
        width: 80px;
        height: 80px;
    }
    
    .header-text {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0px 0px 30px rgba(0, 201, 255, 0.5);
        margin: 0;
        line-height: 1.2;
    }

    /* Mobile Media Query */
    @media only screen and (max-width: 600px) {
        .header-container {
            flex-direction: column; /* Stack vertically on phone */
            gap: 10px;
            padding: 10px;
        }
        .header-icon {
            width: 50px;
            height: 50px;
        }
        .header-text {
            font-size: 1.8rem; /* Smaller text for mobile */
            text-align: center;
        }
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Responsive Header HTML
st.markdown("""
<div class="header-container">
    <img src="https://img.icons8.com/nolan/96/movie-projector.png" class="header-icon"/>
    <h1 class="header-text">MYANMAR AI STUDIO PRO</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'raw_transcript' not in st.session_state: st.session_state.raw_transcript = ""
if 'final_script' not in st.session_state: st.session_state.final_script = ""
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'processed_audio_path' not in st.session_state: st.session_state.processed_audio_path = None
if 'caption_video_path' not in st.session_state: st.session_state.caption_video_path = None
if 'google_creds' not in st.session_state: st.session_state.google_creds = None
if 'user_api_key' not in st.session_state: st.session_state.user_api_key = ""

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
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

def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please add 'ffmpeg' to packages.txt")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0.0

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
    except Exception as e: st.error(f"Whisper Error (Try reloading): {e}"); return None

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

# ---------------------------------------------------------
# 🧠 AI ENGINE (API KEY FIX)
# ---------------------------------------------------------
# New Logic: Directly use the user-provided key. No secrets loop.
def generate_content(prompt, image_input=None):
    api_key = st.session_state.user_api_key
    if not api_key:
        return "❌ Please enter your Gemini API Key in the sidebar first."
    
    genai.configure(api_key=api_key)
    model_name = st.session_state.get("selected_model", "gemini-1.5-pro")
    
    try:
        model = genai.GenerativeModel(model_name)
        custom_rules = load_custom_dictionary()
        full_prompt = f"RULES:\n{custom_rules}\n\nTASK:\n{prompt}" if custom_rules else prompt
        
        if image_input:
            response = model.generate_content([image_input, full_prompt])
        else:
            response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

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
# 🖥️ MAIN UI & SIDEBAR (User Input API Key)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 🔥 API KEY INPUT (MANDATORY & RED BORDER)
    st.markdown("### 🔑 Enter Your Gemini API Key")
    user_key = st.text_input("Paste your Key here:", type="password", help="Get a free key from Google AI Studio.")
    
    if user_key:
        st.session_state.user_api_key = user_key.strip()
        st.success("✅ Key Saved!")
    else:
        st.error("⚠️ API Key Required!")

    st.divider()

    st.markdown("☁️ **Google Cloud TTS (Optional):**")
    gcp_file = st.file_uploader("Upload service_account.json", type=["json"])
    if gcp_file:
        try:
            gcp_data = json.load(gcp_file)
            st.session_state.google_creds = service_account.Credentials.from_service_account_info(gcp_data)
            st.success("✅ GCP Key Loaded!")
        except: st.error("❌ Invalid JSON File")

    st.divider()
    st.session_state.selected_model = st.selectbox(
        "Model", 
        [
            "gemini-2.0-flash",           # အမြန်ဆုံးနဲ့ နောက်ဆုံးထွက်
            "gemini-2.0-flash-lite-preview-02-05", # စမ်းသပ်ဗားရှင်း (ပေါ့ပါးသည်)
            "gemini-1.5-pro",             # Vision အတွက် အကောင်းဆုံး
            "gemini-1.5-flash",
            "Gemini-2.5-Flash",
            "Gemini-2-Pro-Exp"
            # ပုံမှန်သုံးရန်
        ], 
        index=0
    )

    with st.expander("🚨 Danger Zone", expanded=False):
        if st.button("🗑️ Clear My Data"):
            try:
                if os.path.exists(USER_SESSION_DIR):
                    shutil.rmtree(USER_SESSION_DIR)
                    os.makedirs(USER_SESSION_DIR, exist_ok=True)
                    st.success("Data cleared!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e: st.error(str(e))

    if st.button("🔴 Reset System", use_container_width=True):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# ⚠️ STOP IF NO KEY
if not st.session_state.user_api_key:
    st.warning("⚠️ Please enter your Gemini API Key in the Sidebar to continue.")
    st.stop()

t1, t2, t3 = st.tabs(["🎙️ DUBBING STUDIO", "📝 AUTO CAPTION", "🚀 VIRAL SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    col_up, col_set = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    with col_set:
        task_mode = st.radio("Mode", ["🗣️ Translate (Dubbing)", "👀 AI Narration (Silent Video)"])
        
        if task_mode == "🗣️ Translate (Dubbing)":
            in_lang = st.selectbox("Input Language", ["English", "Burmese", "Japanese", "Chinese", "Thai"])
        else:
            vibe = st.selectbox("Narration Style", ["Vlog/Casual", "Tutorial/Explainer", "Relaxing/ASMR", "Exciting/Unboxing"])
            
        out_lang = st.selectbox("Output Language", ["Burmese", "English"], index=0)
    
    if uploaded:
        with open(FILE_INPUT, "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("🚀 Start Magic", use_container_width=True):
            check_requirements()
            p_bar = st.progress(0, text="Starting...")

            # PATH A: TRANSLATION
            if task_mode == "🗣️ Translate (Dubbing)":
                p_bar.progress(20, text="🎤 Listening to Audio...")
                subprocess.run(['ffmpeg', '-y', '-i', FILE_INPUT, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', FILE_AUDIO_RAW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                model = load_whisper_safe()
                if model:
                    lang_map = {"Burmese": "my", "English": "en", "Japanese": "ja", "Chinese": "zh", "Thai": "th"}
                    lang_code = lang_map.get(in_lang, "en")
                    raw = model.transcribe(FILE_AUDIO_RAW, language=lang_code)['text']
                    st.session_state.raw_transcript = raw
                    p_bar.progress(50, text="🧠 Translating...")
                    
                    recap_style_guide = """
                    ROLE: You are a famous Myanmar Movie Recap Narrator.
                    TONE: Dramatic, Flowing, Suspenseful.
                    STRICT WRITING RULES:
                    1. Use dramatic vocabulary ('မျက်ဝါးထင်ထင် တွေ့ရှိလိုက်ရတယ်').
                    2. Connect sentences smoothly using Cause & Effect.
                    3. End sentences naturally with 'ပါတော့တယ်', 'ခဲ့ပါတယ်', ''.
                    4. Do not use robotic fillers.
                    """
                    
                    if in_lang == out_lang:
                        prompt = f"""{recap_style_guide}\nTASK: Rewrite into flowing Recap script.\nInput: '{raw}'"""
                    else:
                        prompt = f"""{recap_style_guide}\nTASK: Translate to Burmese Recap script.\nInput: '{raw}'"""
                    
                    st.session_state.final_script = generate_content(prompt)

            # PATH B: AI NARRATION (VISION)
            else:
                p_bar.progress(20, text="👀 AI is watching video...")
                try:
                    genai.configure(api_key=st.session_state.user_api_key)
                    video_file = genai.upload_file(path=FILE_INPUT)
                    
                    while video_file.state.name == "PROCESSING":
                        time.sleep(2)
                        video_file = genai.get_file(video_file.name)

                    p_bar.progress(50, text="✍️ Writing Script...")
                    prompt = f"""
                    ROLE: Professional Video Narrator.
                    TASK: Write a voiceover script in {out_lang}.
                    STYLE: {vibe}.
                    RULES: Describe actions naturally. Match video pacing. Use engaging language.
                    """
                    st.session_state.final_script = generate_content(prompt, image_input=video_file)
                    genai.delete_file(video_file.name)
                except Exception as e:
                    st.error(f"AI Vision Error: {e}")
                    st.stop()

            p_bar.progress(100, text="✅ Script Ready!")
            st.rerun()
        
        txt = st.text_area("Final Script", st.session_state.final_script, height=200)

        st.markdown("---")
        st.markdown("#### ⚙️ Rendering Options")
        
        tts_engine = st.radio("Voice Engine", ["Edge TTS (Free)", "Google Cloud TTS (Pro)"], horizontal=True)
        c_fmt, c_spd = st.columns([1, 1.2]) 
        with c_fmt: export_format = st.radio("Export Format:", ["🎬 Video (MP4)", "🎵 Audio Only (MP3)"], horizontal=True)
        with c_spd:
            audio_speed = st.slider("🔊 Audio Speed", 0.5, 2.0, 1.0, 0.05)
            video_speed = st.slider("🎞️ Video Speed", 0.5, 4.0, 1.0, 0.1)

        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Voice Lang", list(VOICE_MAP.keys()), index=0 if out_lang == "Burmese" else 1)
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        zoom_val = st.slider("🔍 Copyright Zoom (Video Only)", 1.0, 1.2, 1.0, 0.01)

        btn_label = "🚀 GENERATE AUDIO" if "Audio" in export_format else "🚀 RENDER FINAL VIDEO"
        
        if st.button(btn_label, use_container_width=True):
            p_bar = st.progress(0, text="🚀 Initializing...")
            
            if not txt.strip(): st.error("❌ Script is empty!"); st.stop()

            p_bar.progress(30, text="🔊 Generating Speech...")
            try:
                success, msg = generate_audio_with_emotions(txt, target_lang, gender, v_mode, FILE_VOICE, engine=tts_engine, base_speed=audio_speed)
                if not success: st.error(f"❌ Audio Failed: {msg}"); st.stop()
                st.session_state.processed_audio_path = FILE_VOICE
            except Exception as e: st.error(f"Audio Error: {e}"); st.stop()
            
            if "Audio" in export_format:
                p_bar.progress(100, text="✅ Done!")
            else:
                p_bar.progress(50, text="🎞️ Rendering Video...")
                pts_val = 1.0 / video_speed
                w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                if w_s % 2 != 0: w_s += 1
                if h_s % 2 != 0: h_s += 1
                
                aud_dur = get_duration(FILE_VOICE)
                vid_dur = get_duration(FILE_INPUT) / video_speed
                
                cmd = ['ffmpeg', '-y', '-i', FILE_INPUT, '-i', FILE_VOICE, 
                       '-filter_complex', f"[0:v]setpts={pts_val}*PTS,scale={w_s}:{h_s},crop=1920:1080[vzoom]", 
                       '-map', '[vzoom]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', FILE_FINAL]
                
                if aud_dur > vid_dur:
                    cmd.insert(4, '-stream_loop'); cmd.insert(5, '-1')
                    cmd.append('-shortest')

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(FILE_FINAL) and os.path.getsize(FILE_FINAL) > 1000:
                    st.session_state.processed_video_path = FILE_FINAL
                    p_bar.progress(100, text="🎉 Done!")
                else: st.error("❌ Video Generation Failed")

    if st.session_state.processed_video_path and "Video" in export_format:
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f: st.download_button("🎬 Download Video", f, "dubbed.mp4", use_container_width=True)

    if st.session_state.processed_audio_path:
        st.audio(st.session_state.processed_audio_path)
        with open(st.session_state.processed_audio_path, "rb") as f: st.download_button("🎵 Download Audio", f, "voice.mp3", use_container_width=True)

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
