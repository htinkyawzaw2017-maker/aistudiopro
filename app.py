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

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# 🔥 NEW: Google Cloud Imports
from google.cloud import texttospeech
from google.oauth2 import service_account

# ---------------------------------------------------------
# 🎨 UI SETUP (NEON SPACE THEME)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2072&auto=format&fit=crop");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        color: #e0e0e0;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .main-header {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 0px 0px 30px rgba(0, 201, 255, 0.5);
        margin-bottom: 20px;
    }
    div[data-testid="stFileUploader"] {
        background-color: rgba(10, 25, 47, 0.9);
        border: 1px solid #00d2ff;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stExpander"], div[class="stTextArea"] {
        background-color: rgba(10, 25, 47, 0.85);
        border: 1px solid #00d2ff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff00cc, #3333ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255, 0, 204, 0.4);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(255, 0, 204, 0.7);
    }
    input, textarea, select, div[data-baseweb="select"] > div {
        background-color: rgba(10, 25, 47, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid #00d2ff !important;
        border-radius: 8px !important;
    }
    div[role="listbox"] ul {
        background-color: #050b14 !important;
        color: white !important;
    }
    p, label {
        color: #e0e0e0 !important;
    }
    div[data-baseweb="tab-list"] {
        background-color: rgba(0,0,0,0.5);
        border-radius: 15px;
        padding: 5px;
        border: 1px solid #333;
    }
    div[data-baseweb="tab"] {
        color: #888;
        font-weight: bold;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #ff4b1f, #ff9068);
        color: white !important;
        border-radius: 10px;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="https://img.icons8.com/nolan/96/movie-projector.png" width="80"/>
    <div class="main-header"></div>
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
if 'api_keys' not in st.session_state: st.session_state.api_keys = []
if 'google_creds' not in st.session_state: st.session_state.google_creds = None

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
    font_filename = "Padauk-Bold.ttf"
    if not os.path.exists(font_filename):
        url = "https://github.com/googlefonts/padauk/raw/main/fonts/ttf/Padauk-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_filename, 'wb') as f: f.write(r.content)
        except: pass
    return os.path.abspath(font_filename)

def load_whisper_safe():
    try: return whisper.load_model("base")
    except Exception as e: st.error(f"Whisper Error: {e}"); return None

# ---------------------------------------------------------
# 🔊 DUAL AUDIO ENGINE (EDGE & GOOGLE)
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
    "[action]": {"rate": "+20%", "pitch": "+0Hz"},
    "[whisper]": {"rate": "-10%", "pitch": "-20Hz"},
}

# 1. Edge TTS Generator (With Retry)
def generate_edge_chunk(text, lang, gender, rate_str, pitch_str, output_file):
    voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
    cmd = ["edge-tts", "--voice", voice_id, "--text", text, f"--rate={rate_str}", f"--pitch={pitch_str}", "--write-media", output_file]
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return True
        except: time.sleep(1); continue
    return False

# 2. Google Cloud TTS Generator
def generate_google_chunk(text, lang, gender, rate_val, pitch_val, output_file, creds):
    try:
        client = texttospeech.TextToSpeechClient(credentials=creds)
        s_input = texttospeech.SynthesisInput(text=text)
        
        g_voice_name = GOOGLE_VOICE_MAP.get(lang, {}).get(gender, "en-US-Neural2-F")
        lang_code = "my-MM" if lang == "Burmese" else "en-US"
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=g_voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=rate_val,
            pitch=pitch_val
        )

        response = client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        return True
    except Exception as e:
        print(f"Google TTS Error: {e}")
        return False


# 3. Main Audio Logic (Switchable) - UPDATED FOR GLITCH FIX
def generate_audio_with_emotions(full_text, lang, gender, base_mode, output_file, engine="Edge TTS", base_speed=1.0):
    base_settings = VOICE_MODES.get(base_mode, VOICE_MODES["Normal"])
    
    # Calculate Base Params
    base_r_int = int(base_settings['rate'].replace('%', ''))
    base_p_int = int(base_settings['pitch'].replace('Hz', ''))
    slider_adj = int((base_speed - 1.0) * 100)
    current_rate = base_r_int + slider_adj
    current_pitch = base_p_int

    # Regex split to keep tags
    parts = re.split(r'(\[.*?\])', full_text)
    audio_segments = []
    chunk_idx = 0

    for part in parts:
        part = part.strip()
        
        # 🔥 LOGIC 1: Filter Empty Spaces (Glitch Fix)
        # စာမရှိရင်၊ Space ပဲရှိရင် လုံးဝ ကျော်မယ်
        if not part: 
            continue

        part_lower = part.lower()

        # 🔥 LOGIC 2: Handle [p] as Silence (Not Text)
        if part_lower == "[p]":
            chunk_filename = f"chunk_{chunk_idx}_silence.mp3"
            # FFmpeg ကိုသုံးပြီး ၁ စက္ကန့်စာ အသံတိတ်ဖိုင် ထုတ်မယ်
            cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', '1', '-q:a', '9', chunk_filename]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(chunk_filename):
                audio_segments.append(chunk_filename)
                chunk_idx += 1
            continue

        # Handle Emotion Tags
        if part_lower in EMOTION_MAP:
            emo = EMOTION_MAP[part_lower]
            base_r = int(base_settings['rate'].replace('%', '')) + slider_adj
            base_p = int(base_settings['pitch'].replace('Hz', ''))
            current_rate = base_r + int(emo['rate'].replace('%', ''))
            current_pitch = base_p + int(emo['pitch'].replace('Hz', ''))
            continue
        
        # 🔥 LOGIC 3: Skip Unknown Tags
        # Tag ပုံစံ [...] ဖြစ်ပြီး EMOTION_MAP ထဲမရှိရင် စာလိုမဖတ်ဘဲ ကျော်မယ်
        if part.startswith("[") and part.endswith("]"):
            continue
        
        # Normal Text Processing
        chunk_filename = f"chunk_{chunk_idx}.mp3"
        processed_text = normalize_text_for_tts(part)
        
        # စာသားသန့်စင်ပြီးလို့ ဘာမှမကျန်ရင် (ဥပမာ သင်္ကေတသက်သက်) အသံမထုတ်ဘူး
        if not processed_text.strip():
            continue
        
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
            if engine == "Edge TTS": time.sleep(0.1) # Stability Delay

    if not audio_segments: return False, "No audio generated"
    
    concat_list = "audio_concat.txt"
    with open(concat_list, "w") as f:
        for seg in audio_segments: f.write(f"file '{seg}'\n")
            
    try:
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Cleanup
        for seg in audio_segments: 
            if os.path.exists(seg): os.remove(seg)
        if os.path.exists(concat_list): os.remove(concat_list)
        return True, "Success"
    except Exception as e: return False, str(e)


# ---------------------------------------------------------
# 🔢 TEXT NORMALIZATION
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    try:
        if "." in num_str:
            parts = num_str.split(".")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                left = num_to_burmese_spoken(parts[0])
                right = num_to_burmese_spoken(parts[1]) 
                return f"{left} ဒသမ {right}"

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
        fixed_sound = pron_dict[original]
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(fixed_sound, text)
    text = text.replace("၊", ", ").replace("။", ". ").replace("[p]", "... ") 
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda x: num_to_burmese_spoken(x.group()), text)
    text = text.replace("လုံလောက် တဲ့", "လုံလောက်တဲ့").replace("လုံလောက်သော", "လုံလောက်တဲ့")
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 🧠 AI ENGINE
# ---------------------------------------------------------
def get_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_with_retry(prompt):
    keys = st.session_state.api_keys
    model_name = st.session_state.get("selected_model", "gemini-1.5-flash")
    custom_rules = load_custom_dictionary()
    if custom_rules:
        prompt = f"RULES:\n{custom_rules}\n\nTASK:\n{prompt}"
    for i, key in enumerate(keys):
        try:
            model = get_model(key, model_name)
            response = model.generate_content(prompt)
            return response.text
        except: continue
    return "AI Error: All keys failed."

# ---------------------------------------------------------
# 📝 .ASS SUBTITLE
# ---------------------------------------------------------
def generate_ass_file(segments, font_path):
    filename = "captions.ass"
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
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            start = seconds_to_ass(seg['start'])
            end = seconds_to_ass(seg['end'])
            raw_text = seg['text'].strip()
            wrapped_lines = textwrap.wrap(raw_text, width=40)
            final_text = "\\N".join(wrapped_lines) 
            f.write(f"Dialogue: 0,{start},{end},CapCut,,0,0,0,,{final_text}\n")
    return filename

# ---------------------------------------------------------
# 🖥️ MAIN UI & SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 🔥 FIXED: Moved JSON Uploader to TOP for visibility
    st.markdown("☁️ **Google Cloud TTS (Optional):**")
    gcp_file = st.file_uploader("Upload service_account.json", type=["json"], help="Upload your GCP Key here to unlock Pro voices.")
    if gcp_file:
        try:
            gcp_data = json.load(gcp_file)
            st.session_state.google_creds = service_account.Credentials.from_service_account_info(gcp_data)
            st.success("✅ GCP Key Loaded!")
        except: st.error("❌ Invalid JSON File")

    st.divider()

    with st.expander("🔑 API Keys & Model", expanded=True):
        try:
            if "GOOGLE_API_KEYS" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEYS"]
            elif "GOOGLE_API_KEY" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEY"]
            else: default_keys = ""
        except: default_keys = ""
        api_key_input = st.text_input("Gemini API Keys", value=default_keys, type="password")
        if api_key_input:
            st.session_state.api_keys = [k.strip() for k in api_key_input.split(",") if k.strip()]
        
        st.session_state.selected_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash-exp"], index=0)

    with st.expander("📚 Knowledge & Pronunciation", expanded=False):
        st.info("📂 System is using internal files.")
        if st.button("🔄 Reload Files"): st.rerun()

    if st.button("🔴 Reset System", use_container_width=True):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_keys: st.warning("⚠️ Enter Gemini API Keys"); st.stop()

t1, t2, t3 = st.tabs(["🎙️ DUBBING STUDIO", "📝 AUTO CAPTION", "🚀 VIRAL SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    col_up, col_set = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    with col_set:
        # 🔥 UI FIX: Input နဲ့ Output ကို ခွဲခြားလိုက်ခြင်း
        in_lang = st.selectbox("Input (Video Language)", ["English", "Burmese", "Japanese", "Chinese", "Thai"])
        out_lang = st.selectbox("Output (Script & Voice)", ["Burmese", "English", "Japanese", "Chinese", "Thai"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("📝 Extract & Process", use_container_width=True):
            check_requirements()
            p_bar = st.progress(0, text="Starting...")
            p_bar.progress(20, text="🎤 Transcribing Audio...")
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                # 1. Whisper အတွက် Language Code သတ်မှတ်ခြင်း
                lang_map = {"Burmese": "my", "English": "en", "Japanese": "ja", "Chinese": "zh", "Thai": "th"}
                lang_code = lang_map.get(in_lang, "en")

                # Transcribe လုပ်ခြင်း
                raw = model.transcribe("temp.wav", language=lang_code)['text']
                st.session_state.raw_transcript = raw
                
                p_bar.progress(60, text="🧠 AI Processing...")
                
                # 🔥 LOGIC FIX: Input နဲ့ Output တူ/မတူ ကြည့်ပြီး Prompt ပြောင်းခြင်း
                if in_lang == out_lang:
                    # ဘာသာစကားတူရင် Refine ပဲလုပ်မယ်
                    prompt = f"Act as a professional {out_lang} editor. Refine this text for clarity and flow. Do not translate. Input: '{raw}'"
                else:
                    # ဘာသာစကားမတူရင် Translate လုပ်မယ်
                    prompt = f"Translate the following {in_lang} text into {out_lang}. Ensure the tone is natural and suitable for a video script. Input: '{raw}'"
                
                st.session_state.final_script = generate_with_retry(prompt)
                p_bar.progress(100, text="✅ Done!")
                st.rerun()
        
        txt = st.session_state.final_script if st.session_state.final_script else ""
        word_count = len(txt.split())
        est_min = round(word_count / 250, 1)
        st.caption(f"⏱️ Est. Duration: ~{est_min} mins")
        
        if st.session_state.final_script:
            st.markdown("### 🎬 Script & Production")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            # 🔥 Dynamic Refine Button (Output Language ပေါ်မူတည်ပြီး အလုပ်လုပ်မည်)
            refine_label = f"✨ Refine: {out_lang} Recap Style"
            if st.button(refine_label, use_container_width=True):
                with st.spinner("Refining..."):
                    prompt = f"""Act as a professional {out_lang} Movie Recap Narrator. 
                    Rewrite the input text into an engaging Recap Script in {out_lang}.
                    RULES: 1. Use about 250 words per minute. 2. Add tags like [p], [action], [sad]. 3. Do not summarize.
                    Input: "{st.session_state.final_script}" """
                    
                    st.session_state.final_script = generate_with_retry(prompt)
                    st.rerun()

        with c_opt2:
            if st.button("↩️ Reset Script", use_container_width=True):
                st.session_state.final_script = st.session_state.raw_transcript
                st.rerun()

        txt = st.text_area("Final Script", st.session_state.final_script, height=200)

        st.markdown("---")
        st.markdown("#### ⚙️ Rendering Options")
        
        tts_engine = st.radio("Voice Engine", ["Edge TTS (Free)", "Google Cloud TTS (Pro)"], horizontal=True)
        if tts_engine == "Google Cloud TTS (Pro)" and not st.session_state.google_creds:
            st.error("⚠️ Please upload 'service_account.json' in Settings Sidebar to use Google Cloud TTS.")

        c_fmt, c_spd = st.columns([1, 1.2]) 
        with c_fmt:
            export_format = st.radio("Export Format:", ["🎬 Video (MP4)", "🎵 Audio Only (MP3)"], horizontal=True)
        with c_spd:
            audio_speed = st.slider("🔊 Audio Speed", 0.5, 2.0, 1.0, 0.05)
            video_speed = st.slider("🎞️ Video Speed", 0.5, 4.0, 1.0, 0.1)

        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: 
            # Voice Language ကို Output Language နဲ့ တိုက်ရိုက်ချိတ်ဆက်လိုက်ခြင်း
            target_lang = st.selectbox("Voice Lang", list(VOICE_MAP.keys()), index=0 if out_lang == "Burmese" else 1)
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        zoom_val = st.slider("🔍 Copyright Zoom (Video Only)", 1.0, 1.2, 1.0, 0.01)

        btn_label = "🚀 GENERATE AUDIO" if "Audio" in export_format else "🚀 RENDER FINAL VIDEO"
        
        if st.button(btn_label, use_container_width=True):
            p_bar = st.progress(0, text="🚀 Initializing...")
            p_bar.progress(30, text="🔊 Generating Neural Speech...")
            try:
                generate_audio_with_emotions(txt, target_lang, gender, v_mode, "voice.mp3", engine=tts_engine, base_speed=audio_speed)
                st.session_state.processed_audio_path = "voice.mp3"
            except Exception as e:
                st.error(f"Audio Error: {e}"); st.stop()
            
            if "Audio" in export_format:
                p_bar.progress(100, text="✅ Audio Generated!")
            else:
                p_bar.progress(50, text="🎞️ Adjusting Video Speed & Logic...")
                input_vid = "input.mp4"
                raw_vid_dur = get_duration(input_vid)
                vid_dur = raw_vid_dur / video_speed 
                aud_dur = get_duration("voice.mp3")

                # 🔥 Unique Filename သတ်မှတ်ခြင်း
                out_vid = f"final_{st.session_state.session_id}.mp4"

                p_bar.progress(80, text="🎬 Merging & Finalizing...")
                pts_val = 1.0 / video_speed
                w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                if w_s % 2 != 0: w_s += 1
                if h_s % 2 != 0: h_s += 1
                
                if aud_dur > vid_dur:
                    cmd = ['ffmpeg', '-y', '-stream_loop', '-1', '-i', input_vid, '-i', "voice.mp3", '-filter_complex', f"[0:v]setpts={pts_val}*PTS,scale={w_s}:{h_s},crop=1920:1080[vzoom]", '-map', '[vzoom]', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', out_vid]
                else:
                    cmd = ['ffmpeg', '-y', '-i', input_vid, '-i', "voice.mp3", '-filter_complex', f"[0:v]setpts={pts_val}*PTS,scale={w_s}:{h_s},crop=1920:1080[vzoom]", '-map', '[vzoom]', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', out_vid]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 🔥 ဖိုင်တကယ်ထွက်မထွက် စစ်ဆေးခြင်း
                if os.path.exists(out_vid):
                    p_bar.progress(100, text="🎉 Video Complete!")
                    st.session_state.processed_video_path = out_vid
                else:
                    st.error("❌ Rendering Fail: ဗီဒီယိုဖိုင် ထွက်မလာပါဘူး။")



    if st.session_state.processed_video_path and "Video" in export_format:
        # 🔥 ဖိုင်တကယ်ရှိမရှိ အရင်စစ်မယ်
        if os.path.exists(st.session_state.processed_video_path):
            try:
                st.video(st.session_state.processed_video_path)
            except Exception as e:
                st.error("ဗီဒီယိုကို ပြသလို့မရပါဘူး။ ဖိုင်ဆိုဒ် အရမ်းကြီးနေတာ ဖြစ်နိုင်ပါတယ်။")
        else:
            st.error("❌ ဗီဒီယိုဖိုင် ထွက်မလာပါဘူး။ Rendering လုပ်တာ တစ်ခုခု မှားယွင်းသွားပါတယ်။")

    if st.session_state.processed_audio_path:
        st.audio(st.session_state.processed_audio_path)
        with open(st.session_state.processed_audio_path, "rb") as f: st.download_button("🎵 Download Audio", f, "voice.mp3", use_container_width=True)

# === TAB 2: AUTO CAPTION ===
with t2:
    st.subheader("📝 Auto Caption")
    cap_up = st.file_uploader("Upload Video", type=['mp4','mov'], key="cap")
    if cap_up:
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        if st.button("Generate Captions", use_container_width=True):
            check_requirements(); font_path = download_font()
            p_bar = st.progress(0, text="Processing...")
            subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'cap.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                segments = model.transcribe("cap.wav", task="transcribe")['segments']
                trans_segments = []
                total_seg = len(segments)
                for i, seg in enumerate(segments):
                    p_bar.progress(30 + int((i/total_seg)*50), text=f"🧠 Translating {i+1}/{total_seg}")
                    txt = seg['text'].strip()
                    if txt:
                        burmese = generate_with_retry(f"Translate to Burmese. Short. Input: '{txt}'")
                        trans_segments.append({'start': seg['start'], 'end': seg['end'], 'text': burmese})
                        time.sleep(0.3)
                p_bar.progress(90, text="✍️ Burning Subtitles...")
                ass_file = generate_ass_file(trans_segments, font_path)
                font_dir = os.path.dirname(font_path)
                subprocess.run(['ffmpeg', '-y', '-i', "cap_input.mp4", '-vf', f"ass={ass_file}:fontsdir={font_dir}", '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', "captioned_final.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state.caption_video_path = "captioned_final.mp4"
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
                seo_result = generate_with_retry(prompt)
                st.success("SEO Generated!")
                st.code(seo_result, language="markdown")
    else:
        st.info("Please generate a script in Tab 1 first.")
