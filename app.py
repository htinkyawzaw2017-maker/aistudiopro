
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

# ---------------------------------------------------------
# 🎨 UI SETUP (NEON SPACE THEME)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* 1. Global Background (Space Theme) */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2072&auto=format&fit=crop");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        color: #e0e0e0;
    }
    
    /* 2. Hide Standard Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3. NEON Title Style */
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
    
    /* 4. Custom Containers (Glassmorphism + Neon Border) */
    div[data-testid="stFileUploader"], div[data-testid="stExpander"], div[class="stTextArea"] {
        background-color: rgba(10, 25, 47, 0.85); /* Semi-transparent dark blue */
        border: 1px solid #00d2ff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* 5. Glowing Buttons */
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

   /* 6. Text Inputs & Select Boxes (Fix for Visibility) */
    input, textarea, select, div[data-baseweb="select"] > div {
        background-color: rgba(10, 25, 47, 0.9) !important;
        color: #ffffff !important; /* စာလုံးအဖြူ */
        border: 1px solid #00d2ff !important;
        border-radius: 8px !important;
    }
    /* Dropdown စာလုံးများအတွက် */
    div[role="listbox"] ul {
        background-color: #050b14 !important;
        color: white !important;
    }
    p, label {
        color: #e0e0e0 !important; /* Label စာလုံးများ */
    }
    }

    /* 7. Tabs Styling */
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

# Title with Icon
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

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
# 🔊 SIMPLE AUDIO GENERATOR (For single chunk)

# --- LOADERS FOR DICTIONARIES ---
def load_custom_dictionary():
    # For Gemini (Text Generation)
    dict_file = "dictionary.txt"
    if os.path.exists(dict_file):
        with open(dict_file, "r", encoding="utf-8") as f: return f.read()
    return ""

def load_pronunciation_dict():
    # Pronunciation Dictionary Loader
    pron_file = "pronunciation.txt"
    replacements = {}
    if os.path.exists(pron_file):
        with open(pron_file, "r", encoding="utf-8") as f:
            for line in f:
                # "Original = Sound" ပုံစံကို ရှာပြီး ခွဲခြားခြင်း
                if "=" in line and not line.startswith("#"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        replacements[parts[0].strip()] = parts[1].strip()
    return replacements

def generate_single_chunk(text, lang, gender, rate_str, pitch_str, output_file):
    if not text.strip(): return False
    processed_text = normalize_text_for_tts(text) 
    
    voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
    cmd = ["edge-tts", "--voice", voice_id, "--text", processed_text, f"--rate={rate_str}", f"--pitch={pitch_str}", "--write-media", output_file]

    # 🔥 FIX: Retry Logic (၃ ခါအထိ ကြိုးစားမည်)
    for attempt in range(3):
        try:
            # Run command and check for errors
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            # File ထွက်မထွက် စစ်မယ် (Size 0 ဖြစ်နေရင် မအောင်မြင်ဘူး)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return True # အောင်မြင်ရင် ပြီးမယ်
        except:
            time.sleep(1) # Error တက်ရင် ၁ စက္ကန့်နားပြီး ပြန်လုပ်မယ်
            continue

    return False # ၃ ခါလုပ်လို့မရမှ လက်လျှော့မယ်


# Helper Functions
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

# 🧠 SMART EMOTION AUDIO ENGINE (Parses tags and merges audio)
def generate_audio_with_emotions(full_text, lang, gender, base_mode, output_file, base_speed=1.0):
    # 1. Default Settings from Base Mode
    base_settings = VOICE_MODES.get(base_mode, VOICE_MODES["Normal"])
    current_rate = int(base_settings['rate'].replace('%', ''))
    current_pitch = int(base_settings['pitch'].replace('Hz', ''))
    
    # Apply Slider Speed to Base Rate
    slider_adj = int((base_speed - 1.0) * 100)
    current_rate += slider_adj

    # 2. Split Text by Tags (e.g., [sad], [happy])
    # Regex to capture tags like [sad] and keep them as separators
    parts = re.split(r'(\[.*?\])', full_text)
    
    audio_segments = []
    chunk_idx = 0

    for part in parts:
        part = part.strip()
        if not part: continue

        # Check if this part is a Tag
        if part.lower() in EMOTION_MAP:
            # Update settings for NEXT chunks
            emo = EMOTION_MAP[part.lower()]
            # Reset to base first, then apply emotion offset
            base_r = int(base_settings['rate'].replace('%', '')) + slider_adj
            base_p = int(base_settings['pitch'].replace('Hz', ''))
            
            emo_r = int(emo['rate'].replace('%', ''))
            emo_p = int(emo['pitch'].replace('Hz', ''))
            
            current_rate = base_r + emo_r
            current_pitch = base_p + emo_p
            continue # Tag itself is not spoken

                

        
        # If it's Text -> Generate Audio with current settings
        time.sleep(0.5) # Server မပိတ်အောင် ခဏနားမယ်
        chunk_filename = f"chunk_{chunk_idx}.mp3"
        rate_str = f"{current_rate:+d}%"
        pitch_str = f"{current_pitch:+d}Hz"
        
        success = generate_single_chunk(part, lang, gender, rate_str, pitch_str, chunk_filename)
        if success:
            audio_segments.append(chunk_filename)
            chunk_idx += 1

    # 3. Merge All Chunks using FFmpeg
    if not audio_segments: return False, "No audio generated"
    
    concat_list = "audio_concat.txt"
    with open(concat_list, "w") as f:
        for seg in audio_segments:
            f.write(f"file '{seg}'\n")
            
    try:
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Cleanup temp files
        for seg in audio_segments: os.remove(seg)
        os.remove(concat_list)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# ---------------------------------------------------------
# 🔢 NUMBER & TEXT NORMALIZATION
# ---------------------------------------------------------
def num_to_burmese_spoken(num_str):
    try:
        # 🔥 DECIMAL FIX: 3.3 -> သုံး ဒသမ သုံး
        if "." in num_str:
            parts = num_str.split(".")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                left = num_to_burmese_spoken(parts[0]) # ရှေ့ခြမ်း (၃)
                # နောက်ခြမ်း (၃) ကို တစ်လုံးချင်းဖတ်မလား၊ ဂဏန်းလိုဖတ်မလား
                # ရိုးရိုးရှင်းရှင်း ဂဏန်းလိုပဲ ပြန်ခေါ်လိုက်မယ်
                right = num_to_burmese_spoken(parts[1]) 
                return f"{left} ဒသမ {right}"

        # Normal Integer Logic
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
        
        # အသံထွက်ချောအောင် ပြင်ခြင်း
        result = result.replace("ထောင်", "ထောင့်").replace("ရာ", "ရာ့").replace("ဆယ်", "ဆယ့်")
        if result.endswith("ထောင့်"): result = result[:-1] + "င်"
        if result.endswith("ရာ့"): result = result[:-1]
        if result.endswith("ဆယ့်"): result = result[:-1]
        
        return result
    except: return num_str

def normalize_text_for_tts(text):
    if not text: return ""
    
    # 🔥 STEP 0: NUCLEAR COMMA REMOVAL (ဂဏန်းကြားက ကော်မာများကို အရင်ဆုံး သတ်မယ်)
    # 20,000 -> 20000 ဖြစ်သွားအောင် လုပ်သော အဆင့်
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # 1. Basic Symbol Cleaning
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    
    # 2. Pronunciation Fix (Dictionary Check)
    pron_dict = load_pronunciation_dict()
    sorted_keys = sorted(pron_dict.keys(), key=len, reverse=True)
    
    for original in sorted_keys:
        fixed_sound = pron_dict[original]
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(fixed_sound, text)
        
    # 3. Pause Logic
    text = text.replace("၊", ", ") 
    text = text.replace("။", ". ")
    text = text.replace("[p]", "... ") 
        
    # 4. Number Conversion (Regex Updated)
    # \d+ ဆိုတာ ကော်မာမရှိတော့တဲ့ ဂဏန်းတွေကို ရှာတာပါ
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda x: num_to_burmese_spoken(x.group()), text)
    
    # 5. Fix "Lone Lauk Tae" (Specific Patch)
    text = text.replace("လုံလောက် တဲ့", "လုံလောက်တဲ့") 
    text = text.replace("လုံလောက်သော", "လုံလောက်တဲ့")
    
    # 6. Final Clean
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
    # 🔥 AI MODEL SELECTION USAGE
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
# 🔊 AUDIO ENGINE
# ---------------------------------------------------------
# 🔥 EMOTION SETTINGS (Tag -> Pitch/Rate Adjustments)
EMOTION_MAP = {
    "[normal]": {"rate": "+0%", "pitch": "+0Hz"},
    "[sad]":    {"rate": "-15%", "pitch": "-15Hz"},  # နှေးပြီး အသံနိမ့်မယ်
    "[angry]":  {"rate": "+15%", "pitch": "+5Hz"},   # မြန်ပြီး အသံမာမယ်
    "[happy]":  {"rate": "+10%", "pitch": "+15Hz"},  # သွက်ပြီး အသံမြင့်မယ်
    "[action]": {"rate": "+30%", "pitch": "+0Hz"},   # အက်ရှင်ခန်း လိုမျိုး အရမ်းမြန်မယ်
    "[whisper]": {"rate": "-10%", "pitch": "-20Hz"}, # တိုးတိုးလေး ပြောသလို
}

VOICE_MAP = {
    "Burmese": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
    "English": {"Male": "en-US-ChristopherNeural", "Female": "en-US-AriaNeural"},
}
VOICE_MODES = {
    "Normal": {"rate": "+0%", "pitch": "+0Hz"},
    "Story": {"rate": "-5%", "pitch": "-2Hz"}, 
    "Recap": {"rate": "+5%", "pitch": "+0Hz"},
    "Motivation": {"rate": "+10", "pitch": "+2Hz"},
}

def generate_audio_cli(text, lang, gender, mode_name, output_file, speed_multiplier=1.0):
    if not text.strip(): return False, "Empty text"
    processed_text = normalize_text_for_tts(text)
    try:
        voice_id = VOICE_MAP.get(lang, {}).get(gender, "en-US-AriaNeural")
        settings = VOICE_MODES.get(mode_name, VOICE_MODES["Normal"])
        base_rate = int(settings['rate'].replace('%', ''))
        slider_rate = int((speed_multiplier - 1.0) * 100)
        final_rate_str = f"{base_rate + slider_rate:+d}%"
        cmd = ["edge-tts", "--voice", voice_id, "--text", processed_text, f"--rate={final_rate_str}", f"--pitch={settings['pitch']}", "--write-media", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e: return False, str(e)

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
# 🖥️ MAIN UI & SIDEBAR (UPDATED)
# ---------------------------------------------------------
st.markdown("""<div class="main-header">🎬 Myanmar AI Studio Pro</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 🔥 DROPDOWN MENU BAR (EXPANDER)
    with st.expander("🔑 API & System Settings", expanded=True):
        # 1. API KEY (HIDDEN PASSWORD MODE)
        try:
            if "GOOGLE_API_KEYS" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEYS"]
            elif "GOOGLE_API_KEY" in st.secrets: default_keys = st.secrets["GOOGLE_API_KEY"]
            else: default_keys = ""
        except: default_keys = ""
        
        api_key_input = st.text_input("API Keys (Comma separated)", value=default_keys, type="password", help="Enter multiple keys separated by comma for auto-rotation")
        
        if api_key_input:
            st.session_state.api_keys = [k.strip() for k in api_key_input.split(",") if k.strip()]
            st.success(f"✅ {len(st.session_state.api_keys)} Keys Loaded")
        else: st.session_state.api_keys = []
        
        st.divider()
        
        # 2. AI MODEL SELECTOR (INCLUDED 2.0 FLASH)
        st.markdown("🤖 **Select AI Model:**")
        st.session_state.selected_model = st.selectbox(
            "Model Version",
            ["gemini-2.5-flash", "gemini-2.0-flash-exp"],
            index=0,
            label_visibility="collapsed"
        )
        st.caption("Tip: Use '2.0-flash-exp' for better logic.")
        
    # 3. AUTO-LOAD KNOWLEDGE BASE (No Upload Needed)
    with st.expander("📚 Knowledge & Pronunciation", expanded=False):
        st.info("📂 System is using internal files.")

        # Check Dictionary.txt
        if os.path.exists("dictionary.txt"):
            st.success("✅ dictionary.txt: Active")
        else:
            st.error("❌ dictionary.txt: Missing")

        st.divider()

        # Check Pronunciation.txt
        if os.path.exists("pronunciation.txt"):
            st.success("✅ pronunciation.txt: Active")
        else:
            st.error("❌ pronunciation.txt: Missing")
            
        if st.button("🔄 Reload Files"):
            st.rerun()


    if st.button("🔴 Reset System", use_container_width=True):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

if not st.session_state.api_keys: st.warning("⚠️ Enter API Keys in Settings"); st.stop()

# 🔥 TABS AS BOXES (Buttons)
t1, t2, t3 = st.tabs(["🎙️ DUBBING STUDIO", "📝 AUTO CAPTION", "🚀 VIRAL SEO"])

# === TAB 1: DUBBING STUDIO ===
with t1:
    col_up, col_set = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload Video", type=['mp4','mov'], key="dub")
    with col_set:
        source_lang = st.selectbox("Original Lang", ["English", "Japanese", "Chinese", "Thai"])
    
    if uploaded:
        with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
        if st.button("📝 Extract & Translate", use_container_width=True):
            check_requirements()
            p_bar = st.progress(0, text="Starting...")
            p_bar.progress(20, text="🎤 Transcribing Audio...")
            subprocess.run(['ffmpeg', '-y', '-i', "input.mp4", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = load_whisper_safe()
            if model:
                raw = model.transcribe("temp.wav")['text']
                st.session_state.raw_transcript = raw
                p_bar.progress(60, text="🧠 AI Translating...")
                prompt = f"Translate {source_lang} to Burmese. Input: '{raw}'. Rules: Keep Proper Nouns in English."
                st.session_state.final_script = generate_with_retry(prompt)
                p_bar.progress(100, text="✅ Done!")
                st.rerun()
        
        # Duration Estimation (Fixed Variable Scope)
        txt = st.session_state.final_script if st.session_state.final_script else ""
        word_count = len(txt.split())
        est_min = round(word_count / 250, 1)
        st.caption(f"⏱️ Est. Duration: ~{est_min} mins")
        
        if st.session_state.final_script:
            st.markdown("### 🎬 Script & Production")
        
        # SCRIPT REFINING & EDITOR (Removed Duplicates)
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            # ✨ Refine with Auto-Tagging
            if st.button("✨ Refine: Recap Style (Full Length)", use_container_width=True):
                with st.spinner("Refining with Auto-Tagging & Sync Logic..."):
                    prompt = f"""
                    Act as a professional Myanmar Movie Recap Narrator. 
                    Your goal is to rewrite the input text into an engaging Recap Script that fits the exact duration of the video.
                    Input Text: "{st.session_state.final_script}"
                    **STRICT DUBBING & TAGGING RULES:**
                    1. **TIME MATCHING:** Use approximately 250 Burmese words per 1 minute of video. 
                    2. **AUTO-TAGGING:** Insert [p], [action], [sad], [happy], [whisper] based on context.
                    3. **FLOW:** Insert [p] every 2-3 sentences.
                    4. **NO SUMMARIZATION:** Match the full length of the original content.
                    5. **CORRECTION:** 'Fall' is 'ပြုတ်ကျ'.
                    """
                    st.session_state.final_script = generate_with_retry(prompt)
                    st.rerun()

        with c_opt2:
            if st.button("↩️ Reset Script", use_container_width=True):
                st.session_state.final_script = st.session_state.raw_transcript
                st.rerun()

        txt = st.text_area("Final Script", st.session_state.final_script, height=200)

        # Rendering Options
        st.markdown("---")
        st.markdown("#### ⚙️ Rendering Options")
        
        c_fmt, c_spd = st.columns([1, 1.2]) 
        with c_fmt:
            export_format = st.radio("Export Format:", ["🎬 Video (MP4)", "🎵 Audio Only (MP3)"], horizontal=True)
        with c_spd:
            audio_speed = st.slider("🔊 Audio Speed", 0.5, 2.0, 1.0, 0.05)
            video_speed = st.slider("🎞️ Video Speed", 0.5, 4.0, 1.0, 0.1)

        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        zoom_val = st.slider("🔍 Copyright Zoom (Video Only)", 1.0, 1.2, 1.0, 0.01)
        
        auto_freeze = None; manual_freeze = None
        if "Video" in export_format:
            with st.expander("❄️ Freeze Frame Settings (Sync Helper)"):
                c1, c2 = st.columns(2)
                with c1:
                    if st.checkbox("Every 30s"): auto_freeze = 30
                    if st.checkbox("Every 60s"): auto_freeze = 60
                with c2:
                    manual_freeze = st.text_input("Manual Command", placeholder="freeze 10,3")

        # RENDER BUTTON LOGIC
        btn_label = "🚀 GENERATE AUDIO" if "Audio" in export_format else "🚀 RENDER FINAL VIDEO"
        
        if st.button(btn_label, use_container_width=True):
            p_bar = st.progress(0, text="🚀 Initializing...")
            
            # 1. Generate Audio
            p_bar.progress(30, text="🔊 Generating Neural Speech...")
            try:
                generate_audio_with_emotions(txt, target_lang, gender, v_mode, "voice.mp3", base_speed=audio_speed)
                st.session_state.processed_audio_path = "voice.mp3"
            except Exception as e:
                st.error(f"Audio Error: {e}"); st.stop()
            
            if "Audio" in export_format:
                p_bar.progress(100, text="✅ Audio Generated!")
            else:
                # 2. Video Processing with Speed & Sync
                p_bar.progress(50, text="🎞️ Adjusting Video Speed & Logic...")
                input_vid = "input.mp4"
                
                # Update Duration based on speed
                raw_vid_dur = get_duration(input_vid)
                vid_dur = raw_vid_dur / video_speed 
                aud_dur = get_duration("voice.mp3")

                p_bar.progress(80, text="🎬 Merging & Finalizing...")
                pts_val = 1.0 / video_speed
                w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
                if w_s % 2 != 0: w_s += 1
                if h_s % 2 != 0: h_s += 1
                
                if aud_dur > vid_dur:
                    cmd = ['ffmpeg', '-y', '-stream_loop', '-1', '-i', input_vid, '-i', "voice.mp3", '-filter_complex', f"[0:v]setpts={pts_val}*PTS,scale={w_s}:{h_s},crop=1920:1080[vzoom]", '-map', '[vzoom]', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', "dubbed_final.mp4"]
                else:
                    cmd = ['ffmpeg', '-y', '-i', input_vid, '-i', "voice.mp3", '-filter_complex', f"[0:v]setpts={pts_val}*PTS,scale={w_s}:{h_s},crop=1920:1080[vzoom]", '-map', '[vzoom]', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', "dubbed_final.mp4"]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                p_bar.progress(100, text="🎉 Video Complete!")
                st.session_state.processed_video_path = "dubbed_final.mp4"

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
        with open("cap_input.mp4", "wb") as f: f.write(cap_up.getbuffer())
        if st.button("Generate Captions", use_container_width=True):
            check_requirements(); font_path = download_font()
            p_bar = st.progress(0, text="Processing...")
            p_bar.progress(30, text="🎤 Transcribing...")
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
