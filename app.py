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
# 🎨 UI SETUP (CUSTOM CSS FOR HYPER-REALISM)
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* 1. HIDE DEFAULT STREAMLIT NAVBAR & FOOTER */
    .stApp > header {visibility: hidden;}
    .stApp > footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* 2. GLOBAL THEME */
    .stApp { background-color: #050505; color: #e0e0e0; }
    
    /* 3. STYLING TABS AS BOXES (BUTTONS) */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    div[data-baseweb="tab"] {
        background-color: #112240;
        border-radius: 8px;
        padding: 10px 20px;
        color: #8892b0;
        border: 1px solid #233554;
        font-weight: bold;
        flex-grow: 1; /* Full width boxes */
        text-align: center;
    }
    div[data-baseweb="tab"]:hover {
        background-color: #1a2f55;
        color: white;
        border-color: #64ffda;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #00d2ff; /* Neon Blue Active */
        color: black !important;
        border-color: #00d2ff;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }
    
    /* 4. MAIN HEADER STYLE */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px; text-align: center; margin-bottom: 20px;
        font-size: 3rem; font-weight: 900;
        text-shadow: 0px 0px 20px rgba(0,255,136,0.3);
    }
    
    /* 5. INPUT FIELDS STYLE */
    input[type="text"], input[type="password"], textarea { 
        background-color: #0a192f !important; color: #64ffda !important; 
        border: 1px solid #112240 !important; border-radius: 8px !important;
    }
    
    /* 6. INFO BOX */
    .info-box { background: #112240; padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff; margin: 10px 0; color: white;}
    </style>
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

def load_custom_dictionary():
    dict_file = "dictionary.txt"
    if os.path.exists(dict_file):
        with open(dict_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# ---------------------------------------------------------
# 🔢 NUMBER & TEXT NORMALIZATION
# ---------------------------------------------------------
# 🔢 NUMBER & TEXT NORMALIZATION (UPDATED)
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
    
    # 1. Basic Symbol Cleaning
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    
    # 2. Pronunciation Fix (Dictionary Check)
    pron_dict = load_pronunciation_dict()
    # Sort keys by length (Longer first) to prevent partial replacement
    # e.g., fix "40,000" before "4"
    sorted_keys = sorted(pron_dict.keys(), key=len, reverse=True)
    
    for original in sorted_keys:
        fixed_sound = pron_dict[original]
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(fixed_sound, text)
        
    # 3. Pause Logic
    text = text.replace("၊", ", ") 
    text = text.replace("။", ". ")
    text = text.replace("[p]", "... ") 
        
    # 4. 🔥 REGEX UPDATE FOR DECIMALS 🔥
    # အရင်က \b\d+\b (ဂဏန်းသီးသန့်) ပဲ ရှာတယ်
    # အခု \d+(\.\d+)? (ဒသမ ပါတာရော ရှာမယ်)
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda x: num_to_burmese_spoken(x.group()), text)
    
    # 5. Fix "Lone Lauk Tae" pause issue specifically in code if Dict fails
    text = text.replace("လုံလောက် တဲ့", "လုံလောက်တဲ့") 
    
    # 6. Final Clean
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
    # 1. Basic Symbol Cleaning (မလိုတဲ့ သင်္ကေတတွေ ဖယ်မယ်)
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', "").replace("'", "")
    
    # 2. PRONUNCIATION FIX (အသံထွက် ပြင်ဆင်ခြင်း)
    pron_dict = load_pronunciation_dict()
    for original, fixed_sound in pron_dict.items():
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(fixed_sound, text)
        
    # 3. 🔥 PAUSE LOGIC (အဖြတ်အတောက် သင်ပေးခြင်း) 🔥
    # မြန်မာ '၊' ကို English ',' (ကော်မာ) ပြောင်းမှ AI က ခဏရပ်တတ်တယ်
    text = text.replace("၊", ", ") 
    text = text.replace("။", ". ")
    
    # User က [p] လို့ရေးရင် အကြာကြီးရပ်မယ့် logic (ဥပမာ - ... ထည့်ပေးလိုက်ခြင်း)
    text = text.replace("[p]", "... ") 
        
    # 4. Number Conversion
    text = re.sub(r'\b\d+\b', lambda x: num_to_burmese_spoken(x.group()), text)
    
    # 5. Remove Newlines (ဒါပေမဲ့ ပုဒ်ဖြတ်တွေ ကျန်ခဲ့မယ်)
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

    

        # ⚠️ ဒီ code block က 'with t1:' ရဲ့ အောက်မှာ ရှိနေရမယ် (Space 4 ချက် ဝင်နေရမယ်)
    if st.session_state.final_script:
        st.markdown("### 🎬 Script & Production")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            # 🔥 REFINEMENT BUTTON (FORCES BURMESE OUTPUT)
            if st.button("✨ Refine: Storytelling Mode", use_container_width=True):
                with st.spinner("Refining Script into Burmese Storytelling Style..."):
                    prompt = f"""
                    Act as a professional Myanmar Movie Narrator.
                    Rewrite the following input text into natural, engaging **Burmese spoken language** (Storytelling Style).
                    
                    Input Text: "{st.session_state.final_script}"
                    
                    **STRICT RULES:**
                    1. **OUTPUT LANGUAGE:** MUST BE BURMESE (မြန်မာစာ) ONLY. Do not output English.
                    2. **STYLE:** Storytelling/Recap style. Use natural endings like 'တယ်', 'မယ်', 'ပါ'.
                    3. **FORBIDDEN:** Do NOT use 'ဗျ', 'ရှင့်', 'ခင်ဗျာ', 'သူ၏', '၎င်း', 'သည်', '၍'.
                    4. **FLOW:** Make it continuous and exciting.
                    """
                    # AI Call
                    st.session_state.final_script = generate_with_retry(prompt)
                    st.rerun()

        with c_opt2:
             if st.button("↩️ Reset Script", use_container_width=True): pass

        txt = st.text_area("Final Script", st.session_state.final_script, height=200)
        
        # Duration Estimation
        word_count = len(txt.split())
        est_min = round(word_count / 250, 1)
        st.markdown(f"<div class='info-box'>⏱️ Est. Duration: ~{est_min} mins</div>", unsafe_allow_html=True)
        
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1: target_lang = st.selectbox("Output Lang", list(VOICE_MAP.keys()))
        with c_v2: gender = st.selectbox("Gender", ["Male", "Female"])
        with c_v3: v_mode = st.selectbox("Voice Mode", list(VOICE_MODES.keys()))
        
        audio_speed = st.slider("🔊 Audio Speed", 0.8, 1.5, 1.0, 0.05)
        zoom_val = st.slider("🔍 Copyright Zoom", 1.0, 1.2, 1.0, 0.01)
        
        with st.expander("❄️ Freeze Frame Settings"):
            c1, c2 = st.columns(2)
            auto_freeze = None; manual_freeze = None
            with c1:
                if st.checkbox("Every 30s"): auto_freeze = 30
                if st.checkbox("Every 60s"): auto_freeze = 60
            with c2: manual_freeze = st.text_input("Manual Command", placeholder="freeze 10,3")
        
        if st.button("🚀 RENDER FINAL VIDEO", use_container_width=True):
            p_bar = st.progress(0, text="🚀 Initializing...")
            
            p_bar.progress(30, text="🔊 Generating Neural Speech (Applied Pronunciation Fix)...")
            generate_audio_cli(txt, target_lang, gender, v_mode, "voice.mp3", speed_multiplier=audio_speed)
            st.session_state.processed_audio_path = "voice.mp3"
            
            p_bar.progress(50, text="❄️ Applying Visual Effects...")
            input_vid = "input.mp4"
            if auto_freeze or manual_freeze:
                freeze_pts = []
                dur = get_duration(input_vid)
                if auto_freeze: freeze_pts = [(t, 3) for t in range(auto_freeze, int(dur), auto_freeze)]
                elif manual_freeze:
                    match = re.search(r'freeze\s*[:=]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', manual_freeze)
                    if match: freeze_pts = [(float(match.group(1)), float(match.group(2)))]
                if freeze_pts:
                    concat_file = "concat_list.txt"
                    prev_t = 0
                    with open(concat_file, "w") as f:
                        for idx, (ft, fd) in enumerate(freeze_pts):
                            subprocess.run(['ffmpeg', '-y', '-ss', str(prev_t), '-t', str(ft-prev_t), '-i', input_vid, '-c', 'copy', f"c_{idx}.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            f.write(f"file 'c_{idx}.mp4'\n")
                            subprocess.run(['ffmpeg', '-y', '-ss', str(ft), '-i', input_vid, '-vframes', '1', 'f.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', 'f.jpg', '-t', str(fd), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f"fr_{idx}.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            f.write(f"file 'fr_{idx}.mp4'\n")
                            prev_t = ft
                        subprocess.run(['ffmpeg', '-y', '-ss', str(prev_t), '-i', input_vid, '-c', 'copy', "c_end.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        f.write(f"file 'c_end.mp4'\n")
                    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', 'frozen.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    input_vid = "frozen.mp4"

            p_bar.progress(80, text="🎬 Merging...")
            w_s = int(1920 * zoom_val); h_s = int(1080 * zoom_val)
            if w_s % 2 != 0: w_s += 1
            if h_s % 2 != 0: h_s += 1
            
            subprocess.run([
                'ffmpeg', '-y', '-i', input_vid, '-i', "voice.mp3",
                '-filter_complex', f"[0:v]scale={w_s}:{h_s},crop=1920:1080[vzoom]",
                '-map', '[vzoom]', '-map', '1:a',
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', "dubbed_final.mp4"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            p_bar.progress(100, text="🎉 Complete!")
            st.session_state.processed_video_path = "dubbed_final.mp4"
            st.success("Dubbing Complete!")

    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path)
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            with open(st.session_state.processed_video_path, "rb") as f: st.download_button("🎬 Download Video", f, "dubbed.mp4", use_container_width=True)
        with c_d2:
            if st.session_state.processed_audio_path and os.path.exists(st.session_state.processed_audio_path):
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
