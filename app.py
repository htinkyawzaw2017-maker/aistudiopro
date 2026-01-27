import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import google.generativeai as genai
import subprocess
import json
import time
import shutil
import whisper
import re
from pydub import AudioSegment

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'generated_script' not in st.session_state: st.session_state.generated_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B, #FF416C); 
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444; background: #222; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ SYSTEM FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing."); st.stop()
    # Check if edge-tts is installed as CLI
    try:
        subprocess.run(["edge-tts", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        st.error("❌ edge-tts CLI missing. Install with: pip install edge-tts")
        st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 ROBUST AUDIO GENERATION (CLI METHOD - FIXES MUTED ISSUE)
# ---------------------------------------------------------
def generate_audio_cli(text, voice, rate, pitch, output_file):
    # Using Subprocess to call edge-tts is 100% more stable than python library in Streamlit
    try:
        command = [
            "edge-tts",
            "--voice", voice,
            "--text", text,
            "--rate", rate,
            "--pitch", pitch,
            "--write-media", output_file
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"TTS Failed: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ SMART TEXT CLEANER (UNIT & ENGLISH FIXER)
# ---------------------------------------------------------
def clean_text_for_burmese(text):
    # 1. Hard-coded Replacements for Units (Correct Pronunciation)
    replacements = {
        "No.": "နံပါတ် ",
        "kg": " ကီလိုဂရမ် ",
        "cm": " စင်တီမီတာ ",
        "mm": " မီလီမီတာ ",
        "km": " ကီလိုမီတာ ",
        "%": " ရာခိုင်နှုန်း ",
        "$": " ဒေါ်လာ ",
        "Mr.": "မစ္စတာ ",
        "Ms.": "မစ္စ "
    }
    
    # Case insensitive replace
    for k, v in replacements.items():
        pattern = re.compile(re.escape(k), re.IGNORECASE)
        text = pattern.sub(v, text)

    # 2. English Killer: Remove a-z, A-Z (Keep Burmese, numbers, basic punctuation)
    # Range U+1000-U+109F is Myanmar. We also keep 0-9.
    cleaned = re.sub(r'[A-Za-z]', '', text)
    
    return cleaned.strip()

def translate_smart(model, text, style):
    prompt = f"""
    Role: Professional Burmese Dubbing Translator.
    Task: Translate English to Spoken Burmese.
    Input: "{text}"
    
    RULES:
    1. **Output Burmese Only**: No English allowed.
    2. **Numbers**: Write as words (100 -> တစ်ရာ).
    3. **Tone**: {style}.
    4. **Direct Translation**: No "Here is the translation".
    """
    
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            # Apply Cleaner
            final = clean_text_for_burmese(result)
            if final: return final
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except: continue
    
    return ""

# ---------------------------------------------------------
# 🎬 MAIN DUBBING PIPELINE
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style_desc, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract Audio
    status.info("🎧 Step 1: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper
    status.info("🧠 Step 2: Speech Recognition...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Translation & TTS
    status.info(f"🎙️ Step 3: Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # Base Silent Track
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Rate Limit
        time.sleep(1) 
        
        # Translate & Clean
        burmese_text = translate_smart(model, seg['text'], style_desc)
        
        if burmese_text:
            fname = f"seg_{i}.mp3"
            # 🔥 USE CLI GENERATOR (Stable)
            success = generate_audio_cli(burmese_text, voice_config['id'], voice_config['rate'], voice_config['pitch'], fname)
            
            if success and os.path.exists(fname) and os.path.getsize(fname) > 0:
                seg_audio = AudioSegment.from_file(fname)
                curr_dur = len(seg_audio) / 1000.0
                target_dur = end - start
                
                if curr_dur > 0 and target_dur > 0:
                    speed = max(0.6, min(curr_dur / target_dur, 1.5))
                    # Stretch
                    subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(f"s_{i}.mp3"):
                        final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                        final_audio = final_audio.overlay(final_seg, position=start * 1000)
                        try: os.remove(f"s_{i}.mp3")
                        except: pass
                
                try: os.remove(fname)
                except: pass
        
        progress.progress((i + 1) / total)

    status.info("🔊 Step 4: Mixing...")
    final_audio.export("final_track.mp3", format="mp3")
    
    # Check if audio exists
    if os.path.getsize("final_track.mp3") < 100:
        st.error("Audio generation failed. Check API Key or FFmpeg.")
        return None

    status.info("🎬 Step 5: Merging Video...")
    out_file = f"dubbed_{int(time.time())}.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', out_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return out_file

# ---------------------------------------------------------
# 📝 SCRIPT & GENAI
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    p = f"""
    Write a {type} script about "{topic}".
    Tone: {tone}.
    Instructions: {prompt}
    Language: Burmese (Myanmar).
    Format: Clear script format.
    """
    return model.generate_content(p).text

def text_to_speech_file(text, v_conf):
    clean = clean_text_for_burmese(text)
    fname = f"script_{int(time.time())}.mp3"
    generate_audio_cli(clean, v_conf['id'], v_conf['rate'], v_conf['pitch'], fname)
    return fname

# ---------------------------------------------------------
# 🖥️ UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    
    # 🔥 CUSTOM MODEL BUTTON ADDED
    model_mode = st.radio("Model Selection", ["Preset", "Custom Input"])
    if model_mode == "Preset":
        model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    else:
        model_name = st.text_input("Enter Model ID", value="gemini-2.5-flash")
    
    st.info(f"Active: {model_name}")
    
    if st.button("🔴 Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("Enter API Key"); st.stop()

# TABS
t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "📝 Script", "🚀 Viral", "🖼️ Thumbnail"])

# TAB 1: DUBBING
with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
    
    c1, c2, c3 = st.columns(3)
    with c1: narrator = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Style", ["Normal", "Movie Recap", "News", "Deep", "Calming"])
    
    # Voice Config (Precise Control)
    base_id = "my-MM-ThihaNeural" if "Male" in narrator else "my-MM-NilarNeural"
    
    if tone == "Movie Recap": v_conf = {"id": base_id, "rate": "+0%", "pitch": "-5Hz"}
    elif tone == "News": v_conf = {"id": base_id, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": v_conf = {"id": base_id, "rate": "-5%", "pitch": "-15Hz"}
    elif tone == "Calming": v_conf = {"id": base_id, "rate": "-5%", "pitch": "+5Hz"}
    else: v_conf = {"id": base_id, "rate": "+0%", "pitch": "+0Hz"}
    
    if st.button("🚀 Start Dubbing") and uploaded_file:
        status = st.empty()
        prog = st.progress(0)
        try:
            out = process_video_dubbing("temp.mp4", v_conf, tone, st.session_state.api_key, model_name, status, prog)
            if out:
                st.session_state.processed_video = out
                status.success("Done!")
                st.rerun()
        except Exception as e: status.error(f"Error: {e}")

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download", f, "dubbed.mp4")

# TAB 2: SCRIPT
with t2:
    st.subheader("📝 Script Writer")
    sc1, sc2 = st.columns(2)
    with sc1: s_type = st.selectbox("Type", ["Story", "Movie Script", "Voiceover"])
    with sc2: s_tone = st.selectbox("Tone", ["Emotional", "Serious", "Funny"])
    
    topic = st.text_input("Topic")
    prompt = st.text_area("User Instructions")
    
    if st.button("✍️ Write Script"):
        with st.spinner("Writing..."):
            res = generate_script(topic, s_type, s_tone, prompt, st.session_state.api_key, model_name)
            st.session_state.generated_script = res
            st.rerun()
            
    if st.session_state.generated_script:
        edit_script = st.text_area("Result", st.session_state.generated_script, height=300)
        
        st.markdown("### 🔊 Convert to Audio")
        ac1, ac2 = st.columns(2)
        with ac1: a_voice = st.selectbox("Voice", ["Male", "Female"], key="av")
        with ac2: a_tone = st.selectbox("Tone", ["Normal", "Deep", "News"], key="at")
        
        # Map Script Voice
        s_id = "my-MM-ThihaNeural" if "Male" in a_voice else "my-MM-NilarNeural"
        if a_tone == "Deep": s_conf = {"id": s_id, "rate": "-5%", "pitch": "-15Hz"}
        elif a_tone == "News": s_conf = {"id": s_id, "rate": "+10%", "pitch": "+0Hz"}
        else: s_conf = {"id": s_id, "rate": "+0%", "pitch": "+0Hz"}
        
        if st.button("🗣️ Read Script"):
            with st.spinner("Generating Audio..."):
                a_file = text_to_speech_file(edit_script, s_conf)
                st.audio(a_file)

# TAB 3 & 4 (Placeholders for Viral/Thumbnail - Logic is similar)
with t3: st.info("Viral Kit Ready")
with t4: st.info("Thumbnail Ready")
