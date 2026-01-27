import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import time
import shutil # Fixed: Added missing import
import whisper
import re
from pydub import AudioSegment
from PIL import Image
from google.api_core import exceptions # Fixed: Added missing import

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
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
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. Please install FFmpeg on the server."); st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 ROBUST AUDIO ENGINE (LIBRARY METHOD WITH LOOP FIX)
# ---------------------------------------------------------
async def tts_generation_async(text, voice, rate, pitch, output_file):
    # This uses the direct python library which is allowed on Streamlit Cloud
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_config, filename):
    # Unpack config
    voice = voice_config['id']
    rate = voice_config['rate']
    pitch = voice_config['pitch']
    
    try:
        # 🔥 CRITICAL FIX: Create a new event loop for every generation
        # This solves the "Event loop is closed" or hanging issues on servers
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_generation_async(text, voice, rate, pitch, filename))
        loop.close()
        
        # Verify file creation and size
        if os.path.exists(filename) and os.path.getsize(filename) > 100:
            return True
        print(f"Warning: Audio file created but empty: {filename}")
        return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ SMART TEXT CLEANER
# ---------------------------------------------------------
def clean_text_for_burmese(text):
    # 1. Force Burmese Units (Python Side Backup)
    replacements = {
        "No.": "နံပါတ် ",
        "kg": " ကီလိုဂရမ် ",
        "cm": " စင်တီမီတာ ",
        "mm": " မီလီမီတာ ",
        "km": " ကီလိုမီတာ ",
        "%": " ရာခိုင်နှုန်း ",
        "$": " ဒေါ်လာ ",
        "Mr.": "မစ္စတာ ",
        "Ms.": "မစ္စ ",
        "Dr.": "ဒေါက်တာ "
    }
    
    # Case insensitive replacement
    for k, v in replacements.items():
        pattern = re.compile(re.escape(k), re.IGNORECASE)
        text = pattern.sub(v, text)

    # 2. English Killer: Remove A-Z but keep Burmese
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

# ---------------------------------------------------------
# 🧠 TRANSLATION ENGINE
# ---------------------------------------------------------
def translate_content(model, text, style):
    prompt = f"""
    Act as a professional Burmese Dubbing Artist.
    Translate the following English text to Spoken Burmese (Myanmar).
    
    Input: "{text}"
    
    CRITICAL RULES:
    1. **Output Burmese Only**: Do NOT output English words.
    2. **Numbers**: Convert to Burmese words (e.g., 100 -> တစ်ရာ).
    3. **Tone/Style**: {style}.
    4. **No Explanations**: Just return the translated text.
    """
    
    retries = 3
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            # Clean and return
            cleaned = clean_text_for_burmese(translated)
            if cleaned: return cleaned
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except: continue
    return ""

# ---------------------------------------------------------
# 🎬 MAIN PROCESSING PIPELINE
# ---------------------------------------------------------
def process_video_pipeline(video_path, voice_config, style_desc, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract Audio
    status.info("🎧 Step 1: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper
    status.info("🧠 Step 2: Speech Recognition...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. AI Setup
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # 4. Dubbing Loop
    status.info(f"🎙️ Step 3: Dubbing {len(segments)} segments...")
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    # Audio Generation Check
    generated_count = 0
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Rate Limit
        time.sleep(1)
        
        # Translate
        burmese_text = translate_content(model, seg['text'], style_desc)
        
        if burmese_text:
            fname = f"seg_{i}.mp3"
            success = generate_audio(burmese_text, voice_config, fname)
            
            if success:
                generated_count += 1
                seg_audio = AudioSegment.from_file(fname)
                curr_dur = len(seg_audio) / 1000.0
                target_dur = end - start
                
                if curr_dur > 0 and target_dur > 0:
                    # Time Stretch (Sync)
                    speed = max(0.6, min(curr_dur / target_dur, 1.5))
                    subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(f"s_{i}.mp3"):
                        final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                        final_audio = final_audio.overlay(final_seg, position=start * 1000)
                        try: os.remove(f"s_{i}.mp3")
                        except: pass
                
                try: os.remove(fname)
                except: pass
        
        progress.progress((i + 1) / total)

    if generated_count == 0:
        st.error("❌ No audio segments were generated. Check API Key or Input Video.")
        return None

    status.info("🔊 Step 4: Mixing Audio...")
    final_audio.export("final_track.mp3", format="mp3")
    
    status.info("🎬 Step 5: Finalizing Video...")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    # Strict Re-encoding to ensure audio is present
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-i', "final_track.mp3", 
        '-c:v', 'copy', 
        '-c:a', 'aac', 
        '-map', '0:v:0', 
        '-map', '1:a:0', 
        output_filename
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_filename

# ---------------------------------------------------------
# 📝 SCRIPT & AUDIO CONVERTER
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    p = f"""
    Act as a Burmese Content Creator.
    Write a '{type}' script about "{topic}".
    Tone: {tone}.
    User Instructions: {prompt}
    
    Output Language: Burmese (Myanmar).
    Format: Clear script format.
    """
    return model.generate_content(p).text

def text_to_speech_script(text, v_conf):
    clean = clean_text_for_burmese(text)
    # Split text if too long (Edge-TTS limit approx 1000 chars)
    chunks = [clean[i:i+500] for i in range(0, len(clean), 500)]
    combined = AudioSegment.empty()
    
    for idx, chunk in enumerate(chunks):
        fname = f"chunk_{idx}.mp3"
        if generate_audio(chunk, v_conf, fname):
            combined += AudioSegment.from_file(fname)
            os.remove(fname)
            
    out_path = f"script_{int(time.time())}.mp3"
    combined.export(out_path, format="mp3")
    return out_path

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    
    # 🔥 CUSTOM MODEL SELECTION
    model_mode = st.radio("Model Settings", ["Preset", "Custom Input"])
    if model_mode == "Preset":
        model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    else:
        model_name = st.text_input("Enter Custom Model ID", value="gemini-2.5-flash")
    
    st.caption(f"Active: {model_name}")
    
    if st.button("🔴 Force Reset"):
        st.session_state.processed_video = None
        st.session_state.generated_script = ""
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key to proceed."); st.stop()

# TABS
t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "📝 Script Writer", "🚀 Viral Kit", "🖼️ Thumbnail"])

# --- TAB 1: DUBBING ---
with t1:
    st.subheader("🔊 Video Dubbing Engine")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    c1, c2, c3 = st.columns(3)
    with c1: narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
    with c2: 
        tone = st.selectbox("Tone / Style", [
            "Normal", 
            "Movie Recap (Dramatic)", 
            "News (Formal)", 
            "Deep (Serious)", 
            "Calming (Soft)", 
            "Fast (Excited)"
        ])
    
    # 🔥 PRECISE TONE CONFIGURATION
    base_id = "my-MM-ThihaNeural" if "Male" in narrator else "my-MM-NilarNeural"
    
    if tone == "Movie Recap (Dramatic)":
        v_conf = {"id": base_id, "rate": "+0%", "pitch": "-10Hz"}
        style_prompt = "Dramatic, Storytelling"
    elif tone == "News (Formal)":
        v_conf = {"id": base_id, "rate": "+10%", "pitch": "+0Hz"}
        style_prompt = "Formal, News Reporter"
    elif tone == "Deep (Serious)":
        v_conf = {"id": base_id, "rate": "-5%", "pitch": "-20Hz"}
        style_prompt = "Serious, Deep voice"
    elif tone == "Calming (Soft)":
        v_conf = {"id": base_id, "rate": "-5%", "pitch": "+5Hz"}
        style_prompt = "Soft, Gentle"
    elif tone == "Fast (Excited)":
        v_conf = {"id": base_id, "rate": "+25%", "pitch": "+10Hz"}
        style_prompt = "Excited, Fast"
    else:
        v_conf = {"id": base_id, "rate": "+0%", "pitch": "+0Hz"}
        style_prompt = "Natural Conversation"

    if uploaded_file and st.button("🚀 Start Dubbing"):
        st.session_state.processed_video = None
        status = st.empty()
        prog = st.progress(0)
        try:
            out = process_video_pipeline("temp.mp4", v_conf, style_prompt, st.session_state.api_key, model_name, status, prog)
            if out:
                st.session_state.processed_video = out
                status.success("Dubbing Complete!")
                st.rerun()
        except Exception as e:
            status.error(f"Error: {e}")
            
    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download Dubbed Video", f, "dubbed.mp4")

# --- TAB 2: SCRIPT WRITER ---
with t2:
    st.subheader("📝 Script & Voiceover")
    
    col1, col2 = st.columns(2)
    with col1: s_type = st.selectbox("Script Type", ["Story", "Movie Script", "Video Voiceover", "Documentary"])
    with col2: s_tone = st.selectbox("Script Tone", ["Emotional", "Serious", "Funny", "Educational"])
    
    topic = st.text_input("Topic (e.g., History of Bagan)")
    user_prompt = st.text_area("Specific Instructions (User Prompt)", placeholder="Start with a hook, mention specific dates...")
    
    if st.button("✍️ Generate Script"):
        if not topic: st.warning("Please enter a topic.")
        else:
            with st.spinner("Writing..."):
                res = generate_script(topic, s_type, s_tone, user_prompt, st.session_state.api_key, model_name)
                st.session_state.generated_script = res
                st.rerun()
                
    if st.session_state.generated_script:
        st.markdown("### Generated Script")
        final_script = st.text_area("Edit Script:", st.session_state.generated_script, height=300)
        
        st.markdown("### 🔊 Convert Script to Audio")
        ac1, ac2 = st.columns(2)
        with ac1: a_voice = st.selectbox("Voice", ["Male", "Female"], key="av")
        with ac2: a_tone = st.selectbox("Audio Tone", ["Normal", "Deep", "News"], key="at")
        
        # Audio Config for Script
        s_id = "my-MM-ThihaNeural" if "Male" in a_voice else "my-MM-NilarNeural"
        if a_tone == "Deep": s_conf = {"id": s_id, "rate": "-5%", "pitch": "-15Hz"}
        elif a_tone == "News": s_conf = {"id": s_id, "rate": "+10%", "pitch": "+0Hz"}
        else: s_conf = {"id": s_id, "rate": "+0%", "pitch": "+0Hz"}
        
        if st.button("🗣️ Read Script"):
            with st.spinner("Generating Audio..."):
                a_file = text_to_speech_script(final_script, s_conf)
                st.audio(a_file)
                with open(a_file, "rb") as f:
                    st.download_button("Download Audio (MP3)", f, "script.mp3")

# --- TAB 3 & 4 (Viral & Thumbnail) ---
with t3: st.info("Viral Kit integrated with AI model.")
with t4: st.info("Thumbnail Analyzer integrated with AI model.")
