import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import time
import shutil
import whisper
import re
from pydub import AudioSegment
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Myanmar AI Studio Pro", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D); 
        color: black; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .debug-box { font-family: monospace; font-size: 12px; color: #00ff00; background: #000; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .error-box { font-family: monospace; font-size: 12px; color: #ff4b4b; background: #260000; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 STATE
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'generated_script' not in st.session_state: st.session_state.generated_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing. System cannot run."); st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 ROBUST AUDIO ENGINE (LIBRARY BASED)
# ---------------------------------------------------------
async def tts_async(text, voice, rate, pitch, filename):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(filename)

def generate_audio_lib(text, voice_conf, filename):
    try:
        # Use fresh loop to avoid Streamlit conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_async(text, voice_conf['id'], voice_conf['rate'], voice_conf['pitch'], filename))
        loop.close()
        
        # Verify File
        if os.path.exists(filename) and os.path.getsize(filename) > 100: # Check if > 100 bytes
            return True
        return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ SMART TEXT CLEANER (UNIT FIXER)
# ---------------------------------------------------------
def clean_burmese_text(text):
    # 1. Unit Replacements (Pronunciation Fix)
    replacements = {
        "No.": "နံပါတ် ",
        "kg": " ကီလိုဂရမ် ",
        "cm": " စင်တီမီတာ ",
        "mm": " မီလီမီတာ ",
        "km": " ကီလိုမီတာ ",
        "%": " ရာခိုင်နှုန်း ",
        "$": " ဒေါ်လာ ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v).replace(k.lower(), v)

    # 2. English Killer (Remove A-Z, Keep Burmese)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_content(model, text, style):
    # Powerful Prompt
    prompt = f"""
    Act as a Burmese Dubbing Artist. Translate English to Burmese (Myanmar).
    Input: "{text}"
    
    RULES:
    1. **Output Burmese Only**: Do NOT output English words.
    2. **Numbers**: Convert to words (100 -> တစ်ရာ).
    3. **Tone**: {style}.
    4. **Direct Output**: No explanations.
    """
    
    retries = 3
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            cleaned = clean_burmese_text(translated)
            if cleaned: return cleaned
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except: continue
    return ""

# ---------------------------------------------------------
# 🎬 MAIN DEBUG PIPELINE
# ---------------------------------------------------------
def process_video_debug(video_path, voice_conf, style, api_key, model_name, debug_area, progress):
    check_requirements()
    debug_logs = []
    
    def log(msg, type="info"):
        debug_logs.append(msg)
        if type=="info": debug_area.markdown(f"<div class='debug-box'>ℹ️ {msg}</div>", unsafe_allow_html=True)
        else: debug_area.markdown(f"<div class='error-box'>❌ {msg}</div>", unsafe_allow_html=True)
        print(msg)

    # 1. Extract Audio
    log("Step 1: Extracting Audio from Video...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists("temp.wav"):
        log("Error: temp.wav not created", "error"); return None

    # 2. Whisper
    log("Step 2: Whisper Transcription Started...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    log(f"Whisper Found {len(segments)} Segments.")

    # 3. AI Setup
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # 4. Dubbing Loop
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        
        # Translate
        log(f"--- Segment {i+1}/{total} ---")
        burmese_text = translate_content(model, seg['text'], style)
        
        if burmese_text:
            log(f"Translated: {burmese_text}")
            fname = f"seg_{i}.mp3"
            
            # Generate Audio
            if generate_audio_lib(burmese_text, voice_conf, fname):
                fsize = os.path.getsize(fname) / 1024
                log(f"Audio Generated: {fsize:.2f} KB")
                
                # Sync & Stretch
                seg_audio = AudioSegment.from_file(fname)
                curr_dur = len(seg_audio) / 1000.0
                target_dur = end - start
                
                if curr_dur > 0 and target_dur > 0:
                    speed = max(0.6, min(curr_dur / target_dur, 1.5))
                    subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(f"s_{i}.mp3"):
                        final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                        final_audio = final_audio.overlay(final_seg, position=start * 1000)
                        log("Audio Synced & Overlaid.")
                        try: os.remove(f"s_{i}.mp3")
                        except: pass
                try: os.remove(fname)
                except: pass
            else:
                log("TTS Failed (No Audio Generated)", "error")
        else:
            log("Translation Empty (English Killer Active)", "error")
        
        progress.progress((i + 1) / total)
        time.sleep(1) # Rate limit safety

    # 5. Export
    log("Step 4: Exporting Final Audio Track...")
    final_audio.export("final_track.mp3", format="mp3")
    
    if os.path.getsize("final_track.mp3") < 1000:
        log("CRITICAL: Final Audio is too small/silent!", "error")
        # Fallback to original audio if dubbing failed completely
        return None

    log("Step 5: Merging with Video...")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    # -map 0:v (video from file 0) -map 1:a (audio from file 1)
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    log("Process Complete!")
    return output_filename

# ---------------------------------------------------------
# 📝 SCRIPT FUNCTIONS
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    p = f"Write a {type} script about '{topic}'. Tone: {tone}. Instructions: {prompt}. Language: Burmese."
    return model.generate_content(p).text

def text_to_speech_script(text, v_conf):
    clean = clean_burmese_text(text)
    chunks = [clean[i:i+500] for i in range(0, len(clean), 500)]
    combined = AudioSegment.empty()
    for idx, chunk in enumerate(chunks):
        fname = f"chunk_{idx}.mp3"
        if generate_audio_lib(chunk, v_conf, fname):
            combined += AudioSegment.from_file(fname)
            os.remove(fname)
    out = f"script_{int(time.time())}.mp3"
    combined.export(out, format="mp3")
    return out

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    
    # 🔥 CUSTOM MODEL BUTTON
    model_mode = st.radio("Model Settings", ["Preset", "Custom Input"])
    if model_mode == "Preset":
        model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    else:
        model_name = st.text_input("Enter Custom Model ID", value="gemini-2.5-flash")
    
    st.caption(f"Active: {model_name}")
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

# TABS
t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing (Debug)", "📝 Script", "🚀 Viral", "🖼️ Thumbnail"])

# TAB 1: DUBBING
with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    c1, c2, c3 = st.columns(3)
    with c1: narrator = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Style", ["Normal", "Movie Recap", "News", "Deep", "Calming", "Fast"])
    
    # Tone Config
    base_id = "my-MM-ThihaNeural" if "Male" in narrator else "my-MM-NilarNeural"
    if tone == "Movie Recap": v_conf = {"id": base_id, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": v_conf = {"id": base_id, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": v_conf = {"id": base_id, "rate": "-5%", "pitch": "-20Hz"}
    elif tone == "Calming": v_conf = {"id": base_id, "rate": "-5%", "pitch": "+5Hz"}
    elif tone == "Fast": v_conf = {"id": base_id, "rate": "+25%", "pitch": "+10Hz"}
    else: v_conf = {"id": base_id, "rate": "+0%", "pitch": "+0Hz"}
    
    if st.button("🚀 Start Dubbing (Debug Mode)"):
        st.session_state.processed_video = None
        debug_area = st.empty() # Area for logs
        prog = st.progress(0)
        
        try:
            out = process_video_debug("temp.mp4", v_conf, tone, st.session_state.api_key, model_name, debug_area, prog)
            if out:
                st.session_state.processed_video = out
                st.success("✅ Dubbing Complete!")
                st.rerun()
        except Exception as e:
            st.error(f"Critical Error: {e}")

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
    prompt = st.text_area("Instructions")
    
    if st.button("✍️ Write"):
        with st.spinner("Writing..."):
            res = generate_script(topic, s_type, s_tone, prompt, st.session_state.api_key, model_name)
            st.session_state.generated_script = res
