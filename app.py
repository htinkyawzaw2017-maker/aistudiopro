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
import shutil
import whisper
import re
from pydub import AudioSegment
from PIL import Image
from google.api_core import exceptions 

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
# 🔊 ROBUST AUDIO ENGINE
# ---------------------------------------------------------
async def tts_generation_async(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_config, filename):
    # Unpack config
    voice = voice_config['id']
    rate = voice_config['rate']
    pitch = voice_config['pitch']
    
    try:
        # Create a fresh loop for thread safety
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_generation_async(text, voice, rate, pitch, filename))
        loop.close()
        
        if os.path.exists(filename) and os.path.getsize(filename) > 100:
            return True
        return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ SMART TRANSLATION LOGIC
# ---------------------------------------------------------
def clean_text_for_burmese(text):
    # 1. Unit & Symbol Fixer (Manual Override)
    replacements = {
        "No.": "နံပါတ် ", "kg": " ကီလိုဂရမ် ", "cm": " စင်တီမီတာ ", 
        "mm": " မီလီမီတာ ", "km": " ကီလိုမီတာ ", "%": " ရာခိုင်နှုန်း ", 
        "$": " ဒေါ်လာ ", "Mr.": "မစ္စတာ ", "Ms.": "မစ္စ ", "Dr.": "ဒေါက်တာ "
    }
    for k, v in replacements.items():
        pattern = re.compile(re.escape(k), re.IGNORECASE)
        text = pattern.sub(v, text)

    # 2. Relaxed Cleaner: Only remove English if the whole sentence looks English
    # Instead of deleting A-Z, we just strip them from edges or keep them if mixed.
    # For now, let's just remove A-Z characters but keep everything else to debug.
    cleaned = re.sub(r'[A-Za-z]', '', text) 
    return cleaned.strip()

def translate_content(model, text, style):
    # 🔥 SUPER STRICT PROMPT
    prompt = f"""
    You are a professional Burmese Translator.
    Task: Translate the following English text into natural spoken Burmese (Myanmar).
    
    Source Text: "{text}"
    
    RULES:
    1. **TRANSLATE EVERYTHING**: Do not leave any English words.
    2. **Numbers**: Write numbers as Burmese words (e.g. 1990 -> တစ်ထောင့်ကိုးရာကိုးဆယ်).
    3. **Tone**: {style}.
    4. **Output**: Return ONLY the Burmese translation. Do not say "Here is the translation".
    """
    
    retries = 3
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            
            # Clean it
            cleaned = clean_text_for_burmese(raw_text)
            
            # If cleaned text is empty, it means AI returned English. Retry.
            if not cleaned: 
                time.sleep(2)
                continue
                
            return cleaned, raw_text # Return both for debugging
            
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except: continue
        
    return "", "Failed to Translate"

# ---------------------------------------------------------
# 🎬 MAIN VIDEO PIPELINE
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
    
    # Debug Expander
    debug_log = st.expander("🛠️ Translation Logs (Check Here)", expanded=True)
    
    generated_count = 0
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Rate Limit
        time.sleep(1)
        
        # Translate
        burmese_text, raw_ai_response = translate_content(model, seg['text'], style_desc)
        
        # LOGGING
        with debug_log:
            if burmese_text:
                st.success(f"✅ Seg {i+1}: {burmese_text}")
            else:
                st.error(f"❌ Seg {i+1} Failed: Raw AI said -> '{raw_ai_response}'")

        # Audio Gen
        audio_created = False
        if burmese_text:
            fname = f"seg_{i}.mp3"
            if generate_audio(burmese_text, voice_config, fname):
                audio_created = True
                generated_count += 1
                
                # Sync logic
                seg_audio = AudioSegment.from_file(fname)
                curr_dur = len(seg_audio) / 1000.0
                target_dur = end - start
                
                if curr_dur > 0 and target_dur > 0:
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

    # ⚠️ FALLBACK: If NO audio was generated, create a dummy track to avoid FFmpeg error
    if generated_count == 0:
        st.warning("⚠️ Translation failed for all segments. Using original audio as fallback.")
        shutil.copy("temp.wav", "final_track.mp3")
    else:
        status.info("🔊 Step 4: Mixing Audio...")
        final_audio.export("final_track.mp3", format="mp3")

    status.info("🎬 Step 5: Finalizing Video...")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    
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
# 📝 SCRIPT & GENAI
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    p = f"Write a {type} script in Burmese about {topic}. Tone: {tone}. {prompt}"
    return model.generate_content(p).text

def text_to_speech_script(text, v_conf):
    clean = clean_text_for_burmese(text)
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
# 🖥️ UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    
    # Custom Model Input
    model_mode = st.radio("Model Settings", ["Preset", "Custom Input"])
    if model_mode == "Preset":
        model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    else:
        model_name = st.text_input("Enter Custom Model ID", value="gemini-2.5-flash")
    
    if st.button("🔴 Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "📝 Script", "🚀 Viral", "🖼️ Thumbnail"])

with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
    
    c1, c2 = st.columns(2)
    with c1: narrator = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Style", ["Normal", "Movie Recap", "News", "Deep"])
    
    # Config
    base_id = "my-MM-ThihaNeural" if "Male" in narrator else "my-MM-NilarNeural"
    if tone == "Movie Recap": v_conf = {"id": base_id, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": v_conf = {"id": base_id, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": v_conf = {"id": base_id, "rate": "-5%", "pitch": "-15Hz"}
    else: v_conf = {"id": base_id, "rate": "+0%", "pitch": "+0Hz"}
    
    style_prompt = tone + " style"

    if st.button("🚀 Start Dubbing") and uploaded_file:
        status = st.empty()
        prog = st.progress(0)
        try:
            out = process_video_pipeline("temp.mp4", v_conf, style_prompt, st.session_state.api_key, model_name, status, prog)
            if out:
                st.session_state.processed_video = out
                status.success("Success!")
                st.rerun()
        except Exception as e: status.error(f"Error: {e}")

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download", f, "dubbed.mp4")

with t2:
    st.subheader("📝 Script Writer")
    topic = st.text_input("Topic")
    if st.button("Generate"):
        res = generate_script(topic, "Script", "Normal", "", st.session_state.api_key, model_name)
        st.session_state.generated_script = res
        st.rerun()
    if st.session_state.generated_script:
        st.text_area("Script", st.session_state.generated_script)

with t3: st.info("Viral Kit Ready")
with t4: st.info("Thumbnail Ready")
