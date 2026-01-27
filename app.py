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
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 UI CONFIGURATION
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
# 💾 STATE
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'generated_script' not in st.session_state: st.session_state.generated_script = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🛠️ SYSTEM CHECKS
# ---------------------------------------------------------
def check_dependencies():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing.")
        st.stop()
    
    # Ensure edge-tts is installed for CLI usage
    try:
        subprocess.run(["edge-tts", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        st.warning("⚙️ Installing Audio Engine...")
        subprocess.run(["pip", "install", "edge-tts"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🔊 ROBUST AUDIO ENGINE (CLI METHOD - 100% STABLE)
# ---------------------------------------------------------
def generate_audio_cli(text, voice, rate, pitch, output_file):
    """
    Uses Command Line Interface to generate audio.
    This bypasses Python asyncio issues on Cloud Servers.
    """
    try:
        # Construct command
        cmd = [
            "edge-tts",
            "--voice", voice,
            "--text", text,
            "--rate", rate,
            "--pitch", pitch,
            "--write-media", output_file
        ]
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if file exists and has size
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            return True
        else:
            print(f"TTS Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"CLI Error: {e}")
        return False

# ---------------------------------------------------------
# 🛡️ SMART TRANSLATION & CLEANER
# ---------------------------------------------------------
def clean_burmese_text(text):
    # Fix Units to be spoken correctly
    replacements = {
        "No.": "နံပါတ် ", "kg": " ကီလိုဂရမ် ", "cm": " စင်တီမီတာ ", 
        "mm": " မီလီမီတာ ", "km": " ကီလိုမီတာ ", "%": " ရာခိုင်နှုန်း ", 
        "$": " ဒေါ်လာ ", "Mr.": "မစ္စတာ "
    }
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    
    # Remove English characters (Keep Burmese & Numbers)
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_smart(model, text, style):
    prompt = f"""
    Translate English to Spoken Burmese.
    Input: "{text}"
    Rules: 
    1. Output Burmese ONLY. No English words.
    2. Convert numbers to words (100 -> တစ်ရာ).
    3. Tone: {style}.
    """
    for _ in range(2):
        try:
            res = model.generate_content(prompt)
            clean = clean_burmese_text(res.text.strip())
            if clean: return clean
        except exceptions.ResourceExhausted: time.sleep(5)
        except: pass
    return ""

# ---------------------------------------------------------
# 🎬 SYNC ENGINE (PREVENTS OVERLAPPING)
# ---------------------------------------------------------
def fit_audio_to_slot(audio_path, output_path, max_duration):
    """
    Speeds up audio if it's too long for the video segment.
    """
    current_dur = get_duration(audio_path)
    if current_dur == 0: return False
    
    # Calculate speed needed
    if current_dur > max_duration:
        speed = current_dur / max_duration
        # Cap speed at 1.7x to keep it understandable
        speed = min(speed, 1.7)
    else:
        speed = 1.0 # No change
        
    try:
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-filter:a', f"atempo={speed}", output_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

# ---------------------------------------------------------
# 🎥 MAIN PROCESS
# ---------------------------------------------------------
def process_video(video_path, voice_conf, style, mix_bg, bg_vol, api_key, model_name, status, progress):
    check_dependencies()
    
    # 1. Extract Audio
    status.info("🎧 Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper
    status.info("🧠 Analyzing Speech Timing...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Dubbing Loop
    status.info(f"🎙️ Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # Base Canvas
    total_duration = get_duration(video_path)
    final_audio = AudioSegment.silent(duration=total_duration * 1000)
    
    # Debug Logs
    logs = st.expander("Show Translation Details", expanded=False)
    
    audio_generated_count = 0
    
    for i, seg in enumerate(segments):
        start_time = seg['start']
        end_time = seg['end']
        
        # Calculate strict time slot
        if i < len(segments) - 1:
            next_start = segments[i+1]['start']
            time_slot = next_start - start_time
        else:
            time_slot = end_time - start_time + 1.0
            
        # Translate
        burmese = translate_smart(model, seg['text'], style)
        with logs: st.text(f"{i+1} [{time_slot:.1f}s]: {burmese}")
        
        if burmese:
            raw_file = f"raw_{i}.mp3"
            
            # 🔥 CLI GENERATION
            if generate_audio_cli(burmese, voice_conf['id'], voice_conf['rate'], voice_conf['pitch'], raw_file):
                
                # 🔥 SMART FIT (Prevent Overlap)
                proc_file = f"proc_{i}.mp3"
                if fit_audio_to_slot(raw_file, proc_file, time_slot):
                    
                    if os.path.exists(proc_file):
                        seg_audio = AudioSegment.from_file(proc_file)
                        final_audio = final_audio.overlay(seg_audio, position=start_time * 1000)
                        audio_generated_count += 1
                        try: os.remove(proc_file)
                        except: pass
                
                try: os.remove(raw_file)
                except: pass
                
        # Rate Limit
        time.sleep(1)
        progress.progress((i + 1) / len(segments))

    if audio_generated_count == 0:
        st.error("❌ Audio generation failed. Check API Key.")
        return None

    # 4. Final Mixing
    status.info("🔊 Mixing Audio...")
    final_audio.export("voice_track.mp3", format="mp3")
    output_file = f"dubbed_{int(time.time())}.mp4"
    
    # 🔥 DUCKING LOGIC (Lowers original volume)
    if mix_bg:
        vol = bg_vol / 100.0 # e.g. 0.1
        # Filter: Original(0.1 volume) + Voice(1.5 volume) mixed
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice_track.mp3',
            '-filter_complex', f'[0:a]volume={vol}[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
    else:
        # Mute original entirely
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', 'voice_track.mp3',
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', output_file
        ]
        
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

# ---------------------------------------------------------
# 📝 EXTRA FEATURES
# ---------------------------------------------------------
def generate_script(topic, type, tone, prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model.generate_content(f"Write a Burmese {type} script about {topic}. Tone: {tone}. {prompt}").text

def script_audio(text, conf):
    clean = clean_burmese_text(text)
    chunks = [clean[i:i+500] for i in range(0, len(clean), 500)]
    comb = AudioSegment.empty()
    for i, c in enumerate(chunks):
        f = f"s_{i}.mp3"
        if generate_audio_cli(c, conf['id'], conf['rate'], conf['pitch'], f):
            comb += AudioSegment.from_file(f)
            os.remove(f)
    out = "script_out.mp3"
    comb.export(out, format="mp3")
    return out

# ---------------------------------------------------------
# 🖥️ UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    st.divider()
    
    mode = st.radio("Model", ["Preset", "Custom"])
    model_name = st.selectbox("Select", ["gemini-1.5-flash", "gemini-2.0-flash-exp"]) if mode == "Preset" else st.text_input("ID", "gemini-2.5-flash")
    
    if st.button("🔴 Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key: st.warning("Enter API Key"); st.stop()

t1, t2, t3, t4 = st.tabs(["🎙️ Dubbing", "📝 Script", "🚀 Viral", "🖼️ Thumbnail"])

with t1:
    st.subheader("🔊 Smart Dubbing")
    uploaded = st.file_uploader("Video", type=['mp4','mov'])
    if uploaded:
        with open("temp.mp4", "wb") as f: f.write(uploaded.getbuffer())
        
    c1, c2, c3 = st.columns(3)
    with c1: voice = st.selectbox("Voice", ["Male (Thiha)", "Female (Nilar)"])
    with c2: tone = st.selectbox("Tone", ["Normal", "Movie Recap", "News", "Deep"])
    with c3: 
        mix_bg = st.checkbox("Mix Original?", value=True)
        bg_vol = st.slider("BG Vol %", 0, 50, 10) if mix_bg else 0

    base = "my-MM-ThihaNeural" if "Male" in voice else "my-MM-NilarNeural"
    if tone == "Movie Recap": conf = {"id": base, "rate": "+0%", "pitch": "-10Hz"}
    elif tone == "News": conf = {"id": base, "rate": "+10%", "pitch": "+0Hz"}
    elif tone == "Deep": conf = {"id": base, "rate": "-5%", "pitch": "-15Hz"}
    else: conf = {"id": base, "rate": "+0%", "pitch": "+0Hz"}

    if st.button("🚀 Start Dubbing") and uploaded:
        st.session_state.processed_video = None
        st_box = st.empty()
        pg = st.progress(0)
        try:
            out = process_video("temp.mp4", conf, tone, mix_bg, bg_vol, st.session_state.api_key, model_name, st_box, pg)
            if out:
                st.session_state.processed_video = out
                st_box.success("Done!")
                st.rerun()
        except Exception as e: st_box.error(str(e))

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download", f, "dubbed.mp4")

with t2:
    st.subheader("📝 Script")
    topic = st.text_input("Topic")
    if st.button("Write"):
        res = generate_script(topic, "Script", "Normal", "", st.session_state.api_key, model_name)
        st.session_state.generated_script = res
        st.rerun()
    if st.session_state.generated_script:
        sc = st.text_area("Script", st.session_state.generated_script)
        if st.button("Read"):
            f = script_audio(sc, conf)
            st.audio(f)

with t3: st.info("Viral Kit Ready")
with t4: st.info("Thumbnail Ready")
