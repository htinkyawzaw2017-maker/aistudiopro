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
    textarea { font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing."); st.stop()

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

async def tts_save(text, voice, rate, pitch, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_file)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def generate_audio(text, voice_id, rate, pitch, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(tts_save(text, voice_id, rate, pitch, filename))
        loop.close()
        return result
    except: return False

# ---------------------------------------------------------
# 🛡️ SMART TEXT CLEANER (UNIT FIXER)
# ---------------------------------------------------------
def normalize_burmese(text):
    # 1. Force Unit Conversion (Python Side Backup)
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
        text = text.replace(k, v)
        text = text.replace(k.lower(), v) # case insensitive check
    
    # 2. Remove English Chars (English Killer)
    # Keep Burmese (u1000-u109f), Numbers, and Basic Punctuation
    cleaned = re.sub(r'[A-Za-z]', '', text)
    return cleaned.strip()

def translate_strict(model, text, style):
    prompt = f"""
    Act as a professional Burmese Dubbing Artist.
    Translate the following text to spoken Burmese (Myanmar).
    
    Input: "{text}"
    
    CRITICAL RULES:
    1. **Language**: OUTPUT BURMESE ONLY. No English words.
    2. **Numbers**: Convert to words (e.g., "100" -> "တစ်ရာ").
    3. **Units**: Convert "kg", "mm", "No." to Burmese words (ကီလို, မီလီ, နံပါတ်).
    4. **Tone**: {style}. 
       - Movie Recap: Dramatic, Storytelling.
       - News: Formal, Clear.
       - Deep: Serious.
    5. **Output**: Just the translation.
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            # Normalize and Clean
            final_text = normalize_burmese(translated)
            if final_text: return final_text
        except exceptions.ResourceExhausted:
            time.sleep(5)
            continue
        except:
            continue
            
    return "" # Return empty if failed

# ---------------------------------------------------------
# 🎬 MAIN DUBBING WORKFLOW
# ---------------------------------------------------------
def process_video_dubbing(video_path, voice_config, style, api_key, model_name, status, progress):
    check_requirements()
    
    # Unpack Voice Config
    voice_id = voice_config["id"]
    rate = voice_config["rate"]
    pitch = voice_config["pitch"]
    
    # 1. Extract Audio
    status.info("🎧 Step 1: Extracting Audio...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper Transcription
    status.info("🧠 Step 2: Listening (Whisper)...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. Translate & Dub
    status.info(f"🎙️ Step 3: Dubbing {len(segments)} segments...")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")

    # Create Silent Base Track
    total_duration = get_duration(video_path)
    final_audio = AudioSegment.silent(duration=total_duration * 1000)
    
    # Load Original Audio for Fallback (Low Volume)
    original_audio = AudioSegment.from_wav("temp.wav")
    original_audio = original_audio - 15 # Lower volume by 15dB
    
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        
        # Rate Limit Safety
        time.sleep(1)
        
        # Translate
        burmese_text = translate_strict(model, seg['text'], style)
        
        audio_segment_generated = False
        
        if burmese_text:
            fname = f"seg_{i}.mp3"
            success = generate_audio(burmese_text, voice_id, rate, pitch, fname)
            
            if success and os.path.exists(fname):
                seg_audio = AudioSegment.from_file(fname)
                curr_dur = len(seg_audio) / 1000.0
                target_dur = end - start
                
                if curr_dur > 0 and target_dur > 0:
                    # Time Stretch
                    speed = max(0.6, min(curr_dur / target_dur, 1.5))
                    subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(f"s_{i}.mp3"):
                        final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                        final_audio = final_audio.overlay(final_seg, position=start * 1000)
                        audio_segment_generated = True
                        
                        try: os.remove(f"s_{i}.mp3")
                        except: pass
                
                try: os.remove(fname)
                except: pass

        # Fallback: If AI failed or text was empty, keep original audio for this segment?
        # Better strategy: We leave the silent base. 
        # But if you want background noise, we can overlay original low volume audio globally.
        
        progress.progress((i + 1) / total)

    status.info("🔊 Step 4: Mixing Audio...")
    
    # Optional: Mix with low background music/original audio to prevent total silence gaps
    # final_audio = final_audio.overlay(original_audio) 
    
    final_audio.export("final_track.mp3", format="mp3")
    
    # Check if audio file is valid
    if os.path.getsize("final_track.mp3") < 1000:
        st.warning("⚠️ Warning: Generated audio is very short. Using original audio as backup.")
        shutil.copy("temp.wav", "final_track.mp3")

    status.info("🎬 Step 5: Finalizing...")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_filename

# ---------------------------------------------------------
# 📝 SCRIPT & AUDIO FEATURES
# ---------------------------------------------------------
def generate_script(topic, script_type, user_prompt, api_key, model_name):
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    Act as a Burmese Content Creator. Write a '{script_type}' script.
    Topic: "{topic}"
    Instructions: {user_prompt}
    
    Language: Burmese (Myanmar).
    Format: Professional script format.
    """
    return model.generate_content(prompt).text

def script_to_audio_task(script_text, voice_config):
    # Clean text
    clean_text = normalize_burmese(script_text)
    
    # Split for TTS limit (max 500 chars roughly)
    chunks = [clean_text[i:i+500] for i in range(0, len(clean_text), 500)]
    combined = AudioSegment.empty()
    
    for idx, chunk in enumerate(chunks):
        fname = f"script_part_{idx}.mp3"
        voice_id = voice_config["id"]
        generate_audio(chunk, voice_id, voice_config["rate"], voice_config["pitch"], fname)
        if os.path.exists(fname):
            combined += AudioSegment.from_file(fname)
            try: os.remove(fname)
            except: pass
            
    out_path = f"script_audio_{int(time.time())}.mp3"
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
    model_choice = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro"])
    
    if st.button("🔴 Reset App"):
        st.session_state.processed_video = None
        st.session_state.generated_script = ""
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key."); st.stop()

# TABS
t1, t2, t3, t4 = st.tabs(["🎙️ Video Dubbing", "📝 Script Writer", "🚀 Viral Kit", "🖼️ Thumbnail"])

# --- TAB 1: DUBBING ---
with t1:
    st.subheader("🔊 Video Dubbing")
    uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])
    
    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    c1, c2, c3 = st.columns(3)
    with c1: narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
    with c2: 
        tone = st.selectbox("Tone/Style", [
            "Normal", 
            "Movie Recap (Dramatic)", 
            "News (Formal)", 
            "Deep (Serious)", 
            "Calming (Soft)",
            "Fast (Excited)"
        ])
    
    # Voice Config Logic
    base_id = "my-MM-ThihaNeural" if "Male" in narrator else "my-MM-NilarNeural"
    
    if tone == "Movie Recap (Dramatic)":
        v_conf = {"id": base_id, "rate": "+0%", "pitch": "-10Hz"}
        style_prompt = "Dramatic, Storytelling"
    elif tone == "News (Formal)":
        v_conf = {"id": base_id, "rate": "+10%", "pitch": "+0Hz"}
        style_prompt = "Formal, Professional"
    elif tone == "Deep (Serious)":
        v_conf = {"id": base_id, "rate": "-10%", "pitch": "-20Hz"}
        style_prompt = "Serious, Deep voice"
    elif tone == "Calming (Soft)":
        v_conf = {"id": base_id, "rate": "-5%", "pitch": "+5Hz"}
        style_prompt = "Soft, Gentle"
    elif tone == "Fast (Excited)":
        v_conf = {"id": base_id, "rate": "+25%", "pitch": "+10Hz"}
        style_prompt = "Excited, Fast paced"
    else:
        v_conf = {"id": base_id, "rate": "+0%", "pitch": "+0Hz"}
        style_prompt = "Natural Conversation"

    if uploaded_file and st.button("🚀 Start Dubbing"):
        st.session_state.processed_video = None
        status = st.empty()
        prog = st.progress(0)
        try:
            # FIXED FUNCTION NAME
            out = process_video_dubbing("temp.mp4", v_conf, style_prompt, st.session_state.api_key, model_choice, status, prog)
            st.session_state.processed_video = out
            status.success("Dubbing Complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            
    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)
        with open(st.session_state.processed_video, "rb") as f:
            st.download_button("Download Video", f, "dubbed.mp4")

# --- TAB 2: SCRIPT ---
with t2:
    st.subheader("📝 Script & Voiceover")
    
    sc1, sc2 = st.columns(2)
    with sc1: script_type = st.selectbox("Type", ["Story", "Movie Script", "Video Voiceover", "Documentary"])
    with sc2: topic = st.text_input("Topic")
    
    user_prompt = st.text_area("Specific Instructions (Prompt)", placeholder="Example: Start with a hook about history...")
    
    if st.button("✍️ Generate Script"):
        if not topic: st.warning("Enter a topic!")
        else:
            with st.spinner("Writing..."):
                res = generate_script(topic, script_type, user_prompt, st.session_state.api_key, model_choice)
                st.session_state.generated_script = res
                st.rerun()
    
    if st.session_state.generated_script:
        st.markdown("### Generated Script")
        edited_script = st.text_area("Edit Script:", value=st.session_state.generated_script, height=300)
        
        st.markdown("### 🔊 Convert to Audio")
        ac1, ac2 = st.columns(2)
        with ac1: s_voice = st.selectbox("Audio Voice", ["Male (Thiha)", "Female (Nilar)"], key="sv")
        with ac2: s_tone = st.selectbox("Audio Tone", ["Normal", "Deep", "News"], key="st")
        
        s_base = "my-MM-ThihaNeural" if "Male" in s_voice else "my-MM-NilarNeural"
        if s_tone == "Deep": s_conf = {"id": s_base, "rate": "-10%", "pitch": "-15Hz"}
        elif s_tone == "News": s_conf = {"id": s_base, "rate": "+10%", "pitch": "+0Hz"}
        else: s_conf = {"id": s_base, "rate": "+0%", "pitch": "+0Hz"}
        
        if st.button("🗣️ Read Script"):
            with st.spinner("Generating Audio..."):
                a_path = script_to_audio_task(edited_script, s_conf)
                st.audio(a_path)
                with open(a_path, "rb") as f:
                    st.download_button("Download MP3", f, "script.mp3")

# --- TAB 3 & 4 (Placeholders) ---
with t3: st.info("Viral Kit Ready.")
with t4: st.info("Thumbnail Ready.")
