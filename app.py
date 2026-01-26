import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
import shutil
import whisper
import re
from pydub import AudioSegment
from PIL import Image

# ---------------------------------------------------------
# 💾 STATE MANAGEMENT
# ---------------------------------------------------------
if 'processed_video' not in st.session_state: st.session_state.processed_video = None
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🇲🇲", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B, #FF416C); 
        color: white; border: none; height: 50px; font-weight: bold; width: 100%;
        border-radius: 10px; font-size: 16px;
    }
    .success-box { padding: 15px; background: #051a05; border: 1px solid #00ff00; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------
def check_requirements():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg is missing."); st.stop()

async def tts_save(text, voice, rate, pitch, output_file):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, voice_id, rate, pitch, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts_save(text, voice_id, rate, pitch, filename))
        loop.close()
        return True
    except: return False

def get_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

# ---------------------------------------------------------
# 🧠 BATCH TRANSLATION (The Fix for Repeating Audio)
# ---------------------------------------------------------
def batch_translate(model, segments, style):
    # 1. Combine all text into one block with separators
    full_text = " ||| ".join([seg['text'].strip() for seg in segments])
    
    prompt = f"""
    Act as a Burmese Dubbing Expert.
    Translate the following text segments from English to Burmese (Myanmar).
    The segments are separated by " ||| ". Keep the separators in your output.
    
    INPUT TEXT:
    "{full_text}"
    
    STRICT RULES:
    1. OUTPUT BURMESE ONLY. NO ENGLISH CHARACTERS.
    2. Maintain the " ||| " separators exactly.
    3. Convert numbers to words (e.g., 500 -> ငါးရာ).
    4. Style: {style}.
    5. NO EXPLANATIONS. JUST THE TRANSLATED STRING.
    """
    
    try:
        response = model.generate_content(prompt)
        translated_string = response.text.strip()
        
        # 2. Clean English characters just in case
        cleaned_string = re.sub(r'[A-Za-z]', '', translated_string)
        
        # 3. Split back into list
        translated_segments = cleaned_string.split("|||")
        
        # Ensure list length matches
        if len(translated_segments) != len(segments):
            # Fallback: If mismatch, return original text to avoid crashing
            return [s['text'] for s in segments]
            
        return [t.strip() for t in translated_segments]
        
    except Exception as e:
        print(f"Batch Error: {e}")
        return [s['text'] for s in segments] # Fail safe: use original text

# ---------------------------------------------------------
# 🎬 MAIN WORKFLOW
# ---------------------------------------------------------
def process_full_workflow(video_path, voice_data, style, api_key, model_name, status, progress):
    check_requirements()
    
    # 1. Extract Audio
    status.update(label="🎧 Extracting Audio...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Whisper Transcription
    status.update(label="🧠 Analyzing Speech...", state="running")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp.wav")
    segments = result['segments']
    
    # 3. BATCH TRANSLATION (ONE API CALL)
    status.update(label="🎙️ Translating (Batch Mode)...", state="running")
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel(model_name)
    except: model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Get all translations at once
    translated_texts = batch_translate(model, segments, style)

    # 4. Generate Audio & Sync
    status.update(label="🔊 Generating Audio...", state="running")
    final_audio = AudioSegment.silent(duration=get_duration(video_path) * 1000)
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        
        # Get corresponding translated text
        try:
            burmese_text = translated_texts[i]
        except:
            burmese_text = seg['text'] # Index fallback
            
        if not burmese_text.strip(): continue # Skip empty
        
        # TTS
        fname = f"seg_{i}.mp3"
        generate_audio(burmese_text, voice_data["id"], voice_data["rate"], voice_data["pitch"], fname)
        
        # Sync Logic
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            curr_dur = len(seg_audio) / 1000.0
            target_dur = end - start
            
            if curr_dur > 0 and target_dur > 0:
                speed = max(0.6, min(curr_dur / target_dur, 1.5))
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(f"s_{i}.mp3"):
                    final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
        
        progress.progress((i + 1) / total)
        try: os.remove(fname); os.remove(f"s_{i}.mp3")
        except: pass

    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. Merge
    status.update(label="🎬 Finalizing...", state="running")
    output_filename = f"dubbed_{int(time.time())}.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_filename

# ---------------------------------------------------------
# 🚀 FEATURES
# ---------------------------------------------------------
def run_genai(prompt, video_path, api_key, model_name):
    genai.configure(api_key=api_key)
    f = genai.upload_file(video_path)
    while f.state.name == "PROCESSING": time.sleep(2); f = genai.get_file(f.name)
    model = genai.GenerativeModel(model_name)
    return model.generate_content([f, prompt]).text

def run_thumbnail(video_path, api_key, model_name):
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vframes', '1', 'thumb.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    genai.configure(api_key=api_key)
    img = Image.open("thumb.jpg")
    model = genai.GenerativeModel(model_name)
    return model.generate_content([img, "Describe this for a Thumbnail prompt."]).text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇲🇲 AI Studio Pro")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    model_name = st.text_input("Model", value="gemini-2.5-flash")
    if st.button("🔴 Force Reset"):
        st.session_state.processed_video = None
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter API Key."); st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    t1, t2, t3, t4 = st.tabs(["🎙️ DUBBING", "🚀 VIRAL KIT", "📝 SCRIPT", "🖼️ THUMBNAIL"])

    # 1. DUBBING
    with t1:
        st.subheader("🔊 Auto Dubbing (Batch Fixed)")
        
        # Display Processed Video from Memory
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.markdown('<div class="success-box"><h3>✅ Video Ready!</h3></div>', unsafe_allow_html=True)
            st.video(st.session_state.processed_video)
            with open(st.session_state.processed_video, "rb") as f:
                st.download_button("💾 Download", f, "final_dub.mp4")

        c1, c2 = st.columns(2)
        with c1:
            narrator = st.selectbox("Narrator", ["Male (Thiha)", "Female (Nilar)"])
        with c2:
            style = st.selectbox("Style", ["Natural", "News", "Dramatic"])
            
        v_data = {
            "Male (Thiha)": {"id": "my-MM-ThihaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Female (Nilar)": {"id": "my-MM-NilarNeural", "rate": "+10%", "pitch": "+10Hz"}
        }

        if st.button("🚀 START DUBBING"):
            status = st.status("Processing...", expanded=True)
            prog = st.progress(0)
            try:
                # Calls the new batch workflow
                out = process_full_workflow("temp.mp4", v_data[narrator], style, st.session_state.api_key, model_name, status, prog)
                st.session_state.processed_video = out
                status.update(label="✅ Done!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    # 2. VIRAL KIT
    with t2:
        if st.button("✨ Generate Viral Info"):
            with st.spinner("Processing..."):
                res = run_genai("Generate 3 Viral Titles (Burmese) & Hashtags.", "temp.mp4", st.session_state.api_key, model_name)
                st.info(res)

    # 3. SCRIPT
    with t3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                res = run_genai("Write a full Burmese script.", "temp.mp4", st.session_state.api_key, model_name)
                st.text_area("Script", res, height=400)

    # 4. THUMBNAIL
    with t4:
        if st.button("🎨 Thumbnail Idea"):
            with st.spinner("Analyzing..."):
                txt, img = run_thumbnail("temp.mp4", st.session_state.api_key, model_name)
                st.image(img)
                st.code(txt)
