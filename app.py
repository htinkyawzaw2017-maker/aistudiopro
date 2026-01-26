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
from google.api_core import exceptions

# ---------------------------------------------------------
# 🎨 PRO UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Global AI Studio", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    .css-1r6slb0, .stFileUploader, div[data-testid="stSidebar"] {
        background: #111; border: 1px solid #333; border-radius: 15px; padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff00cc, #333399); color: white; border: none;
        height: 50px; font-weight: bold; width: 100%; border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); }
    .success-text { color: #00ff00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🌍 VOICE DATABASE (EXPANDED)
# ---------------------------------------------------------
# edge-tts only has 2 Burmese voices, so we create "Styles" by manipulating Pitch/Rate
VOICE_OPTIONS = {
    "Myanmar (Thiha - Male News)": {"voice": "my-MM-ThihaNeural", "pitch": "+0Hz", "rate": "+0%"},
    "Myanmar (Nilar - Female Story)": {"voice": "my-MM-NilarNeural", "pitch": "+0Hz", "rate": "+0%"},
    "Myanmar (Thiha - Deep Narrator)": {"voice": "my-MM-ThihaNeural", "pitch": "-15Hz", "rate": "-10%"},
    "Myanmar (Nilar - Sweet Vlog)": {"voice": "my-MM-NilarNeural", "pitch": "+10Hz", "rate": "+10%"},
    
    "English (US - Guy)": {"voice": "en-US-ChristopherNeural", "pitch": "+0Hz", "rate": "+0%"},
    "English (US - Jenny)": {"voice": "en-US-JennyNeural", "pitch": "+0Hz", "rate": "+0%"},
}

# ---------------------------------------------------------
# 🛠️ SYSTEM FUNCTIONS
# ---------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("❌ FFmpeg Missing! System cannot process audio.")
        st.stop()

def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(r.stdout)['format']['duration'])
    except: return 0

async def _tts_async(text, voice, rate, pitch, output):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output)

def generate_tts(text, voice_config, filename):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_async(text, voice_config['voice'], voice_config['rate'], voice_config['pitch'], filename))
        loop.close()
        return True
    except: return False

def upload_to_gemini(video_path, api_key):
    genai.configure(api_key=api_key)
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    return video_file

def extract_frame(video_path, output_image):
    try:
        duration = get_duration(video_path)
        subprocess.run(['ffmpeg', '-y', '-ss', str(duration/2), '-i', video_path, '-vframes', '1', output_image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except: return False

# ---------------------------------------------------------
# 🤬 STRICT TRANSLATION LOGIC (NO ENGLISH ALLOWED)
# ---------------------------------------------------------
def translate_strict(model, text, target_lang, style):
    # Forced Prompt Engineering
    prompt = f"""
    ROLE: Professional Dubbing Translator.
    TASK: Translate the following text into {target_lang}.
    INPUT TEXT: "{text}"
    
    🚨 STRICT RULES (MUST FOLLOW):
    1. OUTPUT ONLY IN {target_lang} SCRIPT.
    2. ABSOLUTELY NO ENGLISH CHARACTERS ALLOWED in the output.
    3. If the input is "Hello", output "မင်္ဂလာပါ" (Not "Hello").
    4. Translate technical terms phonetically if needed (e.g., "AI" -> "အေအိုင်").
    5. Tone/Style: {style} (Natural spoken flow).
    6. DO NOT add explanations. JUST the translation.
    """
    
    try:
        response = model.generate_content(prompt)
        translated = response.text.strip()
        
        # Double Check: If output contains English letters, force retry (Python-side check)
        if target_lang == "Myanmar" and re.search(r'[a-zA-Z]', translated):
            # Fallback prompt for stubborn AI
            retry_prompt = f"You outputted English. REWRITE THIS IN BURMESE SCRIPT ONLY: '{translated}'"
            response = model.generate_content(retry_prompt)
            translated = response.text.strip()
            
        return translated
    except:
        return text # Worst case fallback

def process_dubbing_workflow(video_path, voice_key, style, api_key, model_id, status_box, progress_bar):
    check_ffmpeg()
    
    # 1. Setup Voice
    voice_config = VOICE_OPTIONS[voice_key]
    target_lang = "Myanmar" if "Myanmar" in voice_key else "English"
    
    # 2. Extract Audio
    status_box.update(label="🎧 Extracting Audio...", state="running")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Whisper Transcription
    status_box.update(label="🧠 Speech Recognition (Whisper)...", state="running")
    model = whisper.load_model("base")
    result = model.transcribe("temp.wav")
    segments = result['segments']
    
    # 4. AI Translation & Dubbing
    status_box.update(label=f"🎙️ Dubbing to {target_lang} ({style})...", state="running")
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model_id)
    
    video_dur = get_duration(video_path)
    final_audio = AudioSegment.silent(duration=video_dur * 1000)
    
    total = len(segments)
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        text = seg['text']
        
        # 🔥 STRICT TRANSLATION
        translated_text = translate_strict(gemini, text, target_lang, style)
        
        # TTS Generation
        fname = f"seg_{i}.mp3"
        generate_tts(translated_text, voice_config, fname)
        
        # Sync & Stretch
        if os.path.exists(fname):
            seg_audio = AudioSegment.from_file(fname)
            seg_dur = end - start
            curr_dur = len(seg_audio) / 1000.0
            
            if curr_dur > 0:
                speed = max(0.6, min(curr_dur / seg_dur, 1.5))
                subprocess.run(['ffmpeg', '-y', '-i', fname, '-filter:a', f"atempo={speed}", f"s_{i}.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(f"s_{i}.mp3"):
                    final_seg = AudioSegment.from_file(f"s_{i}.mp3")
                    final_audio = final_audio.overlay(final_seg, position=start * 1000)
        
        progress_bar.progress((i + 1) / total)
        # Cleanup
        try: 
            os.remove(fname)
            os.remove(f"s_{i}.mp3")
        except: pass

    final_audio.export("final_track.mp3", format="mp3")
    
    # 5. Merge
    status_box.update(label="🎬 Finalizing Video...", state="running")
    output_file = "dubbed_output.mp4"
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-i', "final_track.mp3", '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_file

# ---------------------------------------------------------
# 🚀 FEATURES: VIRAL, SCRIPT, THUMBNAIL
# ---------------------------------------------------------
def feature_viral(video_path, api_key, model_id):
    genai.configure(api_key=api_key)
    video = upload_to_gemini(video_path, api_key)
    model = genai.GenerativeModel(model_id)
    res = model.generate_content([video, "Generate 3 Viral Clickbait Titles (Burmese) and 10 Hashtags for this video."])
    return res.text

def feature_script(video_path, api_key, model_id):
    genai.configure(api_key=api_key)
    video = upload_to_gemini(video_path, api_key)
    model = genai.GenerativeModel(model_id)
    res = model.generate_content([video, "Write a full detailed Blog Post script in Burmese based on this video."])
    return res.text

def feature_thumbnail(video_path, api_key, model_id):
    extract_frame(video_path, "thumb.jpg")
    img = Image.open("thumb.jpg")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    res = model.generate_content([img, "Describe this image for a high-CTR YouTube Thumbnail prompt."])
    return res.text, "thumb.jpg"

# ---------------------------------------------------------
# 🖥️ MAIN DASHBOARD
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎙️ AI STUDIO PRO")
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    if api_key: st.session_state.api_key = api_key
    
    st.divider()
    model_choice = st.selectbox("AI Model", ["gemini-1.5-flash", "gemini-2.0-flash-exp"])
    if st.button("🔴 REBOOT APP"): st.rerun()

# Tabs for Features
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ DUBBING", "🚀 VIRAL KIT", "📝 SCRIPT", "🖼️ THUMBNAIL"])

if not st.session_state.api_key:
    st.warning("⚠️ Please Enter API Key in Sidebar")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video", type=['mp4', 'mov'])

if uploaded_file:
    with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    # --- TAB 1: DUBBING ---
    with tab1:
        st.subheader("🔊 Auto Dubbing Studio")
        
        c1, c2 = st.columns(2)
        with c1:
            # 🔥 NEW: Detailed Narrator Selection
            voice_choice = st.selectbox("🗣️ Choose Narrator", list(VOICE_OPTIONS.keys()))
        with c2:
            style_choice = st.selectbox("🎭 Content Style", ["Natural Conversation", "News Reporter", "Movie Recap (Dramatic)", "Vlogger (Casual)"])
            
        if st.button("🚀 START DUBBING", type="primary"):
            status = st.status("Initializing...", expanded=True)
            progress = st.progress(0)
            try:
                out = process_dubbing_workflow("temp.mp4", voice_choice, style_choice, st.session_state.api_key, model_choice, status, progress)
                status.update(label="✅ Dubbing Complete!", state="complete")
                st.video(out)
                with open(out, "rb") as f: st.download_button("💾 Download Video", f, "dubbed.mp4")
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    # --- TAB 2: VIRAL KIT ---
    with tab2:
        if st.button("✨ Generate Viral Data"):
            with st.spinner("Analyzing..."):
                res = feature_viral("temp.mp4", st.session_state.api_key, model_choice)
                st.info(res)

    # --- TAB 3: SCRIPT ---
    with tab3:
        if st.button("✍️ Write Script"):
            with st.spinner("Writing..."):
                res = feature_script("temp.mp4", st.session_state.api_key, model_choice)
                st.text_area("Script", res, height=400)

    # --- TAB 4: THUMBNAIL ---
    with tab4:
        if st.button("🎨 Analyze Thumbnail"):
            with st.spinner("Processing..."):
                txt, img = feature_thumbnail("temp.mp4", st.session_state.api_key, model_choice)
                st.image(img)
                st.code(txt)
