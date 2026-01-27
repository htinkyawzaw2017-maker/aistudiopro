import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import re
from pydub import AudioSegment

# Setup
st.set_page_config(page_title="AI Dubbing Debugger", layout="wide")
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# --- 1. SYSTEM CHECK ---
def check_system():
    if shutil.which("ffmpeg") is None:
        st.error("🚨 CRITICAL: FFmpeg is NOT installed on this server.")
        st.stop()

# --- 2. PYTHON NATIVE TTS (NO CLI) ---
async def generate_audio_native(text, voice, rate, pitch, output_file):
    try:
        # Using Python Library directly - Works better on Cloud
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_file)
        return True
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return False

# --- 3. TRANSLATION ---
def translate_text(model, text):
    try:
        prompt = f"""
        Translate to Burmese (Myanmar).
        Input: "{text}"
        Output: Burmese text only. No English. No explanations.
        Numbers: Convert to words.
        """
        response = model.generate_content(prompt)
        # Clean English chars
        clean = re.sub(r'[A-Za-z]', '', response.text).strip()
        return clean if clean else "အသံမထွက်ပါ"
    except:
        return "အသံမထွက်ပါ"

# --- 4. MAIN PROCESS ---
def debug_dubbing(video_path, voice, rate, pitch, api_key, model_name):
    check_system()
    status = st.status("🛠️ Starting Debug Process...", expanded=True)
    
    # Step 1: Extract Audio
    status.write("🎧 Extracting audio from video...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists("temp.wav"):
        status.update(label="❌ Failed to extract audio!", state="error")
        return None

    # Step 2: Whisper
    status.write("🧠 Transcribing with Whisper...")
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe("temp.wav")
    segments = result['segments']
    
    status.write(f"✅ Found {len(segments)} segments.")
    
    # Step 3: Loop & Dub
    genai.configure(api_key=api_key)
    ai_model = genai.GenerativeModel(model_name)
    
    # Create empty audio container
    video_len = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]).strip())
    final_audio = AudioSegment.silent(duration=video_len * 1000)
    
    for i, seg in enumerate(segments):
        start = seg['start'] * 1000 # to ms
        text = seg['text']
        
        status.write(f"🔹 Segment {i+1}: '{text}'")
        
        # Translate
        burmese = translate_text(ai_model, text)
        status.write(f"   ↳ Burmese: '{burmese}'")
        
        if not burmese or len(burmese) < 2:
            continue

        # Generate Audio
        fname = f"seg_{i}.mp3"
        
        # 🔥 RUN ASYNC FUNCTION IN SYNC LOOP
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_audio_native(burmese, voice, rate, pitch, fname))
        loop.close()
        
        # CHECK FILE SIZE
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            status.write(f"   ↳ Audio Generated: {size} bytes")
            
            if size > 100: # If audio is valid
                seg_audio = AudioSegment.from_file(fname)
                final_audio = final_audio.overlay(seg_audio, position=start)
            else:
                status.warning("   ⚠️ Audio file is empty!")
        else:
            status.error("   ❌ Audio file creation failed!")

    # Step 4: Export Audio
    status.write("🔊 Exporting final audio track...")
    final_audio.export("final_track.mp3", format="mp3")
    
    if os.path.getsize("final_track.mp3") < 1000:
        status.error("❌ Final audio is empty. Something is wrong with TTS.")
        return None

    # Step 5: Merge
    status.write("🎬 Merging with video...")
    out_file = "final_dubbed.mp4"
    # STRICT FFMPEG COMMAND
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', 'final_track.mp3',
        '-c:v', 'copy', # Copy video stream directly
        '-c:a', 'aac',  # Re-encode audio to AAC
        '-map', '0:v:0', # Take video from file 0
        '-map', '1:a:0', # Take audio from file 1
        out_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    status.update(label="✅ Process Complete!", state="complete")
    return out_file

# --- UI ---
st.title("🛠️ AI Dubbing Debugger")
st.warning("ဒီ Mode က အသံထွက်မထွက် စစ်ဆေးဖို့ဖြစ်ပါတယ်။")

api_key = st.text_input("API Key", type="password")
if api_key: st.session_state.api_key = api_key

uploaded = st.file_uploader("Upload Video", type=['mp4'])
if uploaded:
    with open("input.mp4", "wb") as f: f.write(uploaded.getbuffer())
    
    if st.button("🔴 Start Debugging"):
        if not st.session_state.api_key: st.error("No API Key"); st.stop()
        
        # Voice Settings
        voice = "my-MM-ThihaNeural"
        rate = "+0%"
        pitch = "-5Hz"
        model = "gemini-1.5-flash"
        
        out = debug_dubbing("input.mp4", voice, rate, pitch, st.session_state.api_key, model)
        
        if out and os.path.exists(out):
            st.success("Video Generated!")
            st.video(out)
