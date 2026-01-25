import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import subprocess
import json
import os
import time
from PIL import Image

# ---------------------------------------------------------
# 🎨 UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Studio Pro", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white; border-radius: 12px; height: 50px; border: none; font-weight: bold; width: 100%;
    }
    .stButton>button:hover { transform: scale(1.02); }
    [data-testid='stFileUploader'] { background-color: #1E1E1E; border: 1px dashed #555; border-radius: 10px; padding: 15px; }
    .stFileUploader label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #333; }
    
    .progress-box {
        padding: 20px; border-radius: 10px; background-color: #1a1a1a; border: 1px solid #00e5ff;
        text-align: center; margin-bottom: 20px;
    }
    .status-text { color: #00e5ff; font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 SESSION STATE
# ---------------------------------------------------------
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'current_task' not in st.session_state: st.session_state.current_task = ""
if 'api_key_storage' not in st.session_state: st.session_state.api_key_storage = ""

# Result Storage
if 'dub_result' not in st.session_state: st.session_state.dub_result = None
if 'viral_result' not in st.session_state: st.session_state.viral_result = None
if 'script_result' not in st.session_state: st.session_state.script_result = None
if 'thumb_img' not in st.session_state: st.session_state.thumb_img = None
if 'thumb_prompt' not in st.session_state: st.session_state.thumb_prompt = None

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS (ROBUST)
# ---------------------------------------------------------
def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def get_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except:
        return 0

def extract_frame(video_path, output_image):
    try:
        duration = get_duration(video_path)
        mid_point = duration / 2
        cmd = ['ffmpeg', '-y', '-ss', str(mid_point), '-i', video_path, '-vframes', '1', output_image]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

# ---------------------------------------------------------
# 🧠 ASYNC PROCESSING FUNCTIONS
# ---------------------------------------------------------
async def process_dubbing(video_path, gender, style, tone, progress_bar, status_text, api_key):
    try:
        if not check_ffmpeg():
            raise Exception("FFmpeg not found! Please install FFmpeg first (brew install ffmpeg).")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') 

        # Step 1: Upload
        status_text.write("📤 Step 1/5: Uploading & Analyzing Video...")
        progress_bar.progress(10)
        video_file = genai.upload_file(video_path)
        
        start_time = time.time()
        while video_file.state.name == "PROCESSING":
            if time.time() - start_time > 600: raise Exception("Timeout: Video upload took too long.")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED": raise Exception("Gemini Failed to process video.")

        # Step 2: Translate
        status_text.write("🧠 Step 2/5: Translating Dialogue...")
        progress_bar.progress(30)
        
        prompt = f"""
        Translate the spoken dialogue into Burmese (Myanmar).
        STYLE: {style}
        RULES:
        1. Keep sentences SHORT to match video timing.
        2. No timestamps. No scene descriptions.
        3. Only spoken words.
        """
        response = model.generate_content([video_file, prompt])
        burmese_text = response.text.strip()
        
        if not burmese_text:
            raise Exception("AI did not generate any translation text.")

        # Step 3: TTS
        status_text.write(f"🎙️ Step 3/5: Generating Audio ({tone})...")
        progress_bar.progress(50)
        
        voice = "my-MM-ThihaNeural" if gender == "Male" else "my-MM-NilarNeural"
        pitch, rate = "+0Hz", "+0%"
        if tone == "Deep": pitch = "-10Hz"
        elif tone == "Fast": rate = "+10%"
        elif tone == "Calm": pitch, rate = "-5Hz", "-5%"
        
        raw_audio = "temp_raw_audio.mp3"
        communicate = edge_tts.Communicate(burmese_text, voice, rate=rate, pitch=pitch)
        await communicate.save(raw_audio)

        # Step 4: Audio Processing
        status_text.write("✂️ Step 4/5: Cleaning Audio...")
        progress_bar.progress(70)
        clean_audio = "temp_clean_audio.mp3"
        
        # Remove Silence
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_audio,
            '-af', 'silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-30dB',
            clean_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 5: Merge
        status_text.write("🎬 Step 5/5: Rendering Final Video...")
        progress_bar.progress(90)
        
        final_audio = "temp_sync.mp3"
        output_video = os.path.abspath("final_output.mp4") # Use Absolute Path
        
        # Sync Logic
        vid_dur = get_duration(video_path)
        aud_dur = get_duration(clean_audio)
        
        if vid_dur > 0 and aud_dur > 0:
            speed = aud_dur / vid_dur
            speed = max(0.7, min(speed, 1.3)) # Limit speed change
            subprocess.run(['ffmpeg', '-y', '-i', clean_audio, '-filter:a', f"atempo={speed}", final_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            final_audio = clean_audio

        # Final Merge (Capture Errors)
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', final_audio,
            '-vf', 'scale=-2:720', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_video
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg Merge Failed:\n{result.stderr}")
            
        if not os.path.exists(output_video):
            raise Exception("Output file was not created by FFmpeg.")

        progress_bar.progress(100)
        return output_video, final_audio

    except Exception as e:
        raise e

# ... (Other generator functions remain same) ...
def generate_human_script(video_path, format_type, api_key, progress_bar, status_text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    status_text.write("Analyzing...")
    progress_bar.progress(30)
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
    status_text.write("Writing...")
    prompt = f"Write a {format_type} in Burmese based on this video."
    response = model.generate_content([video_file, prompt])
    progress_bar.progress(100)
    return response.text

def generate_viral_content(video_path, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    video_file = genai.upload_file(video_path)
    while video_file.state.name == "PROCESSING": time.sleep(1); video_file = genai.get_file(video_file.name)
    response = model.generate_content([video_file, "Generate 3 Viral Captions & 15 Hashtags in Burmese."])
    return response.text

def generate_thumbnail_idea(video_path, api_key):
    extract_frame(video_path, "thumb.jpg")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    img = Image.open("thumb.jpg")
    response = model.generate_content([img, "Describe a high-quality AI Art prompt for this video thumbnail."])
    return response.text, "thumb.jpg"

# ---------------------------------------------------------
# ⚙️ SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    if not st.session_state.api_key_storage:
        key_input = st.text_input("🔑 Enter Gemini API Key", type="password")
        if key_input:
            st.session_state.api_key_storage = key_input
            st.rerun()
    else:
        real_key = st.session_state.api_key_storage
        st.text_input("✅ API Key Active", value=f"{real_key[:6]} •••••••••", disabled=True)
        if st.button("🔄 Change Key"):
            st.session_state.api_key_storage = ""
            st.rerun()

    st.markdown("---")
    st.markdown("### 🚑 Troubleshooting")
    if st.button("⚠️ Force Unlock / Reset"):
        st.session_state.is_processing = False
        st.session_state.current_task = ""
        st.rerun()
    
    st.markdown("---")
    menu_selection = st.selectbox("Navigate Menu:", ["🎙️ Auto Dubbing", "🚀 Viral Kit", "🖼️ Thumbnail", "📝 Script Writer"], disabled=st.session_state.is_processing)

# ---------------------------------------------------------
# 🖥️ MAIN APP
# ---------------------------------------------------------
st.title("⚡ AI Video Studio Pro")

if not st.session_state.api_key_storage:
    st.warning("⚠️ Please enter API Key in Sidebar.")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Video (Max 200MB)", type=['mp4', 'mov'], disabled=st.session_state.is_processing)

if uploaded_file:
    if not st.session_state.is_processing:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

    # 🔒 PROCESSING LOCK
    if st.session_state.is_processing:
        st.markdown(f"""
            <div class="progress-box">
                <h3 class="status-text">⚙️ Processing: {st.session_state.current_task}</h3>
                <p>Please wait... (Do not close tab)</p>
            </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if st.session_state.current_task == "Dubbing":
                params = st.session_state.temp_params
                vid, aud = asyncio.run(process_dubbing(
                    "temp.mp4", params['gender'], params['style'], params['tone'], 
                    progress_bar, status_text, st.session_state.api_key_storage
                ))
                st.session_state.dub_result = (vid, aud)
            
            elif st.session_state.current_task == "Script":
                res = generate_human_script("temp.mp4", st.session_state.temp_params['format'], st.session_state.api_key_storage, progress_bar, status_text)
                st.session_state.script_result = res
                
            elif st.session_state.current_task == "ViralKit":
                status_text.write("Thinking...")
                progress_bar.progress(50)
                res = generate_viral_content("temp.mp4", st.session_state.api_key_storage)
                st.session_state.viral_result = res
                progress_bar.progress(100)
                
            elif st.session_state.current_task == "Thumbnail":
                status_text.write("Analyzing...")
                progress_bar.progress(50)
                txt, img = generate_thumbnail_idea("temp.mp4", st.session_state.api_key_storage)
                st.session_state.thumb_prompt = txt
                st.session_state.thumb_img = img
                progress_bar.progress(100)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            st.session_state.is_processing = False
            st.session_state.current_task = ""
            st.rerun()

    # 🔓 NORMAL SCREEN
    else:
        st.video("temp.mp4")

        # --- DUBBING ---
        if "Dubbing" in menu_selection:
            st.subheader("🎙️ Auto Dubbing")
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Voice", ["Male", "Female"])
                style = st.selectbox("Style", ["Narrator", "News Reporter", "Vlogger"])
            with c2:
                tone = st.selectbox("Tone", ["Natural", "Deep", "Fast", "Calm"])
            
            if st.button("🔴 Start Dubbing"):
                st.session_state.temp_params = {'gender': gender, 'style': style, 'tone': tone}
                st.session_state.current_task = "Dubbing"
                st.session_state.is_processing = True
                st.rerun()

            # 🔥 SAFE DOWNLOAD FIX
            if st.session_state.dub_result:
                vid_path = st.session_state.dub_result[0]
                aud_path = st.session_state.dub_result[1]
                
                # Check if file exists BEFORE showing button
                if os.path.exists(vid_path):
                    st.success("✅ Complete!")
                    c1, c2 = st.columns(2)
                    with c1: st.download_button("🎬 Video", open(vid_path, 'rb'), "dub.mp4", "video/mp4")
                    with c2: st.download_button("🎵 Audio", open(aud_path, 'rb'), "aud.mp3", "audio/mpeg")
                else:
                    st.error("❌ Output file missing. Try 'Force Unlock' and run again.")

        # ... (Other menus remain same) ...
        elif "Script" in menu_selection:
            st.subheader("📝 Script Writer")
            fmt = st.radio("Format", ["Word-for-Word Transcript", "Human Blog Post", "Video Script (Re-write)"])
            if st.button("✍️ Generate Text"):
                st.session_state.temp_params = {'format': fmt}
                st.session_state.current_task = "Script"
                st.session_state.is_processing = True
                st.rerun()
            if st.session_state.script_result:
                st.text_area("Result", st.session_state.script_result, height=400)
                st.download_button("📥 Download", st.session_state.script_result, "script.txt")

        elif "Viral" in menu_selection:
            if st.button("⚡ Generate Content"):
                st.session_state.current_task = "ViralKit"
                st.session_state.is_processing = True
                st.rerun()
            if st.session_state.viral_result: st.info(st.session_state.viral_result)
        
        elif "Thumbnail" in menu_selection:
            if st.button("🖼️ Generate Prompt"):
                st.session_state.current_task = "Thumbnail"
                st.session_state.is_processing = True
                st.rerun()
            if st.session_state.thumb_img:
                st.image(st.session_state.thumb_img, width=300)
                st.code(st.session_state.thumb_prompt)