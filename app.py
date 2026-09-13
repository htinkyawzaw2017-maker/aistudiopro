import streamlit as st
import os
import tempfile
import asyncio
from groq import Groq
import google.generativeai as genai
import edge_tts

st.set_page_config(
    page_title="Transcript Master Clone",
    page_icon="🎙️",
    layout="wide"
)

# Custom Styling (Transcript Master Dark/Modern Look)
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2563eb; color: white; font-weight: bold; }
    .badge { background-color: #1e293b; padding: 4px 8px; border-radius: 4px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions -----------------

def convert_seconds_to_srt_time(seconds: float) -> str:
    millisec = int((seconds % 1) * 1000)
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millisec:03d}"

def generate_srt(segments) -> str:
    srt_output = []
    for i, seg in enumerate(segments, 1):
        # seg သည် dictionary ဖြစ်နေနိုင်သလို object လည်း ဖြစ်နိုင်သောကြောင့် ၂ မျိုးလုံးအဆင်ပြေအောင် ရေးသားခြင်း
        start_sec = seg.get('start') if isinstance(seg, dict) else getattr(seg, 'start', 0)
        end_sec = seg.get('end') if isinstance(seg, dict) else getattr(seg, 'end', 0)
        text = seg.get('text') if isinstance(seg, dict) else getattr(seg, 'text', '')
        
        start = convert_seconds_to_srt_time(start_sec)
        end = convert_seconds_to_srt_time(end_sec)
        srt_output.append(f"{i}\n{start} --> {end}\n{text.strip()}\n")
    return "\n".join(srt_output)

async def generate_voiceover(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# ----------------- Sidebar Configurations -----------------
with st.sidebar:
    st.title("⚙️ Engine Settings")
    
    # API Keys can be configured in Render Environment Variables or entered manually
    groq_api_key = st.text_input("Groq API Key (Whisper)", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    gemini_api_key = st.text_input("Gemini API Key (Recap/AI)", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.markdown("**🎙️ AI Voice Settings (Edge-TTS)**")
    voice_options = {
        "English (Guy - Male)": "en-US-GuyNeural",
        "English (Jenny - Female)": "en-US-JennyNeural",
        "Burmese (Nilar - Female)": "my-MM-NilarNeural",
        "Burmese (Thiha - Male)": "my-MM-ThihaNeural"
    }
    selected_voice = st.selectbox("Select Voice", list(voice_options.keys()))
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Groq API သည် Whisper Transcription အတွက် အခမဲ့နှင့် အလွန်လျင်မြန်ပါသည်။")

# ----------------- Main Interface -----------------
st.title("🎬 Transcript Master Studio")
st.caption("AI-Powered Transcription, Video Recap, Subtitles (.SRT) & Voiceover Studio")

uploaded_file = st.file_uploader("Upload Video or Audio File (.mp3, .wav, .m4a, .mp4, .mov)", type=["mp3", "wav", "m4a", "mp4", "mov"])

if uploaded_file:
    # Save temporary file
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    col1, col2 = st.columns([1, 1])
    with col1:
        if suffix in [".mp4", ".mov"]:
            st.video(tmp_file_path)
        else:
            st.audio(tmp_file_path)

    if st.button("🚀 Transcribe & Analyze Media"):
        if not groq_api_key:
            st.error("ကျေးဇူးပြု၍ Sidebar တွင် Groq API Key အရင်ထည့်သွင်းပေးပါ။")
        else:
            with st.spinner("AI ဖြင့် စာသားနှင့် Subtitles များ ထုတ်ယူနေပါသည်..."):
                try:
                    groq_client = Groq(api_key=groq_api_key)
                    with open(tmp_file_path, "rb") as file_to_transcribe:
                        transcription_data = groq_client.audio.transcriptions.create(
                            file=(uploaded_file.name, file_to_transcribe.read()),
                            model="whisper-large-v3",
                            response_format="verbose_json"
                        )
                    
                    full_text = transcription_data.text
                    srt_content = generate_srt(transcription_data.segments)
                    
                    st.session_state["transcript"] = full_text
                    st.session_state["srt"] = srt_content
                    st.success("Transcription အောင်မြင်စွာ ပြီးဆုံးပါပြီ!")
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")

# Display Tabs if transcript exists
if "transcript" in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Full Transcript", "⏱️ SRT Subtitles", "🧠 AI Recap & Hooks", "🔊 AI Voiceover"])

    # Tab 1: Full Transcript
    with tab1:
        st.subheader("Extracted Script")
        st.text_area("Full Transcript", st.session_state["transcript"], height=250)
        st.download_button("📥 Download Transcript (.txt)", st.session_state["transcript"], file_name="transcript.txt")

    # Tab 2: Subtitles (SRT)
    with tab2:
        st.subheader("Subtitles with Timestamps (.srt)")
        st.text_area("SRT Output", st.session_state["srt"], height=250)
        st.download_button("📥 Download .SRT File", st.session_state["srt"], file_name="subtitles.srt")

    # Tab 3: AI Video Recap & Hooks
    with tab3:
        st.subheader("AI Content Multiplier (Gemini)")
        if st.button("✨ Generate Recap, Timestamps & Social Hooks"):
            if not gemini_api_key:
                st.error("ကျေးဇူးပြု၍ Sidebar တွင် Gemini API Key ထည့်ပေးပါ။")
            else:
                with st.spinner("အနှစ်ချုပ်နှင့် Hook စာသားများ ရေးသားနေပါသည်..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt = f"""
                        Analyze this transcript and generate in Burmese (or matching language):
                        1. **Short Executive Summary** (2-3 sentences)
                        2. **Key Takeaways** (Bullet points)
                        3. **Viral Social Media Hooks** (For TikTok/Facebook Reels)
                        4. **YouTube Description & Timestamps**

                        Transcript:
                        {st.session_state['transcript']}
                        """
                        response = model.generate_content(prompt)
                        st.session_state["ai_recap"] = response.text
                    except Exception as e:
                        st.error(f"AI Generation Error: {str(e)}")

        if "ai_recap" in st.session_state:
            st.markdown(st.session_state["ai_recap"])
            st.download_button("📥 Download Recap (.txt)", st.session_state["ai_recap"], file_name="recap.txt")

    # Tab 4: AI Voiceover (TTS)
    with tab4:
        st.subheader("AI Text-to-Speech Generation")
        tts_input = st.text_area("Voiceover လုပ်မည့် စာသားရိုက်ပါ/ထည့်ပါ", value=st.session_state.get("transcript", "")[:500], height=150)
        
        if st.button("🎙️ Generate Voiceover Audio"):
            if tts_input.strip():
                with st.spinner("AI အသံဖိုင် ဖန်တီးနေပါသည်..."):
                    try:
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        temp_audio_path = temp_audio.name
                        temp_audio.close()

                        asyncio.run(generate_voiceover(tts_input, voice_options[selected_voice], temp_audio_path))
                        
                        st.audio(temp_audio_path, format="audio/mp3")
                        with open(temp_audio_path, "rb") as f:
                            st.download_button("📥 Download Audio (.mp3)", f.read(), file_name="voiceover.mp3", mime="audio/mp3")
                    except Exception as e:
                        st.error(f"TTS Error: {str(e)}")
            else:
                st.warning("စာသား အနည်းငယ် ထည့်သွင်းပေးပါ။")
