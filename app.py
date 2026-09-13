import streamlit as st
import os
import tempfile
import asyncio
from groq import Groq
import google.generativeai as genai
import edge_tts

st.set_page_config(
    page_title="English to Myanmar AI Recap Studio",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2563eb; color: white; font-weight: bold; }
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

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("⚙️ Engine API Keys")
    groq_api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    gemini_api_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.markdown("**🎙️ မြန်မာ AI အသံထွက် (Voiceover)**")
    voice_options = {
        "မြန်မာမလေး အသံ (Nilar)": "my-MM-NilarNeural",
        "မြန်မာလူငယ် အသံ (Thiha)": "my-MM-ThihaNeural",
        "English (Jenny)": "en-US-JennyNeural"
    }
    selected_voice = st.selectbox("Select Voice", list(voice_options.keys()))

# ----------------- Main UI -----------------
st.title("🎬 English Video to Myanmar Recap Studio")
st.caption("အင်္ဂလိပ်ဗီဒီယို/အသံဖိုင် တင်လိုက်ရုံဖြင့် မြန်မာလို အပြည့်အစုံ Transcript လုပ်ပြီး Recap ထုတ်ပေးသည့်စနစ်")

uploaded_file = st.file_uploader("Upload English Video or Audio (.mp3, .wav, .m4a, .mp4, .mov)", type=["mp3", "wav", "m4a", "mp4", "mov"])

if uploaded_file:
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

    if st.button("🚀 Process: English အသံမှ မြန်မာ Transcript နှင့် Recap ချက်ချင်းထုတ်ယူမည်"):
        if not groq_api_key or not gemini_api_key:
            st.error("ကျေးဇူးပြု၍ Sidebar တွင် Groq API Key နှင့် Gemini API Key နှစ်ခုလုံး ထည့်သွင်းပေးပါ။")
        else:
            # အဆင့် ၁ - Whisper ဖြင့် မူရင်းအသံဖိုင်ကို အတိအကျ transcribe လုပ်ခြင်း
            with st.spinner("အဆင့် (၁/၂): အင်္ဂလိပ်အသံကို ဖတ်ယူနေပါသည်..."):
                try:
                    groq_client = Groq(api_key=groq_api_key)
                    with open(tmp_file_path, "rb") as file_to_transcribe:
                        transcription_data = groq_client.audio.transcriptions.create(
                            file=(uploaded_file.name, file_to_transcribe.read()),
                            model="whisper-large-v3",
                            response_format="verbose_json"
                        )
                    
                    if isinstance(transcription_data, dict):
                        en_transcript = transcription_data.get("text", "")
                        segments = transcription_data.get("segments", [])
                    else:
                        en_transcript = getattr(transcription_data, "text", "")
                        segments = getattr(transcription_data, "segments", [])
                        
                    st.session_state["en_transcript"] = en_transcript
                    st.session_state["en_srt"] = generate_srt(segments)
                except Exception as e:
                    st.error(f"Groq Transcription Error: {str(e)}")
                    st.stop()

            # အဆင့် ၂ - Gemini ဖြင့် မြန်မာလို Transcript တိုက်ရိုက်ပြန်ဆိုခြင်းနှင့် Recap ထုတ်ခြင်း
            with st.spinner("အဆင့် (၂/၂): မြန်မာဘာသာသို့ ပြန်ဆိုပြီး Video Recap ထုတ်ပေးနေပါသည်..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel("gemini-pro")
                    
                    prompt = f"""
                    You are an expert video content creator and bilingual translator.
                    Below is the English transcript of a video. 
                    Please perform two major tasks in clear, engaging, natural Myanmar language (Burmese):

                    ---
                    ### Task 1: FULL MYANMAR TRANSCRIPT (မြန်မာလို စာသားအပြည့်အစုံ)
                    Translate the entire English speech into natural, fluent Burmese spoken/narrative style (Do not use stiff robotic machine translation).

                    ---
                    ### Task 2: VIDEO RECAP & VIRAL HOOKS (အနှစ်ချုပ်နှင့် အဓိကအချက်များ)
                    1. 📌 **ဗီဒီယို ဇာတ်လမ်း/အကြောင်းအရာ အကျဉ်းချုပ် (Short Executive Summary)**: အဓိကဆိုလိုရင်းကို ၃ ကြောင်းဖြင့် ရှင်းပြပါ။
                    2. 💡 **အဓိကသင်ခန်းစာ/မှတ်သားဖွယ်အချက်များ (Key Takeaways)**: Bullet points ဖြင့် ဖော်ပြပါ။
                    3. 🔥 **Viral Social Media Hooks (Facebook / TikTok အတွက်)**: လူစိတ်ဝင်စားအောင် စတင်မည့် ဆွဲဆောင်မှုရှိသော စာကြောင်း ၃ ကြောင်း။
                    4. 🎬 **Recap Voiceover Script (အသံသွင်းရန် အနှစ်ချုပ်စကားပြေ)**: ၁ မိနစ်စာ အနှစ်ချုပ် ပြောပြရန်အတွက် စကားပြောဟန် စာသား။

                    English Transcript:
                    {en_transcript}
                    """
                    response = model.generate_content(prompt)
                    st.session_state["myanmar_content"] = response.text
                    st.success("မြန်မာ Transcript နှင့် Recap အောင်မြင်စွာ ဖန်တီးပြီးပါပြီ!")
                except Exception as e:
                    st.error(f"Gemini Recap Error: {str(e)}")

# ရလဒ်များ ပြသခြင်း
if "myanmar_content" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["🇲🇲 မြန်မာ Recap & Transcript", "🔊 မြန်မာ AI အသံထွက်သွင်းမည်", "📄 မူရင်း English Transcript"])

    with tab1:
        st.subheader("🇲🇲 မြန်မာလို အနှစ်ချုပ် (Recap) နှင့် Transcript အပြည့်အစုံ")
        st.markdown(st.session_state["myanmar_content"])
        st.download_button(
            label="📥 Download Myanmar Recap & Transcript (.txt)",
            data=st.session_state["myanmar_content"],
            file_name="myanmar_recap_transcript.txt"
        )

    with tab2:
        st.subheader("🔊 မြန်မာ Voiceover အသံထုတ်ယူခြင်း")
        st.caption("အပေါ်မှ Recap သို့မဟုတ် နှစ်သက်ရာ မြန်မာစာသားကို အသံဖိုင် (.mp3) အဖြစ် အလိုအလျောက် ထုတ်ပေးမည်။")
        tts_text = st.text_area("အသံသွင်းမည့် မြန်မာစာသား", height=150, placeholder="အသံထုတ်လိုသော မြန်မာစာသားများကို ဤနေရာတွင် ထည့်ပါ...")
        
        if st.button("🎙️ မြန်မာ အသံဖိုင် ထုတ်ယူမည်"):
            if tts_text.strip():
                with st.spinner("AI မြန်မာအသံ ထုတ်လုပ်နေပါသည်..."):
                    try:
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        temp_audio_path = temp_audio.name
                        temp_audio.close()

                        asyncio.run(generate_voiceover(tts_text, voice_options[selected_voice], temp_audio_path))
                        
                        st.audio(temp_audio_path, format="audio/mp3")
                        with open(temp_audio_path, "rb") as f:
                            st.download_button("📥 Download Audio (.mp3)", f.read(), file_name="myanmar_voiceover.mp3", mime="audio/mp3")
                    except Exception as e:
                        st.error(f"TTS Error: {str(e)}")
            else:
                st.warning("ကျေးဇူးပြု၍ အသံထုတ်ရန် စာသား အနည်းငယ် ထည့်သွင်းပေးပါ။")

    # Tab 3: AI Video Recap & Hooks
    with tab3:
        st.subheader("AI Content Multiplier (Groq Llama-3)")
        if st.button("✨ Generate Recap, Timestamps & Social Hooks"):
            if not groq_api_key:
                st.error("ကျေးဇူးပြု၍ Sidebar တွင် Groq API Key ထည့်ပေးပါ။")
            else:
                with st.spinner("အနှစ်ချုပ်နှင့် Hook စာသားများ ရေးသားနေပါသည်..."):
                    try:
                        groq_client = Groq(api_key=groq_api_key)
                        prompt = f"""
                        Analyze this transcript and generate in Burmese:
                        1. **Short Executive Summary** (2-3 sentences)
                        2. **Key Takeaways** (Bullet points)
                        3. **Viral Social Media Hooks** (For TikTok/Facebook Reels)
                        4. **YouTube Description & Timestamps**

                        Transcript:
                        {st.session_state['transcript']}
                        """
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a professional social media and video content creator."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama-3.3-70b-versatile",
                        )
                        
                        st.session_state["ai_recap"] = chat_completion.choices[0].message.content
                    except Exception as e:
                        st.error(f"AI Generation Error: {str(e)}")

        if "ai_recap" in st.session_state:
            st.markdown(st.session_state["ai_recap"])
            st.download_button("📥 Download Recap (.txt)", st.session_state["ai_recap"], file_name="recap.txt")
