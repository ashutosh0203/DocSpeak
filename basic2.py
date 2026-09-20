import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
import threading
import os
from datetime import datetime
import pyttsx3
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# --- HF Model Config ---
HF_TOKEN = os.getenv("HF_TOKEN")  # replace with your actual token
QA_MODEL = "deepset/roberta-base-squad2"  # or tinyroberta-squad2
client = InferenceClient(model=QA_MODEL, token=HF_TOKEN)

# --- Audio Config ---
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def generate_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

# --- Recorder Class ---
class SpeechRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        self.stream = None
        self.filename = None

    def start_recording(self):
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=FRAMES_PER_BUFFER)
        self.frames = []
        self.recording = True
        self.filename = generate_filename()
        while self.recording:
            data = self.stream.read(FRAMES_PER_BUFFER)
            self.frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        return self.filename

# --- Speak Output ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# --- Transcribe Audio ---
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError:
        return "❌ API error."

# # --- QA model with context ---
# def query_llm(question, context):
#     try:
#         result = client.question_answering(question=question, context=context)
#         return result.get("answer", "❌ No answer found.")
#     except Exception as e:
#         return f"❌ QA Error: {e}"

from huggingface_hub import InferenceClient
MODEL = "deepset/roberta-base-squad2"

context = """
    India, a land of vibrant contrasts and rich heritage, is a nation renowned for its diverse cultures, ancient traditions, and captivating history. From the towering Himalayas in the north to the serene backwaters of Kerala in the south, India's geography is as diverse as its people. The country is home to multiple languages, religions, and festivals, all coexisting in a unique blend of unity and diversity. This harmonious coexistence of different traditions and modernity is a hallmark of India, drawing visitors from across the globe. 
India's history stretches back thousands of years, with its civilization emerging as one of the world's oldest. From the Indus Valley civilization to the Mughal Empire and British rule, India has witnessed the rise and fall of many empires, each leaving its indelible mark on the nation's cultural landscape. This rich historical tapestry is reflected in the numerous forts, temples, and monuments that dot the country, each narrating tales of bygone eras. 
Culturally, India is a kaleidoscope of traditions, customs, and artistic expressions. From classical dance forms like Bharatanatyam and Kathak to vibrant festivals like Diwali, Holi, and Eid, India's cultural diversity is a spectacle to behold. The concept of "Atithi Devo Bhava," which translates to "the guest is equivalent to God," is deeply ingrained in Indian culture, reflecting the warmth and hospitality extended to visitors. 
India's contribution to the world extends beyond its cultural heritage. It has been a cradle of spirituality and philosophy, with religions like Hinduism, Buddhism, Jainism, and Sikhism originating here. The country has also made significant contributions to science, mathematics, and literature, with inventions like the concept of zero and the decimal system. In modern times, India has emerged as a global leader in information technology, with its thriving software industry and innovative startups. 
India's unity in diversity is perhaps its most defining characteristic. Despite the multitude of languages, religions, and customs, Indians share a common identity as citizens of one nation. This spirit of unity is celebrated through various festivals and cultural events, fostering a sense of belonging and national pride. 
Looking ahead, India is poised to play a significant role on the global stage. With a young and dynamic population, a thriving economy, and a rich cultural heritage, India has the potential to become a global superpower in the 21st century. However, challenges like poverty, inequality, and environmental sustainability need to be addressed to ensure that India's growth benefits all its citizens. 
In conclusion, India is a land of captivating beauty, rich history, and vibrant culture. Its unity in diversity, its ancient traditions, and its modern aspirations make it a unique and fascinating nation, one that continues to inspire and captivate the world. 
    """
def query_llm(question: str, context: str) -> str:
    try:
        result = client.question_answering(question=question, context=context)
        return result.get("answer", "❌ No answer found.")
    except Exception as e:
        return f"❌ Error: {e}"
# --- Streamlit UI ---
st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")
st.title("🎙️ Voice-Controlled Chatbot with Context")
st.caption("Speak a question and get an answer based on previous conversation.")

# --- Session State ---
if "thread" not in st.session_state:
    st.session_state.thread = None
if "active_recorder" not in st.session_state:
    st.session_state.active_recorder = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_qa" not in st.session_state:
    st.session_state.latest_qa = None

# --- Chat History ---
st.subheader("💬 Conversation History")
if st.session_state.chat_history:
    for i, (q, r, path) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**Response:** {r}")
        st.audio(path, format="audio/wav")
        st.markdown("---")
else:
    st.info("No previous conversations yet.")

# --- Recording Controls ---
st.subheader("🎤 Record & Ask")
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording"):
        st.session_state.active_recorder = SpeechRecorder()
        st.session_state.thread = threading.Thread(
            target=st.session_state.active_recorder.start_recording)
        st.session_state.thread.start()
        st.info("🎙️ Recording... Press Stop when done.")

with col2:
    if st.button("⏹️ Stop & Ask"):
        if st.session_state.active_recorder and st.session_state.active_recorder.recording:
            audio_path = st.session_state.active_recorder.stop_recording()
            st.success("✅ Recording saved.")
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(audio_path)
                st.info("📝 Transcription: " + transcription)

            if transcription and "❌" not in transcription:
                # ✅ Build context from chat history
                with st.spinner("Generating response..."):
                    response = query_llm(transcription, context)
                    st.success(response)
                    st.session_state.chat_history.append((transcription, response, audio_path))
                    st.session_state.latest_qa = (transcription, response, audio_path)
                    speak(response)
                st.rerun()
            else:
                st.session_state.latest_qa = (transcription, None, audio_path)
                st.error("Speech not clear or not recognized.")
        else:
            st.warning("❗ Start recording first.")
