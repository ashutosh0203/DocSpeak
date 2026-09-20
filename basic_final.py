import os
import atexit
import glob
import platform
import queue
import re
import shutil
import threading
import uuid
import wave
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
import pyaudio
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# ============================================================
# Environment
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
).strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# Directories
# ============================================================

RECORDINGS_DIR = "recordings"
DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"
VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")

for directory in (RECORDINGS_DIR, DATA_DIR, VECTOR_DB_DIR):
    os.makedirs(directory, exist_ok=True)


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="DocSpeak",
    page_icon="🎤",
    layout="wide"
)

st.title("🎙️ DocSpeak: AI Voice Assistant with Document Q&A")
st.caption(
    "Upload a document, ask questions using text or voice, and receive "
    "answers grounded in the document."
)


# ============================================================
# Audio configuration
# ============================================================

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def generate_filename() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(
        RECORDINGS_DIR,
        f"recording_{timestamp}.wav"
    )


# ============================================================
# Speech Recorder
# ============================================================

class SpeechRecorder:
    def __init__(self):
        self.p = None
        self.frames = []
        self.recording = False
        self.stream = None
        self.filename = None

    def start_recording(self):
        try:
            self.p = pyaudio.PyAudio()

            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER
            )

            self.frames = []
            self.recording = True
            self.filename = generate_filename()

            print(f"Recording started: {self.filename}")

            while self.recording:
                try:
                    data = self.stream.read(
                        FRAMES_PER_BUFFER,
                        exception_on_overflow=False
                    )
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break

        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.recording = False

    def stop_recording(self) -> Optional[str]:
        self.recording = False

        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            if self.p:
                self.p.terminate()
                self.p = None

            if not self.frames or not self.filename:
                print("No audio frames to save.")
                return None

            audio = pyaudio.PyAudio()
            sample_width = audio.get_sample_size(FORMAT)
            audio.terminate()

            with wave.open(self.filename, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(sample_width)
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.frames))

            print(f"Recording saved: {self.filename}")
            return self.filename

        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None


# ============================================================
# Speech to Text
# ============================================================

def transcribe_audio(filename: str) -> str:
    if not filename or not os.path.exists(filename):
        return "❌ Audio file not found."

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)

        return recognizer.recognize_google(audio)

    except sr.UnknownValueError:
        return "❌ Could not understand the audio."

    except sr.RequestError as e:
        return f"❌ Speech recognition API error: {e}"

    except Exception as e:
        return f"❌ Transcription error: {e}"


# ============================================================
# Text to Speech
# ============================================================

class ThreadSafeTTS:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()

    def _init_engine(self) -> bool:
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.engine.setProperty("volume", 1.0)
            return True
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            return False

    def _worker(self):
        if not self._init_engine():
            print("Failed to initialize TTS engine.")
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)

                if text is None:
                    break

                if self.engine and text.strip():
                    try:
                        self.engine.stop()
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"TTS playback error: {e}")

                self.speech_queue.task_done()

            except queue.Empty:
                continue

            except Exception as e:
                print(f"TTS worker error: {e}")

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        try:
            with self.lock:
                if not self.running:
                    self.running = True
                    self.worker_thread = threading.Thread(
                        target=self._worker,
                        daemon=True
                    )
                    self.worker_thread.start()

            self.speech_queue.put(text)
            return True

        except Exception as e:
            print(f"TTS queue error: {e}")
            return False

    def stop(self):
        with self.lock:
            if not self.running:
                return

            self.running = False
            self.speech_queue.put(None)

            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2)


_tts_instance = None


def get_tts_instance() -> ThreadSafeTTS:
    global _tts_instance

    if _tts_instance is None:
        _tts_instance = ThreadSafeTTS()

    return _tts_instance


def speak_system(text: str) -> bool:
    try:
        system = platform.system().lower()

        if system == "windows":
            safe_text = (
                text
                .replace("&", " ")
                .replace("'", " ")
                .replace('"', " ")
            )

            command = (
                'powershell -Command '
                '"Add-Type -AssemblyName System.Speech; '
                '$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                '$synth.Rate = 0; '
                '$synth.Volume = 100; '
                f'$synth.Speak(\'{safe_text}\')"'
            )

            return os.system(command) == 0

        if system == "darwin":
            safe_text = text.replace("'", " ")
            return os.system(f"say '{safe_text}'") == 0

        if system == "linux":
            if shutil.which("espeak"):
                safe_text = text.replace("'", " ")
                return os.system(f"espeak '{safe_text}'") == 0

        return False

    except Exception as e:
        print(f"System TTS error: {e}")
        return False


def speak_with_fallback(text: str) -> bool:
    if not text or not text.strip():
        return False

    try:
        if get_tts_instance().speak(text):
            return True
    except Exception as e:
        print(f"Primary TTS error: {e}")

    return speak_system(text)


def stop_speaking():
    global _tts_instance

    try:
        if _tts_instance:
            _tts_instance.stop()
            _tts_instance = None
    except Exception as e:
        print(f"Error stopping TTS: {e}")


# ============================================================
# Document Processing
# ============================================================

def process_document_with_tables(file_path: str):
    """Extract document content and preserve table-like content."""
    documents = []

    try:
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()

        for doc in docs:
            content = doc.page_content or ""
            metadata = doc.metadata or {}

            if (
                "table" in content.lower()
                or "|" in content
                or "\t" in content
            ):
                processed_lines = []

                for line in content.split("\n"):
                    if "|" in line or "\t" in line:
                        cleaned_line = " ".join(line.split())
                        processed_lines.append(
                            f"TABLE ROW: {cleaned_line}"
                        )
                    else:
                        processed_lines.append(line)

                content = "\n".join(processed_lines)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "contains_tables": (
                            "|" in content
                            or "TABLE ROW:" in content
                        ),
                        "file_path": file_path,
                        "processed_time": datetime.now().isoformat()
                    }
                )
            )

    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        return []

    return documents


# ============================================================
# Embeddings and FAISS
# ============================================================

@st.cache_resource(show_spinner=False)
def create_embeddings():
    device = EMBEDDING_DEVICE

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                st.warning(
                    "CUDA was requested but is unavailable. "
                    "Using CPU for embeddings."
                )
                device = "cpu"

        except ImportError:
            st.warning(
                "PyTorch CUDA support was not detected. "
                "Using CPU for embeddings."
            )
            device = "cpu"

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )


def initialize_or_load_vectorstore(embeddings):
    if not os.path.exists(VECTOR_DB_PATH):
        return None, False

    if not os.listdir(VECTOR_DB_PATH):
        return None, False

    try:
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        sample_vector = embeddings.embed_query("dimension test")

        if len(sample_vector) != vectorstore.index.d:
            print(
                "Embedding dimension mismatch. Rebuilding vector database."
            )
            shutil.rmtree(VECTOR_DB_PATH)
            return None, False

        return vectorstore, True

    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None, False


def add_documents_to_vectorstore(
    documents,
    embeddings,
    existing_vectorstore=None
):
    if not documents:
        raise ValueError("No document chunks were provided.")

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    if existing_vectorstore is not None:
        try:
            existing_vectorstore.add_documents(documents)
            existing_vectorstore.save_local(VECTOR_DB_PATH)
            return existing_vectorstore

        except Exception as e:
            print(f"Error adding documents to existing index: {e}")

            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore


# ============================================================
# Relevance Checking
# ============================================================

def is_context_relevant(
    query: str,
    context: str,
    min_similarity: float = 0.2
) -> bool:
    if not context or not context.strip():
        return False

    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    context_words = set(re.findall(r"\b\w+\b", context.lower()))

    if not query_words:
        return False

    table_keywords = {
        "table",
        "row",
        "column",
        "data",
        "value",
        "list",
        "compare",
        "comparison",
        "versus",
        "vs"
    }

    if any(keyword in query.lower() for keyword in table_keywords):
        return True

    overlap = len(query_words.intersection(context_words))
    similarity = overlap / len(query_words)

    return similarity >= min_similarity or len(context.strip()) > 30


# ============================================================
# OpenRouter
# ============================================================

def call_openrouter(
    prompt: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.2
) -> str:
    if not OPENROUTER_API_KEY:
        return (
            "❌ OPENROUTER_API_KEY is missing. "
            "Add it to your .env file and restart Streamlit."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AskAloud"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error",
                    {}
                ).get(
                    "message",
                    response.text
                )
            except Exception:
                error_message = response.text

            return (
                f"❌ OpenRouter API error "
                f"({response.status_code}): {error_message}"
            )

        result = response.json()

        choices = result.get("choices", [])

        if not choices:
            return "❌ OpenRouter returned no answer."

        message = choices[0].get("message", {})
        answer = message.get("content", "")

        if not answer:
            return "❌ OpenRouter returned an empty answer."

        return answer.strip()

    except requests.Timeout:
        return "❌ OpenRouter request timed out. Please try again."

    except requests.RequestException as e:
        return f"❌ Network error while contacting OpenRouter: {e}"

    except Exception as e:
        return f"❌ OpenRouter error: {e}"


# ============================================================
# Session State
# ============================================================

defaults = {
    "chat_history": [],
    "retriever": None,
    "vectorstore": None,
    "embeddings": None,
    "documents_loaded": [],
    "current_answer": None,
    "speech_enabled": True,
    "active_recorder": None,
    "thread": None,
    "initialized": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Initialize Embeddings and Existing FAISS Index
# ============================================================

if not st.session_state.initialized:
    try:
        st.session_state.embeddings = create_embeddings()

        vectorstore, loaded = initialize_or_load_vectorstore(
            st.session_state.embeddings
        )

        if loaded:
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )

        st.session_state.initialized = True

    except Exception as e:
        st.error(f"❌ Initialization error: {e}")


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("🔊 Speech Settings")

st.session_state.speech_enabled = st.sidebar.toggle(
    "Enable Speech Response",
    value=st.session_state.speech_enabled,
    help="Enable or disable spoken answers."
)

if st.session_state.speech_enabled:
    st.sidebar.success("🔊 Speech ON")
else:
    st.sidebar.info("🔇 Speech OFF")


st.sidebar.header("🗂️ Session Management")

if st.sidebar.button(
    "🗑️ Clear All Data",
    help="Delete uploaded documents, recordings, and the FAISS index."
):
    stop_speaking()

    for folder in (RECORDINGS_DIR, DATA_DIR):
        for file in glob.glob(os.path.join(folder, "*")):
            try:
                if os.path.isfile(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    st.session_state.documents_loaded = []
    st.session_state.current_answer = None

    st.sidebar.success("🗑️ All data cleared.")
    st.rerun()


if st.session_state.documents_loaded:
    st.sidebar.subheader("📑 Loaded Documents")

    for i, doc_name in enumerate(
        st.session_state.documents_loaded,
        start=1
    ):
        st.sidebar.text(f"{i}. {doc_name}")


# st.sidebar.divider()

# st.sidebar.subheader("🤖 AI Configuration")
# st.sidebar.caption(f"Model: {OPENROUTER_MODEL}")
# st.sidebar.caption(f"Embeddings: {EMBEDDING_MODEL}")
# st.sidebar.caption(f"Device: {EMBEDDING_DEVICE}")


# # ============================================================
# # API Status
# # ============================================================

# if OPENROUTER_API_KEY:
#     st.sidebar.success("OpenRouter API key loaded")
# else:
#     st.sidebar.error("OpenRouter API key missing")


# ============================================================
# Document Upload
# ============================================================

st.header("📂 Add Documents to Knowledge Base")

mode = st.radio(
    "Choose input type:",
    ["Upload File", "Enter Link"],
    horizontal=True
)


def process_and_add_document(
    file_path: str,
    doc_name: str
) -> bool:

    if not OPENROUTER_API_KEY:
        st.error(
            "❌ OPENROUTER_API_KEY not found. "
            "Add it to .env and restart Streamlit."
        )
        return False

    try:
        with st.spinner(f"Processing {doc_name}..."):

            documents = process_document_with_tables(file_path)

            if not documents:
                st.error("❌ No content extracted from the document.")
                return False

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )

            chunks = []

            for doc in documents:
                split_texts = splitter.split_text(
                    doc.page_content
                )

                for chunk_index, text in enumerate(split_texts):
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                **doc.metadata,
                                "chunk_id": str(uuid.uuid4()),
                                "chunk_index": chunk_index,
                                "source_file": doc_name
                            }
                        )
                    )

            if not chunks:
                st.error("❌ No text chunks were created.")
                return False

            # Add to vectorstore
            st.session_state.vectorstore = (
                add_documents_to_vectorstore(
                    chunks,
                    st.session_state.embeddings,
                    st.session_state.vectorstore
                )
            )

            #Add to retriever
            st.session_state.retriever = (
                st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
            )
            # Track loaded documents
            if doc_name not in st.session_state.documents_loaded:
                st.session_state.documents_loaded.append(doc_name)

            st.success(
                f"✅ {doc_name} added to the knowledge base."
            )

            return True

    except Exception as e:
        st.error(
            f"❌ Error processing {doc_name}: {e}"
        )
        return False


if mode == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, PPT, PPTX, HTML)",
        type=["pdf", "docx", "ppt", "pptx", "html", "htm"]
    )

    if uploaded_file:
        file_path = os.path.join(
            DATA_DIR,
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button(
            f" Add '{uploaded_file.name}' to Knowledge Base"
        ):
            process_and_add_document(
                file_path,
                uploaded_file.name
            )

else:

    link = st.text_input(
        "Enter document link (http/https)"
    )

    if link and st.button(" Add Link to Knowledge Base"):

        if not link.startswith(("http://", "https://")):
            st.error("❌ Please enter a valid http/https URL.")

        else:
            try:
                safe_name = re.sub(
                    r"[^a-zA-Z0-9_.-]",
                    "_",
                    link
                )

                if "." not in os.path.basename(link):
                    safe_name += ".html"

                file_path = os.path.join(
                    DATA_DIR,
                    safe_name
                )

                with st.spinner("Downloading document..."):
                    response = requests.get(
                        link,
                        timeout=20,
                        headers={
                            "User-Agent": "AskAloud/1.0"
                        }
                    )
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        f.write(response.content)

                process_and_add_document(
                    file_path,
                    safe_name
                )

            except Exception as e:
                st.error(
                    f"❌ Failed to download document: {e}"
                )


# ============================================================
# Conversation Context
# ============================================================

def build_chat_history_context(max_turns: int = 5) -> str:
    if not st.session_state.chat_history:
        return ""

    recent_history = st.session_state.chat_history[-max_turns:]

    lines = []

    for question, answer, _ in recent_history:
        lines.append(
            f"User: {question}\nAssistant: {answer}"
        )

    return "\n\n".join(lines)


# ============================================================
# Document Question Answering
# ============================================================

def ask_document(query: str) -> str:

    query = query.strip()

    if not query:
        return "❌ Please enter a question."

    retriever = st.session_state.retriever

    if not retriever:
        return (
            "❌ Please add at least one document "
            "to the knowledge base first."
        )

    try:
        docs = retriever.invoke(query)

        if not docs:
            return (
                "❌ I couldn't find relevant information "
                "in the loaded documents."
            )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
            if doc.page_content
        )

        if not context.strip():
            return "❌ No usable content was retrieved."

        if not is_context_relevant(
            query,
            context,
            min_similarity=0.2
        ):
            return (
                "❌ I couldn't find relevant information in "
                "the loaded documents to answer your question."
            )

        history = build_chat_history_context()

        prompt = f"""
You are AskAloud, a document question-answering assistant.

Answer the user's question using only the information contained
in the retrieved document context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain the answer, say so clearly.
4. Keep the answer concise and conversational.
5. For tables, preserve the important values and relationships.
6. For comparisons, present the differences clearly.
7. Use Markdown when a list or table improves readability.
8. Use previous conversation only to understand references such as
   "it", "they", "the previous result", or follow-up questions.
9. If the answer is not supported by the document context,
   explicitly say that the document does not provide enough information.

PREVIOUS CONVERSATION:
{history if history else "No previous conversation."}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        answer = call_openrouter(
            prompt=prompt,
            model=OPENROUTER_MODEL,
            max_tokens=500,
            temperature=0.2
        )

        return answer

    except Exception as e:
        return f"❌ Error processing your question: {e}"


# ============================================================
# Conversation History
# ============================================================

st.subheader("📝 Conversation History")

if st.session_state.chat_history:

    for i, (question, answer, audio_path) in enumerate(
        st.session_state.chat_history,
        start=1
    ):

        with st.expander(
            f"Q{i}: {question[:70]}...",
            expanded=True
        ):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")

            if audio_path and os.path.exists(audio_path):
                st.audio(
                    audio_path,
                    format="audio/wav"
                )

    st.markdown("---")

else:
    st.info(
        "No conversations yet. "
        "Add a document and start asking questions."
    )


# ============================================================
# Statistics
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "📚 Documents Loaded",
        len(st.session_state.documents_loaded)
    )

with col2:
    st.metric(
        "💬 Questions Asked",
        len(st.session_state.chat_history)
    )


# ============================================================
# Text Question
# ============================================================

st.subheader("✏️ Ask Your Question")

text_query = st.text_input(
    "Ask about tables, data, or any document content:"
)

if text_query and st.button("🤖 Ask Question"):

    if not st.session_state.retriever:
        st.warning(
            "Please add at least one document "
            "to the knowledge base first."
        )

    else:

        with st.spinner(
            "🤖 Analyzing documents and generating answer..."
        ):
            response = ask_document(text_query)

        st.session_state.current_answer = response

        st.session_state.chat_history.append(
            (
                text_query,
                response,
                None
            )
        )

        st.markdown("### Answer")
        st.markdown(response)

        if (
            st.session_state.speech_enabled
            and "❌" not in response
        ):
            with st.spinner("🔊 Speaking response..."):
                try:
                    speak_with_fallback(response)
                except Exception as e:
                    st.warning(
                        f"Speech error: {e}"
                    )


# ============================================================
# Voice Input
# ============================================================

st.subheader("🎤 Voice Input")

col1, col2 = st.columns(2)

with col1:

    if st.button("▶️ Start Recording"):

        try:
            stop_speaking()

            st.session_state.active_recorder = (
                SpeechRecorder()
            )

            st.session_state.thread = threading.Thread(
                target=st.session_state.active_recorder.start_recording,
                daemon=True
            )

            st.session_state.thread.start()

            st.info(
                "🎙️ Recording... Press Stop when finished."
            )

        except Exception as e:
            st.error(
                f"❌ Failed to start recording: {e}"
            )


with col2:

    if st.button("⏹️ Stop & Ask"):

        recorder = st.session_state.get(
            "active_recorder"
        )

        if not recorder or not recorder.recording:
            st.warning(
                "❗ Start recording first."
            )

        else:

            try:
                audio_path = recorder.stop_recording()

                if not audio_path:
                    st.error(
                        "❌ Failed to save recording."
                    )

                else:

                    st.success(
                        "✅ Recording saved."
                    )

                    with st.spinner(
                        "🔎 Transcribing..."
                    ):
                        transcription = transcribe_audio(
                            audio_path
                        )

                    st.info(
                        f"📝 Transcription: {transcription}"
                    )

                    if (
                        transcription
                        and "❌" not in transcription
                    ):

                        if not st.session_state.retriever:
                            st.warning(
                                "❗ Please add at least one "
                                "document first."
                            )

                        else:

                            with st.spinner(
                                "🤖 Analyzing documents and generating answer..."
                            ):
                                response = ask_document(
                                    transcription
                                )

                            st.session_state.current_answer = (
                                response
                            )

                            st.session_state.chat_history.append(
                                (
                                    transcription,
                                    response,
                                    audio_path
                                )
                            )

                            st.markdown("### Answer")
                            st.markdown(response)

                            if (
                                st.session_state.speech_enabled
                                and "❌" not in response
                            ):
                                with st.spinner(
                                    "🔊 Speaking response..."
                                ):
                                    try:
                                        speak_with_fallback(
                                            response
                                        )
                                    except Exception as e:
                                        st.warning(
                                            f"Speech error: {e}"
                                        )

                    else:
                        st.error(
                            "❌ Speech was not clear or "
                            "could not be recognized."
                        )

            except Exception as e:
                st.error(
                    f"❌ Error processing recording: {e}"
                )


# ============================================================
# Cleanup
# ============================================================

@atexit.register
def cleanup_on_exit():
    try:
        stop_speaking()

        # Keep the FAISS vector database between app restarts.
        # Use "Clear All Data" when you want to delete it.
        for file in glob.glob(
            os.path.join(RECORDINGS_DIR, "*")
        ):
            try:
                if os.path.isfile(file):
                    os.remove(file)
            except Exception:
                pass

    except Exception as e:
        print(f"Cleanup error: {e}")
