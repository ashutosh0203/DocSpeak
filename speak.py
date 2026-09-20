import os
import pyaudio
import wave
import speech_recognition as sr
from datetime import datetime
import pyttsx3
import threading
import queue
import time
import platform

# ---------------------------
# Audio Config
# ---------------------------
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def generate_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

# ---------------------------
# Improved Recorder Class
# ---------------------------
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
                    data = self.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break
                    
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.recording = False

    def stop_recording(self):
        self.recording = False
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
                
            if self.frames and self.filename:
                with wave.open(self.filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.frames))
                print(f"Recording saved: {self.filename}")
                return self.filename
            else:
                print("No frames to save")
                return None
                
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None

# ---------------------------
# Speech to Text
# ---------------------------
def transcribe_audio(filename):
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
        return f"❌ API error: {e}"
    except Exception as e:
        return f"❌ Transcription error: {e}"

# ---------------------------
# Thread-Safe Text to Speech
# ---------------------------
class ThreadSafeTTS:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()

    def _init_engine(self):
        """Initialize TTS engine in worker thread"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)
            return True
        except Exception as e:
            print(f"TTS engine initialization failed: {e}")
            return False

    def _worker(self):
        """Worker thread for TTS operations"""
        if not self._init_engine():
            print("Failed to initialize TTS engine")
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break

                if self.engine and text.strip():
                    try:
                        self.engine.stop()  # Stop any ongoing speech
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
        """Add text to speech queue"""
        if not text or not text.strip():
            return False

        try:
            with self.lock:
                if not self.running:
                    self.running = True
                    self.worker_thread = threading.Thread(target=self._worker, daemon=True)
                    self.worker_thread.start()

            self.speech_queue.put(text)
            return True
        except Exception as e:
            print(f"Error adding to speech queue: {e}")
            return False

    def stop(self):
        """Stop TTS worker thread"""
        with self.lock:
            if self.running:
                self.running = False
                self.speech_queue.put(None)  # Shutdown signal
                if self.worker_thread and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=2)

# Global TTS instance
_tts_instance = None

def get_tts_instance():
    """Get or create thread-safe TTS instance"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = ThreadSafeTTS()
    return _tts_instance

# ---------------------------
# System-based TTS (Fallback)
# ---------------------------
def speak_system(text: str) -> bool:
    """Fallback TTS using system commands"""
    try:
        system = platform.system().lower()
        
        # Clean text for system commands
        safe_text = text.replace('"', '\\"').replace("'", "\\'")
        
        if system == "windows":
            # Use Windows SAPI via PowerShell
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 0; $synth.Volume = 100; $synth.Speak(\'{safe_text}\')"'
            os.system(cmd)
        elif system == "darwin":  # macOS
            os.system(f"say '{safe_text}'")
        elif system == "linux":
            # Try different TTS engines on Linux
            if os.system("which espeak > /dev/null 2>&1") == 0:
                os.system(f"espeak '{safe_text}'")
            elif os.system("which festival > /dev/null 2>&1") == 0:
                os.system(f"echo '{safe_text}' | festival --tts")
            elif os.system("which spd-say > /dev/null 2>&1") == 0:
                os.system(f"spd-say '{safe_text}'")
            else:
                return False
        else:
            return False
        return True
    except Exception as e:
        print(f"System TTS error: {e}")
        return False

# ---------------------------
# Main TTS Functions
# ---------------------------
def speak(text: str) -> bool:
    """Primary TTS function using thread-safe pyttsx3"""
    try:
        tts = get_tts_instance()
        return tts.speak(text)
    except Exception as e:
        print(f"Primary TTS error: {e}")
        return False

def speak_with_fallback(text: str) -> bool:
    """Try thread-safe TTS first, then system TTS"""
    if not text or not text.strip():
        return False
        
    # Try primary method first
    if speak(text):
        return True
    
    print("Primary TTS failed, trying system fallback...")
    # Fallback to system commands
    return speak_system(text)

def stop_speaking():
    """Stop all TTS operations"""
    global _tts_instance
    try:
        if _tts_instance:
            _tts_instance.stop()
            _tts_instance = None
    except Exception as e:
        print(f"Error stopping TTS: {e}")

# ---------------------------
# Legacy compatibility
# ---------------------------
# Keep the old interface for backward compatibility
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
except:
    engine = None

def speak_legacy(text: str):
    """Legacy speak function - use speak_with_fallback instead"""
    if engine:
        try:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Legacy TTS error: {e}")
            # Fallback to new method
            speak_with_fallback(text)
    else:
        speak_with_fallback(text)

# import os
# import pyaudio
# import wave
# import speech_recognition as sr
# from datetime import datetime
# import pyttsx3

# # ---------------------------
# # Audio Config
# # ---------------------------
# FRAMES_PER_BUFFER = 3200
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# RECORDINGS_DIR = "recordings"
# os.makedirs(RECORDINGS_DIR, exist_ok=True)

# def generate_filename():
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     return os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

# # Recorder Class
# class SpeechRecorder:
#     def __init__(self):
#         self.p = pyaudio.PyAudio()
#         self.frames = []
#         self.recording = False
#         self.stream = None
#         self.filename = None

#     def start_recording(self):
#         self.stream = self.p.open(format=FORMAT,
#                                   channels=CHANNELS,
#                                   rate=RATE,
#                                   input=True,
#                                   frames_per_buffer=FRAMES_PER_BUFFER)
#         self.frames = []
#         self.recording = True
#         self.filename = generate_filename()
#         while self.recording:
#             data = self.stream.read(FRAMES_PER_BUFFER)
#             self.frames.append(data)

#     def stop_recording(self):
#         self.recording = False
#         self.stream.stop_stream()
#         self.stream.close()
#         self.p.terminate()
#         with wave.open(self.filename, 'wb') as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(self.p.get_sample_size(FORMAT))
#             wf.setframerate(RATE)
#             wf.writeframes(b''.join(self.frames))
#         return self.filename


# # ---------------------------
# # Speech to Text
# # ---------------------------
# def transcribe_audio(filename):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(filename) as source:
#         audio = recognizer.record(source)
#     try:
#         return recognizer.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "❌ Could not understand the audio."
#     except sr.RequestError:
#         return "❌ API error."


# # ---------------------------
# # Text to Speech
# # ---------------------------
# engine = pyttsx3.init()
# engine.setProperty('rate', 160)
# engine.setProperty('volume', 1.0)

# def speak(text: str):
#     engine.stop()   # stop any ongoing speech loop
#     engine.say(text)
#     engine.runAndWait()

# def stop_speaking():
#     engine.stop()