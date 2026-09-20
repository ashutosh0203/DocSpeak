import speech_recognition as sr

def transcribe_audio(filename="output.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("📝 Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError:
        print("❌ Could not request results.")
transcribe_audio()