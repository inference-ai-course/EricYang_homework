# services/asr.py
import whisper

# Load once at import time
asr_model = whisper.load_model("small")  # "base" is faster, "small" is more accurate

def transcribe_audio(wav_path: str) -> str:
    """
    Transcribe audio file at wav_path using Whisper.
    Returns recognized text.
    """
    result = asr_model.transcribe(wav_path)
    return result.get("text", "").strip()
