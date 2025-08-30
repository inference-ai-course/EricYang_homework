# utils/audio.py
import wave
import io
import tempfile

def write_temp_wav(audio_bytes: bytes) -> str:
    """
    Best-effort: if already WAV, just write. If webm/ogg, you'd normally convert.
    For the assignment, we assume the UI records WAV/PCM.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="in_", suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name
