# services/tts.py
import os
import tempfile
from datetime import datetime
import asyncio

# Try CozyVoice first (local, high quality)
USE_COZY = True
try:
    from cozyvoice import CozyVoice
    cozy = CozyVoice()
except Exception:
    USE_COZY = False
    cozy = None

def _unique_name(base="response", ext=".wav") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base}_{ts}{ext}"

async def synthesize_speech(text: str, filename: str | None = None) -> str:
    """
    Generate speech from text. Returns path to an audio file.
    Order: CozyVoice (wav) → Edge-TTS (mp3) → pyttsx3 (wav, offline).
    """
    if filename is None:
        folder = tempfile.gettempdir()
        filename = os.path.join(folder, _unique_name(ext=".wav"))

    loop = asyncio.get_running_loop()

    # 1) CozyVoice
    if USE_COZY and cozy is not None:
        def _cozy_generate():
            cozy.generate(text, output_file=filename)
            return filename
        return await loop.run_in_executor(None, _cozy_generate)

    # 2) Edge-TTS (network)
    try:
        import edge_tts  # pip install edge-tts
        voice = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
        out_mp3 = filename.replace(".wav", ".mp3")
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(out_mp3)
        return out_mp3
    except Exception as e_edge:
        # 3) Offline fallback: pyttsx3 (Windows SAPI5)
        try:
            import pyttsx3  # pip install pyttsx3
        except Exception as e_import:
            # No TTS available at all
            raise RuntimeError(
                f"TTS unavailable. Install CozyVoice or edge-tts or pyttsx3. Edge-TTS error: {e_edge}"
            ) from e_import

        def _pyttsx3_generate():
            engine = pyttsx3.init()
            # You can tweak voice/rate/volume here if you want
            # engine.setProperty('rate', 180)
            engine.save_to_file(text, filename)  # SAPI5 writes WAV on Windows
            engine.runAndWait()
            return filename

        return await loop.run_in_executor(None, _pyttsx3_generate)
