# services/tts.py
import os
from datetime import datetime

USE_EDGE_TTS_FALLBACK = False
try:
    from cozyvoice import CozyVoice
    tts_engine = CozyVoice()         # cozyvoice is sync/blocking
except Exception:
    USE_EDGE_TTS_FALLBACK = True

def _unique_name(base="response", ext=".wav") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base}_{ts}{ext}"

async def synthesize_speech(text: str, filename: str | None = None) -> str:
    """
    Async-friendly TTS. Returns path to an audio file (wav if CozyVoice, mp3 if Edge-TTS fallback).
    """
    if filename is None:
        filename = _unique_name()

    if not USE_EDGE_TTS_FALLBACK:
        # CozyVoice path (blocking) -> run in a worker thread so we don't block the event loop
        import asyncio
        loop = asyncio.get_running_loop()

        def _cozy_generate():
            tts_engine.generate(text, output_file=filename)  # Teacher API (sync)
            return filename

        return await loop.run_in_executor(None, _cozy_generate)

    # Edge-TTS fallback (native async)
    import asyncio, edge_tts
    out_mp3 = filename.replace(".wav", ".mp3")
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await communicate.save(out_mp3)
    return out_mp3
