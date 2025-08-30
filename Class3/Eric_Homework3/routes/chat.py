# routes/chat.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from services.asr import transcribe_audio
from services.llm import generate_response
from services.tts import synthesize_speech
from utils.audio import write_temp_wav
import os

router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/m4a", "audio/mp4", "audio/webm", "audio/ogg"):
        raise HTTPException(status_code=400, detail="Upload an audio file (wav/m4a/mp4/webm/ogg).")
    audio_bytes = await file.read()

    temp_wav = write_temp_wav(audio_bytes)

    # ASR
    user_text = transcribe_audio(temp_wav)

    # LLM
    bot_text = generate_response(user_text)

    # TTS (async)
    audio_path = await synthesize_speech(bot_text)

    # Content type depends on engine (wav for CozyVoice, mp3 for Edge-TTS fallback)
    ext = os.path.splitext(audio_path)[1].lower()
    media_type = "audio/wav" if ext == ".wav" else "audio/mpeg"

    return FileResponse(audio_path, media_type=media_type)
