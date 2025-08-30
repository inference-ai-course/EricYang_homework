# schemas/messages.py
from pydantic import BaseModel

class ChatResponse(BaseModel):
    text: str
    audio_path: str
