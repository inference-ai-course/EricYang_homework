# 1. Clone & setup
git clone <your-repo-url>
cd <your-repo-folder>
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate      # or source .venv312/bin/activate on Linux/Mac
pip install --upgrade pip
pip install fastapi uvicorn[standard] python-multipart
pip install openai-whisper
pip install "transformers>=4.41" accelerate safetensors einops
pip install cozyvoice || pip install edge-tts

# 2. Run API
uvicorn main:app --reload --reload-exclude ".venv312*"
# → open http://127.0.0.1:8000/static/ui.html in your browser

# 3. Try it out
- Click **Record** → speak → **Stop**
- Server transcribes (Whisper), generates reply (LLM), and speaks back (TTS)
- Conversation memory keeps last 5 turns

# 4. Deliverables
- FastAPI project with `/chat/` endpoint
- Voice agent with 5-tur
