# Week 6 – Voice Agent with Function Calling (Windows-friendly)

## 1) Clone & Setup
```bash
git clone <your-repo-url>
cd <your-repo-folder>/Eric_Homework6
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate         # or source .venv312/bin/activate on Linux/Mac
pip install --upgrade pip

# Core deps (same as Week 3)
pip install fastapi "uvicorn[standard]" python-multipart
pip install openai-whisper
pip install "transformers>=4.41" accelerate safetensors einops
pip install cozyvoice || pip install edge-tts

# New for Week 6 tools
pip install sympy arxiv
