# services/llm.py
from typing import List, Dict
import re
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
llm = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)

# ---- Conversation + simple user facts memory ----
conversation_history: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "text": "..."}]
USER_FACTS: Dict[str, str] = {}                  # e.g., {"favorite sport": "hockey"}

# Keep FIVE TURNS (i.e., 5 user+assistant pairs = 10 messages)
TURNS_TO_KEEP = 5
WINDOW_MSGS = TURNS_TO_KEEP * 2  # 10

SYSTEM_PROMPT = (
    "You are a concise assistant. Use only information in the conversation and the supplied user facts.\n"
    "If the user asks about their preferences, rely on previously stated facts. "
    "If a fact is unknown, say you don't know—do not invent.\n"
    "Only speak as the assistant. Never invent lines that start with 'user:'."
)

FACT_PATTERNS = [
    # very small demo extractor: add more patterns if you want
    (re.compile(r"\bmy favorite ([a-zA-Z ]+?) is ([^\.!\n\r]+)", re.I), lambda m: (f"favorite {m.group(1).strip().lower()}", m.group(2).strip()))
]

def _update_user_facts(user_text: str) -> None:
    for pat, fn in FACT_PATTERNS:
        m = pat.search(user_text)
        if m:
            key, val = fn(m)
            USER_FACTS[key] = val

def _windowed_history(history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    """Keep last 5 exchanges (10 msgs) + current user turn."""
    return (history + [{"role": "user", "text": user_text}])[-WINDOW_MSGS:]

def _facts_block() -> str:
    if not USER_FACTS:
        return "No known user facts."
    lines = [f"- {k}: {v}" for k, v in USER_FACTS.items()]
    return "Known user facts:\n" + "\n".join(lines)

def _build_chat_prompt(messages: List[Dict[str, str]]) -> str:
    # Compose messages with a system message up front plus a facts block
    chat_msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _facts_block()}]
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        chat_msgs.append({"role": role, "content": m["text"]})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)

    # Fallback if no template
    text = ""
    for m in chat_msgs:
        text += f"{m['role']}: {m['content']}\n"
    text += "assistant: "
    return text

def _postprocess(full_text: str, prompt_text: str) -> str:
    gen = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
    lower = gen.lower()
    cut = lower.find("\nuser:")
    if cut != -1:
        gen = gen[:cut]
    return gen.strip()

def generate_response(user_text: str) -> str:
    # 1) Update facts before building prompt (so this turn can be referenced next time)
    _update_user_facts(user_text)

    # 2) Build prompt with 5-turn (10 message) window + facts
    msgs = _windowed_history(conversation_history, user_text)
    prompt = _build_chat_prompt(msgs)

    # 3) Generate
    outputs = llm(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=True,
    )
    full = outputs[0]["generated_text"]
    bot = _postprocess(full, prompt)

    # 4) Append this turn
    conversation_history.append({"role": "user", "text": user_text})
    conversation_history.append({"role": "assistant", "text": bot})
    return bot

def reset_history():
    conversation_history.clear()
    USER_FACTS.clear()
