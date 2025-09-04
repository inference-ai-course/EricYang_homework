# services/llm.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os, re, json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def _likely_math_query(text: str) -> bool:
    t = text.lower()
    has_digits = any(ch.isdigit() for ch in t)
    has_op = any(op in t for op in ["+", "-", "*", "/", "^"])
    math_words = any(w in t for w in ["compute", "calculate", "evaluate", "sin", "cos", "tan", "sqrt", "log", "pi"])
    return (has_digits and has_op) or math_words

def _extract_math_expression(text: str) -> str:
    """
    Extract a clean math expression from free-form text.
    Keeps only tokens Sympy understands (digits, + - * / ^, parentheses, known funcs/constants).
    Removes leading verbs like 'compute', 'calculate', 'evaluate', 'what is', etc.
    """
    # 1) Drop common leading verbs/phrases
    import re
    t = re.sub(r'^\s*(what\s*is|compute|calculate|eval(?:uate)?)\s*[:\-]?\s*', '', text, flags=re.I).strip()

    # 2) Tokenize: numbers, operators, parentheses, known functions, constants
    tokens = re.findall(
        r'(?:'                # group of alternatives:
        r'sin|cos|tan|sqrt|log|'   # functions
        r'pi|\bPI\b|\bPi\b|\be\b|' # constants
        r'\d+(?:\.\d+)?|'          # numbers (ints/floats)
        r'[+\-*/^()]'              # operators and parentheses
        r')',
        t, flags=re.I
    )

    expr = ''.join(tokens)

    # 3) Optional: normalize multiple upper-case PI variants
    expr = expr.replace('PI', 'pi').replace('Pi', 'pi')

    return expr.strip()


def _likely_arxiv_query(text: str) -> bool:
    t = text.lower()
    keywords = ["arxiv", "paper", "summarize", "overview", "diffusion model", "transformer", "retrieval-augmented generation"]
    return any(k in t for k in keywords)



# ------------------------------
# 1) MODEL: force CPU load (no auto offload) to avoid "device disk is invalid"
# ------------------------------
MODEL_NAME = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=False,   # avoid meta init + offload path
    torch_dtype=None,          # eliminates the deprecation warning
    device_map=None            # stay off accelerate mapping
)

llm = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,                 # -1 = CPU
)

# ---- Conversation + simple user facts memory ----
conversation_history: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "text": "..."}]
USER_FACTS: Dict[str, str] = {}                  # e.g., {"favorite sport": "hockey"}

# Keep FIVE TURNS (i.e., 5 user+assistant pairs = 10 messages)
TURNS_TO_KEEP = 5
WINDOW_MSGS = TURNS_TO_KEEP * 2  # 10

# ------------------------------
# 2) SYSTEM PROMPT: add tool-calling instructions + few-shot
# ------------------------------
SYSTEM_PROMPT = (
    "You are a concise assistant. If the user asks for arithmetic or evaluation, you MUST call the calculate tool.\n"
    "If the user asks for a paper summary/overview, you MUST call the search_arxiv tool.\n"
    "Output ONLY a single JSON object on one line when calling a tool: "
    '{"function":"...","arguments":{...}}. Otherwise reply in one short sentence.\n'
    "Use only the conversation and the supplied user facts; if a fact is unknown, say you don't know.\n"
)

TOOL_SPEC = (
    "TOOLS YOU CAN CALL (emit JSON as above):\n"
    "- search_arxiv(query: str) -> str\n"
    "- calculate(expression: str) -> str\n"
)

FEWSHOT = (
    "User: what's 2+2?\n"
    'Assistant: {"function":"calculate","arguments":{"expression":"2+2"}}\n'
    "User: summarize diffusion models\n"
    'Assistant: {"function":"search_arxiv","arguments":{"query":"diffusion models overview"}}\n'
    "User: hello there!\n"
    "Assistant: Hi!\n"
)

FACT_PATTERNS = [
    # very small demo extractor: add more patterns if you want
    (re.compile(r"\bmy favorite ([a-zA-Z ]+?) is ([^\.!\n\r]+)", re.I),
     lambda m: (f"favorite {m.group(1).strip().lower()}", m.group(2).strip()))
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
    # Compose messages with a system message up front; include tool spec + few-shot in system for simplicity
    sys_content = SYSTEM_PROMPT + "\n" + TOOL_SPEC + "\nExamples:\n" + FEWSHOT + "\n\n" + _facts_block()
    chat_msgs = [{"role": "system", "content": sys_content}]
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
    # Keep only the last "Assistant:" block if the model echoes roles
    idx = gen.rfind("Assistant:")
    if idx != -1:
        gen = gen[idx + len("Assistant:"):].strip()
    # Trim if it starts inventing a 'User:' line
    low = gen.lower()
    cut = low.find("\nuser:")
    if cut != -1:
        gen = gen[:cut]
    return gen.strip()

# ------------------------------
# 3) Tool-call detection & routing
# ------------------------------
_JSON_OBJ = re.compile(r"\{.*\}", re.S)

def _extract_json(text: str) -> Tuple[bool, str]:
    """Try to grab a single JSON object from the reply."""
    fence = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
    if fence:
        return True, fence.group(1).strip()
    after = re.search(r"json\s*({.*})", text, flags=re.S | re.I)
    if after:
        return True, after.group(1).strip()
    m = re.search(_JSON_OBJ, text)
    if m:
        return True, m.group(0)
    return False, text

def _route_llm_output(llm_output: str, user_text: str):
    """
    If llm_output is a function call (JSON), call the tool and return tool result.
    Otherwise, try heuristic intent detection on the original user_text.
    Returns (final_text, debug_dict).
    """
    debug = {"raw_llm": llm_output, "tool_called": None, "tool_args": None, "tool_output": None, "fallback_used": None}

    # 1) Try to parse JSON from model output
    ok, json_str = _extract_json(llm_output)
    if ok:
        try:
            obj = json.loads(json_str)
            func_name = obj.get("function")
            args = obj.get("arguments", {}) or {}
        except Exception as e:
            debug["json_error"] = f"{type(e).__name__}: {e}"
            func_name, args = None, None

        if func_name:
            from .tools import search_arxiv, calculate  # lazy import
            tool_registry = {"search_arxiv": search_arxiv, "calculate": calculate}
            if func_name in tool_registry:
                try:
                    result = tool_registry[func_name](**args)
                except TypeError:
                    result = tool_registry[func_name](args.get("query") or args.get("expression") or "")
                debug["tool_called"] = func_name
                debug["tool_args"] = args
                debug["tool_output"] = result
                debug["final"] = str(result)
                return debug["final"], debug

    # 2) Heuristic fallback based on user_text (assignment allows fallback behavior)
    from .tools import search_arxiv, calculate
    if _likely_math_query(user_text):
        expr = _extract_math_expression(user_text) or user_text
        res = calculate(expr)
        debug["fallback_used"] = "math-intent"
        debug["tool_called"] = "calculate"
        debug["tool_args"] = {"expression": expr}
        debug["tool_output"] = res
        debug["final"] = str(res)
        return debug["final"], debug

    if _likely_arxiv_query(user_text):
        res = search_arxiv(user_text)
        debug["fallback_used"] = "arxiv-intent"
        debug["tool_called"] = "search_arxiv"
        debug["tool_args"] = {"query": user_text}
        debug["tool_output"] = res
        debug["final"] = str(res)
        return debug["final"], debug

    # 3) No tool: return the model text as-is
    debug["final"] = llm_output.strip()
    return debug["final"], debug


# ------------------------------
# 4) Public functions
# ------------------------------
def generate_response(user_text: str) -> str:
    # Update facts before building prompt (so this turn can be referenced next time)
    _update_user_facts(user_text)

    # Build prompt with 5-turn (10 message) window + facts + tool spec
    msgs = _windowed_history(conversation_history, user_text)
    prompt = _build_chat_prompt(msgs)

    # Deterministic decoding helps JSON formatting when tools are needed
    outputs = llm(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = outputs[0]["generated_text"]
    raw = _postprocess(full, prompt)

    final, dbg = _route_llm_output(raw, user_text)

    # Append this turn
    conversation_history.append({"role": "user", "text": user_text})
    conversation_history.append({"role": "assistant", "text": final})
    return final

def generate_debug(user_text: str) -> Dict[str, Any]:
    """Return a dict with prompt, raw LLM, any tool call, tool output, and final text (for /chat_text logs)."""
    _update_user_facts(user_text)
    msgs = _windowed_history(conversation_history, user_text)
    prompt = _build_chat_prompt(msgs)

    outputs = llm(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = outputs[0]["generated_text"]
    raw = _postprocess(full, prompt)

    final, dbg = _route_llm_output(raw, user_text)
    dbg["prompt"] = prompt

    conversation_history.append({"role": "user", "text": user_text})
    conversation_history.append({"role": "assistant", "text": final})
    return dbg

def reset_history():
    conversation_history.clear()
    USER_FACTS.clear()
