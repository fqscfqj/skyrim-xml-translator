"""Real-glossary end-to-end check after prompt v2."""
import contextlib
import io
import sys
import time

sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.rag.engine import RAGEngine
from src.translation.translator import Translator

CASES = [
    ("Restore 20 points of Health.", {"text_kind": "ui"}),
    ("Wait 24 hours for the renovations to finish", {"text_kind": "quest"}),
    ("Report back to Jarl Balgruuf.", {"text_kind": "quest"}),
]

config = ConfigManager("config.json")
config.config.setdefault("general", {})["log_level"] = "CRITICAL"
client = LLMClient(config)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    engine = RAGEngine(config, client)
translator = Translator(client, engine)

for text, ctx in CASES:
    started = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            translation, debug = translator.translate_text(
                text, use_rag=True, max_retries=1, return_debug_info=True,
                context_hint=dict(ctx))
        status = debug.get("result_status")
        details = debug.get("result_details", "")
        attempts = len(debug.get("translation_attempts") or [])
        matched = list((debug.get("matched_terms") or {}).items())[:8]
        glossary_len = len(debug.get("glossary_context") or "")
        system_len = len(debug.get("system_prompt") or "")
        user_len = len(debug.get("user_prompt") or "")
        error = ""
    except Exception as exc:
        translation, status, details, attempts = "", "failed", "", 0
        matched, glossary_len, system_len, user_len = [], 0, 0, 0
        error = f"{type(exc).__name__}: {exc}"
    print("=" * 90)
    print(f"SRC: {text!r}  {time.time()-started:.1f}s status={status} attempts={attempts}")
    print(f"OUT: {translation!r}")
    print(f"matched={matched} glossary={glossary_len} system={system_len} user={user_len}")
    if details:
        print("DETAIL:", details[:300])
    if error:
        print("ERROR:", error[:300])
