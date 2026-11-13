"""End-to-end translation probe with the real model and no glossary."""
import json
import sys
import time

sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.prompt.prompt_manager import PromptManager
from src.rag.glossary_manager import GlossaryManager
from src.rag.keyword_extractor import KeywordExtractor
from src.rag.search import RAGSearcher
from src.translation.translator import Translator
from types import SimpleNamespace

CASES = [
    ("font", "<font face='$HandwrittenFont'>Hello, traveler. Take a seat.</font>", {"text_kind": "document"}),
    ("percent", "Press %s to continue.", {"text_kind": "ui"}),
    ("bracket", "[PlayerName] said, \"Wait here.\"", {"text_kind": "dialogue"}),
    ("paragraph", "Line one.\n\nLine two continues the story, and it is longer than the first.", {"text_kind": "book"}),
    ("quest", "Find the Dragon Stone in Bleak Falls Barrow.", {"text_kind": "quest"}),
    ("onehand", "One-Handed", {"text_kind": "ui"}),
    ("sneak", "Sneak", {"text_kind": "ui"}),
    ("armor", "Armor Rating", {"text_kind": "ui"}),
    ("she", "She said she'd come, but she never did.", {"text_kind": "dialogue"}),
    ("alias", "<Alias=Player> has been spotted!", {"text_kind": "message"}),
    ("dmg", "Deals %d points of fire damage.", {"text_kind": "ui"}),
]

class _FakeRAG:
    def __init__(self, config):
        self.config = config
        self.glossary = {}
        self.llm_client = None
    def get_glossary_fingerprint(self):
        return "empty"

def main():
    prompts_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    out_path = sys.argv[2] if len(sys.argv) > 2 else "audit/e2e_results.json"
    style = sys.argv[3] if len(sys.argv) > 3 else "default"
    config = ConfigManager("config.json")
    config.config.setdefault("general", {})["log_level"] = "CRITICAL"
    client = LLMClient(config)
    rag = _FakeRAG(config)
    translator = Translator(client, rag)
    if prompts_dir:
        translator.prompt_manager.prompts_dir = prompts_dir
        translator.prompt_manager.load()
        translator._prompt_builder.prompt_manager = translator.prompt_manager
        translator._response_parser.prompt_manager = translator.prompt_manager
    results = []
    for name, text, ctx in CASES:
        started = time.time()
        try:
            translation, debug = translator.translate_text(
                text, use_rag=False, max_retries=1,
                return_debug_info=True, context_hint=dict(ctx),
            )
            status = debug.get("result_status")
            details = debug.get("result_details")
            attempts = len(debug.get("translation_attempts") or [])
            error = ""
        except Exception as exc:
            translation, status, details, attempts = "", "failed", "", 0
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        print("=" * 90)
        print(f"[{name}] style={style} {elapsed:.1f}s status={status} attempts={attempts}")
        print("SRC :", repr(text))
        print("OUT :", repr(translation))
        if details:
            print("DETAIL:", details[:300])
        if error:
            print("ERROR:", error[:300])
        results.append({"id": name, "source": text, "translation": translation,
                        "status": status, "details": details, "error": error,
                        "attempts": attempts, "elapsed": elapsed})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved", out_path)

if __name__ == "__main__":
    main()
