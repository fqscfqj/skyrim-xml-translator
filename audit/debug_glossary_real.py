import contextlib, io, sys, json
sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.rag.engine import RAGEngine
from src.translation.translator import Translator

config = ConfigManager("config.json")
config.config.setdefault("general", {})["log_level"] = "CRITICAL"
client = LLMClient(config)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    engine = RAGEngine(config, client)
tr = Translator(client, engine)
text = "Restore 20 points of Health."
ctx = {"text_kind": "ui"}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    rag = tr._run_rag_phase(text)
meta = tr._collect_rich_meta(rag["matched_terms"])
print("matched:", rag["matched_terms"])
print("meta:", json.dumps(meta, ensure_ascii=False))
print("get_rich_meta Health:", engine.get_rich_meta("Health"))
print("get_rich_meta Restore:", engine.get_rich_meta("Restore"))
pb = tr._prompt_builder
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    user = pb.build_glossary_context(text, rag["matched_terms"], rich_meta=meta,
                                     reference_terms=rag.get("reference_terms"),
                                     context_hint=ctx)
print("USER CONTEXT:")
print(user)
print("DROP Health?", pb._should_drop_glossary_entry("Health", meta.get("Health", {}), text, ctx, config))
