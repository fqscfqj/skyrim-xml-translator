import sys, json, time, io, contextlib
sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.rag.engine import RAGEngine
from src.translation.translator import Translator

config = ConfigManager("config.json")
config.config.setdefault("general", {})["log_level"] = "CRITICAL"
client = LLMClient(config)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    engine = RAGEngine(config, client)
init_s = time.time() - t0
translator = Translator(client, engine)

cases = [
    ("Health", {"text_kind": "ui", "whitespace_policy": "strict"}),
    ("The Dragonborn entered Whiterun, where Jarl Balgruuf waited with Erik the Slayer.",
     {"text_kind": "book", "whitespace_policy": "strict"}),
    ("Wait 24 hours for the renovations to finish",
     {"text_kind": "quest", "whitespace_policy": "strict"}),
]
print(f"RAG init: {init_s:.1f}s terms={len(engine.glossary)}")
for text, ctx in cases:
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rag = translator._run_rag_phase(text)
        pb = translator._prompt_builder
        prompt_style = translator._resolve_prompt_style_for_context(ctx)
        system, user = pb.build(text, rag["matched_terms"], prompt_style, context_hint=ctx,
                                glossary_source_text=text,
                                rich_meta=translator._collect_rich_meta(rag["matched_terms"]),
                                reference_terms=rag.get("reference_terms"))
    print("=" * 90)
    print(f"TEXT: {text!r}  rag={time.time()-t0:.1f}s")
    print(f"keywords={rag['keywords']!r}")
    print(f"matched={list(rag['matched_terms'].items())[:12]}")
    print(f"reference count={len(rag.get('reference_terms') or {})}")
    print(f"glossary_ctx={len(rag['glossary_context'])}ch  system={len(system)}ch  user={len(user)}ch  total={len(system)+len(user)}")
    print("--- USER ---")
    print(user[:3000])
