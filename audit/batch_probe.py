"""Probe short-text batch translation reliability with the live flash model."""
import json
import sys
import time

sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder
from src.translation.response_parser import ResponseParser

ITEMS = ["Health", "Magicka", "Stamina", "Sneak", "Lockpicking",
         "Restore 20 points of Health.", "Iron Sword", "Whiterun"]
EXPECTED = {
    "Health": "生命值", "Magicka": "魔法值", "Stamina": "耐力",
    "Sneak": "潜行", "Lockpicking": "开锁", "Restore 20 points of Health.": "恢复20点生命值。",
    "Iron Sword": "铁剑", "Whiterun": "白漫城",
}
STYLE = sys.argv[1] if len(sys.argv) > 1 else "default"

config = ConfigManager("config.json")
client = LLMClient(config)
pm = PromptManager(config)
pb = PromptBuilder(pm, config)
rp = ResponseParser(config)

entries = [{"id": i, "text": text, "context_hint": {"text_kind": "ui"}}
           for i, text in enumerate(ITEMS)]
for run in range(2):
    system, user = pb.build_batch(entries, STYLE)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    started = time.time()
    raw = client.chat_completion(messages)
    parsed = rp.parse_batch(raw, expected_ids=set(range(len(ITEMS))))
    print("=" * 90)
    print(f"run {run} style={STYLE} elapsed={time.time()-started:.1f}s prompt={len(system)+len(user)}")
    print("RAW:", raw[:2000])
    if parsed is None:
        print("PARSE: FAILED")
        continue
    for i, text in enumerate(ITEMS):
        got = parsed.get(i)
        mark = "OK" if got and got.strip() else "MISS"
        print(f"  [{i}] {text!r} -> {got!r} {mark}")
