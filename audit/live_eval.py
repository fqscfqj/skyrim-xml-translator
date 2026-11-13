"""Live translation-quality probe using the configured OpenAI-compatible endpoint."""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from src.config.manager import ConfigManager
from src.llm.client import LLMClient
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder
from src.translation.response_parser import ResponseParser

CASES = [
    {
        "id": "pronoun",
        "style": "default",
        "kind": "dialogue",
        "text": "Gerdur said the dragon took her son, but he was hidden in the keep.",
    },
    {
        "id": "idiom",
        "style": "default",
        "kind": "dialogue",
        "text": "You're a stone's throw away from the Jarl's longhouse. Don't get lost.",
    },
    {
        "id": "negscope",
        "style": "default",
        "kind": "book",
        "text": "No one but the Thalmor would dare enter that ruin, and not even they return.",
    },
    {
        "id": "ellipsis",
        "style": "default",
        "kind": "dialogue",
        "text": "Took you long enough. I was starting to think you'd gone and got yourself eaten.",
    },
    {
        "id": "complex",
        "style": "default",
        "kind": "book",
        "text": "The blade that slew the last dragon priest now rests in the hands of a farmer who has never drawn it.",
    },
    {
        "id": "ui",
        "style": "default",
        "kind": "ui",
        "text": "Restore 20 points of Health.",
    },
    {
        "id": "nsfw1",
        "style": "nsfw",
        "kind": "dialogue",
        "text": "I want you inside me. Now. Don't make me beg.",
    },
    {
        "id": "nsfw2",
        "style": "nsfw",
        "kind": "dialogue",
        "text": "Ah... yes, right there. Harder. Gods, don't stop.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="")
    parser.add_argument("--style", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cases-file", default="")
    parser.add_argument("--prompts-dir", default="")
    args = parser.parse_args()

    config = ConfigManager("config.json")
    client = LLMClient(config)
    pm = PromptManager(config)
    if args.prompts_dir:
        pm.prompts_dir = args.prompts_dir
        pm.load()
    pb = PromptBuilder(pm, config)
    rp = ResponseParser(config)

    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    cases = CASES
    if args.cases_file:
        with open(args.cases_file, "r", encoding="utf-8") as f:
            cases = json.load(f)
    cases = [c for c in cases if not wanted or c["id"] in wanted]
    if args.style:
        for c in cases:
            c["style"] = args.style

    results = []
    for rep in range(max(1, args.repeat)):
        for case in cases:
            ctx = dict(case.get("context") or {"text_kind": case.get("kind", "")})
            system, user = pb.build(
                case["text"], case.get("glossary") or {}, case["style"],
                context_hint=ctx,
                reference_terms=case.get("reference_terms") or None,
                rich_meta=case.get("rich_meta") or None,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            started = time.time()
            try:
                raw = client.chat_completion(messages)
            except Exception as exc:
                print(f"[{case['id']}] request failed: {type(exc).__name__}: {exc}")
                continue
            elapsed = time.time() - started
            try:
                parsed = rp.parse(raw, case["text"], messages)
            except Exception as exc:
                parsed = f"<parse error: {type(exc).__name__}: {exc}>"
            print("=" * 80)
            print(f"[{case['id']}] style={case['style']} elapsed={elapsed:.1f}s "
                  f"prompt={len(system)+len(user)}ch")
            print(f"SRC : {case['text']}")
            print(f"RAW : {raw}")
            print(f"OUT : {parsed}")
            results.append({
                "id": case["id"], "repeat": rep, "style": case["style"],
                "source": case["text"], "raw": raw, "parsed": parsed,
                "elapsed": elapsed, "system": system, "user": user,
            })
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
