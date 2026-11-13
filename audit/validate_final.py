import json
import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class Cfg:
    def get(self, s, k, d=None):
        return d

pm = PromptManager()
pm.prompts_dir = r"audit/prompt_v2/final/prompts"
pm.load()
print("roots:", sorted(pm.prompts.keys()))
print("translator keys:", sorted(pm.prompts.get("translator", {}).keys()))
for style in ("default", "nsfw"):
    joined = "\n".join(pm.get(f"translator.system_prompts.{style}", []))
    print(style, "system chars:", len(joined))
for rel in ("translator/batch.json", "translator/retry.json", "translator/reformat.json",
            "translator/templates.json", "translator/style_profiles.json", "rag/keywords.json"):
    with open("audit/prompt_v2/final/prompts/" + rel, encoding="utf-8") as f:
        json.load(f)
print("all JSON valid")

pb = PromptBuilder(pm, Cfg())
for style, ctx, text in (
    ("default", {"text_kind": "dialogue"}, "Took you long enough. I was starting to think you'd gone and got yourself eaten."),
    ("default", {"text_kind": "ui"}, "Health"),
    ("nsfw", {"text_kind": "dialogue"}, "I want you inside me. Now."),
):
    sp, up = pb.build(text, {}, style, context_hint=ctx)
    print(f"style={style} ctx={ctx} system={len(sp)} user={len(up)} total={len(sp)+len(up)}")
