import sys, json
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager

live = PromptManager()
staged = PromptManager()
staged.prompts_dir = "audit/prompt_v2/final/prompts"
staged.load()

def walk(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(walk(v, p))
        else:
            out[p] = v
    return out

a = walk(live.prompts)
b = walk(staged.prompts)
print("live keys", len(a), "staged keys", len(b))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print("DIFF", k)
        if isinstance(a.get(k), str) and isinstance(b.get(k), str):
            import difflib
            for line in difflib.unified_diff(a[k].split("\n"), b[k].split("\n"), lineterm="", n=0):
                print(line)
print("done")
