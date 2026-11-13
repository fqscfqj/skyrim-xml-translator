import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
pm = PromptManager()
print("translator keys:", sorted(pm.prompts["translator"].keys()))
print("style profiles:", sorted(pm.get("translator.style_profiles", {}).keys()))
print("default lines:", len(pm.get("translator.system_prompts.default", [])))
print("nsfw lines:", len(pm.get("translator.system_prompts.nsfw", [])))
print("batch:", pm.get("translator.batch.output_anchor", "")[:30])
print("retry:", sorted(pm.get("translator.retry", {}).keys()))
print("keywords:", sorted(pm.get("rag.keywords", {}).keys()))
