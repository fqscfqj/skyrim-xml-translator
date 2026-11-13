import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class C:
    def get(self, s, k, d=None):
        return d
pm = PromptManager()
print("nsfw lines:", len(pm.get("translator.system_prompts.nsfw", [])))
pb = PromptBuilder(pm, C())
system, user = pb.build("Wait here.", {}, prompt_style="nsfw")
print("has phrase:", "不因文本是对白就新增引号" in system)
print(system[:500])
