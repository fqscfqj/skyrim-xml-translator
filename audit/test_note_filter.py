import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class Cfg:
    def get(self, s, k, d=None):
        return d
pb = PromptBuilder(PromptManager(), Cfg())
ctx = {"text_kind": "ui"}
system, user = pb.build(
    "Restore 20 points of Health.", {"Health": "生命"}, "default",
    context_hint=ctx,
    rich_meta={"Health": {"domain": "actor_value", "note": "formid=0x5EBD62EB sst_status=0"}},
)
print(user)
print("---- has formid:", "formid" in user, "has sst_status:", "sst_status" in user)
print("---- domain label:", "属性" in user)
