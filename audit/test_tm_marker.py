import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class Cfg:
    def get(self, s, k, d=None):
        return d
pb = PromptBuilder(PromptManager(), Cfg())
system, user = pb.build(
    "Wait 24 hours for the renovations to finish",
    {"Wait 24 hours for the renovations to finish": "等待24小时装修完毕",
     "Harvest": "收获"},
    "default",
    context_hint={"text_kind": "quest"},
    rich_meta={"Wait 24 hours for the renovations to finish": {"domain": "quest", "note": "formid=0x28016BA9 sst_status=1"},
               "Harvest": {"domain": "flora", "note": "formid=0x1234"}},
)
print(user)
