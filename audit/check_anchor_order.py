import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class C:
    def get(self, s, k, d=None):
        return d
pb = PromptBuilder(PromptManager(), C())
for style, ctx in (("default", {"text_kind": "dialogue"}), ("nsfw", {"text_kind": "dialogue"}), ("default", {"text_kind": "ui"})):
    system, user = pb.build("Wait here.", {}, style, context_hint=ctx)
    print("=" * 60)
    print(style, ctx, "system chars", len(system))
    print("LAST 2 LINES:", repr("\n".join(system.splitlines()[-2:])))
    print("HAS STYLE:", "文本类型与文体上下文" in system)
    print("FORMAT AFTER STYLE:", system.rfind("格式锚点") > system.rfind("文本类型与文体上下文"))
