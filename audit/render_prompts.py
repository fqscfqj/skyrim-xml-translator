import sys
sys.path.insert(0, ".")
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder

class Cfg:
    def __init__(self, vals=None):
        self.v = dict(vals or {})
    def get(self, s, k, d=None):
        return self.v.get((s,k), d)

pm = PromptManager()
def show(name, style, text, ctx=None, glossary=None, refs=None, mcm=False, prev=None):
    cfg = Cfg()
    pb = PromptBuilder(pm, cfg)
    sp, up = pb.build(text, glossary or {}, style, mcm_ui_mode=mcm,
                      context_hint=ctx, previous_source=prev, reference_terms=refs)
    print("="*100)
    print(f"## {name}: system={len(sp)} chars, user={len(up)} chars, total={len(sp)+len(up)} source={len(text)}")
    print("--- SYSTEM ---")
    print(sp)
    print("--- USER ---")
    print(up)

show("short UI default", "default", "Health")
show("dialogue default", "default", '"Where is the Dragonborn?" the guard asked.', {"text_kind":"dialogue"})
show("nsfw adult line", "nsfw", '"Ah... yes, right there, harder," she moaned.', {"text_kind":"dialogue"})
g = {"Dragonborn":"龙裔","Whiterun":"雪漫","Erik the Slayer":"弑龙者埃里克","Iron Sword":"铁剑","Jarl Balgruuf":"领主巴尔古夫"}
show("lore with glossary", "default", "The Dragonborn entered Whiterun, where Jarl Balgruuf waited with Erik the Slayer.", {"text_kind":"book"}, glossary=g)
