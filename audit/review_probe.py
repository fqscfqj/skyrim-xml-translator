"""Test a semantic review pass on existing translations."""
import json
import sys
sys.path.insert(0, ".")
from src.config.manager import ConfigManager
from src.llm.client import LLMClient

CASES = [
    ("complex", "The blade that slew the last dragon priest now rests in the hands of a farmer who has never drawn it.",
     "那柄杀死最后一位龙祭司的利刃，如今落在一个从未出鞘挥斩过的农夫手中。"),
    ("ellipsis", "Took you long enough. I was starting to think you'd gone and got yourself eaten.",
     "可真够慢的。我还开始以为你跑去喂了怪物、把自己搭进去了呢。"),
    ("nsfw1", "I want you inside me. Now. Don't make me beg.",
     "我要你进来。现在。别逼我求你。"),
    ("hard3", "She was not the woman he remembered, though the scar above her eye said otherwise.",
     "她已不是他记忆中的那个女人，尽管她眼睛上方的伤疤说的却是另一回事。"),
]

SYSTEM = """你是资深游戏本地化审校。用户会给出一句英文原文和一句中文初译。你的任务是检查初译是否准确、自然，并给出修正后的最终译文。

重点检查：
1. 理解错误：主语/宾语错位、修饰语挂错、代词指错、否定或数量范围错、时态/虚拟语气错。
2. 中文不自然：翻译腔、搭配错误、语序生硬。
3. 成人内容强度：原文直白则译文必须同样直白，不得用含糊词弱化。

如果没有问题，原样返回初译。只输出 JSON：{"translation":"最终译文"}。"""

def main():
    config = ConfigManager("config.json")
    client = LLMClient(config)
    for name, src, draft in CASES:
        user = f"原文：\n{src}\n\n初译：\n{draft}\n\n请审校并输出最终译文。"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        raw = client.chat_completion(messages)
        print("=" * 80)
        print(f"[{name}]")
        print("SRC  :", src)
        print("DRAFT:", draft)
        print("RAW  :", raw)

if __name__ == "__main__":
    main()
