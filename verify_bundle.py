# -*- coding: utf-8 -*-
"""Verify the prompt actually bundled inside the onefile exe."""
import json
from PyInstaller.archive.readers import CArchiveReader

r = CArchiveReader("dist/SkyrimXMLTranslator.exe")
names = [n for n in r.toc if "system_prompts" in n or "nsfw" in n]
print("matching entries:", names)

for n in names:
    data = r.extract(n)
    if isinstance(data, tuple):
        data = data[0]
    if not isinstance(data, bytes):
        continue
    try:
        text = data.decode("utf-8")
    except Exception:
        continue
    if "nsfw" not in n:
        continue
    print(f"\n--- {n} ({len(data)} bytes) ---")
    d = json.loads(text)
    joined = "\n".join(d["nsfw"])
    print("  '不补原文没有的解剖' present:", "不补原文没有的解剖" in joined)
    print("  '绝不变软' present:", "绝不变软" in joined)
    print("  total chars:", sum(len(s) for s in d["nsfw"]))
    print("  lines:", len(d["nsfw"]))
