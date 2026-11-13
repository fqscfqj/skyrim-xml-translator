import json, collections
print("loading...")
with open("glossary/glossary.rich.json", encoding="utf-8-sig") as f:
    data = json.load(f)
print(type(data), list(data.keys())[:10] if isinstance(data, dict) else len(data))
# figure structure
if isinstance(data, dict):
    terms = data.get("terms") or data.get("rich_meta") or data
else:
    terms = data
print("entries", len(terms))
cnt = collections.Counter()
keys = collections.Counter()
samples = []
if isinstance(terms, dict):
    items = terms.items()
else:
    items = [(x.get("term"), x) for x in terms if isinstance(x, dict)]
for i, (k, v) in enumerate(items):
    if not isinstance(v, dict):
        continue
    for kk in v.keys():
        keys[kk] += 1
    d = v.get("domain")
    cnt[str(d)] += 1
    if len(samples) < 8:
        samples.append((k, v))
print("keys", keys.most_common(30))
print("domains", cnt.most_common(40))
for k, v in samples:
    print("SAMPLE", k, "->", json.dumps(v, ensure_ascii=False)[:500])
