import json, sys
sys.path.insert(0, ".")
from src.translation.quality_checker import QualityChecker, is_model_refusal

qc = QualityChecker()
files = ["audit/live_baseline_all.json", "audit/live_current_hard.json",
         "audit/live_current_nsfw_fresh.json", "audit/live_benign_nsfw.json"]
for path in files:
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except FileNotFoundError:
        continue
    print("=" * 90)
    print(path)
    for row in rows:
        issues = qc.check(row["source"], row.get("parsed", ""))
        refusal = is_model_refusal(row.get("parsed", ""), row["source"])
        print(f"[{row['id']}] issues={[(i.rule_id, i.severity, i.details[:40]) for i in issues]} refusal={refusal}")
