import json
from pathlib import Path
from collections import Counter

DATASET = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"

stats = Counter()
for cat_dir in sorted(DATASET.glob("penn_*")):
    n = len(list(cat_dir.glob("*.json")))
    stats[cat_dir.name.replace("penn_", "")] = n

print(f"{'类别':<20s} {'视频数':>6s}")
print("-" * 30)
total = 0
for cat, n in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"{cat:<20s} {n:>6d}")
    total += n
print("-" * 30)
print(f"{'总计':<20s} {total:>6d}")