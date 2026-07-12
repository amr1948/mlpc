import json
from pathlib import Path

ROOT = Path("/Users/amrabdullah/AI_2026ss/mlpc")   # folder containing config.json

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

a_dir = ROOT / CONFIG["paths"]["strategy_a_artifacts"]
b_dir = ROOT / CONFIG["paths"]["strategy_b_artifacts"]

for name, d in [("Strategy A", a_dir), ("Strategy B", b_dir)]:
    print(f"\n{name} artifacts: {d}  (exists: {d.exists()})")
    if d.exists():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                print(f"   {p.relative_to(d)}  ({p.stat().st_size/1e6:.1f} MB)")