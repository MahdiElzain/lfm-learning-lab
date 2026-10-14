"""Generate the robot-dispatch datasets (JSONL) with scene-level leakage control.

    python scripts/text/make_robot_data.py            # default sizes, seed 0
    python scripts/text/make_robot_data.py --n-train 4000

Writes data/robot/{train,val,test,test_ood}.jsonl, data/robot/overfit_tiny.jsonl
and data/robot/stats.json.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab import robot_task as rt  # noqa: E402
from lab.common import DATA, write_json, write_jsonl  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-train", type=int, default=6000)
p.add_argument("--n-val", type=int, default=200)
p.add_argument("--n-test", type=int, default=300)
p.add_argument("--n-test-ood", type=int, default=200)
args = p.parse_args()

out = DATA / "robot"
splits = rt.build_splits(args.seed, args.n_train, args.n_val, args.n_test, args.n_test_ood)

# Separate miniature dataset for the overfitting demo: 16 fresh scenes that are
# checked against every other split so the demo cannot leak into evaluation.
import random  # noqa: E402

seen = {rt.scene_signature(ex) for rows in splits.values() for ex in rows}
rng = random.Random(args.seed + 999)
tiny = []
while len(tiny) < 16:
    ex = rt.make_example(rng, rng.randint(3, 5), want_trap=len(tiny) % 2 == 0)
    if rt.scene_signature(ex) in seen:
        continue
    seen.add(rt.scene_signature(ex))
    ex["id"], ex["split"] = f"overfit-{len(tiny):03d}", "overfit_tiny"
    tiny.append(ex)
splits["overfit_tiny"] = tiny

stats = {"seed": args.seed}
for name, rows in splits.items():
    write_jsonl(out / f"{name}.jsonl", rows)
    stats[name] = {
        "n": len(rows),
        "trap_rate": round(sum(r["trap"] for r in rows) / len(rows), 3),
        "kinds": dict(Counter(r["kind"] for r in rows)),
        "n_robots": dict(sorted(Counter(r["n_robots"] for r in rows).items())),
        "answer_distance_mean": round(sum(r["answer"]["distance"] for r in rows) / len(rows), 2),
    }
sigs = {name: {rt.scene_signature(r) for r in rows} for name, rows in splits.items()}
names = list(sigs)
stats["cross_split_overlap"] = {f"{a}&{b}": len(sigs[a] & sigs[b])
                                for i, a in enumerate(names) for b in names[i + 1:]}
write_json(out / "stats.json", stats)
print(f"Wrote {out}")
for name in splits:
    print(f"  {name:13s} n={stats[name]['n']:5d} trap_rate={stats[name]['trap_rate']} robots={stats[name]['n_robots']}")
print("  overlap:", stats["cross_split_overlap"])
print("\nExample prompt:\n" + rt.format_user(splits['train'][0]))
print("Answer:", rt.format_answer(splits['train'][0]['answer']))
