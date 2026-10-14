"""Generate shape-scene images and question/answer files, split by scene.

    python scripts/vlm/make_shapes_data.py                 # default sizes
    python scripts/vlm/make_shapes_data.py --n-train 1200

Writes data/shapes/images/<scene_id>.png, data/shapes/{train,val,test}.jsonl
(one row per question) and data/shapes/stats.json.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab import shapes_task as st  # noqa: E402
from lab.common import DATA, write_json, write_jsonl  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-train", type=int, default=800, help="training scenes (x3 questions each)")
p.add_argument("--n-val", type=int, default=60)
p.add_argument("--n-test", type=int, default=150)
p.add_argument("--size", type=int, default=256)
args = p.parse_args()

out = DATA / "shapes"
(out / "images").mkdir(parents=True, exist_ok=True)
splits = st.build_scene_splits(args.seed, args.n_train, args.n_val, args.n_test, size=args.size)
stats = {"seed": args.seed, "image_size": args.size}
for name, scenes in splits.items():
    rows = []
    for sc in scenes:
        st.render(sc).save(out / "images" / f"{sc['scene_id']}.png")
        for qi, q in enumerate(sc["questions"]):
            rows.append({"id": f"{sc['scene_id']}-q{qi}", "scene_id": sc["scene_id"], "split": name,
                         "image": f"images/{sc['scene_id']}.png", "type": q["type"], "question": q["question"],
                         "answer": q["answer"], "q": q, "scene": {k: v for k, v in sc.items() if k != "questions"}})
    write_jsonl(out / f"{name}.jsonl", rows)
    stats[name] = {"scenes": len(scenes), "questions": len(rows),
                   "types": dict(Counter(r["type"] for r in rows)),
                   "answers": dict(Counter(r["answer"] for r in rows).most_common()),
                   "majority_answer_rate": round(Counter(r["answer"] for r in rows).most_common(1)[0][1] / len(rows), 3)}
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], Counter())[r["answer"]] += 1
    stats[name]["majority_rate_by_type"] = {t: round(c.most_common(1)[0][1] / sum(c.values()), 3) for t, c in by_type.items()}
ids = {n: {sc["scene_id"] for sc in s} for n, s in splits.items()}
keys = {n: {st.scene_key(sc) for sc in s} for n, s in splits.items()}
stats["scene_key_overlap"] = {"train&test": len(keys["train"] & keys["test"]), "train&val": len(keys["train"] & keys["val"])}
write_json(out / "stats.json", stats)
for n in splits:
    print(f"{n:6s} scenes={stats[n]['scenes']:4d} questions={stats[n]['questions']:5d} "
          f"majority-answer rate={stats[n]['majority_answer_rate']}")
print("overlap:", stats["scene_key_overlap"])
