"""Evaluate a VLM on shape questions under correct / swapped / blank image conditions.

    .venv/bin/python scripts/vlm/evaluate.py --name vlm_base
    .venv/bin/python scripts/vlm/evaluate.py --adapter outputs/runs/vlm_sft/lora_quick/adapter --name vlm_sft
    .venv/bin/python scripts/vlm/evaluate.py --model vlm_small --name smolvlm_base
    .venv/bin/python scripts/vlm/evaluate.py --set-processor max_image_tokens=64 --name vlm_base_64tok

Visual-dependence logic
  correct  image of the question's own scene             -> scored against that scene's answer
  swapped  image of a DIFFERENT test scene, same question -> scored against the SWAPPED scene's answer
           (only question types that are defined for any image), and we report how often the
           prediction stays identical to the correct-image prediction ("unchanged rate")
  blank    empty canvas                                   -> accuracy vs the original answer shows what can
           be guessed from the text alone; for counts the truly correct blank answer is 0 / "no"
A model that ignores the image keeps its answers under swapping and scores near the
question-only prior. Majority-answer baselines (from the TRAIN split) are reported too.
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import yaml  # noqa: E402

from lab import shapes_task as st  # noqa: E402
from lab.common import (DATA, OUTPUTS, GpuMemory, append_experiment_index, environment_info, read_jsonl,  # noqa: E402
                        set_seed, write_json, write_jsonl)
from lab.models import load_vlm, resolve  # noqa: E402
from lab.vlm_utils import VLM_MAX_NEW_TOKENS, generate, load_image  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="vlm")
p.add_argument("--adapter", default=None)
p.add_argument("--split", default="test")
p.add_argument("--limit", type=int, default=None, help="first N questions")
p.add_argument("--conditions", nargs="+", default=["correct", "swapped", "blank"])
p.add_argument("--set-processor", action="append", default=[], help="image processor override key=value")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--name", required=True)
args = p.parse_args()
set_seed(0)

proc_kwargs = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set_processor)}
mem = GpuMemory()
mem.reset()
model, processor = load_vlm(args.model, processor_kwargs=proc_kwargs)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

rows = read_jsonl(DATA / "shapes" / f"{args.split}.jsonl")[: args.limit]
rng = random.Random(0)
scene_ids = sorted({r["scene_id"] for r in rows})
scene_row = {r["scene_id"]: r for r in rows}
swap_of = {}
for sid in scene_ids:                 # derangement: every scene gets another scene's image
    other = sid
    while other == sid:
        other = rng.choice(scene_ids)
    swap_of[sid] = scene_row[other]

train_rows = read_jsonl(DATA / "shapes" / "train.jsonl")
majority = {t: Counter(r["answer"] for r in train_rows if r["type"] == t).most_common(1)[0][0]
            for t in {r["type"] for r in train_rows}}
summary = {"name": args.name, "model": resolve(args.model), "adapter": args.adapter, "split": args.split,
           "processor_overrides": proc_kwargs, "max_new_tokens": VLM_MAX_NEW_TOKENS, "decoding": "greedy",
           "env": environment_info(), "conditions": {}}
majority_acc = sum(majority[r["type"]] == r["answer"] for r in rows) / len(rows)
summary["majority_baseline_accuracy"] = round(majority_acc, 4)
preds_correct = None
out_dir = OUTPUTS / "evals" / args.name
for cond in args.conditions:
    use = rows if cond != "swapped" else [r for r in rows if r["type"] in st.ANY_IMAGE_TYPES]
    images = [load_image(r, cond, swap_of.get(r["scene_id"])) for r in use]
    g = generate(model, processor, images, [r["question"] for r in use], batch_size=args.batch_size)
    recs, by_type = [], defaultdict(list)
    for r, text, ntok, nimg in zip(use, g["texts"], g["new_tokens"], g["image_tokens"] or [None] * len(use)):
        if cond == "swapped":
            expected = st.answer_for(swap_of[r["scene_id"]]["scene"], r["q"])
        elif cond == "blank":
            expected = st.answer_for({"objects": []}, r["q"]) if r["type"] in st.ANY_IMAGE_TYPES else None
        else:
            expected = r["answer"]
        v = st.verify(r["type"], expected, text)
        v_orig = st.verify(r["type"], r["answer"], text)
        rec = {"id": r["id"], "type": r["type"], "condition": cond, "question": r["question"], "raw_output": text,
               "pred": v["pred"], "expected_for_this_image": expected, "original_answer": r["answer"],
               "valid": v["valid"], "correct": v["correct"], "matches_original_answer": v_orig["correct"],
               "new_tokens": ntok, "image_tokens": nimg}
        recs.append(rec)
        by_type[r["type"]].append(rec)
    write_jsonl(out_dir / f"{args.split}_{cond}.jsonl", recs)
    n = len(recs)
    c = {"n": n, "accuracy": round(sum(x["correct"] for x in recs) / n, 4) if cond != "blank" else None,
         "valid_rate": round(sum(x["valid"] for x in recs) / n, 4),
         "matches_original_answer": round(sum(x["matches_original_answer"] for x in recs) / n, 4),
         "by_type": {t: round(sum(x["correct"] for x in v) / len(v), 4) for t, v in sorted(by_type.items())},
         "seconds": round(g["seconds"], 1), "ms_per_question": round(1000 * g["seconds"] / n, 1),
         "mean_image_tokens": round(sum(g["image_tokens"]) / len(g["image_tokens"]), 1) if g["image_tokens"] else None}
    if cond == "blank":
        defined = [x for x in recs if x["expected_for_this_image"] is not None]
        c["blank_truth_accuracy (counts=0, exists=no)"] = round(sum(x["correct"] for x in defined) / max(1, len(defined)), 4)
    if cond == "correct":
        preds_correct = {x["id"]: x["pred"] for x in recs}
    elif preds_correct:
        same = [preds_correct[x["id"]] == x["pred"] for x in recs if x["id"] in preds_correct]
        c["unchanged_vs_correct_image"] = round(sum(same) / max(1, len(same)), 4)
    summary["conditions"][cond] = c
    print(f"{args.name:24s} {cond:8s} n={n:4d} acc={c['accuracy']} valid={c['valid_rate']} "
          f"matches_orig={c['matches_original_answer']} unchanged={c.get('unchanged_vs_correct_image')} "
          f"img_tokens={c['mean_image_tokens']} {c['ms_per_question']}ms/q")
    print("   by type:", c["by_type"])
print(f"majority-answer baseline (train priors): {summary['majority_baseline_accuracy']}")
summary["vram"] = mem.snapshot()
write_json(out_dir / "metrics.json", summary)
append_experiment_index({"experiment": "vlm_eval", "name": args.name, "adapter": args.adapter,
                         "processor_overrides": proc_kwargs,
                         **{f"{k}_acc": v["accuracy"] for k, v in summary["conditions"].items()},
                         **{f"{k}_matches_orig": v["matches_original_answer"] for k, v in summary["conditions"].items()},
                         "swapped_unchanged": summary["conditions"].get("swapped", {}).get("unchanged_vs_correct_image"),
                         "majority_baseline": summary["majority_baseline_accuracy"],
                         "peak_allocated_gb": summary["vram"].get("peak_allocated_gb")})
print("saved", out_dir)
