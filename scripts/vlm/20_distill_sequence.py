"""VLM Lesson C: visual sequence distillation (teacher sees the SAME images and questions).

    .venv/bin/python scripts/vlm/20_distill_sequence.py --preset quick

  1. Load the VLM teacher (Qwen3.5-2B, multimodal, Apache-2.0) and evaluate it on the
     test scenes with the correct image AND a blank image (does the teacher look?).
  2. Generate teacher answers for the training questions WITH the real images,
     recording provenance. Text-only guesses are never used as supervision.
  3. Filter without gold labels: keep answers inside the allowed vocabulary
     (a number for counts, yes/no, or a colour).
  4. Unload the teacher; train the student (LFM2.5-VL LoRA) in a separate process.
"""
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab import shapes_task as st  # noqa: E402
from lab.common import (DATA, ROOT, GpuMemory, append_experiment_index, base_argparser, environment_info,  # noqa: E402
                        load_config, make_run_dir, read_jsonl, set_seed, write_json, write_jsonl)
from lab.models import load_vlm, resolve  # noqa: E402
from lab.vlm_utils import generate, load_image  # noqa: E402

parser = base_argparser(__doc__, "vlm_distill.yaml")
parser.add_argument("--no-train", action="store_true")
args = parser.parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
run_dir = make_run_dir("vlm_distill_sequence", args.run_name)
write_json(run_dir / "config.json", cfg)
repo, rev = resolve(cfg["teacher_model"])
metrics = {"config": cfg, "env": environment_info(), "teacher": {"repo": repo, "revision": rev}}
mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
teacher, processor = load_vlm(cfg["teacher_model"])
teacher.eval()
metrics["teacher"].update(load_seconds=round(time.perf_counter() - t0, 1),
                          params=sum(p.numel() for p in teacher.parameters()), vram=mem.snapshot())
tkw = {"enable_thinking": False} if "qwen" in repo.lower() else {}
print("teacher:", metrics["teacher"])

# ------------------------------------------------------------ evaluate ----
test = read_jsonl(DATA / "shapes" / "test.jsonl")[: cfg["teacher_eval_n"]]
for cond in ("correct", "blank"):
    g = generate(teacher, processor, [load_image(r, cond) for r in test], [r["question"] for r in test],
                 batch_size=cfg["batch_size"], template_kwargs=tkw)
    res = [st.verify(r["type"], r["answer"], t) for r, t in zip(test, g["texts"])]
    by_type = {}
    for r, v in zip(test, res):
        by_type.setdefault(r["type"], []).append(v["correct"])
    metrics[f"teacher_eval_{cond}"] = {
        "accuracy_vs_original_answer": round(sum(v["correct"] for v in res) / len(res), 4),
        "valid_rate": round(sum(v["valid"] for v in res) / len(res), 4),
        "by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())},
        "ms_per_question": round(1000 * g["seconds"] / len(test), 1)}
    write_jsonl(run_dir / f"teacher_eval_{cond}.jsonl",
                [{"id": r["id"], "raw": t, **v} for r, t, v in zip(test, g["texts"], res)])
    print(f"teacher [{cond}]:", metrics[f"teacher_eval_{cond}"])

# ------------------------------------------------- generate + filter ----
train = read_jsonl(DATA / "shapes" / "train.jsonl")[: cfg["n_train"]]
t0 = time.perf_counter()
g = generate(teacher, processor, [load_image(r) for r in train], [r["question"] for r in train],
             batch_size=cfg["batch_size"], template_kwargs=tkw)
kept, rejected = [], Counter()
for r, text in zip(train, g["texts"]):
    pred = st.normalize(text)
    if pred not in st.allowed_answers(r["type"]):
        rejected[r["type"]] += 1
        continue
    new = dict(r)
    new["gold_answer"], new["answer"] = r["answer"], pred
    new["provenance"] = {"teacher": repo, "revision": rev, "decoding": "greedy", "raw": text,
                         "image_condition": "correct", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    kept.append(new)
out_file = DATA / "shapes_distill" / f"{cfg['dataset_name']}.jsonl"
write_jsonl(out_file, kept)
metrics["distill_data"] = {
    "generated": len(train), "kept": len(kept), "keep_rate": round(len(kept) / len(train), 4),
    "rejected_by_type": dict(rejected), "generation_seconds": round(time.perf_counter() - t0, 1),
    "kept_accuracy_vs_gold (analysis only)": round(sum(k["answer"] == k["gold_answer"] for k in kept) / max(1, len(kept)), 4),
    "file": str(out_file.relative_to(ROOT))}
print("distillation data:", metrics["distill_data"])

del teacher
torch.cuda.empty_cache()
metrics["unload_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
write_json(run_dir / "metrics.json", metrics)
if not args.no_train:
    run_name = f"distilled_{cfg['dataset_name']}_{cfg['_preset']}"
    cmd = [sys.executable, str(ROOT / "scripts/vlm/10_sft.py"), "--preset", cfg["_preset"], "--train-file",
           str(out_file.relative_to(DATA)), "--run-name", run_name, "--set", f"n_train={len(kept)}"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    metrics["student_run"] = str(ROOT / "outputs/runs/vlm_sft" / run_name)
    write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "vlm_distill_sequence", "run_dir": str(run_dir), "teacher": repo,
                         "teacher_correct_acc": metrics["teacher_eval_correct"]["accuracy_vs_original_answer"],
                         "teacher_blank_acc": metrics["teacher_eval_blank"]["accuracy_vs_original_answer"],
                         **{k: v for k, v in metrics["distill_data"].items() if k != "rejected_by_type"}})
print("saved", run_dir)
