"""Lesson D1: sequence-level distillation from a different-tokenizer teacher.

    .venv/bin/python scripts/text/20_distill_sequence.py --preset quick
    .venv/bin/python scripts/text/20_distill_sequence.py --set teacher_thinking=true --run-name qwen_thinking

Pipeline
  1. Profile the teacher (load time, VRAM, latency) and evaluate it on held-out test scenes.
  2. Generate teacher answers for the SAME training prompts that gold SFT uses,
     recording provenance (model, revision, decoding settings, raw text).
  3. Filter WITHOUT gold labels: keep answers that parse, name an eligible
     robot, and report that robot's true distance (lab.robot_task.label_free_check).
     Gold labels are used afterwards only to *measure* how good the kept labels are.
  4. Unload the teacher and confirm VRAM is released.
  5. Train the student with the ordinary SFT script on the teacher's answers
     (separate process, so no teacher memory can leak into training).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab import robot_task as rt  # noqa: E402
from lab.common import (DATA, ROOT, GpuMemory, append_experiment_index, base_argparser, environment_info,  # noqa: E402
                        load_config, make_run_dir, read_jsonl, set_seed, write_json, write_jsonl)
from lab.models import resolve  # noqa: E402
from lab.text_eval import generate  # noqa: E402

parser = base_argparser(__doc__, "distill_sequence.yaml")
parser.add_argument("--no-train", action="store_true", help="only build the distillation dataset")
parser.add_argument("--eval-only", action="store_true", help="only profile + evaluate the teacher (no labels)")
args = parser.parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
run_dir = make_run_dir("text_distill_sequence", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info()}
repo, rev = resolve(cfg["teacher_model"])
metrics["teacher"] = {"repo": repo, "revision": rev}

# ------------------------------------------------- 1. load + profile teacher ----
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
try:
    teacher = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).cuda()
    metrics["teacher"]["loaded_as"] = type(teacher).__name__
except (ValueError, KeyError) as e:   # multimodal-only checkpoints: load the full VLM, use it text-only
    teacher = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).cuda()
    metrics["teacher"]["loaded_as"] = type(teacher).__name__ + f" (text-only use; CausalLM failed: {e!s:.80})"
teacher.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
metrics["teacher"]["load_seconds"] = round(time.perf_counter() - t0, 1)
metrics["teacher"]["params"] = sum(p.numel() for p in teacher.parameters())
metrics["teacher"]["vram_after_load"] = mem.snapshot()
print("teacher:", metrics["teacher"])

template_kwargs = {"enable_thinking": bool(cfg["teacher_thinking"])}


def teacher_prompts(rows):
    return [tok.apply_chat_template(rt.chat_messages(r), add_generation_prompt=True, tokenize=False,
                                    **template_kwargs) for r in rows]


def final_answer(text):
    """Strip a reasoning block if the teacher produced one; keep the final answer."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ----------------------------------------------------- 2. evaluate teacher ----
test_rows = read_jsonl(DATA / "robot" / "test.jsonl")[: cfg["teacher_eval_n"]]
mem.reset()
g = generate(teacher, tok, teacher_prompts(test_rows), max_new_tokens=cfg["max_new_tokens"],
             batch_size=cfg["batch_size"])
res = [rt.verify(r, final_answer(t[0])) for r, t in zip(test_rows, g["texts"])]
metrics["teacher_eval_test"] = {**rt.aggregate(res), "seconds": round(g["seconds"], 1),
                                "ms_per_example": round(1000 * g["seconds"] / len(test_rows), 1),
                                "mean_new_tokens": round(sum(n[0] for n in g["new_tokens"]) / len(test_rows), 1),
                                **mem.snapshot()}
write_jsonl(run_dir / "teacher_eval_test.jsonl",
            [{"id": r["id"], "raw_output": t[0], **v} for r, t, v in zip(test_rows, g["texts"], res)])
print("teacher on test:", {k: metrics["teacher_eval_test"][k] for k in ("exact", "robot_correct", "parsed",
                                                                         "ms_per_example", "mean_new_tokens")})

if args.eval_only:
    write_json(run_dir / "metrics.json", metrics)
    append_experiment_index({"experiment": "text_teacher_eval", "run_dir": str(run_dir), "teacher": repo,
                             "thinking": cfg["teacher_thinking"], "n": len(test_rows),
                             **{k: metrics["teacher_eval_test"][k] for k in ("exact", "robot_correct", "parsed",
                                                                            "ms_per_example", "mean_new_tokens")}})
    raise SystemExit(0)

# ------------------------------------------------- 3. generate + filter labels ----
train_rows = read_jsonl(DATA / "robot" / "train.jsonl")[: cfg["n_train"]]
t0 = time.perf_counter()
g = generate(teacher, tok, teacher_prompts(train_rows), max_new_tokens=cfg["max_new_tokens"],
             batch_size=cfg["batch_size"])
gen_seconds = time.perf_counter() - t0
kept, all_rows = [], []
for r, t in zip(train_rows, g["texts"]):
    ans_text = final_answer(t[0])
    check = rt.label_free_check(r, ans_text)
    obj, _ = rt.parse_completion(ans_text)
    gold_ok = rt.verify(r, ans_text)["exact"]
    rec = {"id": r["id"], "teacher_raw": t[0], "label_free": check, "gold_exact_for_analysis": gold_ok}
    all_rows.append(rec)
    if check["ok"]:
        new = dict(r)
        new["gold_answer"] = r["answer"]
        new["answer"] = {"robot": obj["robot"], "distance": obj["distance"]}
        new["provenance"] = {"teacher": repo, "revision": rev, "thinking": cfg["teacher_thinking"],
                             "decoding": "greedy", "max_new_tokens": cfg["max_new_tokens"],
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": t[0]}
        kept.append(new)
write_jsonl(run_dir / "teacher_train_outputs.jsonl", all_rows)
out_file = DATA / "distill" / f"{cfg['dataset_name']}.jsonl"
write_jsonl(out_file, kept)
metrics["distill_data"] = {
    "generated": len(train_rows), "kept_by_label_free_filter": len(kept),
    "keep_rate": round(len(kept) / len(train_rows), 4),
    "kept_label_accuracy_vs_gold (analysis only)": round(sum(k["answer"] == k["gold_answer"] for k in kept) /
                                                         max(1, len(kept)), 4),
    "all_outputs_accuracy_vs_gold": round(sum(a["gold_exact_for_analysis"] for a in all_rows) / len(all_rows), 4),
    "generation_seconds": round(gen_seconds, 1), "file": str(out_file.relative_to(ROOT)),
}
print("distillation data:", metrics["distill_data"])

# --------------------------------------------------------- 4. unload teacher ----
before = torch.cuda.memory_allocated() / 1e9
del teacher
torch.cuda.empty_cache()
metrics["unload"] = {"allocated_before_gb": round(before, 3),
                     "allocated_after_gb": round(torch.cuda.memory_allocated() / 1e9, 3)}
print("teacher unloaded:", metrics["unload"])
write_json(run_dir / "metrics.json", metrics)

# ------------------------------------------------------ 5. train the student ----
if not args.no_train:
    student_run = f"distilled_{cfg['dataset_name']}_{cfg['_preset']}"
    cmd = [sys.executable, str(ROOT / "scripts/text/10_sft.py"), "--preset", cfg["_preset"],
           "--train-file", str(out_file.relative_to(DATA)), "--run-name", student_run,
           "--experiment", "text_sft_distilled", "--set", f"n_train={len(kept)}"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    metrics["student_run"] = str(ROOT / "outputs/runs/text_sft_distilled" / student_run)
    write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "text_distill_sequence", "run_dir": str(run_dir), "teacher": repo,
                         "thinking": cfg["teacher_thinking"], "teacher_test_exact": metrics["teacher_eval_test"]["exact"],
                         **metrics["distill_data"]})
print("saved", run_dir)
