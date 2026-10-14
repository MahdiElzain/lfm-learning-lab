"""Lesson E: Direct Preference Optimization on SYNTHETIC, verifier-derived preferences.

    .venv/bin/python scripts/text/30_dpo.py --set sft_adapter=outputs/runs/text_sft/lora_quick/adapter

Step 1 - build pairs (no human labels; everything comes from the verifier):
    chosen   = the verified-correct answer (canonical JSON)
    rejected = (a) the SFT model's own sampled answer that the verifier marks wrong
               ("on_policy_error"), or, if every sample was right,
               (b) a rule-based corruption: nearest-but-ineligible robot,
               eligible-but-farther robot, or right robot with an off-by-k distance.
    The source of every rejected answer is recorded in data/robot/dpo_pairs.jsonl.

Step 2 - DPO. The SFT LoRA is merged into the weights; a fresh LoRA is the
trainable policy pi_theta, and disabling it gives the reference pi_ref (= SFT
model). At step 0 the two are identical, so the loss must be log(2) = 0.693.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

from lab import robot_task as rt  # noqa: E402
from lab.common import (DATA, GpuMemory, append_experiment_index, base_argparser, count_parameters,  # noqa: E402
                        environment_info, load_config, make_run_dir, read_jsonl, require_disk, set_seed, write_json,
                        write_jsonl)
from lab.models import load_text_model, resolve  # noqa: E402
from lab.text_eval import evaluate_robot, generate, robot_prompts  # noqa: E402

args = base_argparser(__doc__, "dpo.yaml").parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
require_disk(3.0)
run_dir = make_run_dir("text_dpo", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "model": resolve(cfg["model"])}

from peft import LoraConfig, PeftModel  # noqa: E402

model, tok = load_text_model(cfg["model"])
if cfg.get("sft_adapter"):
    model = PeftModel.from_pretrained(model, cfg["sft_adapter"]).merge_and_unload()


# ------------------------------------------------------- 1. build pairs ----
def corruptions(ex, rng):
    robots, (tx, ty) = ex["robots"], ex["target"]
    ans = ex["answer"]
    allowed = rt.TASK_KINDS[ex["kind"]][1]
    dist = {r["id"]: rt.manhattan(r["x"], r["y"], tx, ty) for r in robots}
    out = []
    inel = [r for r in robots if r["status"] != "idle" or r["type"] not in allowed]
    if inel:
        r = min(inel, key=lambda r: dist[r["id"]])
        out.append(("nearest_ineligible", {"robot": r["id"], "distance": dist[r["id"]]}))
    farther = [r for r in robots if r["status"] == "idle" and r["type"] in allowed and r["id"] != ans["robot"]]
    if farther:
        r = rng.choice(farther)
        out.append(("eligible_farther", {"robot": r["id"], "distance": dist[r["id"]]}))
    k = rng.choice([-2, -1, 1, 2])
    out.append(("distance_off_by_k", {"robot": ans["robot"], "distance": max(0, ans["distance"] + k)}))
    return out


rows = read_jsonl(DATA / "robot" / "train.jsonl")[cfg["prompt_offset"]: cfg["prompt_offset"] + cfg["n_pairs"]]
rng = random.Random(cfg["seed"])
gen = generate(model, tok, robot_prompts(tok, rows), do_sample=True, temperature=cfg["sample_temperature"],
               top_p=1.0, num_return_sequences=cfg["samples_per_prompt"], seed=cfg["seed"], batch_size=16)
pairs, sources = [], Counter()
for ex, samples in zip(rows, gen["texts"]):
    chosen = rt.format_answer(ex["answer"])
    wrong = [s for s in samples if not rt.verify(ex, s)["exact"]]
    if wrong and cfg["use_on_policy_errors"]:
        rejected, src = rng.choice(wrong), "on_policy_error"
    else:
        src, obj = rng.choice(corruptions(ex, rng))
        rejected = json.dumps(obj)
    assert rt.verify(ex, chosen)["exact"] and not rt.verify(ex, rejected)["exact"]
    sources[src] += 1
    pairs.append({"id": ex["id"], "prompt": rt.chat_messages(ex), "chosen": [{"role": "assistant", "content": chosen}],
                  "rejected": [{"role": "assistant", "content": rejected}], "rejected_source": src,
                  "synthetic": True, "label_rule": "verifier exact-match: chosen passes, rejected fails"})
write_jsonl(DATA / "robot" / "dpo_pairs.jsonl", pairs)
metrics["pairs"] = {"n": len(pairs), "rejected_sources": dict(sources),
                    "sft_sample_accuracy": round(sum(rt.verify(ex, s)["exact"] for ex, ss in zip(rows, gen["texts"])
                                                     for s in ss) / (len(rows) * cfg["samples_per_prompt"]), 4)}
print("pairs:", metrics["pairs"])

# ------------------------------------------------------------- 2. DPO ----
from trl import DPOConfig, DPOTrainer  # noqa: E402

ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
dargs = DPOConfig(
    output_dir=str(run_dir / "trainer"), beta=cfg["beta"], learning_rate=cfg["learning_rate"],
    loss_type=cfg["loss_type"], loss_weights=cfg["loss_weights"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"], num_train_epochs=cfg["num_train_epochs"],
    max_steps=cfg["max_steps"], lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=2, bf16=True,
    save_strategy="no", report_to="none", seed=cfg["seed"], max_length=cfg["max_length"],
)
peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                         target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
trainer = DPOTrainer(model=model, ref_model=None, args=dargs, train_dataset=ds, processing_class=tok,
                     peft_config=peft_config)
metrics["params"] = count_parameters(trainer.model)
mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
out = trainer.train()
metrics["train"] = {**out.metrics, "wall_seconds": round(time.perf_counter() - t0, 1), **mem.snapshot()}
hist = trainer.state.log_history
write_jsonl(run_dir / "train_log.jsonl", hist)
logged = [h for h in hist if "loss" in h]
if logged:
    metrics["first_logged"] = logged[0]
    metrics["last_logged"] = logged[-1]
    print("first log:", {k: round(v, 4) for k, v in logged[0].items() if isinstance(v, float)})
    print("last  log:", {k: round(v, 4) for k, v in logged[-1].items() if isinstance(v, float)})

pm = trainer.model
pm.eval()
pm.save_pretrained(run_dir / "adapter")
metrics["eval"] = {}
for split, limit in (("test", cfg["eval_limit_test"]), ("test_ood", cfg["eval_limit_ood"])):
    agg, raw = evaluate_robot(pm, tok, read_jsonl(DATA / "robot" / f"{split}.jsonl")[:limit])
    write_jsonl(run_dir / f"eval_{split}.jsonl", raw)
    metrics["eval"][split] = agg
    print(f"eval {split:9s} exact={agg['exact']:.3f} robot={agg['robot_correct']:.3f} dist={agg['distance_correct']:.3f}")
write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "text_dpo", "run_dir": str(run_dir), "beta": cfg["beta"],
                         "lr": cfg["learning_rate"], "n_pairs": len(pairs), "sources": dict(sources),
                         "train_seconds": metrics["train"]["wall_seconds"],
                         "peak_allocated_gb": metrics["train"].get("peak_allocated_gb"),
                         **{f"{s}_{k}": v[k] for s, v in metrics["eval"].items()
                            for k in ("exact", "robot_correct", "distance_correct")}})
print("saved", run_dir)
