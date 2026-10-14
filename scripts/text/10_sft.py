"""Lesson C: supervised fine-tuning (LoRA or bounded full fine-tuning) with TRL's SFTTrainer.

    .venv/bin/python scripts/text/10_sft.py --preset quick                       # LoRA
    .venv/bin/python scripts/text/10_sft.py --set lora_r=4 --run-name lora_r4      # one-variable change
    .venv/bin/python scripts/text/10_sft.py --set method=full --set learning_rate=2e-5 --set optim=adamw_8bit \
        --run-name full_ft                                                        # full fine-tuning
    .venv/bin/python scripts/text/10_sft.py --run-name myrun --resume              # resume from last checkpoint

What it records (outputs/runs/text_sft/<run>/):
    config.json, metrics.json, train_log.jsonl, batch_inspection.json,
    eval_test.jsonl / eval_test_ood.jsonl (raw outputs), adapter/ (LoRA weights)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

from lab import robot_task as rt  # noqa: E402
from lab.common import (DATA, GpuMemory, append_experiment_index, base_argparser, count_parameters,  # noqa: E402
                        environment_info, free_gpu, load_config, make_run_dir, read_jsonl, require_disk, set_seed,
                        write_json, write_jsonl)
from lab.models import load_text_model, resolve  # noqa: E402
from lab.text_eval import evaluate_robot  # noqa: E402

parser = base_argparser(__doc__, "sft.yaml")
parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint of --run-name")
parser.add_argument("--train-file", default="robot/train.jsonl", help="relative to data/")
parser.add_argument("--experiment", default="text_sft")
args = parser.parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
require_disk(3.0, "training outputs")
run_dir = make_run_dir(args.experiment, args.run_name or f"{cfg['method']}_{cfg['_preset']}")
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "model": resolve(cfg["model"])}


def to_pc(ex):
    """Conversational prompt-completion format: TRL applies the chat template and
    (with completion_only_loss) masks every prompt token out of the loss."""
    return {"prompt": rt.chat_messages(ex),
            "completion": [{"role": "assistant", "content": rt.format_answer(ex["answer"])}]}


train_rows = read_jsonl(DATA / args.train_file)[: cfg["n_train"]]
val_rows = read_jsonl(DATA / "robot" / "val.jsonl")[: cfg["n_val"]]
train_ds = Dataset.from_list([to_pc(r) for r in train_rows])
val_ds = Dataset.from_list([to_pc(r) for r in val_rows])
print(f"train {len(train_ds)} | val {len(val_ds)}")

# --------------------------------------------------------------- model ----
full = cfg["method"] == "full"
# Full fine-tuning keeps fp32 master weights (bf16 weights would round away
# small updates); LoRA can keep the frozen base in bf16.
if cfg.get("load_in_4bit"):
    # QLoRA: frozen base stored in 4-bit NF4, LoRA adapters trained in higher precision.
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    repo, rev = resolve(cfg["model"])
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, device_map={"": 0}, quantization_config=
                                                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                                                    bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
else:
    model, tok = load_text_model(cfg["model"], dtype="float32" if full else "bfloat16")
peft_config = None
if not full:
    from peft import LoraConfig
    peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                             target_modules=cfg["target_modules"], task_type="CAUSAL_LM", bias="none")

from trl import SFTConfig, SFTTrainer  # noqa: E402

sft_args = SFTConfig(
    output_dir=str(run_dir / "trainer"),
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    max_steps=cfg["max_steps"],
    lr_scheduler_type=cfg["lr_scheduler_type"],
    warmup_steps=cfg["warmup_ratio"],  # a float < 1 means a fraction of total steps
    weight_decay=cfg["weight_decay"],
    optim=cfg["optim"],
    bf16=True,
    gradient_checkpointing=cfg["gradient_checkpointing"],
    completion_only_loss=cfg["completion_only_loss"],
    max_length=cfg["max_length"],
    logging_steps=cfg["logging_steps"],
    eval_strategy="steps",
    eval_steps=cfg["eval_steps"],
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=1,
    report_to="none",
    seed=cfg["seed"],
    dataloader_num_workers=0,
)
trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=val_ds,
                     processing_class=tok, peft_config=peft_config)
pm = trainer.model
metrics["params"] = count_parameters(pm)
print(f"trainable parameters: {metrics['params']['trainable_params']:,} of {metrics['params']['total_params']:,} "
      f"({metrics['params']['trainable_percent']}%)")
if not full:
    lora_names = [n for n, _ in pm.named_parameters() if "lora_" in n]
    metrics["lora_modules_wrapped"] = sorted({n.split(".lora_")[0].split(".")[-1] for n in lora_names})
    print("LoRA wrapped module types:", metrics["lora_modules_wrapped"])

# ---------------------------------------------- inspect one collated batch ----
batch = next(iter(trainer.get_train_dataloader()))
ids, labels = batch["input_ids"], batch["labels"]
row_labels = labels[0][labels[0] != -100]
inspection = {
    "keys": list(batch.keys()), "input_ids_shape": list(ids.shape),
    "attention_mask_zeros (padding)": int((batch["attention_mask"] == 0).sum()),
    "trained_tokens_row0": int((labels[0] != -100).sum()), "total_tokens_row0": int(batch["attention_mask"][0].sum()),
    "row0_full_text": tok.decode(ids[0][batch["attention_mask"][0].bool()]),
    "row0_trained_text": tok.decode(row_labels),
    "labels_equal_input_ids_where_trained": bool(torch.equal(labels[0][labels[0] != -100], ids[0][labels[0] != -100])),
}
write_json(run_dir / "batch_inspection.json", inspection)
print(f"\nbatch input_ids {inspection['input_ids_shape']}; row 0 trains on {inspection['trained_tokens_row0']} of "
      f"{inspection['total_tokens_row0']} tokens")
print("row 0 tokens that receive loss:", repr(inspection["row0_trained_text"]))
print("(the model shifts labels internally: logits at position t are scored against labels at t+1)")

# --------------------------------------- gradient sanity check (one batch) ----
pm.train()
dev_batch = {k: v.to(pm.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss0 = pm(**dev_batch).loss
loss0.backward()
grad_report = {"loss_before_training": loss0.item()}
named = dict(pm.named_parameters())
if not full:
    a_name = next(n for n in named if "lora_A" in n and named[n].requires_grad)
    b_name = a_name.replace("lora_A", "lora_B")
    grad_report.update({
        "example_lora_A": a_name, "grad_norm_A": named[a_name].grad.norm().item(),
        "grad_norm_B": named[b_name].grad.norm().item(),
        "B_is_zero_at_init": bool((named[b_name] == 0).all()),
        "frozen_params_with_grad": sum(1 for n, p_ in named.items() if not p_.requires_grad and p_.grad is not None),
    })
    watch_name = b_name
    base_name = next(n for n in named if "base_layer.weight" in n)
    base_before = named[base_name].detach().clone()
else:
    watch_name = next(n for n in named if "self_attn.q_proj.weight" in n)
    grad_report["grad_norm_q_proj"] = named[watch_name].grad.norm().item()
grads = [p_.grad.norm().item() for p_ in named.values() if p_.grad is not None]
grad_report["n_params_with_grad"] = len(grads)
grad_report["n_zero_grad"] = sum(g == 0 for g in grads)
pm.zero_grad(set_to_none=True)
watch_before = named[watch_name].detach().float().clone()
metrics["gradient_check"] = grad_report
print("\ngradient check:", grad_report)

# ------------------------------------------------------------------ train ----
mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
train_out = trainer.train(resume_from_checkpoint=True if args.resume else None)
train_seconds = time.perf_counter() - t0
metrics["train"] = {**train_out.metrics, "wall_seconds": round(train_seconds, 1), **mem.snapshot()}
write_jsonl(run_dir / "train_log.jsonl", trainer.state.log_history)
named = dict(pm.named_parameters())
metrics["param_change"] = {"watched": watch_name,
                           "l2_change": (named[watch_name].detach().float() - watch_before).norm().item()}
if not full:
    metrics["param_change"]["frozen_base_weight_unchanged"] = bool(torch.equal(named[base_name].detach(), base_before))
print("\ntrain:", {k: metrics["train"][k] for k in ("train_runtime", "train_loss", "peak_allocated_gb",
                                                       "peak_reserved_gb") if k in metrics["train"]})
print("parameter change:", metrics["param_change"])
# TRL counts the (non-padding) tokens it has trained on in `num_tokens`.
seen_tokens = max((h.get("num_tokens", 0) for h in trainer.state.log_history), default=0)
metrics["train"]["tokens_seen"] = seen_tokens
metrics["train"]["tokens_per_second"] = round(seen_tokens / train_seconds, 1)

# ------------------------------------------------------------------- save ----
save_dir = run_dir / ("model" if full else "adapter")
if full and not cfg["save_full_model"]:
    save_dir = None
else:
    pm.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    metrics["saved_to"] = str(save_dir)
    metrics["saved_mb"] = round(sum(f.stat().st_size for f in Path(save_dir).rglob("*") if f.is_file()) / 1e6, 1)

# ------------------------------------------------------------------- eval ----
pm.eval()
if hasattr(pm, "config"):
    pm.config.use_cache = True
eval_sets = {"test": read_jsonl(DATA / "robot" / "test.jsonl")[: cfg["eval_limit_test"]],
             "test_ood": read_jsonl(DATA / "robot" / "test_ood.jsonl")[: cfg["eval_limit_ood"]]}
if "overfit" in args.train_file:
    eval_sets["train_tiny"] = train_rows
metrics["eval"] = {}
raw_test = None
for split, rows in eval_sets.items():
    agg, raw = evaluate_robot(pm, tok, rows)
    write_jsonl(run_dir / f"eval_{split}.jsonl", raw)
    metrics["eval"][split] = agg
    raw_test = raw if split == "test" else raw_test
    print(f"eval {split:10s} exact={agg['exact']:.3f} robot={agg['robot_correct']:.3f} "
          f"dist={agg['distance_correct']:.3f} strict={agg['valid_strict']:.3f} trap_exact={agg['trap_exact']:.3f}")

# ------------------------------------------- reload from disk and compare ----
if save_dir is not None and not full:
    free_gpu(trainer, pm, model)
    del trainer, pm, model
    torch.cuda.empty_cache()
    from peft import PeftModel
    t0 = time.perf_counter()
    base, tok2 = load_text_model(cfg["model"])
    reloaded = PeftModel.from_pretrained(base, save_dir)
    reload_s = time.perf_counter() - t0
    n_check = 64
    agg_r, raw_r = evaluate_robot(reloaded, tok2, eval_sets["test"][:n_check])
    same = sum(a["raw_output"] == b["raw_output"] for a, b in zip(raw_r, raw_test[:n_check]))
    metrics["reload_check"] = {"seconds": round(reload_s, 2), "n": n_check, "identical_outputs": same,
                               "exact_after_reload": agg_r["exact"]}
    print(f"reload check: {same}/{n_check} identical greedy outputs after reloading the adapter "
          f"({reload_s:.1f}s to load)")

write_json(run_dir / "metrics.json", metrics)
append_experiment_index({
    "experiment": args.experiment, "run_dir": str(run_dir), "method": cfg["method"], "preset": cfg["_preset"],
    "overrides": cfg["_overrides"], "n_train": len(train_rows), "trainable_params": metrics["params"]["trainable_params"],
    "train_seconds": metrics["train"]["wall_seconds"], "train_loss": metrics["train"].get("train_loss"),
    "peak_allocated_gb": metrics["train"].get("peak_allocated_gb"), "peak_reserved_gb": metrics["train"].get("peak_reserved_gb"),
    **{f"{s}_{k}": v[k] for s, v in metrics["eval"].items() for k in ("exact", "robot_correct", "distance_correct",
                                                                         "valid_strict", "violation_status", "violation_type")}})
print("saved", run_dir)
