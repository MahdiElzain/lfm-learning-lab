"""VLM Lesson B: LoRA SFT of LFM2.5-VL on shape questions (vision encoder frozen).

    .venv/bin/python scripts/vlm/10_sft.py --preset quick --run-name lora_quick
    .venv/bin/python scripts/vlm/10_sft.py --set train_connector=true --run-name lora_connector_quick
    .venv/bin/python scripts/vlm/10_sft.py --train-file shapes_distill/qwen_vlm.jsonl --run-name distilled

Trainable parts:
    LoRA on language-model linear layers (attention q/k/v/out, conv in/out, MLP w1/w2/w3)
    + optionally the whole multi_modal_projector (PEFT modules_to_save; saved with the adapter)
    The SigLIP2 vision tower is always frozen.
After training, the adapter is reloaded from disk by scripts/vlm/evaluate.py in a
fresh process (correct / swapped / blank conditions).
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from lab.common import (DATA, ROOT, GpuMemory, append_experiment_index, base_argparser, environment_info,  # noqa: E402
                        load_config, make_run_dir, read_json, read_jsonl, require_disk, set_seed, write_json,
                        write_jsonl)
from lab.models import load_vlm, resolve  # noqa: E402
from lab.vlm_utils import LFM2VL_LM_TARGETS, SftCollator  # noqa: E402

parser = base_argparser(__doc__, "vlm_sft.yaml")
parser.add_argument("--train-file", default="shapes/train.jsonl", help="relative to data/")
parser.add_argument("--no-eval", action="store_true")
args = parser.parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
require_disk(3.0)
run_dir = make_run_dir("vlm_sft", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "model": resolve(cfg["model"])}

model, processor = load_vlm(cfg["model"], processor_kwargs={"max_image_tokens": cfg["max_image_tokens"]})


class ShapeQA(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img_path = DATA / (r["image"] if r["image"].startswith("shapes") else "shapes/" + r["image"])
        return {"image": Image.open(img_path).convert("RGB"), "question": r["question"], "answer": r["answer"]}


train_rows = read_jsonl(DATA / args.train_file)[: cfg["n_train"]]
val_rows = read_jsonl(DATA / "shapes" / "val.jsonl")[: cfg["n_val"]]
train_ds, val_ds = ShapeQA(train_rows), ShapeQA(val_rows)
print(f"train questions {len(train_ds)} (scenes {len({r['scene_id'] for r in train_rows})}) | val {len(val_ds)}")

# ------------------------------------------------------------ adapters ----
from peft import LoraConfig, get_peft_model  # noqa: E402

lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                  target_modules=LFM2VL_LM_TARGETS,
                  modules_to_save=["multi_modal_projector"] if cfg["train_connector"] else None)
model = get_peft_model(model, lcfg)
by_part = {"vision_tower": 0, "multi_modal_projector": 0, "language_model": 0, "other": 0}
for n, prm in model.named_parameters():
    if prm.requires_grad:
        part = next((k for k in by_part if k in n), "other")
        by_part[part] += prm.numel()
metrics["trainable_by_component"] = by_part
metrics["trainable_params"] = sum(by_part.values())
metrics["total_params"] = sum(p.numel() for p in model.parameters())
print("trainable parameters by component:", by_part)
assert by_part["vision_tower"] == 0, "vision tower must stay frozen"

# ------------------------------------------------------ inspect collator ----
collator = SftCollator(processor, max_length=cfg["max_length"])
probe = collator([train_ds[i] for i in range(4)])
check = collator.check(probe)
metrics["collator_check"] = check
print("collator check:", {k: v for k, v in check.items() if k != "trained_text_per_row"})
print("trained text per row:", check["trained_text_per_row"])
assert check["image_tokens_receiving_loss"] == 0 and check["prompt_is_prefix_of_full"]


def train_collate(examples):
    b = collator(examples)
    b.pop("_prefix_ok", None)
    if "pixel_values" in b:
        b["pixel_values"] = b["pixel_values"].to(torch.bfloat16)
    return b


# ------------------------------------------------------- gradient check ----
model.train()
gb = {k: v.to(model.device) for k, v in train_collate([train_ds[i] for i in range(4)]).items()}
loss0 = model(**gb).loss
loss0.backward()
named = dict(model.named_parameters())
lora_b = [n for n in named if "lora_B" in n and named[n].grad is not None]
metrics["gradient_check"] = {
    "loss_before_training": loss0.item(),
    "lora_B_with_nonzero_grad": sum(named[n].grad.norm().item() > 0 for n in lora_b), "lora_B_total": len(lora_b),
    "vision_params_with_grad": sum(1 for n, p_ in named.items() if "vision_tower" in n and p_.grad is not None),
}
watch = lora_b[0]
model.zero_grad(set_to_none=True)
before = named[watch].detach().float().clone()
print("gradient check:", metrics["gradient_check"])

# ----------------------------------------------------------------- train ----
from transformers import Trainer, TrainingArguments  # noqa: E402

targs = TrainingArguments(
    output_dir=str(run_dir / "trainer"), per_device_train_batch_size=cfg["batch_size"],
    per_device_eval_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=cfg["learning_rate"], num_train_epochs=cfg["epochs"], max_steps=cfg["max_steps"],
    lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=5, eval_strategy="steps",
    eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"], save_total_limit=1,
    bf16=True, report_to="none", remove_unused_columns=False, seed=cfg["seed"], dataloader_num_workers=2,
    gradient_checkpointing=cfg["gradient_checkpointing"],
)
trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=train_collate)
mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
out = trainer.train()
metrics["train"] = {**out.metrics, "wall_seconds": round(time.perf_counter() - t0, 1), **mem.snapshot()}
write_jsonl(run_dir / "train_log.jsonl", trainer.state.log_history)
metrics["param_change"] = {"watched": watch, "l2_change": (named[watch].detach().float() - before).norm().item()}
print("train:", {k: metrics["train"].get(k) for k in ("train_runtime", "train_loss", "peak_allocated_gb",
                                                       "peak_reserved_gb")})
adapter_dir = run_dir / "adapter"
model.save_pretrained(adapter_dir)
saved = sorted(p_.name for p_ in adapter_dir.iterdir())
metrics["adapter_files"] = saved
metrics["adapter_mb"] = round(sum(f.stat().st_size for f in adapter_dir.iterdir()) / 1e6, 2)
if cfg["train_connector"]:
    from safetensors import safe_open
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        metrics["connector_tensors_in_adapter"] = [k for k in f.keys() if "multi_modal_projector" in k]
    print("connector weights saved with adapter:", metrics["connector_tensors_in_adapter"])
write_json(run_dir / "metrics.json", metrics)

# ------------------------------------------- evaluate in a fresh process ----
if not args.no_eval:
    del trainer, model
    torch.cuda.empty_cache()
    name = f"vlm_sft_{run_dir.name}"
    subprocess.run([sys.executable, str(ROOT / "scripts/vlm/evaluate.py"), "--adapter", str(adapter_dir),
                    "--name", name, "--set-processor", f"max_image_tokens={cfg['max_image_tokens']}"], check=True)
    metrics["eval"] = read_json(ROOT / "outputs/evals" / name / "metrics.json")["conditions"]
    write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "vlm_sft", "run_dir": str(run_dir), "train_connector": cfg["train_connector"],
                         "train_file": args.train_file, "n_train": len(train_rows),
                         "trainable_params": metrics["trainable_params"],
                         "train_seconds": metrics["train"]["wall_seconds"],
                         "peak_allocated_gb": metrics["train"].get("peak_allocated_gb"),
                         **{f"{k}_acc": v.get("accuracy") for k, v in metrics.get("eval", {}).items()}})
print("saved", run_dir)
