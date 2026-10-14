"""Lesson F: GRPO on robot dispatch, starting from the SFT adapter.

    .venv/bin/python scripts/text/40_grpo.py --set sft_adapter=outputs/runs/text_sft/lora_quick/adapter
    .venv/bin/python scripts/text/40_grpo.py --set reward_mode=binary --run-name grpo_binary
    .venv/bin/python scripts/text/40_grpo.py --set beta=0.04 --run-name grpo_kl

Setup:
  * The SFT LoRA is merged into the base weights, so "the SFT model" becomes a
    plain set of weights. A NEW LoRA is trained on top. With beta > 0 the
    reference policy is that merged SFT model (the new adapter disabled), so no
    second copy of the weights is needed.
  * Reward = lab.robot_task.reward (deterministic verifier; shaped or binary).

See lessons/F_grpo.md for the exact objective used by the installed TRL version.
"""
import json
import sys
import time
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
from lab.text_eval import evaluate_robot  # noqa: E402

args = base_argparser(__doc__, "grpo.yaml").parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
require_disk(3.0)
run_dir = make_run_dir("text_grpo", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "model": resolve(cfg["model"])}

# ------------------------------------------------------------- policy ----
from peft import LoraConfig, PeftModel  # noqa: E402

model, tok = load_text_model(cfg["model"])
if cfg.get("sft_adapter"):
    model = PeftModel.from_pretrained(model, cfg["sft_adapter"]).merge_and_unload()
    print("merged SFT adapter:", cfg["sft_adapter"])
peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                         target_modules=cfg["target_modules"], task_type="CAUSAL_LM")

# --------------------------------------------------------------- data ----
# Use training scenes the SFT run did NOT see, so GRPO gets fresh prompts.
rows = read_jsonl(DATA / "robot" / "train.jsonl")[cfg["prompt_offset"]: cfg["prompt_offset"] + cfg["n_prompts"]]
ds = Dataset.from_list([{"prompt": rt.chat_messages(r), "example": json.dumps(r)} for r in rows])
reward_log = []


def robot_reward(completions, example, **kwargs):
    """completions: list of [{'role': 'assistant', 'content': str}] (conversational prompts)."""
    out = []
    for comp, ex_json in zip(completions, example):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        ex = json.loads(ex_json)
        r = rt.reward(ex, text, mode=cfg["reward_mode"])
        out.append(r)
        reward_log.append({"id": ex["id"], "completion": text, "reward": r})
    return out


from trl import GRPOConfig, GRPOTrainer  # noqa: E402

gargs = GRPOConfig(
    output_dir=str(run_dir / "trainer"),
    learning_rate=cfg["learning_rate"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    num_generations=cfg["num_generations"],
    max_completion_length=cfg["max_completion_length"],
    temperature=cfg["temperature"],
    beta=cfg["beta"],
    epsilon=cfg["epsilon"],
    loss_type=cfg["loss_type"],
    scale_rewards=cfg["scale_rewards"],
    num_iterations=cfg["num_iterations"],
    max_steps=cfg["max_steps"],
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=0.05,
    logging_steps=1,
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=1,
    bf16=True,
    report_to="none",
    seed=cfg["seed"],
    gradient_checkpointing=cfg["gradient_checkpointing"],
)
trainer = GRPOTrainer(model=model, reward_funcs=robot_reward, args=gargs, train_dataset=ds,
                      processing_class=tok, peft_config=peft_config)
metrics["params"] = count_parameters(trainer.model)
eff = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
metrics["batch_geometry"] = {
    "completions_per_optimizer_step": eff, "num_generations_G": cfg["num_generations"],
    "prompts_per_optimizer_step": eff // cfg["num_generations"],
    "divisible": eff % cfg["num_generations"] == 0,
}
print("batch geometry:", metrics["batch_geometry"])
print(f"trainable parameters: {metrics['params']['trainable_params']:,}")

mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
out = trainer.train()
metrics["train"] = {**out.metrics, "wall_seconds": round(time.perf_counter() - t0, 1), **mem.snapshot()}
write_jsonl(run_dir / "train_log.jsonl", trainer.state.log_history)
write_jsonl(run_dir / "reward_samples.jsonl", reward_log)

# --------------------------------------------- inspect reward dynamics ----
logs = [h for h in trainer.state.log_history if "reward" in h]
if logs:
    keys = [k for k in logs[0] if k.startswith(("reward", "frac_reward_zero_std", "completions/", "kl", "entropy",
                                                 "clip_ratio"))]
    metrics["train_dynamics_first"] = {k: logs[0].get(k) for k in keys}
    metrics["train_dynamics_last"] = {k: logs[-1].get(k) for k in keys}
G = cfg["num_generations"]
groups = [reward_log[i: i + G] for i in range(0, len(reward_log) - G + 1, G)]
metrics["groups"] = {
    "n_groups_logged": len(groups),
    "equal_reward_groups_frac": round(sum(len({x["reward"] for x in g_}) == 1 for g_ in groups) / max(1, len(groups)), 3),
    "all_zero_groups_frac": round(sum(all(x["reward"] == 0 for x in g_) for g_ in groups) / max(1, len(groups)), 3),
    "unique_completions_per_group_first10": [len({x["completion"] for x in g_}) for g_ in groups[:10]],
    "unique_completions_per_group_last10": [len({x["completion"] for x in g_}) for g_ in groups[-10:]],
}
print("group statistics:", metrics["groups"])

# --------------------------------------------------------------- eval ----
pm = trainer.model
pm.eval()
pm.save_pretrained(run_dir / "adapter")
metrics["eval"] = {}
for split, limit in (("test", cfg["eval_limit_test"]), ("test_ood", cfg["eval_limit_ood"])):
    agg, raw = evaluate_robot(pm, tok, read_jsonl(DATA / "robot" / f"{split}.jsonl")[:limit])
    write_jsonl(run_dir / f"eval_{split}.jsonl", raw)
    metrics["eval"][split] = agg
    print(f"eval {split:9s} exact={agg['exact']:.3f} robot={agg['robot_correct']:.3f} "
          f"dist={agg['distance_correct']:.3f} strict={agg['valid_strict']:.3f}")
write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "text_grpo", "run_dir": str(run_dir), "reward_mode": cfg["reward_mode"],
                         "beta": cfg["beta"], "G": G, "steps": cfg["max_steps"], "sft_adapter": cfg.get("sft_adapter"),
                         "train_seconds": metrics["train"]["wall_seconds"],
                         "peak_allocated_gb": metrics["train"].get("peak_allocated_gb"),
                         **{f"{s}_{k}": v[k] for s, v in metrics["eval"].items()
                            for k in ("exact", "robot_correct", "distance_correct", "valid_strict")}})
print("saved", run_dir)
