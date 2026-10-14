"""VLM Lesson D: multimodal GRPO with TRL's GRPOTrainer (images + questions -> short answers).

    .venv/bin/python scripts/vlm/40_grpo.py --preset quick --run-name grpo_quick
    .venv/bin/python scripts/vlm/40_grpo.py --set model=vlm_small --run-name grpo_smolvlm   # fallback model

The policy sees the real image for every sampled completion (the dataset has an
`image` column and a multimodal prompt), so this is genuinely multimodal RL,
not text-only RL. Reward = lab.shapes_task.reward (1 correct, 0.1 valid-but-wrong,
0 invalid, -0.1 if longer than 3 words).

LFM2.5-VL is not on TRL's list of tested VLMs. If the trainer fails, the full
traceback is saved to <run>/error.txt and the run is reported as blocked.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from lab import shapes_task as st  # noqa: E402
from lab.common import (DATA, GpuMemory, append_experiment_index, base_argparser, count_parameters,  # noqa: E402
                        environment_info, load_config, make_run_dir, read_jsonl, set_seed, write_json, write_jsonl)
from lab.models import load_vlm, resolve  # noqa: E402

args = base_argparser(__doc__, "vlm_grpo.yaml").parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
run_dir = make_run_dir("vlm_grpo", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "model": resolve(cfg["model"]), "status": "started"}

model, processor = load_vlm(cfg["model"], processor_kwargs={"max_image_tokens": cfg["max_image_tokens"]}
                            if cfg["model"] == "vlm" else None)
if cfg.get("sft_adapter"):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, cfg["sft_adapter"]).merge_and_unload()

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402

rows = [r for r in read_jsonl(DATA / "shapes" / "train.jsonl") if r["type"] in cfg["question_types"]]
rows = rows[cfg["prompt_offset"]: cfg["prompt_offset"] + cfg["n_prompts"]]
ds = Dataset.from_list([{
    "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]}],
    # blank_images=true is an ABLATION: same questions, empty canvas. If rewards collapse,
    # the trainer really conditions on the images (i.e. this is multimodal RL).
    "image": (st.blank_image(r["scene"]["size"]) if cfg.get("blank_images")
              else Image.open(DATA / "shapes" / r["image"]).convert("RGB")),
    "qtype": r["type"], "answer": r["answer"], "qid": r["id"]} for r in rows])
reward_log = []


def shape_reward(completions, qtype, answer, qid, **kwargs):
    out = []
    for c, t, a, i in zip(completions, qtype, answer, qid):
        text = c[0]["content"] if isinstance(c, list) else c
        r = st.reward(t, a, text)
        out.append(r)
        reward_log.append({"qid": i, "completion": text, "reward": r})
    return out


from trl import GRPOConfig, GRPOTrainer  # noqa: E402

target = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|conv\.(in_proj|out_proj)"
          r"|feed_forward\.(w1|w2|w3))") if cfg["model"] == "vlm" else \
    r".*text_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
gargs = GRPOConfig(
    output_dir=str(run_dir / "trainer"), learning_rate=cfg["learning_rate"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"], num_generations=cfg["num_generations"],
    max_completion_length=cfg["max_completion_length"], temperature=cfg["temperature"], beta=cfg["beta"],
    max_steps=cfg["max_steps"], logging_steps=1, save_strategy="no", bf16=True, report_to="none",
    seed=cfg["seed"], remove_unused_columns=False,
)
mem = GpuMemory()
try:
    trainer = GRPOTrainer(model=model, reward_funcs=shape_reward, args=gargs, train_dataset=ds,
                          processing_class=processor,
                          peft_config=LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                                 target_modules=target, lora_dropout=0.0))
    metrics["params"] = count_parameters(trainer.model)
    mem.reset()
    t0 = time.perf_counter()
    out = trainer.train()
    metrics["train"] = {**out.metrics, "wall_seconds": round(time.perf_counter() - t0, 1), **mem.snapshot()}
    write_jsonl(run_dir / "train_log.jsonl", trainer.state.log_history)
    metrics["status"] = "trained"
    logs = [h for h in trainer.state.log_history if "reward" in h]
    if logs:
        metrics["reward_first_steps"] = [round(h["reward"], 3) for h in logs[:5]]
        metrics["reward_last_steps"] = [round(h["reward"], 3) for h in logs[-5:]]
        metrics["frac_reward_zero_std_mean"] = round(sum(h.get("frac_reward_zero_std", 0) for h in logs) / len(logs), 3)
    trainer.model.save_pretrained(run_dir / "adapter")
    if not cfg.get("blank_images") and not cfg.get("skip_eval"):
        import subprocess
        from lab.common import ROOT, read_json
        del trainer, model
        torch.cuda.empty_cache()
        name = f"vlm_grpo_{run_dir.name}"
        subprocess.run([sys.executable, str(ROOT / "scripts/vlm/evaluate.py"), "--adapter", str(run_dir / "adapter"),
                        "--name", name, "--model", cfg["model"]], check=True)
        metrics["eval"] = read_json(ROOT / "outputs/evals" / name / "metrics.json")["conditions"]
except Exception as e:  # preserve the evidence instead of hiding it
    metrics["status"] = "blocked"
    metrics["error"] = f"{type(e).__name__}: {e}"
    (run_dir / "error.txt").write_text(traceback.format_exc())
    print("GRPO FAILED:", metrics["error"])
write_jsonl(run_dir / "reward_samples.jsonl", reward_log)
G = cfg["num_generations"]
groups = [reward_log[i: i + G] for i in range(0, len(reward_log) - G + 1, G)]
metrics["groups"] = {"n": len(groups),
                     "equal_reward_frac": round(sum(len({x["reward"] for x in g}) == 1 for g in groups) / max(1, len(groups)), 3),
                     "example_group": groups[0] if groups else None}
write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "vlm_grpo", "run_dir": str(run_dir), "model": cfg["model"],
                         "status": metrics["status"], "error": metrics.get("error"),
                         "train_seconds": metrics.get("train", {}).get("wall_seconds"),
                         "peak_allocated_gb": metrics.get("train", {}).get("peak_allocated_gb"),
                         "reward_first": metrics.get("reward_first_steps"), "reward_last": metrics.get("reward_last_steps")})
print(json.dumps({k: metrics.get(k) for k in ("status", "error", "reward_first_steps", "reward_last_steps",
                                              "frac_reward_zero_std_mean")}, indent=2))
print("saved", run_dir)
