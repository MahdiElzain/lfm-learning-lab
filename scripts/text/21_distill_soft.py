"""Lesson D2: soft-target (logit) distillation with a temperature-scaled KL loss.

The loop is written out by hand so every step of the math is visible:

    p_T = softmax(z_T / tau),   p_S = softmax(z_S / tau)          (per completion position)
    forward KL  = sum_v p_T(v) * (log p_T(v) - log p_S(v))        (default; "mean-seeking")
    reverse KL  = sum_v p_S(v) * (log p_S(v) - log p_T(v))        ("mode-seeking")
    loss = alpha * tau^2 * KL  +  (1 - alpha) * CE(z_S, hard labels)

alpha=0 turns this into ordinary hard-label SFT with the *same* loop, so the
comparison changes exactly one thing.

    .venv/bin/python scripts/text/21_distill_soft.py --run-name kd_soft
    .venv/bin/python scripts/text/21_distill_soft.py --set alpha=0 --run-name kd_hard_baseline
    .venv/bin/python scripts/text/21_distill_soft.py --set temperature=1 --run-name kd_tau1

Teacher and student MUST share a tokenizer; the script checks the vocabulary,
special tokens and the exact token IDs of templated prompts before training.
Teacher logits are computed on the fly for each batch: nothing is cached, so
memory does not grow with dataset size.
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lab import robot_task as rt  # noqa: E402
from lab.common import (DATA, GpuMemory, append_experiment_index, base_argparser, count_parameters,  # noqa: E402
                        environment_info, load_config, make_run_dir, read_jsonl, require_disk, set_seed, write_json,
                        write_jsonl)
from lab.models import load_text_model, resolve  # noqa: E402
from lab.text_eval import evaluate_robot  # noqa: E402

args = base_argparser(__doc__, "distill_soft.yaml").parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
require_disk(3.0)
run_dir = make_run_dir("text_distill_soft", args.run_name)
write_json(run_dir / "config.json", cfg)
metrics = {"config": cfg, "env": environment_info(), "teacher": resolve(cfg["teacher_model"]),
           "student": resolve(cfg["student_model"])}
dev = "cuda"

# ------------------------------------------------------------ models ----
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

teacher, ttok = load_text_model(cfg["teacher_model"])
if cfg.get("teacher_adapter"):
    teacher = PeftModel.from_pretrained(teacher, cfg["teacher_adapter"]).merge_and_unload()
teacher.eval().requires_grad_(False)
student, tok = load_text_model(cfg["student_model"])
student = get_peft_model(student, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                                             target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
metrics["student_params"] = count_parameters(student)
metrics["teacher_total_params"] = sum(p.numel() for p in teacher.parameters())
print(f"teacher {metrics['teacher_total_params']/1e6:.0f}M params | student trainable "
      f"{metrics['student_params']['trainable_params']:,}")

# ------------------------------------------------ tokenizer compatibility ----
train_rows = read_jsonl(DATA / "robot" / "train.jsonl")[: cfg["n_train"]]
checks = {
    "same_vocab": ttok.get_vocab() == tok.get_vocab(),
    "same_special_tokens": ttok.all_special_tokens == tok.all_special_tokens,
    "teacher_logit_dim": teacher.get_output_embeddings().weight.shape[0],
    "student_logit_dim": student.get_output_embeddings().weight.shape[0],
    "same_template_ids_on_5_prompts": all(
        ttok.apply_chat_template(rt.chat_messages(r), add_generation_prompt=True, tokenize=False) ==
        tok.apply_chat_template(rt.chat_messages(r), add_generation_prompt=True, tokenize=False)
        for r in train_rows[:5]),
}
metrics["tokenizer_checks"] = checks
print("tokenizer checks:", checks)
if not (checks["same_vocab"] and checks["same_template_ids_on_5_prompts"]
        and checks["teacher_logit_dim"] == checks["student_logit_dim"]):
    raise SystemExit("Teacher and student do not share a tokenizer: token-level KL would compare unrelated "
                     "distributions. Use sequence distillation (20_distill_sequence.py) instead.")


# --------------------------------------------------------------- data ----
def encode(ex, answer_text):
    """Prompt+completion token IDs and labels (-100 on prompt and padding)."""
    msgs = rt.chat_messages(ex)
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    full_ids = tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer_text}], tokenize=True,
                                       return_dict=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, "chat template is not prefix-consistent"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids[: cfg["max_length"]], labels[: cfg["max_length"]]


source = cfg["sequence_source"]
if source == "gold":
    pairs = [encode(r, rt.format_answer(r["answer"])) for r in train_rows]
else:  # completions produced by a teacher during sequence distillation
    pairs = [encode(r, rt.format_answer(r["answer"])) for r in read_jsonl(DATA / source)[: cfg["n_train"]]]
example_labels = [t for t in pairs[0][1] if t != -100]
metrics["example_trained_text"] = tok.decode(example_labels)
print("tokens that receive loss in example 0:", repr(metrics["example_trained_text"]))


def collate(batch):
    T = max(len(ids) for ids, _ in batch)
    pad = tok.pad_token_id
    ids = torch.full((len(batch), T), pad)
    lab_ = torch.full((len(batch), T), -100)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)], lab_[i, : len(y)], att[i, : len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(dev), lab_.to(dev), att.to(dev)


# ------------------------------------------------------------ the loss ----
def kd_loss(s_logits, t_logits, labels, tau, alpha, direction):
    """Return (total, kl, ce, top1_agreement) over completion positions only.

    Causal alignment: logits at position t predict the token at t+1, so we drop
    the last logit and the first label. Teacher and student see the SAME input
    IDs, so their positions line up one-to-one.
    """
    s = s_logits[:, :-1]
    t = t_logits[:, :-1]
    y = labels[:, 1:]
    mask = y != -100                                  # (B, T-1) True on completion tokens
    s, t, y = s[mask].float(), t[mask].float(), y[mask]   # (M, V), (M, V), (M,)
    log_ps = F.log_softmax(s / tau, dim=-1)
    log_pt = F.log_softmax(t / tau, dim=-1)
    if direction == "forward":
        kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1).mean()
    else:
        kl = (log_ps.exp() * (log_ps - log_pt)).sum(-1).mean()
    ce = F.cross_entropy(s, y)
    total = alpha * tau ** 2 * kl + (1 - alpha) * ce
    agree = (s.argmax(-1) == t.argmax(-1)).float().mean()
    return total, kl.detach(), ce.detach(), agree.detach()


# ------------------------------------------------------------ training ----
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg["learning_rate"])
B, accum = cfg["batch_size"], cfg["grad_accum"]
steps_per_epoch = math.ceil(len(pairs) / (B * accum))
total_steps = steps_per_epoch * cfg["epochs"]
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total_steps))) *
                                          0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
g = torch.Generator().manual_seed(cfg["seed"])
log, step = [], 0
mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
student.train()
for epoch in range(cfg["epochs"]):
    order = torch.randperm(len(pairs), generator=g).tolist()
    for start in range(0, len(order), B * accum):
        micro = [order[i: i + B] for i in range(start, min(start + B * accum, len(order)), B)]
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "agree": 0.0}
        for idx in micro:
            ids, labels, att = collate([pairs[i] for i in idx])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits = teacher(input_ids=ids, attention_mask=att).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = student(input_ids=ids, attention_mask=att).logits
            loss, kl, ce, agree = kd_loss(s_logits, t_logits, labels, cfg["temperature"], cfg["alpha"],
                                          cfg["kl_direction"])
            (loss / len(micro)).backward()
            for k, v in zip(acc, (loss.item(), kl.item(), ce.item(), agree.item())):
                acc[k] += v / len(micro)
        if step == 0:
            gn = [p.grad.norm().item() for n, p in student.named_parameters() if p.requires_grad and p.grad is not None]
            metrics["first_step_grad"] = {"n_with_grad": len(gn), "n_nonzero": sum(x > 0 for x in gn)}
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        acc.update(step=step, lr=sched.get_last_lr()[0])
        log.append(acc)
        if step % 10 == 0 or step == 1:
            print(f"step {step:4d}/{total_steps} loss {acc['loss']:.4f} KL {acc['kl']:.4f} CE {acc['ce']:.4f} "
                  f"top1-agree {acc['agree']:.3f}")
train_seconds = time.perf_counter() - t0
metrics["train"] = {"steps": step, "seconds": round(train_seconds, 1), **mem.snapshot()}
write_jsonl(run_dir / "train_log.jsonl", log)

# ---------------------------------------------- unload teacher, evaluate ----
del teacher
torch.cuda.empty_cache()
student.eval()
student.save_pretrained(run_dir / "adapter")
metrics["eval"] = {}
for split, limit in (("test", cfg["eval_limit_test"]), ("test_ood", cfg["eval_limit_ood"])):
    rows = read_jsonl(DATA / "robot" / f"{split}.jsonl")[:limit]
    agg, raw = evaluate_robot(student, tok, rows)
    write_jsonl(run_dir / f"eval_{split}.jsonl", raw)
    metrics["eval"][split] = agg
    print(f"eval {split:9s} exact={agg['exact']:.3f} robot={agg['robot_correct']:.3f} dist={agg['distance_correct']:.3f}")
write_json(run_dir / "metrics.json", metrics)
append_experiment_index({"experiment": "text_distill_soft", "run_dir": str(run_dir), "alpha": cfg["alpha"],
                         "temperature": cfg["temperature"], "kl_direction": cfg["kl_direction"],
                         "teacher": cfg["teacher_model"], "teacher_adapter": cfg.get("teacher_adapter"),
                         "train_seconds": metrics["train"]["seconds"],
                         "peak_allocated_gb": metrics["train"].get("peak_allocated_gb"),
                         **{f"{s}_{k}": v[k] for s, v in metrics["eval"].items()
                            for k in ("exact", "robot_correct", "distance_correct")}})
print("saved", run_dir)
