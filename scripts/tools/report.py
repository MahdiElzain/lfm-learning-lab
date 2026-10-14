"""Rebuild results tables and plots from the saved run/eval files.

    .venv/bin/python scripts/tools/report.py

Reads only files under outputs/ (never re-runs anything). Missing runs are
skipped, so this works at any point of the lab. Writes:
    results/tables/text_main.md, results/tables/vlm_main.md, results/tables/all_runs.md
    results/plots/*.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.common import OUTPUTS, RESULTS, read_jsonl  # noqa: E402
from lab.plotting import color, plt, style  # noqa: E402

style()
T = RESULTS / "tables"
P = RESULTS / "plots"
T.mkdir(parents=True, exist_ok=True)
P.mkdir(parents=True, exist_ok=True)


def load(path):
    p = OUTPUTS / path
    return json.loads(p.read_text()) if p.exists() else None


def log(path):
    p = OUTPUTS / path
    return read_jsonl(p) if p.exists() else []


# --------------------------------------------------------------- text table ----
def text_row(label, kind, path):
    m = load(path)
    if m is None:
        return None
    if kind == "eval":
        t, o = m["splits"]["test"], m["splits"]["test_ood"]
        return dict(label=label, test=t, ood=o, params=None, secs=None, gb=None)
    if kind == "teacher":
        t = m["teacher_eval_test"]
        return dict(label=label, test=t, ood=None, params=None, secs=None, gb=t.get("peak_allocated_gb"), n=t["n"])
    t, o = m["eval"]["test"], m["eval"]["test_ood"]
    tr = m.get("train", {})
    params = (m.get("params") or m.get("student_params") or {}).get("trainable_params")
    return dict(label=label, test=t, ood=o, params=params, secs=tr.get("wall_seconds") or tr.get("seconds"),
                gb=tr.get("peak_allocated_gb"))


TEXT_ROWS = [
    ("LFM2.5-350M instruct, zero-shot", "eval", "evals/instruct_zero_shot/metrics.json"),
    ("LFM2.5-350M-Base, 2-shot plain prompt", "eval", "evals/base_2shot/metrics.json"),
    ("SmolLM2-135M-Instruct, zero-shot", "eval", "evals/smollm2_135m_zero_shot/metrics.json"),
    ("SFT LoRA, quick (1k ex)", "run", "runs/text_sft/lora_quick/metrics.json"),
    ("SFT LoRA, full-sequence loss, quick", "run", "runs/text_sft/lora_fullseq_quick/metrics.json"),
    ("Full fine-tune, quick (8x2 batch)", "run", "runs/text_sft/full_quick/metrics.json"),
    ("Full fine-tune, quick (4x4 + grad ckpt)", "run", "runs/text_sft/full_quick_ckpt/metrics.json"),
    ("SFT LoRA, overfit demo (16 ex x 40 ep)", "run", "runs/text_sft/overfit_tiny/metrics.json"),
    ("SFT LoRA, standard (3k ex x 2 ep)", "run", "runs/text_sft/lora_standard/metrics.json"),
    ("Teacher: Qwen3.5-2B no-thinking (200 test)", "teacher",
     "runs/text_distill_sequence/qwen_nothink_standard/metrics.json"),
    ("Teacher: Qwen3.5-2B thinking (40 test)", "teacher",
     "runs/text_distill_sequence/qwen_thinking_profile/metrics.json"),
    ("Seq-distilled from Qwen no-think (294 kept labels)", "run",
     "runs/text_sft_distilled/distilled_qwen35_2b_seq_standard/metrics.json"),
    ("Control: same 294 prompts, gold labels", "run", "runs/text_sft_distilled/gold_same294_standard/metrics.json"),
    ("Seq-distilled from Qwen thinking", "run", "runs/text_sft_distilled/distilled_qwen35_2b_think_standard/metrics.json"),
    ("Teacher: LFM2.5-1.2B + LoRA (standard)", "run", "runs/text_sft/teacher_1p2b_lora_standard/metrics.json"),
    ("Soft KD from 1.2B (alpha=1, tau=2)", "run", "runs/text_distill_soft/kd_soft_standard/metrics.json"),
    ("Hard labels, same KD loop (alpha=0)", "run", "runs/text_distill_soft/kd_hard_standard/metrics.json"),
    ("Mixed KD (alpha=0.5, tau=2)", "run", "runs/text_distill_soft/kd_mix_alpha05_standard/metrics.json"),
    ("QLoRA (4-bit NF4 base), quick", "run", "runs/text_sft/qlora_quick/metrics.json"),
    ("DPO on SFT-standard, lr 1e-5", "run", "runs/text_dpo/dpo_quick/metrics.json"),
    ("DPO on SFT-standard, lr 5e-5", "run", "runs/text_dpo/dpo_lr5e-5_quick/metrics.json"),
    ("DPO + SFT term, lr 5e-5", "run", "runs/text_dpo/dpo_sft_lr5e-5_quick/metrics.json"),
    ("GRPO 60 steps on SFT-standard (shaped reward)", "run", "runs/text_grpo/grpo_quick/metrics.json"),
    ("GRPO 200 steps on SFT-standard (shaped reward)", "run", "runs/text_grpo/grpo_standard/metrics.json"),
    ("GRPO 60 steps on SFT-standard (binary reward)", "run", "runs/text_grpo/grpo_binary_quick/metrics.json"),
]
rows = [r for r in (text_row(*s) for s in TEXT_ROWS) if r]
lines = ["| Model / method | test exact | robot | distance | strict JSON | violations (busy/type) | OOD exact | "
         "trainable params | train s | peak alloc GB |", "|---|---|---|---|---|---|---|---|---|---|"]
def fmt(value, spec=".3f"):
    return "" if value is None else format(value, spec)


for r in rows:
    t, o = r["test"], r["ood"]
    viol = f"{t['violation_status']:.3f} / {t['violation_type']:.3f}" if "violation_status" in t else ""
    params = f"{r['params'] / 1e6:.1f}M" if r["params"] else ""
    secs = "" if r["secs"] is None else str(round(r["secs"]))
    lines.append(f"| {r['label']} | {t['exact']:.3f} | {t['robot_correct']:.3f} | {fmt(t.get('distance_correct'))} | "
                 f"{fmt(t.get('valid_strict'))} | {viol} | {fmt(None if o is None else o['exact'])} | {params} | "
                 f"{secs} | {fmt(r['gb'], '')} |")
bon = []
for name in ("sft_best_of_8", "grpo_best_of_8"):
    m = load(f"evals/{name}/best_of_n.json")
    if m:
        bon.append(m)
text_md = ["# Text track: main results", "",
           "Robot dispatch, 300 test scenes (3-5 robots) and 200 OOD scenes (6 robots); greedy decoding unless noted. "
           "Single seed: differences of a few points are within run-to-run noise.", ""] + lines
if bon:
    text_md += ["", "## Multiple-sample selection (best-of-n) on the test split", "",
                "| starting model | k | T | greedy | first label-free-valid | majority of valid | oracle pass@k | "
                "tokens vs greedy | time vs greedy |", "|---|---|---|---|---|---|---|---|---|"]
    for m in bon:
        a, c = m["accuracy"], m["cost"]
        text_md.append(f"| {m['name']} | {m['k']} | {m['temperature']} | {a['greedy']:.3f} | {a['first_valid']:.3f} | "
                       f"{a['majority_valid']:.3f} | {a['oracle_pass@k']:.3f} | {c['token_ratio']}x | {c['time_ratio']}x |")
(T / "text_main.md").write_text("\n".join(text_md) + "\n")

# -------------------------------------------------------------- text plots ----
if rows:
    shown = [r for r in rows if r["ood"] is not None and "overfit" not in r["label"]]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(shown) + 1.2))
    y = list(range(len(shown)))[::-1]
    h = 0.38
    ax.barh([v + h / 2 for v in y], [r["test"]["exact"] for r in shown], height=h, color=color(0), label="test (3-5 robots)")
    ax.barh([v - h / 2 for v in y], [r["ood"]["exact"] for r in shown], height=h, color=color(1), label="OOD (6 robots)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in shown], fontsize=8)
    ax.set_xlabel("exact match (robot and distance both right)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Robot dispatch: exact match by method (single seed)")
    fig.tight_layout()
    fig.savefig(P / "text_methods_exact.png", dpi=130)
    plt.close(fig)

curves = [("LoRA quick (1k ex)", "runs/text_sft/lora_quick/train_log.jsonl"),
          ("LoRA standard (3k ex x2)", "runs/text_sft/lora_standard/train_log.jsonl"),
          ("Full FT quick", "runs/text_sft/full_quick/train_log.jsonl"),
          ("Overfit demo (16 ex)", "runs/text_sft/overfit_tiny/train_log.jsonl")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for i, (label, path) in enumerate(curves):
    L = log(path)
    tr = [(x["step"], x["loss"]) for x in L if "loss" in x and "eval_loss" not in x]
    ev = [(x["step"], x["eval_loss"]) for x in L if "eval_loss" in x]
    if tr:
        axes[0].plot(*zip(*tr), color=color(i), label=label)
    if ev:
        axes[1].plot(*zip(*ev), color=color(i), marker="o", markersize=4, label=label)
axes[0].set_title("training loss (completion tokens)")
axes[1].set_title("validation loss (100 held-out scenes)")
for ax in axes:
    ax.set_xlabel("optimizer step")
axes[0].set_ylabel("cross-entropy (nats/token)")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(P / "text_sft_curves.png", dpi=130)
plt.close(fig)

kd = [("soft KD (alpha=1, tau=2)", "runs/text_distill_soft/kd_soft_standard/train_log.jsonl"),
      ("hard labels (alpha=0)", "runs/text_distill_soft/kd_hard_standard/train_log.jsonl"),
      ("mixed (alpha=0.5)", "runs/text_distill_soft/kd_mix_alpha05_standard/train_log.jsonl")]
if any((OUTPUTS / p).exists() for _, p in kd):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (label, path) in enumerate(kd):
        L = log(path)
        if L:
            axes[0].plot([x["step"] for x in L], [x["kl"] for x in L], color=color(i), label=label, linewidth=1.2)
            axes[1].plot([x["step"] for x in L], [x["agree"] for x in L], color=color(i), label=label, linewidth=1.2)
    axes[0].set_title("KL(teacher || student) at tau, completion tokens")
    axes[1].set_title("student top-1 = teacher top-1 (completion tokens)")
    axes[0].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("optimizer step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(P / "text_kd_curves.png", dpi=130)
    plt.close(fig)

dpo = log("runs/text_dpo/dpo_quick/train_log.jsonl")
if dpo:
    pts = [x for x in dpo if "rewards/chosen" in x]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, k in enumerate(["rewards/chosen", "rewards/rejected", "rewards/margins"]):
        axes[0].plot([x["step"] for x in pts], [x[k] for x in pts], color=color(i), label=k.split("/")[1])
    axes[0].axhline(0, color="#52514e", linewidth=0.8)
    axes[0].set_title("DPO implicit rewards (beta x log-ratio vs reference)")
    axes[0].legend()
    axes[1].plot([x["step"] for x in pts], [x["loss"] for x in pts], color=color(0))
    axes[1].axhline(0.6931, color="#52514e", linewidth=0.8, linestyle="--")
    axes[1].set_title("DPO loss (dashed: log 2, the value at initialisation)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(P / "text_dpo_dynamics.png", dpi=130)
    plt.close(fig)

for tag, path in (("shaped", "runs/text_grpo/grpo_standard/train_log.jsonl"),
                  ("binary", "runs/text_grpo/grpo_binary_quick/train_log.jsonl")):
    G = [x for x in log(path) if "reward" in x]
    if not G:
        continue
    def rolling(v, w=10):
        return [sum(v[max(0, i - w + 1): i + 1]) / len(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]

    steps = [x["step"] for x in G]
    panels = [("reward", f"mean reward per step ({tag})", None),
              ("frac_reward_zero_std", "fraction of groups with equal rewards", (0, 1.05)),
              ("entropy", "policy entropy on sampled tokens (nats)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, (key, title, ylim) in enumerate(panels):
        vals = [x.get(key, 0) for x in G]
        axes[i].plot(steps, vals, color=color(i), linewidth=0.8, alpha=0.35, label="per step")
        axes[i].plot(steps, rolling(vals), color=color(i), linewidth=2, label="10-step mean")
        axes[i].set_title(title)
        if ylim:
            axes[i].set_ylim(*ylim)
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(P / f"text_grpo_{tag}.png", dpi=130)
    plt.close(fig)

# ---------------------------------------------------------------- VLM table ----
VLM_ROWS = [("LFM2.5-VL-450M base", "evals/vlm_base/metrics.json"),
            ("SmolVLM-256M base (no tiling, batch 4)", "evals/smolvlm_base/metrics.json"),
            ("LFM2.5-VL SFT: LoRA on LM", "evals/vlm_sft_lora_quick/metrics.json"),
            ("LFM2.5-VL SFT: LoRA + connector", "evals/vlm_sft_lora_connector_quick/metrics.json"),
            ("LFM2.5-VL SFT on Qwen3.5-2B labels", "evals/vlm_sft_distilled_qwen35_2b_vlm_quick/metrics.json"),
            ("LFM2.5-VL GRPO 30 steps (all counts)", "evals/vlm_grpo_grpo_quick/metrics.json"),
            ("LFM2.5-VL GRPO 100 steps (shape+total counts)", "evals/vlm_grpo_grpo_counts_standard/metrics.json")]
vl = ["# VLM track: main results", "",
      "Shape scenes, 450 test questions on 150 held-out scenes; greedy, 8 new tokens. Swapped = another test scene's "
      "image with the same question, scored against the swapped image (types defined for any image). Unchanged = "
      "prediction identical to the correct-image prediction. Blank: accuracy vs the original answer (= what the text "
      "alone gives). Single seed.", "",
      "| model | correct acc | swapped acc | unchanged under swap | blank: matches original | valid | image tokens | "
      "ms/question | majority baseline |", "|---|---|---|---|---|---|---|---|---|"]
vrows = []
for label, path in VLM_ROWS:
    m = load(path) if path else None
    if m is None:
        continue
    c = m["conditions"]
    vrows.append((label, c))
    vl.append(f"| {label} | {c['correct']['accuracy']:.3f} | {c.get('swapped', {}).get('accuracy', float('nan')):.3f} | "
              f"{c.get('swapped', {}).get('unchanged_vs_correct_image', float('nan')):.3f} | "
              f"{c.get('blank', {}).get('matches_original_answer', float('nan')):.3f} | {c['correct']['valid_rate']:.3f} | "
              f"{c['correct']['mean_image_tokens']} | {c['correct']['ms_per_question']} | {m['majority_baseline_accuracy']:.3f} |")
if vrows:
    types = sorted(vrows[0][1]["correct"]["by_type"])
    vl += ["", "Accuracy by question type (correct image):", "",
           "| model | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for label, c in vrows:
        vl.append(f"| {label} | " + " | ".join(f"{c['correct']['by_type'].get(t, float('nan')):.3f}" for t in types) + " |")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(vrows) + 1.9))
    y = list(range(len(vrows)))[::-1]
    h = 0.26
    for j, (cond, key, lab) in enumerate([("correct", "accuracy", "correct image"),
                                          ("swapped", "accuracy", "swapped image (scored vs swapped)"),
                                          ("blank", "matches_original_answer", "blank image (vs original answer)")]):
        ax.barh([v + h * (1 - j) for v in y], [c.get(cond, {}).get(key) or 0 for _, c in vrows], height=h,
                color=color(j), label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in vrows], fontsize=8)
    ax.set_xlim(0, 1)
    maj = load(VLM_ROWS[0][1])["majority_baseline_accuracy"] if load(VLM_ROWS[0][1]) else None
    if maj:
        ax.axvline(maj, color="#52514e", linestyle="--", linewidth=1)
        ax.text(maj + 0.01, y[0] + 0.55, f"majority-answer baseline {maj:.2f}", fontsize=7, color="#52514e")
    ax.set_xlabel("accuracy")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, fontsize=7)
    ax.set_title("Visual dependence: shape questions under three image conditions")
    fig.tight_layout()
    fig.savefig(P / "vlm_conditions.png", dpi=130)
    plt.close(fig)
(T / "vlm_main.md").write_text("\n".join(vl) + "\n")

# ------------------------------------------------------------ all runs index ----
idx = RESULTS / "experiments.jsonl"
if idx.exists():
    recs = read_jsonl(idx)
    keys = ["finished_at", "experiment", "name", "run_dir", "preset", "test_exact", "test_ood_exact", "correct_acc",
            "train_seconds", "peak_allocated_gb"]
    out = ["# Every finished run (from results/experiments.jsonl)", "", "| " + " | ".join(keys) + " |",
           "|" + "---|" * len(keys)]
    for r in recs:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                v = round(v, 4)
            if k == "run_dir" and v:
                v = "/".join(str(v).split("/")[-2:])
            vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    (T / "all_runs.md").write_text("\n".join(out) + "\n")
print("wrote", sorted(p.name for p in T.iterdir()), sorted(p.name for p in P.iterdir()))
