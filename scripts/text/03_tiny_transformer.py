"""Lesson B2: train a tiny decoder-only Transformer from scratch (mechanics demo).

    python scripts/text/03_tiny_transformer.py --preset quick
    python scripts/text/03_tiny_transformer.py --set n_layer=2 --run-name two_layers

Shows: shapes at every stage, causal masking, shifted labels, cross-entropy,
backprop (non-zero gradients), optimizer updates (parameters change),
validation loss, perplexity, and a deliberate overfitting run.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from lab.common import (RESULTS, base_argparser, count_parameters, load_config, make_run_dir,  # noqa: E402
                        set_seed, timed, write_json, append_experiment_index)
from lab.tiny_gpt import CharTokenizer, TinyGPT, get_batch  # noqa: E402

args = base_argparser(__doc__, "tiny_transformer.yaml").parse_args()
cfg = load_config(args.config, args.preset, args.overrides)
set_seed(cfg["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
run_dir = make_run_dir("tiny_transformer", args.run_name)


# ---------------------------------------------------------------- corpus ----
def robot_log_corpus(n_lines: int, seed: int) -> str:
    """Templated robot mission logs. Distances are correct Manhattan distances,
    so the text has real structure (and arithmetic a tiny model will struggle with)."""
    rng = random.Random(seed)
    names = [f"uav{i}" for i in range(1, 10)] + [f"go2_{i}" for i in range(1, 10)] + [f"ugv{i}" for i in range(1, 10)]
    verbs = ["moves", "drives", "flies", "walks"]
    lines = []
    for _ in range(n_lines):
        n = rng.choice(names)
        x1, y1, x2, y2 = (rng.randrange(10) for _ in range(4))
        kind = rng.random()
        if kind < 0.6:
            lines.append(f"{n} {rng.choice(verbs)} from ({x1},{y1}) to ({x2},{y2}): distance {abs(x1-x2)+abs(y1-y2)}.")
        elif kind < 0.8:
            lines.append(f"{n} is {rng.choice(['idle', 'busy', 'charging'])} at ({x1},{y1}).")
        else:
            lines.append(f"{n} reports battery {rng.randrange(5, 100)} percent.")
    return "\n".join(lines) + "\n"


text = robot_log_corpus(cfg["corpus_lines"], cfg["seed"])
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"corpus: {len(text):,} chars, vocab V={tok.vocab_size}; train {len(train_data):,} / val {len(val_data):,}")
print("first line:", text.splitlines()[0])


def evaluate(model, source, gen):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg["eval_batches"]):
            x, y = get_batch(source, cfg["batch_size"], cfg["block_size"], gen)
            losses.append(model(x.to(dev), y.to(dev))[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(train_src, steps, dropout, tag, inspect=False):
    set_seed(cfg["seed"])
    model = TinyGPT(tok.vocab_size, cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    eval_gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    curve, report = [], {"params": count_parameters(model)}
    print(f"\n[{tag}] parameters: {report['params']['total_params']:,}")
    for step in range(steps + 1):
        x, y = get_batch(train_src, cfg["batch_size"], cfg["block_size"], gen)
        x, y = x.to(dev), y.to(dev)
        logits, loss = model(x, y)
        if inspect and step == 0:
            attn = model.blocks[0].attn.last_attn
            report["shapes"] = {"input_ids x": list(x.shape), "targets y": list(y.shape),
                                "logits": list(logits.shape), "attention weights (block 0)": list(attn.shape)}
            report["shift_check"] = {"x[0,:12]": tok.decode(x[0, :12].tolist()),
                                     "y[0,:12]": tok.decode(y[0, :12].tolist())}
            report["causal_mask_head0_first5"] = [[round(v, 3) for v in row] for row in attn[0, 0, :5, :5].tolist()]
            report["initial_loss"] = loss.item()
            report["uniform_guess_loss_lnV"] = math.log(tok.vocab_size)
            for k, v in report["shapes"].items():
                print(f"  {k:28s} {v}")
            print("  inputs :", repr(report["shift_check"]["x[0,:12]"]))
            print("  targets:", repr(report["shift_check"]["y[0,:12]"]), "(same text shifted by one)")
            print("  attention probs, block 0 head 0, first 5x5 (zeros above diagonal = causal mask):")
            for row in report["causal_mask_head0_first5"]:
                print("   ", row)
            print(f"  initial loss {loss.item():.3f} vs ln(V) = {math.log(tok.vocab_size):.3f} (uniform guessing)")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if inspect and step == 0:
            grads = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            report["grad_norms_step0"] = {n: round(grads[n], 5) for n in
                                          ["tok_emb.weight", "pos_emb.weight", "blocks.0.attn.qkv.weight",
                                           f"blocks.{cfg['n_layer']-1}.mlp.2.weight", "ln_f.weight"]}
            report["params_with_zero_grad"] = [n for n, g in grads.items() if g == 0.0]
            before = model.blocks[0].attn.qkv.weight.detach().clone()
        opt.step()
        if inspect and step == 0:
            report["qkv_weight_change_after_1_step"] = (model.blocks[0].attn.qkv.weight - before).norm().item()
            print("  grad norms:", report["grad_norms_step0"])
            print(f"  ||W_qkv change|| after one AdamW step: {report['qkv_weight_change_after_1_step']:.5f}")
        if step % cfg["eval_every"] == 0:
            tr = evaluate(model, train_src, eval_gen)
            va = evaluate(model, val_data, eval_gen)
            curve.append({"step": step, "train_loss": tr, "val_loss": va})
            print(f"  step {step:5d}  train {tr:.4f}  val {va:.4f}  val ppl {math.exp(va):.3f}")
    return model, curve, report


metrics = {"config": cfg, "device": dev}
with timed(metrics, "main_train_seconds"):
    model, curve, report = train(train_data, cfg["max_steps"], cfg["dropout"], "main", inspect=True)
metrics.update(report)
metrics["curve"] = curve
best = min(curve, key=lambda r: r["val_loss"])
metrics["final"] = {"val_loss": curve[-1]["val_loss"], "val_ppl_per_char": math.exp(curve[-1]["val_loss"]),
                    "val_bits_per_char": curve[-1]["val_loss"] / math.log(2), "best_val_step": best["step"]}
prompt = "uav3 flies from (2,7) to (5,1): distance"
out = model.generate(torch.tensor([tok.encode(prompt)], device=dev), 40, temperature=0.8)
metrics["sample"] = tok.decode(out[0].tolist())
print("\nsample:", metrics["sample"].split("\n")[0], " (true distance is 9)")

# Probe: does the model compute distances? Score 200 held-out prompts greedily.
rng = random.Random(123)
correct = 0
for _ in range(200):
    x1, y1, x2, y2 = (rng.randrange(10) for _ in range(4))
    p = f"uav1 moves from ({x1},{y1}) to ({x2},{y2}): distance "
    ids = torch.tensor([tok.encode(p)], device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            nxt = model(ids[:, -cfg["block_size"]:])[0][:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
    pred = tok.decode(ids[0, len(p):].tolist()).rstrip(".\n ")
    correct += pred == str(abs(x1 - x2) + abs(y1 - y2))
metrics["distance_probe_accuracy"] = correct / 200
print(f"distance probe accuracy (greedy, 200 prompts): {correct/200:.2%}")

# ------------------------------------------------------ overfitting demo ----
tiny_train = train_data[: cfg["overfit_chars"]]
with timed(metrics, "overfit_train_seconds"):
    _, ov_curve, _ = train(tiny_train, cfg["overfit_steps"], cfg["overfit_dropout"], f"overfit ({cfg['overfit_chars']} chars)")
metrics["overfit_curve"] = ov_curve
write_json(run_dir / "metrics.json", metrics)

from lab.plotting import color, plt, style  # noqa: E402

style()
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, c, title in [(axes[0], curve, "Full corpus (train/val)"),
                     (axes[1], ov_curve, f"Overfitting: {cfg['overfit_chars']} training chars, no dropout")]:
    ax.plot([r["step"] for r in c], [r["train_loss"] for r in c], label="train", color=color(0))
    ax.plot([r["step"] for r in c], [r["val_loss"] for r in c], label="validation", color=color(1))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("optimizer step")
axes[0].set_ylabel("cross-entropy (nats per character)")
axes[0].legend()
fig.suptitle("Tiny character-level Transformer (mechanics demo, single seed)", fontsize=11)
fig.tight_layout()
plot = RESULTS / "plots" / "tiny_transformer_curves.png"
fig.savefig(plot, dpi=130)
append_experiment_index({"experiment": "tiny_transformer", "run_dir": str(run_dir), "preset": cfg["_preset"],
                         "val_loss": metrics["final"]["val_loss"], "val_bits_per_char": metrics["final"]["val_bits_per_char"],
                         "overfit_final_train": ov_curve[-1]["train_loss"], "overfit_final_val": ov_curve[-1]["val_loss"],
                         "train_seconds": metrics["main_train_seconds"]})
print(f"\nmain run: {metrics['main_train_seconds']}s, overfit run: {metrics['overfit_train_seconds']}s")
print(f"overfit final train {ov_curve[-1]['train_loss']:.3f} vs val {ov_curve[-1]['val_loss']:.3f}")
print("wrote", run_dir / "metrics.json", "and", plot)
