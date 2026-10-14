"""Lesson B1: inspect LFM2.5-350M's real configuration, modules and tensor shapes.

    .venv/bin/python scripts/text/02_inspect_architecture.py
    .venv/bin/python scripts/text/02_inspect_architecture.py --model text_small   # compare SmolLM2

Writes outputs/runs/lesson_b_arch/<model>/architecture.json.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401  (sets HF cache location)
import torch  # noqa: E402

from lab.common import OUTPUTS, write_json  # noqa: E402
from lab.models import load_text_model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="text_instruct")
args = p.parse_args()

model, tok = load_text_model(args.model)
model.eval()
cfg = model.config
report = {"model": args.model, "class": type(model).__name__}

print("=" * 78 + "\n1. Configuration (the numbers that define the architecture)\n" + "=" * 78)
keys = ["model_type", "vocab_size", "hidden_size", "intermediate_size", "block_ff_dim", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "conv_L_cache", "conv_bias",
        "norm_eps", "rope_theta", "rope_parameters", "tie_word_embeddings", "block_auto_adjust_ff_dim",
        "block_multiple_of", "block_ffn_dim_multiplier", "layer_types", "full_attn_idxs", "hidden_act"]
conf = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}
for k, v in conf.items():
    print(f"  {k:28s} {v}")
report["config"] = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                    for k, v in conf.items()}

print("\n" + "=" * 78 + "\n2. Top-level module tree\n" + "=" * 78)
for name, mod in model.named_children():
    print(f"  {name}: {type(mod).__name__}")
    for n2, m2 in mod.named_children():
        extra = f" x{len(m2)}" if isinstance(m2, torch.nn.ModuleList) else ""
        print(f"    {name}.{n2}: {type(m2).__name__}{extra}")

layers = None
for cand in ("model.layers", "model.language_model.layers"):
    obj = model
    try:
        for part in cand.split("."):
            obj = getattr(obj, part)
        layers = obj
        break
    except AttributeError:
        continue

print("\n" + "=" * 78 + "\n3. Every decoder layer: operator type and submodules\n" + "=" * 78)
layer_rows = []
for i, layer in enumerate(layers):
    children = {n: type(m).__name__ for n, m in layer.named_children()}
    kind = "attention" if any("attn" in n for n in children) else ("conv" if any("conv" in n for n in children)
                                                                     else "other")
    nparams = sum(p.numel() for p in layer.parameters())
    layer_rows.append({"layer": i, "kind": kind, "params": nparams, "children": children})
    print(f"  layer {i:2d} [{kind:9s}] {nparams/1e6:6.2f}M params  {children}")
report["layers"] = layer_rows
print("  counts:", dict(Counter(r["kind"] for r in layer_rows)))

print("\n" + "=" * 78 + "\n4. Detailed view of one conv layer and one attention layer\n" + "=" * 78)
for kind in ("conv", "attention"):
    idx = next(r["layer"] for r in layer_rows if r["kind"] == kind)
    print(f"\n  --- layer {idx} ({kind}) ---")
    for n, prm in layers[idx].named_parameters():
        print(f"    {n:40s} {str(tuple(prm.shape)):18s} {prm.dtype}")
report["example_layers"] = {kind: {n: list(prm.shape) for n, prm in
                                   layers[next(r['layer'] for r in layer_rows if r['kind'] == kind)].named_parameters()}
                            for kind in ("conv", "attention")}

print("\n" + "=" * 78 + "\n5. Parameter budget by component\n" + "=" * 78)
groups = defaultdict(int)
for n, prm in model.named_parameters():
    if "embed" in n:
        g = "embeddings"
    elif "lm_head" in n:
        g = "lm_head (untied)"
    elif "norm" in n:
        g = "norms"
    elif ".conv." in n:
        g = "conv operators"
    elif "attn" in n:
        g = "attention operators"
    elif "feed_forward" in n or "mlp" in n:
        g = "feed-forward (MLP)"
    else:
        g = "other"
    groups[g] += prm.numel()
total = sum(p.numel() for p in model.parameters())
for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {g:22s} {n/1e6:8.2f}M  ({100*n/total:5.1f}%)")
print(f"  {'TOTAL (unique)':22s} {total/1e6:8.2f}M")
tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
print(f"  output head shares the embedding matrix (tied): {tied}")
report["param_groups"] = dict(groups)
report["total_params"] = total
report["tied_embeddings"] = tied

print("\n" + "=" * 78 + "\n6. Shapes flowing through the network (forward hooks)\n" + "=" * 78)
text = "Robot ugv3 hauls the crate to dock B."
x = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
trace = []


def hook(name):
    def fn(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        i0 = inp[0] if inp else None
        trace.append({"module": name, "type": type(mod).__name__,
                      "in": list(i0.shape) if isinstance(i0, torch.Tensor) else None,
                      "out": list(o.shape) if isinstance(o, torch.Tensor) else None})
    return fn


handles = []
conv_i = next(r["layer"] for r in layer_rows if r["kind"] == "conv")
attn_i = next(r["layer"] for r in layer_rows if r["kind"] == "attention")
for name, mod in model.named_modules():
    depth_ok = name.count(".") <= 4
    in_focus = (f"layers.{conv_i}." in name or f"layers.{attn_i}." in name or name.endswith(f"layers.{conv_i}")
                or name.endswith(f"layers.{attn_i}") or "embed" in name or "lm_head" in name)
    if depth_ok and in_focus and name:
        handles.append(mod.register_forward_hook(hook(name)))
with torch.no_grad():
    out = model(**x)
for h in handles:
    h.remove()
print(f"  input_ids {tuple(x['input_ids'].shape)}  ->  logits {tuple(out.logits.shape)}")
for t in trace:
    print(f"  {t['module']:45s} {t['type']:22s} in={t['in']} out={t['out']}")
report["shape_trace"] = trace

print("\n" + "=" * 78 + "\n7. Linear layers (candidate LoRA targets) by short name\n" + "=" * 78)
lin = defaultdict(list)
for n, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        lin[n.split(".")[-1]].append(n)
for short, names in sorted(lin.items()):
    shape = tuple(dict(model.named_modules())[names[0]].weight.shape)
    print(f"  {short:12s} x{len(names):2d}  weight {shape}  e.g. {names[0]}")
report["linear_modules"] = {k: {"count": len(v), "example": v[0]} for k, v in lin.items()}
conv1d = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv1d)]
if conv1d:
    m = dict(model.named_modules())[conv1d[0]]
    print(f"\n  depthwise Conv1d modules: {len(conv1d)}, e.g. {conv1d[0]}: weight {tuple(m.weight.shape)}, "
          f"groups={m.groups}, kernel={m.kernel_size}, padding={m.padding}")
    report["conv1d"] = {"count": len(conv1d), "example": conv1d[0], "weight": list(m.weight.shape),
                        "groups": m.groups, "kernel": list(m.kernel_size)}

out_dir = OUTPUTS / "runs" / "lesson_b_arch" / args.model
write_json(out_dir / "architecture.json", report)
print("\nsaved", out_dir / "architecture.json")
