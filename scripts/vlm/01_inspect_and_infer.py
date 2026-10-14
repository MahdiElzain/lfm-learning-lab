"""VLM Lesson A: inspect LFM2.5-VL-450M and run captioning / VQA / structured answers.

    .venv/bin/python scripts/vlm/01_inspect_and_infer.py
    .venv/bin/python scripts/vlm/01_inspect_and_infer.py --model vlm_small    # SmolVLM-256M

Writes outputs/runs/vlm_lesson_a/<model>/report.json.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab import shapes_task as st  # noqa: E402
from lab.common import OUTPUTS, GpuMemory, read_jsonl, DATA, write_json  # noqa: E402
from lab.models import load_vlm  # noqa: E402
from lab.vlm_utils import conversation, generate, load_image  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="vlm")
args = p.parse_args()
mem = GpuMemory()
mem.reset()
model, processor = load_vlm(args.model)
model.eval()
report = {"model": args.model, "class": type(model).__name__}


def show(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


show("1. Components and parameter counts")
groups = defaultdict(int)
for n, prm in model.named_parameters():
    key = n.split(".")[1] if n.startswith("model.") else n.split(".")[0]
    groups[key] += prm.numel()
total = sum(groups.values())
for k, v in groups.items():
    print(f"  {k:24s} {v/1e6:8.2f}M ({100*v/total:5.1f}%)")
report["param_groups"] = dict(groups)
inner = model.model
for name in ("vision_tower", "multi_modal_projector", "connector", "language_model", "text_model", "vision_model"):
    if hasattr(inner, name):
        print(f"  model.{name}: {type(getattr(inner, name)).__name__}")
if hasattr(inner, "multi_modal_projector"):
    print("  projector:", inner.multi_modal_projector)
cfg = model.config
vc = cfg.vision_config
report["vision_config"] = {k: getattr(vc, k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                                        "patch_size", "num_patches", "intermediate_size")
                           if hasattr(vc, k)}
report["downsample_factor"] = getattr(cfg, "downsample_factor", None)
print("  vision config:", report["vision_config"], "| downsample_factor:", report["downsample_factor"])

show("2. Processor settings")
ip = processor.image_processor
settings = {k: getattr(ip, k) for k in ("min_image_tokens", "max_image_tokens", "do_image_splitting", "tile_size",
                                        "encoder_patch_size", "downsample_factor", "use_thumbnail", "max_tiles",
                                        "min_tiles", "max_pixels_tolerance", "image_mean", "image_std",
                                        "do_resize", "size", "max_image_size") if hasattr(ip, k)}
for k, v in settings.items():
    print(f"  {k:22s} {v}")
report["processor"] = {k: str(v) for k, v in settings.items()}

show("3. One image + question -> tensors")
row = read_jsonl(DATA / "shapes" / "test.jsonl")[0]
image = load_image(row)
conv = conversation(image, row["question"])
text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
enc = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                    return_tensors="pt").to(model.device)
img_id = getattr(processor, "image_token_id", None)
n_img = int((enc["input_ids"] == img_id).sum()) if img_id is not None else None
compact = text
image_token = getattr(processor, "image_token", "<image>")
if n_img:
    compact = text.replace(image_token * n_img, f"{image_token} x {n_img}")
print("template (image placeholders compressed):\n" + compact)
tensors = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in enc.items() if isinstance(v, torch.Tensor)}
for k, v in tensors.items():
    print(f"  {k:22s} {v}")
print(f"  image tokens in input_ids: {n_img} of {enc['input_ids'].shape[1]} total tokens")
if "spatial_shapes" in enc:
    print("  spatial_shapes (patch grid h, w):", enc["spatial_shapes"].tolist())
report["single_example"] = {"question": row["question"], "answer": row["answer"], "template": compact,
                            "tensors": tensors, "image_tokens": n_img}

show("4. Shapes through vision tower -> projector -> language model")
trace = []


def hook(name):
    def fn(m, inp, out):
        o = out[0] if isinstance(out, tuple) else getattr(out, "last_hidden_state", out)
        i0 = inp[0] if inp and isinstance(inp[0], torch.Tensor) else None
        trace.append({"module": name, "in": list(i0.shape) if i0 is not None else None,
                      "out": list(o.shape) if isinstance(o, torch.Tensor) else str(type(o))})
    return fn


handles = []
for name in ("vision_tower", "multi_modal_projector", "connector", "language_model", "text_model"):
    if hasattr(inner, name):
        handles.append(getattr(inner, name).register_forward_hook(hook(f"model.{name}")))
handles.append(model.lm_head.register_forward_hook(hook("lm_head")))
if "pixel_values" in enc:
    enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
with torch.no_grad():
    out = model(**enc)
for h in handles:
    h.remove()
for t in trace:
    print(f"  {t['module']:28s} in={t['in']} out={t['out']}")
report["shape_trace"] = trace

show("5. Captioning, VQA and structured answers on a generated scene")
prompts = [
    ("caption", "Describe this image in one sentence."),
    ("vqa_count", "How many shapes are in this image? Answer with a number."),
    ("vqa_color", "What color is the leftmost shape? Answer with one word."),
    ("structured", 'List every shape as JSON: [{"shape": ..., "color": ...}]. Output JSON only.'),
]
truth = [(o["color"], o["shape"]) for o in sorted(row["scene"]["objects"], key=lambda o: o["cx"])]
print("ground truth (left to right):", truth)
gens = generate(model, processor, [image] * len(prompts), [q for _, q in prompts], max_new_tokens=80, batch_size=4)
report["demos"] = []
for (name, q), ans in zip(prompts, gens["texts"]):
    print(f"\n[{name}] {q}\n  -> {ans}")
    report["demos"].append({"task": name, "question": q, "answer": ans})
report["demo_truth_left_to_right"] = truth
report["vram"] = mem.snapshot()
out_dir = OUTPUTS / "runs" / "vlm_lesson_a" / args.model
write_json(out_dir / "report.json", report)
image.save(out_dir / "example.png")
print("\nVRAM:", report["vram"])
print("saved", out_dir)
