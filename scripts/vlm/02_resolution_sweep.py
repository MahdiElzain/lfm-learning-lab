"""VLM Lesson A2: how resolution, tiling and the image-token budget change cost and accuracy.

    .venv/bin/python scripts/vlm/02_resolution_sweep.py               # default grid, 90 questions
    .venv/bin/python scripts/vlm/02_resolution_sweep.py --n 150

The SAME test scenes are re-rendered at each resolution (shapes are vector
drawings, so nothing is blurred by upscaling). For each setting we record image
tokens per image, latency, peak VRAM and accuracy. Results:
results/tables/vlm_resolution_sweep.md and outputs/runs/vlm_resolution_sweep/<time>/sweep.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab import shapes_task as st  # noqa: E402
from lab.common import DATA, RESULTS, GpuMemory, make_run_dir, read_jsonl, set_seed, write_json  # noqa: E402
from lab.models import load_vlm  # noqa: E402
from lab.vlm_utils import generate  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=90, help="questions (from the test split)")
args = p.parse_args()
set_seed(0)
rows = read_jsonl(DATA / "shapes" / "test.jsonl")[: args.n]
grid = [  # (render size, processor overrides)
    (128, {}), (256, {}), (512, {}), (768, {}), (1024, {}),
    (512, {"max_image_tokens": 64}), (512, {"max_image_tokens": 128}),
    (512, {"max_image_tokens": 64, "do_image_splitting": False}),
    (1024, {"do_image_splitting": False}),
    (1024, {"max_image_tokens": 64, "do_image_splitting": False}),
]
model, processor = load_vlm("vlm")
model.eval()
ip = processor.image_processor
defaults = {k: getattr(ip, k) for k in ("min_image_tokens", "max_image_tokens", "do_image_splitting")}
mem = GpuMemory()
# Untimed warm-up: the first CUDA calls initialise kernels and allocator pools, which
# otherwise inflates whichever setting happens to run first.
generate(model, processor, [st.render(r["scene"], 256) for r in rows[:16]], [r["question"] for r in rows[:16]], batch_size=8)
results = []
for size, over in grid:
    for k, v in defaults.items():
        setattr(ip, k, v)
    for k, v in over.items():
        setattr(ip, k, v)
    images = [st.render(r["scene"], size) for r in rows]
    mem.reset()
    gen = generate(model, processor, images, [r["question"] for r in rows], batch_size=8)
    acc = sum(st.verify(r["type"], r["answer"], t)["correct"] for r, t in zip(rows, gen["texts"])) / len(rows)
    rec = {"render_size": size, "overrides": over or "model-card defaults",
           "image_tokens_mean": round(sum(gen["image_tokens"]) / len(rows), 1),
           "accuracy": round(acc, 4), "ms_per_question": round(1000 * gen["seconds"] / len(rows), 1),
           **{k: v for k, v in mem.snapshot().items() if k.startswith("peak")}}
    results.append(rec)
    print(rec)
    torch.cuda.empty_cache()
run_dir = make_run_dir("vlm_resolution_sweep")
write_json(run_dir / "sweep.json", {"n_questions": len(rows), "defaults": defaults, "results": results})
lines = ["| render size | processor setting | image tokens / image | accuracy | ms / question | peak alloc GB |",
         "|---|---|---|---|---|---|"]
for r in results:
    lines.append(f"| {r['render_size']} | {r['overrides']} | {r['image_tokens_mean']} | {r['accuracy']:.3f} | "
                 f"{r['ms_per_question']} | {r.get('peak_allocated_gb')} |")
note = (f"\nBase LFM2.5-VL-450M, greedy, batch 8, {len(rows)} test questions, single run. "
        "Latency was measured while other GPU/CPU jobs shared the laptop, so treat it as indicative.\n")
(RESULTS / "tables").mkdir(parents=True, exist_ok=True)
(RESULTS / "tables" / "vlm_resolution_sweep.md").write_text("\n".join(lines) + "\n" + note)
print("\n".join(lines))
