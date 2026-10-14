"""Ask a vision-language model a question about a local image.

    .venv/bin/python scripts/vlm/infer.py --image photo.jpg --question "What is in this image?"
    .venv/bin/python scripts/vlm/infer.py --image scene.png --question "How many shapes?" --max-image-tokens 64
    .venv/bin/python scripts/vlm/infer.py --image scene.png --question "..." --adapter outputs/runs/vlm_sft/lora_quick/adapter
    .venv/bin/python scripts/vlm/infer.py --image scene.png --question "..." --model vlm_small    # SmolVLM-256M

Prints the answer, the number of image tokens used, latency and peak VRAM.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
from PIL import Image  # noqa: E402

from lab.common import GpuMemory  # noqa: E402
from lab.models import load_vlm  # noqa: E402
from lab.vlm_utils import generate  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--image", required=True)
p.add_argument("--question", required=True)
p.add_argument("--model", default="vlm")
p.add_argument("--adapter", default=None)
p.add_argument("--max-image-tokens", type=int, default=None)
p.add_argument("--no-tiling", action="store_true", help="disable image splitting (LFM2.5-VL)")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
args = p.parse_args()

proc_kwargs = {}
if args.max_image_tokens:
    proc_kwargs["max_image_tokens"] = args.max_image_tokens
if args.no_tiling:
    proc_kwargs["do_image_splitting"] = False
mem = GpuMemory()
mem.reset()
model, processor = load_vlm(args.model, processor_kwargs=proc_kwargs)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
image = Image.open(args.image).convert("RGB")
t0 = time.perf_counter()
out = generate(model, processor, [image], [args.question], max_new_tokens=args.max_new_tokens, batch_size=1,
               do_sample=args.temperature > 0, temperature=args.temperature or 1.0)
dt = time.perf_counter() - t0
print(out["texts"][0])
print(f"\n[image {image.size[0]}x{image.size[1]} -> {out['image_tokens'][0] if out['image_tokens'] else '?'} image tokens | "
      f"{out['new_tokens'][0]} new tokens | {dt*1000:.0f} ms | peak {mem.snapshot().get('peak_allocated_gb')} GB]")
