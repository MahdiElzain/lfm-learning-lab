"""Evaluate a text model (optionally with a LoRA adapter) on the robot-dispatch splits.

Frozen settings: greedy decoding, max_new_tokens=40 (see lab/text_eval.py).

    # baselines
    .venv/bin/python scripts/text/evaluate.py --model text_instruct --name instruct_zero_shot
    .venv/bin/python scripts/text/evaluate.py --model text_base --mode base_fewshot --name base_2shot
    .venv/bin/python scripts/text/evaluate.py --model text_small --name smollm2_135m_zero_shot
    # a trained adapter
    .venv/bin/python scripts/text/evaluate.py --adapter outputs/runs/text_sft/<run>/adapter --name my_sft

Raw outputs: outputs/evals/<name>/<split>.jsonl ; summary: outputs/evals/<name>/metrics.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab.common import (DATA, OUTPUTS, GpuMemory, append_experiment_index, environment_info, read_jsonl,  # noqa: E402
                        set_seed, write_json, write_jsonl)
from lab.models import load_text_model, resolve  # noqa: E402
from lab.text_eval import EVAL_MAX_NEW_TOKENS, evaluate_robot  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="text_instruct", help="registry key or repo id (base for --adapter)")
p.add_argument("--adapter", default=None, help="path to a saved PEFT adapter")
p.add_argument("--full-model", default=None, help="path to a saved full fine-tuned model")
p.add_argument("--mode", default="chat", choices=["chat", "base_fewshot"])
p.add_argument("--splits", nargs="+", default=["test", "test_ood"])
p.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples per split")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--name", required=True, help="folder name under outputs/evals/")
args = p.parse_args()
set_seed(0)

mem = GpuMemory()
mem.reset()
t0 = time.perf_counter()
if args.full_model:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.full_model)
    model = AutoModelForCausalLM.from_pretrained(args.full_model, dtype=torch.bfloat16).cuda()
else:
    model, tok = load_text_model(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
load_seconds = time.perf_counter() - t0
shots = read_jsonl(DATA / "robot" / "train.jsonl")[:2] if args.mode == "base_fewshot" else None

out_dir = OUTPUTS / "evals" / args.name
summary = {"name": args.name, "model": resolve(args.model), "adapter": args.adapter, "full_model": args.full_model,
           "mode": args.mode, "max_new_tokens": EVAL_MAX_NEW_TOKENS, "decoding": "greedy",
           "load_seconds": round(load_seconds, 2), "env": environment_info(), "splits": {}}
for split in args.splits:
    rows = read_jsonl(DATA / "robot" / f"{split}.jsonl")[: args.limit]
    agg, raw = evaluate_robot(model, tok, rows, mode=args.mode, shots=shots, batch_size=args.batch_size)
    write_jsonl(out_dir / f"{split}.jsonl", raw)
    summary["splits"][split] = agg
    print(f"{args.name:28s} {split:9s} n={agg['n']:4d} exact={agg['exact']:.3f} robot={agg['robot_correct']:.3f} "
          f"dist={agg['distance_correct']:.3f} parsed={agg['parsed']:.3f} strict={agg['valid_strict']:.3f} "
          f"viol(status/type)={agg['violation_status']:.3f}/{agg['violation_type']:.3f} "
          f"trap_exact={agg['trap_exact']:.3f} {agg['ms_per_example']}ms/ex")
summary["vram"] = mem.snapshot()
write_json(out_dir / "metrics.json", summary)
append_experiment_index({"experiment": "text_eval", "name": args.name, "adapter": args.adapter, "mode": args.mode,
                         **{f"{s}_{k}": v[k] for s, v in summary["splits"].items()
                            for k in ("exact", "robot_correct", "distance_correct", "parsed", "valid_strict",
                                      "violation_status", "violation_type", "n")},
                         "peak_allocated_gb": summary["vram"].get("peak_allocated_gb")})
print("saved", out_dir)
