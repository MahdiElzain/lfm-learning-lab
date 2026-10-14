"""Lesson F2: multiple-sample verifier selection (best-of-n) and its cost.

    .venv/bin/python scripts/text/41_best_of_n.py --adapter outputs/runs/text_sft/lora_quick/adapter --name sft_bon
    .venv/bin/python scripts/text/41_best_of_n.py --k 16 --temperature 0.7 --name sft_bon_k16 --adapter ...

Strategies compared on the same prompts:
  greedy           one deterministic answer (the frozen evaluation setting)
  first_valid      first of k samples that passes the LABEL-FREE check
                   (parses, eligible robot, self-consistent distance)
  majority_valid   most common answer among label-free-valid samples
  oracle_pass@k    any of k samples exactly right. Needs gold labels, so it is an
                   UPPER BOUND, not something you could deploy.
Cost is reported as generated tokens and seconds relative to greedy.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401

from lab import robot_task as rt  # noqa: E402
from lab.common import (DATA, OUTPUTS, append_experiment_index, read_jsonl, set_seed, write_json,  # noqa: E402
                        write_jsonl)
from lab.models import load_text_model  # noqa: E402
from lab.text_eval import generate, robot_prompts  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="text_instruct")
p.add_argument("--adapter", default=None)
p.add_argument("--merge-first", default=None, help="adapter to merge before --adapter (e.g. SFT under GRPO)")
p.add_argument("--k", type=int, default=8)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--split", default="test")
p.add_argument("--limit", type=int, default=300)
p.add_argument("--name", required=True)
args = p.parse_args()
set_seed(0)

model, tok = load_text_model(args.model)
if args.merge_first:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.merge_first).merge_and_unload()
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
rows = read_jsonl(DATA / "robot" / f"{args.split}.jsonl")[: args.limit]
prompts = robot_prompts(tok, rows)
greedy = generate(model, tok, prompts)
sampled = generate(model, tok, prompts, do_sample=True, temperature=args.temperature, top_p=1.0,
                   num_return_sequences=args.k, seed=0, batch_size=16)

per_example, counts = [], Counter()
for ex, g_t, s_t in zip(rows, greedy["texts"], sampled["texts"]):
    valid = [s for s in s_t if rt.label_free_check(ex, s)["ok"]]
    first_valid = valid[0] if valid else s_t[0]
    if valid:
        canon = Counter(rt.format_answer(rt.parse_completion(s)[0]) for s in valid)
        majority = canon.most_common(1)[0][0]
    else:
        majority = s_t[0]
    r = {"greedy": rt.verify(ex, g_t[0])["exact"],
         "first_valid": rt.verify(ex, first_valid)["exact"],
         "majority_valid": rt.verify(ex, majority)["exact"],
         "oracle_pass@k": any(rt.verify(ex, s)["exact"] for s in s_t),
         "mean_sample_exact": sum(rt.verify(ex, s)["exact"] for s in s_t) / len(s_t),
         "n_label_free_valid": len(valid)}
    for k_, v in r.items():
        counts[k_] += v
    per_example.append({"id": ex["id"], "greedy_output": g_t[0], "samples": s_t, **r})
n = len(rows)
gen_tok_greedy = sum(x[0] for x in greedy["new_tokens"])
gen_tok_sampled = sum(sum(x) for x in sampled["new_tokens"])
summary = {
    "name": args.name, "adapter": args.adapter, "merge_first": args.merge_first, "k": args.k,
    "temperature": args.temperature, "split": args.split, "n": n,
    "accuracy": {k_: round(counts[k_] / n, 4) for k_ in ("greedy", "first_valid", "majority_valid", "oracle_pass@k",
                                                         "mean_sample_exact")},
    "mean_label_free_valid_per_prompt": round(counts["n_label_free_valid"] / n, 2),
    "cost": {"greedy_tokens": gen_tok_greedy, "sampled_tokens": gen_tok_sampled,
             "token_ratio": round(gen_tok_sampled / max(1, gen_tok_greedy), 2),
             "greedy_seconds": round(greedy["seconds"], 1), "sampled_seconds": round(sampled["seconds"], 1),
             "time_ratio": round(sampled["seconds"] / max(1e-9, greedy["seconds"]), 2)},
}
out = OUTPUTS / "evals" / args.name
write_jsonl(out / f"{args.split}_best_of_n.jsonl", per_example)
write_json(out / "best_of_n.json", summary)
print(summary)
append_experiment_index({"experiment": "text_best_of_n", **{k_: v for k_, v in summary.items() if k_ != "cost"},
                         **summary["cost"]})
