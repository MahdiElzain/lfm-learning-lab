# Start here: the learning path

Work through the lessons in this order. Each one has the same seven parts:
concept, equations (every symbol defined), tensors/shapes/code, a runnable
experiment, the actual results from this machine, an exercise, and a separate
solution file. Run the experiment yourself before reading the results section,
then change one variable and run it again.

| # | Lesson | Command | Time (quick) |
|---|---|---|---|
| 0 | Setup and verification | `make verify && make test` | < 1 min |
| A | Tokens, logits and generation | `make tokens` | ~1 min |
| B1 | LFM2.5 hybrid architecture | `make arch` | < 1 min |
| B2 | Tiny Transformer from scratch | `make tiny` | 35 s (two trainings) |
| C | Robot-dispatch task, baselines, SFT, LoRA, overfitting | `make baselines sft overfit` | ~3 min (SFT itself 39 s) |
| D | Distillation (sequence, soft-target) | `make distill-seq`, then Lesson D steps | 10-20 min (standard size) |
| E | DPO on verifier-derived preferences | `make dpo` | ~2 min |
| F | GRPO, best-of-n, reward pitfalls | `make grpo bon` | ~3 min (quick) / 6 min (standard) |
| V1 | VLM architecture, inference, resolution | `make vlm-inspect vlm-sweep` | ~3 min |
| V2 | Shape scenes, VLM SFT, connector training | `make vlm-baselines vlm-sft vlm-sft-connector` | ~5 min |
| V3 | Visual distillation | `make vlm-distill` | ~5 min |
| V4 | Multimodal GRPO | `make vlm-grpo` | ~1 min (quick) |
| V5 | Visual dependence (correct / swapped / blank) | part of every `scripts/vlm/evaluate.py` run | ~1 min per model |

Times were measured on an RTX 4060 Laptop GPU shared with other jobs; lessons V2-V5 are in
`lessons/V2_vlm_training.md`. Running the scripts is not the same as understanding them. The exercises are
where the learning happens: predict the outcome first, then run, then explain
any gap between prediction and result.

Single-seed results in this lab are **preliminary**. Differences of a few
percentage points between runs can come from the random seed alone. Exercise
C.2 shows you how to check.
