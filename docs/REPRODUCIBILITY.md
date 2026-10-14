# Reproducibility

## What is pinned

| Item | Where |
|---|---|
| Python packages (direct) | `requirements.txt` |
| Python packages (full, resolved) | `requirements-lock.txt` |
| Model revisions (Hub commit SHAs) | `configs/models.yaml` |
| Experiment settings | `configs/*.yaml` presets; each run copies its final config to `outputs/runs/<exp>/<run>/config.json` |
| Data | generated from seeds (`make data`); `data/*/stats.json` records sizes, balance and split overlap (always 0) |
| Evaluation settings | frozen in `lab/text_eval.py` (greedy, 40 new tokens) and `lab/vlm_utils.py` (greedy, 8 new tokens) |
| Machine | `results/environment.json` (written by `scripts/00_verify_env.py`) |

## Seeds and determinism

All scripts call `lab.common.set_seed` (Python, NumPy, PyTorch CPU and CUDA).
That makes data generation and data order deterministic. GPU kernels in BF16
(matrix multiplications, attention, scatter/atomics) are **not** bit-for-bit
deterministic, so rerunning a training job gives close but not identical
losses and can flip a few greedy answers. Every reported result here is a
**single seed** and should be read as preliminary. To measure seed variance,
rerun with `--set seed=1` and `--set seed=2` and compare.

## What each run records

`outputs/runs/<experiment>/<run>/`:

* `config.json`: the merged config, preset name, and command-line overrides
* `metrics.json`: environment versions, trainable parameter counts, gradient
  and parameter-change checks, training time, peak allocated/reserved VRAM,
  evaluation metrics
* `train_log.jsonl`: every logged step (loss, learning rate, grad norm, method-specific metrics)
* `eval_*.jsonl`: one row per test example with the **raw model output** and verifier fields
* `adapter/`: LoRA weights + config (reloadable with `PeftModel.from_pretrained`)

`results/experiments.jsonl` gets one summary line per finished run, and
`make report` rebuilds `results/tables/*.md` and `results/plots/*.png` from it.

## Timing caveat

All timings were measured on a laptop GPU (RTX 4060 Laptop, 8 GB) while other
jobs (a ROS 2 / Gazebo simulation batch) shared the CPU and about 0.8 GB of
VRAM. Treat durations as indicative, not as benchmarks.

## Resuming

Trainer-based scripts save checkpoints every `save_steps` (keeping the last
one). To resume an interrupted run, repeat the command with the same
`--run-name` and add `--resume` (`10_sft.py`). Lesson scripts are short enough
to rerun from scratch.
