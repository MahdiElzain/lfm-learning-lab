# Setup

## Where things live

| Path | What |
|---|---|
| `lab/` | small shared library (tasks, verifiers, evaluation, helpers) |
| `scripts/text/`, `scripts/vlm/` | numbered lesson scripts and trainers |
| `scripts/tools/` | install, model download, reporting |
| `configs/` | one YAML per experiment with `quick` / `standard` / `extended` presets; `models.yaml` pins every model revision |
| `data/` | generated datasets (deterministic from a seed) |
| `models/hf/` | Hugging Face cache for this lab (set in `lab/__init__.py`) |
| `outputs/runs/<experiment>/<run>/` | config, metrics, logs, raw outputs, adapters |
| `outputs/evals/<name>/` | evaluation summaries and raw generations |
| `results/` | experiment index, tables, plots (committed to git) |
| `lessons/`, `exercises/`, `solutions/` | lesson guides and exercises (solutions are separate on purpose) |

`data/`, `models/`, `outputs/` and `.venv/` are not committed; everything in
them can be regenerated with the commands below.

## 1. Environment

Requirements: Linux, an NVIDIA GPU with a recent driver, Python 3.12, and
[`uv`](https://docs.astral.sh/uv/).

```bash
cd lfm-learning-lab                                  # wherever you cloned the repo
bash scripts/tools/install_env.sh                   # creates .venv and installs pinned packages
.venv/bin/python scripts/00_verify_env.py           # versions, GPU, BF16, VRAM, disk
```

The pinned stack is in `requirements.txt` (direct dependencies) and
`requirements-lock.txt` (every installed package, written by the install script):
PyTorch 2.11.0+cu128, Transformers 5.19.0, TRL 1.15.0, PEFT 0.21.2,
Datasets 5.1.0, Accelerate 1.15.0.

**Driver CUDA versus PyTorch CUDA.** `nvidia-smi` prints the newest CUDA the
*driver* supports (13.0 here). The `+cu128` PyTorch wheel bundles its own CUDA
12.8 runtime libraries. Newer drivers run older runtimes, so nothing in the
system CUDA installation is used or changed.

**exFAT drives.** This lab was developed on an exFAT USB drive. exFAT has no
symlinks, so `uv venv` and `python -m venv` fail there. `install_env.sh`
detects this and builds the venv by hand (a copied interpreter plus a
`pyvenv.cfg`, which is all Python needs). The Hugging Face cache also falls
back to real file copies. On ext4 or similar, the normal `uv venv` path is used.
The venv records absolute paths in a few console scripts, so if the drive
mounts at a different path, re-run the install script (or call
`.venv/bin/python -m <module>` instead of the console scripts).

## 2. Models and data

```bash
make download        # LFM2.5-350M, LFM2.5-350M-Base, SmolLM2-135M-Instruct, LFM2.5-VL-450M
make data            # robot-dispatch JSONL + shape-scene images, split by scene
make test            # deterministic verifier tests (no GPU needed)
```

Optional models are downloaded on first use: `LFM2.5-1.2B-Instruct` (soft
distillation teacher), `SmolVLM-256M-Instruct` (VLM fallback). The
`Qwen3.5-2B` teacher is loaded from `models/hf` if present, otherwise downloaded.

## 3. Running lessons

Every training/eval script accepts:

```text
--preset quick|standard|extended    size of the run (quick = minutes)
--set key=value                     change ONE config value (repeatable)
--run-name NAME                     output folder name
```

Examples:

```bash
make tokens                                    # Lesson A
make sft PRESET=quick                          # LoRA SFT
.venv/bin/python scripts/text/10_sft.py --set lora_r=4 --run-name lora_r4
.venv/bin/python scripts/text/10_sft.py --run-name lora_r4 --resume    # continue from the last checkpoint
make report                                    # rebuild tables and plots
```

## 4. GPU memory etiquette

The laptop GPU has 8 GB and is shared with the desktop and other jobs. Every
script prints peak *allocated* and *reserved* VRAM. Runs are sequential; do not
launch two trainings at once. If something else is using the GPU, check with
`nvidia-smi` first and see `docs/TROUBLESHOOTING.md`.
