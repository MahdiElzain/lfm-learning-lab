# Progress log and completion checklists

This file is the resume point for interrupted sessions. Each entry lists what
was done, the exact command, and where the output lives. Checklists at the
bottom are updated only after a step has actually run.

## Verified machine facts (2026-10-14)

| Item | Reported | Measured |
|---|---|---|
| GPU | RTX 4060 Laptop, 8 GB | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, compute capability 8.9 |
| Driver | 580.159.03 | **580.178.04** (driver supports CUDA up to 13.0) |
| PyTorch | 2.11.0+cu128 | Not found in any existing environment on the machine (found 2.7.1+cu126 and 2.10.0+cu128 elsewhere). The lab's own env installs 2.11.0+cu128. |
| CPU | i9-14900HX, 24C/32T | i9-14900HX, 24 cores, 32 threads |
| RAM | — | 30 GiB total, ~16 GiB available, 33 GiB swap (9 GiB in use) |
| Disk | — | **19 GB free of 399 GB (96% used)**: the main constraint |
| OS | Ubuntu 24.04 | Ubuntu 24.04.4 LTS, kernel 7.0.0-38 |
| System Python | — | 3.12.3 |

Shared-machine note: during this work a ROS 2 / Gazebo simulation batch
(headless `gz sim` workers) was running and used ~0.8 GB of VRAM and heavy CPU. The lab does not touch it. Timings in
this lab were measured under that background load unless stated otherwise.

Driver CUDA versus PyTorch CUDA: `nvidia-smi` reports the newest CUDA the
*driver* can serve (13.0). PyTorch ships its own CUDA *runtime* (12.8 for
the `+cu128` wheel). A newer driver runs older runtimes, so 12.8 on a 13.0
driver is fine. Nothing in the system CUDA toolkit is used or changed.

## Log

- 2026-10-14: Inspected the workspace (no project instructions found; unrelated folders left
  untouched). Created the lab directory.
- 2026-10-14: Verified models, revisions, and licenses (see
  `configs/models.yaml`). LFM models use the LFM Open License v1.0, which
  allows redistribution of derivatives with license copy, modification
  notices, and retained attribution; it limits commercial use for entities
  above US$10M annual revenue. SmolLM2/SmolVLM/Qwen3.5 are Apache-2.0.
- 2026-10-14: First install attempt was interrupted by a session restart
  during CUDA library downloads. Relaunched detached:
  `bash scripts/tools/install_env.sh > logs/install.log`.
- 2026-10-14: **Moved the whole lab to the external drive** at the user's request:
  an external USB SSD (1.9 TB, exFAT). The internal copy was deleted after a verified file-count match.
  exFAT has no symlinks/hardlinks and is case-insensitive, so: (1) the venv is
  built by hand (copied interpreter + `pyvenv.cfg`) by `scripts/tools/install_env.sh`;
  (2) packages are installed with `UV_LINK_MODE=copy`; (3) the HF cache lives at
  `models/hf` (set in `lab/__init__.py`) and stores real files instead of symlinks.
  Qwen3.5-2B was copied (not moved) from `~/.cache/huggingface` into `models/hf`.
  The drive must be mounted at the same path for the venv to work; if it mounts
  elsewhere, re-run `scripts/tools/install_env.sh` there.

- 2026-10-14: Environment installed on the drive (`INSTALL_OK`); `scripts/00_verify_env.py` -> ENVIRONMENT OK
  (torch 2.11.0+cu128, CUDA runtime 12.8, BF16 yes, 7.16 GB VRAM free). 25,459 installed files checked: no
  exFAT case collisions. Removed the 9.5 GB uv download cache this install had created on the internal disk.
- 2026-10-14: Models downloaded to `models/hf` (LFM2.5-350M, -Base, SmolLM2-135M, LFM2.5-VL-450M,
  LFM2.5-1.2B-Instruct). A first download attempt wrote 6.6 MB to `~/.cache/huggingface` because
  `download_models.py` imported huggingface_hub before `lab`; fixed, the stray LiquidAI folder removed,
  and an assertion now guards the cache path. Qwen3.5-2B copy verified against the Hub SHA-256.
- 2026-10-14: Lab now runs with HF_HUB_OFFLINE=1 by default (`LAB_ONLINE=1` to override); a PEFT network
  check over flaky Wi-Fi had inflated an adapter reload from 0.4 s to 17 s.
- 2026-10-14: Bugs found by running and fixed: tiny GPT init (initial loss 82 -> 3.75 with std-0.02 init);
  `warmup_ratio` removed in transformers 5.x (now `warmup_steps` float); PyYAML parsed `2e-5` as a string
  (override parser now falls back to float); Lesson A greedy loop now stops at EOS and reports real conv-cache shapes.
- Runs completed (see results/experiments.jsonl): tiny_transformer quick; lesson A; lesson B1; text baselines
  (instruct_zero_shot, base_2shot, smollm2_135m_zero_shot); text_sft lora_quick, lora_standard, overfit_tiny,
  full_quick, full_quick_ckpt, lora_fullseq_quick; sequence distillation qwen_nothink_standard (+ student
  distilled_qwen35_2b_seq_standard, control gold_same294_standard); teacher profile qwen_thinking_profile.
- 1.2B teacher LoRA SFT at batch 8 ran out of memory (6.95 GB in use); rerunning at 4x4 + gradient checkpointing.
- 2026-10-14: Text track core done. Extra runs: soft KD (kd_soft_standard) + same-loop hard baseline
  (kd_hard_standard); DPO x3 (lr 1e-5, lr 5e-5, lr 5e-5 + SFT term); GRPO quick/standard/binary;
  best-of-8 for SFT and SFT+GRPO.
- 2026-10-14: VLM track: inspection + demo (vlm_lesson_a), base eval with correct/swapped/blank,
  LoRA SFT (lora_quick) and connector variant (lora_connector_quick). Visual distillation running.
- Hugging Face CLI: not logged in (needs the user: `hf auth login` in their own terminal).
- 2026-10-14: Found that colour-count questions were 61% "0" (unbalanced). Fixed the generator
  (ask about a present colour 70% of the time; majority answer now 39%), archived the earlier VLM runs
  under `outputs/archive/vlm_unbalanced_count_color/`, and re-ran the whole VLM track with
  `bash scripts/tools/run_vlm_track.sh` (VLM_TRACK_OK).
- 2026-10-14: Multimodal GRPO with TRL 1.15 works on LFM2.5-VL (not on TRL's tested list). Verified
  multimodal conditioning with a blank-image ablation (reward 0.78 real vs 0.28 blank).
- 2026-10-14: Publication prep: model cards built with `scripts/tools/publish_hf.py --dry-run`;
  repository scanned for private paths/names (none); GitHub login is `MahdiElzain` (SSH works).
  User chose: GitHub PRIVATE repo on the personal account; both adapters PUBLIC on HF; suggested bio.
  HF model cards made self-contained (system prompt embedded) because the code repo is private.

## Text-track checklist

- [x] Environment verified on GPU
- [x] A. Tokenization and generation
- [x] B. Architecture inspection
- [x] B. Tiny Transformer mechanics demo
- [x] C. Task + verifier
- [x] C. Baselines (instruct, base, small)
- [x] C. SFT with LoRA (+ profiling, save/reload)
- [x] C. Overfitting demo
- [x] C. Bounded full fine-tuning comparison
- [x] D. Sequence distillation
- [x] D. Soft-target (KL) distillation
- [x] E. DPO
- [x] F. GRPO + best-of-n comparison

## VLM-track checklist

- [x] A. Architecture inspection + inference script
- [x] A. Resolution / image-token measurements
- [x] B. Shapes dataset (scene-level splits)
- [x] B. Baselines
- [x] B. LoRA SFT (vision frozen)
- [x] B. Connector-training comparison
- [x] C. Visual sequence distillation
- [x] D. Multimodal GRPO step verified (or blocked with evidence)
- [x] E. Visual-dependence evaluation

## Publication checklist

- [x] Docs: README, setup, troubleshooting, glossary, lessons, exercises, solutions, reproducibility, licensing
- [x] Results tables and plots
- [x] GitHub repository created and pushed (private, github.com/MahdiElzain/lfm-learning-lab, user's choice)
- [ ] Hugging Face login (requires user action: `hf auth login`)
- [ ] Hugging Face adapter(s) uploaded with model cards
- [ ] Published artifacts re-downloaded and loaded successfully
