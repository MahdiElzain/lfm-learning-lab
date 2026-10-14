# One-word commands for the learning lab. Every target is a thin wrapper around
# a script you can also run directly (see README.md). PRESET=quick|standard|extended
PY ?= .venv/bin/python
PRESET ?= quick
SFT ?= outputs/runs/text_sft/lora_standard/adapter

.PHONY: setup verify test download data tokens arch tiny baselines sft sft-full overfit \
        distill-seq distill-soft dpo grpo bon vlm-inspect vlm-sweep vlm-data vlm-baselines vlm-sft \
        vlm-sft-connector vlm-distill vlm-grpo vlm-dependence report

setup:            ## create .venv and install pinned packages
	bash scripts/tools/install_env.sh
verify:           ## versions, GPU, BF16, VRAM, disk
	$(PY) scripts/00_verify_env.py
test:             ## deterministic verifier / generator tests
	$(PY) tests/test_robot_verifier.py && $(PY) tests/test_shapes.py
download:         ## pinned model snapshots into models/hf
	$(PY) scripts/tools/download_models.py
data:             ## robot-dispatch + shape-scene datasets
	$(PY) scripts/text/make_robot_data.py && $(PY) scripts/vlm/make_shapes_data.py

# ---------------------------------------------------------------- text track
tokens:           ## Lesson A
	$(PY) scripts/text/01_tokens_and_generation.py
arch:             ## Lesson B1
	$(PY) scripts/text/02_inspect_architecture.py
tiny:             ## Lesson B2
	$(PY) scripts/text/03_tiny_transformer.py --preset $(PRESET)
baselines:        ## Lesson C baselines
	$(PY) scripts/text/evaluate.py --model text_instruct --name instruct_zero_shot
	$(PY) scripts/text/evaluate.py --model text_base --mode base_fewshot --name base_2shot
	$(PY) scripts/text/evaluate.py --model text_small --name smollm2_135m_zero_shot
sft:              ## Lesson C LoRA SFT
	$(PY) scripts/text/10_sft.py --preset $(PRESET) --run-name lora_$(PRESET)
sft-full:         ## Lesson C bounded full fine-tuning
	$(PY) scripts/text/10_sft.py --preset $(PRESET) --run-name full_$(PRESET) --set method=full \
	  --set per_device_train_batch_size=4 --set gradient_accumulation_steps=4 --set gradient_checkpointing=true \
	  --set learning_rate=2e-5 --set optim=adamw_8bit
overfit:          ## Lesson C overfitting demo (16 examples, many epochs)
	$(PY) scripts/text/10_sft.py --run-name overfit_tiny --train-file robot/overfit_tiny.jsonl \
	  --set n_train=16 --set num_train_epochs=40 --set per_device_train_batch_size=4 \
	  --set gradient_accumulation_steps=1 --set eval_steps=20 --set save_steps=1000 --set logging_steps=4
distill-seq:      ## Lesson D1
	$(PY) scripts/text/20_distill_sequence.py --preset $(PRESET)
distill-soft:     ## Lesson D2 (set TEACHER_ADAPTER=...)
	$(PY) scripts/text/21_distill_soft.py --preset $(PRESET) --run-name kd_soft --set teacher_adapter=$(TEACHER_ADAPTER)
dpo:              ## Lesson E
	$(PY) scripts/text/30_dpo.py --preset $(PRESET) --run-name dpo_$(PRESET) --set sft_adapter=$(SFT)
grpo:             ## Lesson F
	$(PY) scripts/text/40_grpo.py --preset $(PRESET) --run-name grpo_$(PRESET) --set sft_adapter=$(SFT)
bon:              ## Lesson F2 best-of-n
	$(PY) scripts/text/41_best_of_n.py --adapter $(SFT) --name sft_best_of_8

# ----------------------------------------------------------------- VLM track
vlm-inspect:
	$(PY) scripts/vlm/01_inspect_and_infer.py
vlm-sweep:
	$(PY) scripts/vlm/02_resolution_sweep.py
vlm-baselines:
	$(PY) scripts/vlm/evaluate.py --name vlm_base
vlm-sft:
	$(PY) scripts/vlm/10_sft.py --preset $(PRESET) --run-name lora_$(PRESET)
vlm-sft-connector:
	$(PY) scripts/vlm/10_sft.py --preset $(PRESET) --run-name lora_connector_$(PRESET) --set train_connector=true
vlm-distill:
	$(PY) scripts/vlm/20_distill_sequence.py --preset $(PRESET)
vlm-grpo:
	$(PY) scripts/vlm/40_grpo.py --preset $(PRESET)

report:           ## tables + plots from every finished run
	$(PY) scripts/tools/report.py
