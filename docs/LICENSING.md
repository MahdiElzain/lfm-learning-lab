# Licensing

Code and model weights are licensed separately. Keep them apart when you
copy or publish anything from this lab.

## This repository's code

All source code, configs, lesson text, exercises and generated datasets in this
repository are released under the **Apache License 2.0** (`LICENSE`).
Copyright 2026 maelzain.

## Models used (not redistributed by this repository)

| Model | License | Notes |
|---|---|---|
| LiquidAI/LFM2.5-350M | LFM Open License v1.0 | primary text model |
| LiquidAI/LFM2.5-350M-Base | LFM Open License v1.0 | base comparison |
| LiquidAI/LFM2.5-1.2B-Instruct | LFM Open License v1.0 | soft-distillation teacher |
| LiquidAI/LFM2.5-VL-450M | LFM Open License v1.0 | primary VLM |
| HuggingFaceTB/SmolLM2-135M-Instruct | Apache-2.0 | small reference |
| HuggingFaceTB/SmolVLM-256M-Instruct | Apache-2.0 | VLM fallback |
| Qwen/Qwen3.5-2B | Apache-2.0 | sequence-distillation teacher (text and vision) |

Exact revisions are pinned in `configs/models.yaml`.

### LFM Open License v1.0, in short (read the full text)

The full license ships with each Liquid model (`LICENSE` in the model repo).
The points that matter for this lab:

* **Use, modification and redistribution are allowed**, including of
  derivative works such as LoRA adapters, under the conditions below.
* **Redistribution conditions (Section 4):** give recipients a copy of the
  license; mark modified files prominently; keep the copyright, patent,
  trademark and attribution notices; include any NOTICE file contents.
* **Commercial-use limit (Section 5):** commercial use is licensed only for
  entities below US$10M annual revenue (non-profits have a research exemption).
* **No trademark rights (Section 7)** beyond describing the model's origin.

## Adapters published from this lab

Each adapter published to the Hugging Face Hub is a derivative of an LFM
model, so its repository:

* is marked `license: other`, `license_name: lfm1.0`, with the full LFM Open
  License v1.0 text included as `LICENSE`;
* states the base model and its revision in the model card (`base_model:`);
* states clearly that the files are a LoRA adapter trained by this lab
  (the "prominent notice of modification");
* contains only adapter weights and configuration, never a copy of the base weights.

The code that produced an adapter stays Apache-2.0 in the GitHub repository.
The adapter weights follow the base model's license.

## Generated data

The robot-dispatch and shape-scene datasets are synthetic, produced by code in
this repository, and contain no personal or third-party data. Teacher-generated
labels (sequence distillation) come from Apache-2.0 Qwen3.5-2B outputs and are
used only for local training.
