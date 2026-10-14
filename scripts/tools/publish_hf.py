"""Publish a tested LoRA adapter to the Hugging Face Hub, then verify it loads from the Hub.

    .venv/bin/python scripts/tools/publish_hf.py --which text --repo-id <user>/lfm2.5-350m-robot-dispatch-lora
    .venv/bin/python scripts/tools/publish_hf.py --which vlm  --repo-id <user>/lfm2.5-vl-450m-shapes-lora
    add --dry-run to only build the upload folder (outputs/publish/<which>/) and print the card.

Authentication: run `hf auth login` once in your own terminal (never paste tokens into code or
chat). The token is read from the standard location (~/.cache/huggingface/token) or HF_TOKEN.

What is uploaded: adapter_config.json, adapter_model.safetensors, README.md (model card),
LICENSE (the base model's LFM Open License v1.0, as its Section 4 requires), and NOTICE.md.
The base model weights are never uploaded.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["LAB_ONLINE"] = "1"
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401

from lab.common import OUTPUTS, ROOT, read_json  # noqa: E402
from lab import robot_task as rt  # noqa: E402
from lab.models import resolve  # noqa: E402

GITHUB = "https://github.com/MahdiElzain/lfm-learning-lab"
CODE_NOTE = ("the author's LFM learning lab (source repository currently private; this card is self-contained)")

SPECS = {
    "text": {
        "adapter": OUTPUTS / "runs/text_sft/lora_standard/adapter",
        "metrics": OUTPUTS / "runs/text_sft/lora_standard/metrics.json",
        "base_key": "text_instruct",
        "pipeline": "text-generation",
        "tags": ["lora", "peft", "trl", "sft", "lfm2", "lfm2.5", "robotics", "structured-output", "learning-lab"],
    },
    "vlm": {
        "adapter": OUTPUTS / "runs/vlm_sft/lora_quick/adapter",
        "metrics": OUTPUTS / "runs/vlm_sft/lora_quick/metrics.json",
        "base_key": "vlm",
        "pipeline": "image-text-to-text",
        "tags": ["lora", "peft", "sft", "lfm2-vl", "lfm2.5-vl", "vision-language", "synthetic-data", "learning-lab"],
    },
}


def text_card(repo_id, base, rev, m, bon):
    t, o = m["eval"]["test"], m["eval"]["test_ood"]
    cfg = m["config"]
    return f"""---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: {base}
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: {json.dumps(SPECS['text']['tags'])}
---

# LFM2.5-350M robot-dispatch LoRA (learning-lab artifact)

A **LoRA adapter** for [{base}](https://huggingface.co/{base}) (revision `{rev[:10]}`), trained in
{CODE_NOTE} (Lesson C). It is a teaching artifact for a **synthetic toy task**, not a production
model and not a robotics controller.

**Modification notice (LFM Open License v1.0, Section 4):** these files are a LoRA adapter trained
by the lab's author on top of Liquid AI's LFM2.5-350M; they modify the model's behaviour when
loaded with PEFT. The base weights are not included. The base model is © Liquid AI, Inc.

## Task

Given 3-6 robots (type, grid position, status) and one task, answer with the eligible robot nearest
to the task and its Manhattan distance, as JSON only:

```
Robots:
- go2_3 (quadruped) at (1, 7), charging
- go2_7 (quadruped) at (7, 8), idle
- uav2 (drone) at (8, 6), busy
Task: scan the sector at (5, 5).
-> {{"robot": "go2_7", "distance": 5}}
```

Eligible = idle and of an allowed type (aerial → drone, stairs → quadruped, heavy → rover, scan → any).
Use this **system prompt verbatim** (the adapter was trained with it):

```
{rt.SYSTEM_PROMPT}
```

User message format (one robot per line, then the task):

```
Robots:
- <id> (<drone|quadruped|rover>) at (<x>, <y>), <idle|busy|charging>
Task: <photograph the roof | climb the stairwell | haul the heavy crate | scan the sector> at (<x>, <y>).
```

## Training

* Data: {cfg['n_train']:,} synthetic examples from a seeded generator (3-5 robots on a 10×10 grid),
  half of them "traps" where the nearest robot is ineligible. No personal or third-party data.
* Method: supervised fine-tuning with TRL `SFTTrainer`, completion-only loss, {cfg['num_train_epochs']} epochs.
* LoRA: r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}, targets
  `{", ".join(cfg['target_modules'])}` ({m['params']['trainable_params']:,} trainable parameters, {m['params']['trainable_percent']}%).
* Optimiser: AdamW, lr {cfg['learning_rate']}, cosine schedule, effective batch
  {cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']}, BF16.
* Hardware: one RTX 4060 Laptop GPU; {m['train']['wall_seconds']:.0f} s; peak {m['train']['peak_allocated_gb']} GB allocated.

## Evaluation (measured, single seed, greedy decoding)

| split | exact (robot and distance) | robot correct | distance correct | valid JSON | rule violations (busy / wrong type) |
|---|---|---|---|---|---|
| test, 300 held-out scenes (3-5 robots) | {t['exact']:.3f} | {t['robot_correct']:.3f} | {t['distance_correct']:.3f} | {t['valid_strict']:.3f} | {t['violation_status']:.3f} / {t['violation_type']:.3f} |
| out-of-distribution, 200 scenes (6 robots) | {o['exact']:.3f} | {o['robot_correct']:.3f} | {o['distance_correct']:.3f} | {o['valid_strict']:.3f} | {o['violation_status']:.3f} / {o['violation_type']:.3f} |

Zero-shot base model on the same test set: exact 0.023. With 8 samples (T = 1) and a label-free
verifier + majority vote, this adapter reaches {bon['accuracy']['majority_valid']:.3f} exact at 8× the generated tokens.
Most remaining errors are arithmetic (wrong distance, or a farther eligible robot).

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base}", revision="{rev}", dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("{base}", revision="{rev}")
model = PeftModel.from_pretrained(base, "{repo_id}").eval()
SYSTEM_PROMPT = {rt.SYSTEM_PROMPT!r}
messages = [{{"role": "system", "content": SYSTEM_PROMPT}},
            {{"role": "user", "content": scene_text}}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
out = model.generate(ids, max_new_tokens=40, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

* Trained and evaluated only on a synthetic grid task; it does not understand real robots, maps
  or safety constraints. Do not use it to control hardware.
* Single training seed; differences of a few points between runs are expected.
* Exact match on test is {t['exact']:.1%}: it frequently gets the distance arithmetic wrong.

## License

The adapter is a derivative of an LFM model and is distributed under the **LFM Open License v1.0**
(see `LICENSE`), including its commercial-use limitation (Section 5).
"""


def vlm_card(repo_id, base, rev, m):
    c = m["eval"]
    cfg = m["config"]
    return f"""---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: {base}
library_name: peft
pipeline_tag: image-text-to-text
language: [en]
tags: {json.dumps(SPECS['vlm']['tags'])}
---

# LFM2.5-VL-450M shape-scene LoRA (learning-lab artifact)

A **LoRA adapter** for [{base}](https://huggingface.co/{base}) (revision `{rev[:10]}`), trained in
{CODE_NOTE} (Lesson V2). Teaching artifact for **synthetic images of coloured shapes**; not a
general-purpose vision model.

**Modification notice (LFM Open License v1.0, Section 4):** LoRA adapter on the language-model
layers of Liquid AI's LFM2.5-VL-450M; vision encoder and projector unchanged and not included.
The base model is © Liquid AI, Inc.

## Task

256×256 images with 1-6 coloured circles/squares/triangles; short answers to questions such as
"How many red shapes are there? Answer with a number.", "Is there a blue circle? Answer yes or
no.", "What color is the leftmost shape? Answer with one word.", "Is the red circle to the left of
the blue square? Answer yes or no."

## Training

* Data: {cfg['n_train']:,} questions on 500 synthetic 256×256 scenes (seeded generator: 1-6 shapes,
  5 colours, varied backgrounds), splits by scene (no test image is ever seen in training).
* LoRA r={cfg['lora_r']}, alpha={cfg['lora_alpha']} on language-model attention, convolution and MLP
  projections only ({m['trainable_params']:,} trainable parameters); vision tower frozen.
* {cfg['epochs']} epoch, lr {cfg['learning_rate']}, effective batch {cfg['batch_size'] * cfg['grad_accum']}, BF16,
  {m['train']['wall_seconds']:.0f} s on one RTX 4060 Laptop GPU.

## Evaluation (measured, single seed, greedy, 450 questions on 150 held-out scenes)

| condition | accuracy |
|---|---|
| correct image | {c['correct']['accuracy']:.3f} |
| swapped image (scored against the swapped scene) | {c['swapped']['accuracy']:.3f} |
| answers unchanged when the image is swapped | {c['swapped']['unchanged_vs_correct_image']:.3f} |
| blank image, agreement with original answers | {c['blank']['matches_original_answer']:.3f} |

Accuracy by question type (correct image): {json.dumps(c['correct']['by_type'])}.

## Usage

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("{base}", revision="{rev}", dtype=torch.bfloat16).cuda()
processor = AutoProcessor.from_pretrained("{base}", revision="{rev}")
model = PeftModel.from_pretrained(base, "{repo_id}").eval()
conv = [{{"role": "user", "content": [{{"type": "image", "image": Image.open("scene.png")}},
                                      {{"type": "text", "text": "How many shapes are there? Answer with a number."}}]}}]
inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

## Limitations

* Synthetic flat shapes only; no claim about natural images. Single seed.
* Answers are trained to be one word or number; long-form behaviour of the base model may change.

## License

LFM Open License v1.0 (see `LICENSE`), including its commercial-use limitation.
"""


def build(which, repo_id):
    spec = SPECS[which]
    base, rev = resolve(spec["base_key"])
    m = read_json(spec["metrics"])
    out = OUTPUTS / "publish" / which
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(spec["adapter"] / "adapter_model.safetensors", out / "adapter_model.safetensors")
    acfg = json.loads((spec["adapter"] / "adapter_config.json").read_text())
    acfg["base_model_name_or_path"], acfg["revision"] = base, rev   # pin the exact base commit
    (out / "adapter_config.json").write_text(json.dumps(acfg, indent=2) + "\n")
    snap = next((ROOT / "models/hf/hub").glob(f"models--{base.replace('/', '--')}/snapshots/{rev}"))
    shutil.copy(snap / "LICENSE", out / "LICENSE")
    if which == "text":
        bon = read_json(OUTPUTS / "evals/sft_best_of_8/best_of_n.json")
        card = text_card(repo_id, base, rev, m, bon)
    else:
        card = vlm_card(repo_id, base, rev, m)
    (out / "README.md").write_text(card)
    (out / "NOTICE.md").write_text(
        f"This repository contains a LoRA adapter derived from {base} (Liquid AI, Inc.), "
        f"revision {rev}. It was trained by the repository owner. Base model weights are not included. "
        "Use is subject to the LFM Open License v1.0 in LICENSE.\n")
    return out


def verify(which, repo_id):
    """Download from the Hub into a fresh cache and run a few generations."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    with tempfile.TemporaryDirectory() as tmp:
        local = snapshot_download(repo_id, cache_dir=tmp)
        files = sorted(p.name for p in Path(local).iterdir())
        if which == "text":
            from lab import robot_task as rt
            from lab.common import DATA, read_jsonl
            from lab.models import load_text_model
            from lab.text_eval import evaluate_robot
            model, tok = load_text_model("text_instruct")
            model = PeftModel.from_pretrained(model, local)
            agg, _ = evaluate_robot(model, tok, read_jsonl(DATA / "robot/test.jsonl")[:64])
            return {"files": files, "exact_first64": agg["exact"], "valid_first64": agg["valid_strict"]}
        from lab.common import DATA, read_jsonl
        from lab.models import load_vlm
        from lab import shapes_task as st
        from lab.vlm_utils import generate, load_image
        model, processor = load_vlm("vlm")
        model = PeftModel.from_pretrained(model, local)
        rows = read_jsonl(DATA / "shapes/test.jsonl")[:64]
        g = generate(model, processor, [load_image(r) for r in rows], [r["question"] for r in rows])
        acc = sum(st.verify(r["type"], r["answer"], t)["correct"] for r, t in zip(rows, g["texts"])) / len(rows)
        torch.cuda.empty_cache()
        return {"files": files, "accuracy_first64": acc}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--which", choices=list(SPECS), required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    folder = build(a.which, a.repo_id)
    print("built", folder, sorted(x.name for x in folder.iterdir()))
    if a.dry_run:
        print((folder / "README.md").read_text()[:3000])
        sys.exit(0)
    from huggingface_hub import HfApi
    api = HfApi()
    print("authenticated as:", api.whoami()["name"])
    api.create_repo(a.repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(folder_path=str(folder), repo_id=a.repo_id, repo_type="model",
                               commit_message="Upload LoRA adapter, model card and license")
    print("uploaded:", commit.commit_url)
    print("verification:", verify(a.which, a.repo_id))
