"""Vision-language helpers: conversations, batched generation, and an SFT collator.

FROZEN VLM EVALUATION SETTINGS: greedy decoding, max_new_tokens=8, processor
defaults from the model card unless a script says otherwise
(LFM2.5-VL: min_image_tokens=32, max_image_tokens=256, do_image_splitting=True).
"""
from __future__ import annotations

import time

import torch
from PIL import Image

from . import shapes_task as st
from .common import DATA

VLM_MAX_NEW_TOKENS = 8
# LoRA only on the LANGUAGE model of LFM2.5-VL (vision tower and projector stay frozen).
LFM2VL_LM_TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                     r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))")


def load_image(row: dict, condition: str = "correct", swap_row: dict | None = None) -> Image.Image:
    """condition: correct | swapped (image of another scene) | blank (empty canvas)."""
    if condition == "blank":
        return st.blank_image(row["scene"]["size"])
    src = swap_row if condition == "swapped" else row
    return Image.open(DATA / "shapes" / src["image"]).convert("RGB")


def conversation(image: Image.Image, question: str, answer: str | None = None) -> list[dict]:
    conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    if answer is not None:
        conv.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return conv


@torch.no_grad()
def generate(model, processor, images: list, questions: list[str], max_new_tokens: int = VLM_MAX_NEW_TOKENS,
             batch_size: int = 16, do_sample: bool = False, temperature: float = 1.0,
             template_kwargs: dict | None = None) -> dict:
    tok = processor.tokenizer
    tok.padding_side = "left"
    texts, n_new, n_img_tokens = [], [], []
    image_token_id = getattr(processor, "image_token_id", None)
    t0 = time.perf_counter()
    for i in range(0, len(questions), batch_size):
        convs = [conversation(im, q) for im, q in zip(images[i: i + batch_size], questions[i: i + batch_size])]
        enc = processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt", processor_kwargs={"padding": True},
                                            **(template_kwargs or {})).to(model.device)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
        if image_token_id is not None:
            n_img_tokens += (enc["input_ids"] == image_token_id).sum(1).tolist()
        kw = dict(do_sample=do_sample, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        kw.update(temperature=temperature) if do_sample else kw.update(temperature=None, top_p=None, top_k=None,
                                                                         min_p=None)
        out = model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            texts.append(tok.decode(row, skip_special_tokens=True).strip())
            n_new.append(int((row != tok.pad_token_id).sum()))
    torch.cuda.synchronize()
    return {"texts": texts, "new_tokens": n_new, "image_tokens": n_img_tokens, "seconds": time.perf_counter() - t0}


class SftCollator:
    """Build (input_ids, attention_mask, pixel tensors, labels) for VLM SFT.

    labels = input_ids, except -100 on: prompt tokens (system/user text AND
    image placeholder tokens) and padding. Only the assistant answer and its
    end-of-turn token are trained. `check()` verifies this on a real batch.
    """

    def __init__(self, processor, max_length: int = 2048):
        self.p = processor
        self.max_length = max_length
        self.image_token_id = processor.image_token_id

    def __call__(self, examples: list[dict]) -> dict:
        tok = self.p.tokenizer
        tok.padding_side = "right"
        full = [conversation(e["image"], e["question"], e["answer"]) for e in examples]
        prompt = [conversation(e["image"], e["question"]) for e in examples]
        enc = self.p.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt",
                                         processor_kwargs={"padding": True})
        pr = self.p.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True,
                                        return_tensors="pt", processor_kwargs={"padding": True})
        if enc["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"sequence length {enc['input_ids'].shape[1]} > max_length {self.max_length}: "
                             "truncation could cut image tokens; raise max_length or lower max_image_tokens")
        labels = enc["input_ids"].clone()
        prompt_lens = pr["attention_mask"].sum(1)
        for i, n in enumerate(prompt_lens.tolist()):
            labels[i, :n] = -100
        labels[enc["attention_mask"] == 0] = -100
        labels[enc["input_ids"] == self.image_token_id] = -100
        enc["labels"] = labels
        # Verify that the prompt-only encoding is an exact prefix of the full encoding,
        # so "first n tokens = prompt" is a valid way to find the answer.
        enc["_prefix_ok"] = torch.tensor([bool(torch.equal(enc["input_ids"][i, :n], pr["input_ids"][i, :n]))
                                          for i, n in enumerate(prompt_lens.tolist())])
        return enc

    def check(self, batch: dict) -> dict:
        """Return facts proving the masking is right (used in the lesson output)."""
        tok = self.p.tokenizer
        ids, labels = batch["input_ids"], batch["labels"]
        trained = [tok.decode(ids[i][labels[i] != -100]) for i in range(ids.shape[0])]
        img_per_row = (ids == self.image_token_id).sum(1).tolist()
        img_trained = int(((ids == self.image_token_id) & (labels != -100)).sum())
        return {"input_ids_shape": list(ids.shape),
                "pixel_values_shape": list(batch["pixel_values"].shape) if "pixel_values" in batch else None,
                "image_tokens_per_row": img_per_row, "image_tokens_receiving_loss": img_trained,
                "trained_text_per_row": trained,
                "prompt_is_prefix_of_full": bool(batch["_prefix_ok"].all())}
