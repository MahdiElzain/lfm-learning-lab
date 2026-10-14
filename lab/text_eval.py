"""Batched generation + robot-task evaluation with raw outputs preserved.

FROZEN EVALUATION SETTINGS (used for every reported text-track number):
    greedy decoding (do_sample=False), max_new_tokens=40, repetition_penalty=1.0,
    chat template with add_generation_prompt=True for chat models,
    2-shot plain-text prompt (first line of the completion) for the base model.
"""
from __future__ import annotations

import time

import torch

from . import robot_task as rt

EVAL_MAX_NEW_TOKENS = 40


def chat_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = EVAL_MAX_NEW_TOKENS, batch_size: int = 32,
             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, top_k: int | None = None,
             num_return_sequences: int = 1, seed: int | None = None, repetition_penalty: float = 1.0) -> dict:
    """Generate completions for already-formatted prompt strings.

    Returns {"texts": [[...n per prompt...], ...], "new_tokens": [[...]], "seconds": float}.
    Prompts are LEFT-padded so that every sequence's last real token sits at the
    same position, which is what decoder-only generation needs.
    """
    tok.padding_side = "left"
    texts, counts = [], []
    start = time.perf_counter()
    if seed is not None:
        torch.manual_seed(seed)
    eos_ids = model.generation_config.eos_token_id
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i: i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id,
                      num_return_sequences=num_return_sequences, repetition_penalty=repetition_penalty)
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        for j in range(0, new.shape[0], num_return_sequences):
            group_t, group_n = [], []
            for row in new[j: j + num_return_sequences]:
                ids = row.tolist()
                n = len(ids)
                stops = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                for k, t in enumerate(ids):
                    if t in stops or t == tok.pad_token_id:
                        n = k + 1 if t in stops else k
                        break
                group_n.append(n)
                group_t.append(tok.decode(row[:n], skip_special_tokens=True))
            texts.append(group_t)
            counts.append(group_n)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"texts": texts, "new_tokens": counts, "seconds": time.perf_counter() - start}


def robot_prompts(tok, examples: list[dict], mode: str = "chat", shots: list[dict] | None = None) -> list[str]:
    if mode == "chat":
        return [chat_prompt(tok, rt.chat_messages(ex)) for ex in examples]
    if mode == "base_fewshot":
        bos = tok.bos_token or ""
        return [bos + rt.base_model_prompt(ex, shots or []) for ex in examples]
    raise ValueError(mode)


def evaluate_robot(model, tok, examples: list[dict], mode: str = "chat", shots: list[dict] | None = None,
                   batch_size: int = 32, **gen_kwargs) -> tuple[dict, list[dict]]:
    """Return (aggregate metrics, per-example rows with raw outputs)."""
    model.eval()
    prompts = robot_prompts(tok, examples, mode, shots)
    gen = generate(model, tok, prompts, batch_size=batch_size, **gen_kwargs)
    rows = []
    for ex, texts, ns in zip(examples, gen["texts"], gen["new_tokens"]):
        raw = texts[0]
        answer_text = raw.split("\n")[0] if mode == "base_fewshot" else raw
        v = rt.verify(ex, answer_text)
        rows.append({"id": ex["id"], "split": ex["split"], "trap": ex["trap"], "n_robots": ex["n_robots"],
                     "expected": rt.format_answer(ex["answer"]), "raw_output": raw, "new_tokens": ns[0], **v})
    agg = rt.aggregate(rows)
    agg["trap_exact"] = round(sum(r["exact"] for r in rows if r["trap"]) / max(1, sum(r["trap"] for r in rows)), 4)
    agg["nontrap_exact"] = round(sum(r["exact"] for r in rows if not r["trap"]) /
                                 max(1, sum(not r["trap"] for r in rows)), 4)
    agg["mean_new_tokens"] = round(sum(r["new_tokens"] for r in rows) / max(len(rows), 1), 2)
    agg["seconds"] = round(gen["seconds"], 2)
    agg["ms_per_example"] = round(1000 * gen["seconds"] / max(len(rows), 1), 1)
    agg["decode_mode"] = mode
    return agg, rows
