"""Lesson A: text -> tokens -> tensors -> logits -> probabilities -> generated text.

    .venv/bin/python scripts/text/01_tokens_and_generation.py
    .venv/bin/python scripts/text/01_tokens_and_generation.py --prompt "Name three robot sensors."

Everything printed is also saved to outputs/runs/lesson_a_tokens/<time>/report.json.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lab.common import GpuMemory, make_run_dir, set_seed, write_json  # noqa: E402
from lab.models import load_text_model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prompt", default="The robot reached the charging station and")
p.add_argument("--question", default="In one sentence, what does a LiDAR sensor measure?")
p.add_argument("--top-k", type=int, default=10)
args = p.parse_args()
set_seed(0)
report = {}
mem = GpuMemory()


def show(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


model, tok = load_text_model("text_instruct")
model.eval()

# ------------------------------------------------------------------ 1. tokens
show("1. Text -> tokens -> token IDs -> text")
text = args.prompt
ids = tok.encode(text, add_special_tokens=False)
pieces = tok.convert_ids_to_tokens(ids)
print(f"text      : {text!r}")
print(f"tokens    : {pieces}")
print(f"token IDs : {ids}")
print(f"decoded   : {tok.decode(ids)!r}")
print(f"{len(text)} characters -> {len(ids)} tokens")
print(f"tokenizer defines {len(tok)} tokens; the embedding/logit dimension is {model.config.vocab_size} "
      f"(the extra {model.config.vocab_size - len(tok)} rows are unused padding)")
report["tokenization"] = {"text": text, "tokens": pieces, "ids": ids, "tokenizer_len": len(tok),
                          "logit_dim": model.config.vocab_size}

show("2. Special tokens and the chat template")
special = {"bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token}
special_ids = {k: tok.convert_tokens_to_ids(v) if v else None for k, v in special.items()}
print("special tokens:", special, "ids:", special_ids)
print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)
messages = [{"role": "system", "content": "You are a concise robotics tutor."},
            {"role": "user", "content": args.question}]
templated = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("chat template output (exact string the model sees):\n" + templated)
report["special_tokens"] = {"tokens": special, "ids": special_ids,
                            "generation_eos": model.generation_config.eos_token_id}
report["chat_template_rendered"] = templated

# ------------------------------------------------------- 3. tensors and masks
show("3. Input tensors, padding and attention masks")
tok.padding_side = "left"
batch = tok([text, "Hi"], return_tensors="pt", padding=True, add_special_tokens=False)
for k, v in batch.items():
    print(f"{k:15s} shape={tuple(v.shape)} dtype={v.dtype}\n{v}")
print("Left padding: the short row gets pad IDs on the left and mask 0 there, so the")
print("last real token of every row is at the same (final) position for generation.")
report["batch"] = {k: v.tolist() for k, v in batch.items()}

# ------------------------------------------- 4. logits -> next-token probabilities
show("4. Logits, softmax and next-token probabilities")
x = torch.tensor([ids], device=model.device)
with torch.no_grad():
    out = model(input_ids=x)
logits = out.logits                       # (B=1, T, V)
print(f"input_ids {tuple(x.shape)} -> logits {tuple(logits.shape)}  (batch, positions, vocabulary)")
last = logits[0, -1].float()              # scores for the token AFTER the prompt
rows = []
for temp in (1.0, 0.5, 2.0):
    probs = F.softmax(last / temp, dim=-1)
    top = torch.topk(probs, args.top_k)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    print(f"\ntemperature {temp}: entropy {entropy:.3f} nats")
    for p_, i_ in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   {p_:7.4f}  id={i_:6d}  {tok.decode([i_])!r}")
    rows.append({"temperature": temp, "entropy_nats": entropy,
                 "top": [{"id": i_, "token": tok.decode([i_]), "prob": p_} for p_, i_ in
                         zip(top.values.tolist(), top.indices.tolist())]})
report["next_token"] = rows

# ------------------------------------------- 5. manual greedy loop vs generate
show("5. Autoregressive generation: a manual greedy loop (with and without cache)")
loop_msgs = [{"role": "user", "content": "Explain in three sentences how a LiDAR sensor builds a 3D map."}]
x5 = tok.apply_chat_template(loop_msgs, add_generation_prompt=True, return_tensors="pt",
                             return_dict=True)["input_ids"].to(model.device)
eos_id = model.generation_config.eos_token_id


def manual_greedy(max_new, use_cache):
    """Pick the argmax token, append it, repeat; stop at the end-of-turn token."""
    seq, past = x5.clone(), None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new):
            inp = seq[:, -1:] if (use_cache and past is not None) else seq
            o = model(input_ids=inp, past_key_values=past, use_cache=use_cache)
            past = o.past_key_values if use_cache else None
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            if nxt.item() == eos_id:
                break
    torch.cuda.synchronize()
    return seq, time.perf_counter() - t0, past


seq_nc, t_nc, _ = manual_greedy(120, use_cache=False)
seq_c, t_c, cache = manual_greedy(120, use_cache=True)
same = torch.equal(seq_nc, seq_c)
n_gen = seq_c.shape[1] - x5.shape[1]
print(f"prompt {x5.shape[1]} tokens, generated {n_gen} tokens")
print(f"no cache  : {t_nc*1000:7.1f} ms  (re-runs the whole prefix every step)")
print(f"with cache: {t_c*1000:7.1f} ms  (feeds only the newest token)")
print(f"identical outputs: {same}")
print("greedy answer:", repr(tok.decode(seq_c[0, x5.shape[1]:], skip_special_tokens=True)))
divergence = None
if not same:
    a, b = seq_nc[0, x5.shape[1]:].tolist(), seq_c[0, x5.shape[1]:].tolist()
    k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
    with torch.no_grad():   # logit gap between the two competing tokens at the divergence point
        lg = model(input_ids=seq_c[:, : x5.shape[1] + k]).logits[0, -1].float()
    gap = (lg[a[k]] - lg[b[k]]).abs().item()
    top2 = torch.topk(lg, 2).values
    divergence = {"first_diverging_new_token": k, "no_cache_token": tok.decode([a[k]]),
                  "cache_token": tok.decode([b[k]]), "logit_gap_between_them": round(gap, 4),
                  "top1_minus_top2_logit": round((top2[0] - top2[1]).item(), 4)}
    print("first divergence:", divergence)
    print("BF16 kernels for a full-prefix pass and a one-token cached step round differently; at a near-tie")
    print("(tiny logit gap) the argmax can flip. Re-checking in float32:")
    model32 = load_text_model("text_instruct", dtype="float32")[0].eval()
    model_bf16, model = model, model32
    s32_nc, _, _ = manual_greedy(120, use_cache=False)
    s32_c, _, _ = manual_greedy(120, use_cache=True)
    divergence["identical_in_float32"] = torch.equal(s32_nc, s32_c)
    print("  float32 cached vs uncached identical:", divergence["identical_in_float32"])
    model = model_bf16
    del model32
    torch.cuda.empty_cache()
layer_shapes = []
for li, layer in enumerate(cache.layers):
    entry = {"layer": li, "class": type(layer).__name__}
    for name in ("keys", "values"):
        t = getattr(layer, name, None)
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            entry[name] = list(t.shape)
    conv = getattr(layer, "conv_states", None)
    if isinstance(conv, dict) and conv:
        entry["conv_state"] = list(next(iter(conv.values())).shape)
    layer_shapes.append(entry)
print(f"cache object: {type(cache).__name__} with {len(layer_shapes)} layers")
for e in layer_shapes:
    print("  ", e)
print("Attention layers store K/V of shape (batch, kv_heads, tokens_so_far, head_dim): it grows each step.")
print("Conv layers store a fixed (batch, channels, kernel=3) window: constant size.")
report["manual_generation"] = {"prompt_tokens": x5.shape[1], "generated_tokens": n_gen, "ms_no_cache": t_nc * 1000,
                               "ms_cache": t_c * 1000, "identical": same,
                               "answer": tok.decode(seq_c[0, x5.shape[1]:], skip_special_tokens=True), "divergence": divergence,
                               "cache_type": type(cache).__name__, "cache_layers": layer_shapes}

# ------------------------------------------------- 6. decoding strategies
show("6. Decoding strategies on the chat prompt")
chat_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)["input_ids"].to(model.device)
settings = [
    ("greedy", dict(do_sample=False)),
    ("sample T=0.1 top_k=50 rep=1.05 (model card)", dict(do_sample=True, temperature=0.1, top_k=50,
                                                         repetition_penalty=1.05)),
    ("sample T=0.7 top_p=0.9", dict(do_sample=True, temperature=0.7, top_p=0.9)),
    ("sample T=1.5 top_p=1.0 (too hot)", dict(do_sample=True, temperature=1.5, top_p=1.0, top_k=0)),
    ("greedy, max_new_tokens=8 (truncated)", dict(do_sample=False, max_new_tokens=8)),
]
gens = []
for name, kw in settings:
    torch.manual_seed(0)
    kw = {"max_new_tokens": 60, **kw}
    if not kw.get("do_sample"):
        kw.update(temperature=None, top_p=None, top_k=None)
    t0 = time.perf_counter()
    with torch.no_grad():
        o = model.generate(chat_ids, **kw, pad_token_id=tok.pad_token_id)
    dt = time.perf_counter() - t0
    new = o[0, chat_ids.shape[1]:]
    ended = new[-1].item() == model.generation_config.eos_token_id or \
        (isinstance(model.generation_config.eos_token_id, list) and new[-1].item() in model.generation_config.eos_token_id)
    text_out = tok.decode(new, skip_special_tokens=True)
    print(f"\n[{name}] {len(new)} tokens, {dt*1000:.0f} ms, stopped by {'EOS' if ended else 'length limit'}")
    print("  " + text_out.replace("\n", "\n  "))
    gens.append({"setting": name, "kwargs": {k: v for k, v in kw.items() if v is not None}, "text": text_out,
                 "new_tokens": len(new), "ms": dt * 1000, "stopped_by_eos": bool(ended)})
report["decoding"] = gens

# ------------------------------------------------ 7. base vs instruct
show("7. Base vs instruction-tuned model")
del model
torch.cuda.empty_cache()
base, btok = load_text_model("text_base")
base.eval()
inst, itok = load_text_model("text_instruct")
inst.eval()
comparisons = []
plain = "Question: " + args.question + "\nAnswer:"
for label, m, t, prompt_text, fmt in [
    ("instruct + chat template", inst, itok,
     itok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False), "chat"),
    ("base + chat template", base, btok,
     btok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False), "chat"),
    ("base + plain 'Question/Answer' text", base, btok, (btok.bos_token or "") + plain, "plain"),
    ("instruct + plain text", inst, itok, (itok.bos_token or "") + plain, "plain"),
]:
    enc = t(prompt_text, return_tensors="pt", add_special_tokens=False).to(m.device)
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=50, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=t.pad_token_id)
    out_text = t.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{label}]\n  " + out_text.strip().replace("\n", "\n  "))
    comparisons.append({"setting": label, "prompt_format": fmt, "output": out_text})
report["base_vs_instruct"] = comparisons

# ------------------------------------------------ 8. context and cache size
show("8. Context length and cache memory (from the config)")
cfg = inst.config
n_attn = sum(1 for t_ in cfg.layer_types if "attention" in t_) if hasattr(cfg, "layer_types") else None
head_dim = cfg.hidden_size // cfg.num_attention_heads
kv_bytes_per_token = 2 * n_attn * cfg.num_key_value_heads * head_dim * 2 if n_attn is not None else None
CARD_CONTEXT = 32768   # trained context length stated on the model card
ctx = {"model_card_context_length": CARD_CONTEXT,
       "config_max_position_embeddings (do not rely on)": cfg.max_position_embeddings,
       "attention_layers": n_attn, "num_key_value_heads": cfg.num_key_value_heads, "head_dim": head_dim,
       "kv_cache_bytes_per_token_bf16": kv_bytes_per_token,
       "kv_cache_mb_at_32768_tokens": kv_bytes_per_token * CARD_CONTEXT / 1e6,
       "conv_state_bytes_per_conv_layer_bf16": cfg.hidden_size * cfg.conv_L_cache * 2}
for k, v in ctx.items():
    print(f"  {k:32s} {v}")
report["context"] = ctx
report["vram"] = mem.snapshot()
run_dir = make_run_dir("lesson_a_tokens")
write_json(run_dir / "report.json", report)
print("\nsaved", run_dir / "report.json")
