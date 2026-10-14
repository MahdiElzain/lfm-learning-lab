# Lesson B1: Inside LFM2.5-350M, a hybrid convolution + attention model

Code: `scripts/text/02_inspect_architecture.py`. Output:
`outputs/runs/lesson_b_arch/text_instruct/architecture.json`. All numbers
below were read from the installed model (revision `9e6c6cc`), not from memory.

## 1. Concept and purpose

Most small LMs are pure Transformers: every layer is attention + MLP.
LFM2.5 is **hybrid**: of its 16 layers, 10 use a cheap **gated short
convolution** to mix information between nearby tokens, and only 6 use
**grouped-query attention** to look anywhere in the context. Each layer also
has a SwiGLU MLP. Fewer attention layers mean a smaller generation cache and
faster decoding on edge devices, which is the design goal stated on the model card.

## 2. Equations (all shapes verified)

Width `d = 1024`, sequence length `T`. Every layer `ℓ` is pre-norm residual:

```
h ← h + Operator_ℓ( RMSNorm(h) )        Operator = ShortConv (10 layers) or GQA (6 layers)
h ← h + SwiGLU( RMSNorm(h) )
RMSNorm(u) = g ⊙ u / sqrt(mean(u²) + 1e-5)      g ∈ R^d learned gain
```

**Gated short convolution** (`Lfm2ShortConv.forward`, transformers 5.19):

```
[B, C, x] = split( W_in · u )            W_in ∈ R^{3d×d}   -> three (d)-vectors per position
v = B ⊙ x                                 input gate
w_t = Σ_{k=0..2} K_k ⊙ v_{t-k}            depthwise causal conv: one 3-tap filter per channel, K ∈ R^{d×3}
y = W_out · (C ⊙ w)                       output gate, W_out ∈ R^{d×d}
```

"Depthwise" (`groups = 1024`) means each of the 1,024 channels has its own 3-tap
filter; channels are only mixed by `W_in` and `W_out`. "Causal" means
`w_t` only sees positions `t, t-1, t-2`.

**Grouped-query attention** with RoPE and per-head QK-normalisation:

```
q = RMSNorm_head(W_q u)   16 heads × 64       W_q ∈ R^{1024×1024}
k = RMSNorm_head(W_k u)    8 heads × 64       W_k ∈ R^{512×1024}
v = W_v u                  8 heads × 64       W_v ∈ R^{512×1024}
q, k ← RoPE(q, k; position)                   rotation by position-dependent angles (θ = 1,000,000)
head i uses K/V group ⌊i/2⌋:  A = softmax( q kᵀ/√64 + causal mask ) v
y = W_out · concat(heads)                     W_out ∈ R^{1024×1024}
```

**SwiGLU MLP:** `y = W_2( silu(W_1 u) ⊙ W_3 u )`, `W_1, W_3 ∈ R^{4608×1024}`,
`W_2 ∈ R^{1024×4608}`. (The config says `intermediate_size: 6656`;
`block_auto_adjust_ff_dim` takes ⅔ of it, 4437, and rounds up to a multiple
of 256: 4608.)

**Output head:** final `RMSNorm` (named `embedding_norm` in the code), then
logits `= h Eᵀ` using the **tied** embedding matrix `E ∈ R^{65536×1024}`.

## 3. Results from the inspection

| Item | Value |
|---|---|
| Layer pattern | `conv conv ATTN conv conv ATTN conv conv ATTN conv ATTN conv ATTN conv ATTN conv` |
| Parameters (unique) | 354.48M |
| Feed-forward (all 16 layers) | 226.49M (63.9%) |
| Embeddings (tied with output head) | 67.11M (18.9%) |
| Conv operators (10 layers) | 41.97M (11.8%) |
| Attention operators (6 layers) | 18.87M (5.3%) |
| Norms | 0.03M |
| Per layer | conv layer 18.36M, attention layer 17.30M |
| Context length | 32,768 per the model card; the config's `max_position_embeddings` field says 128,000, so do not rely on the field |

Shapes traced with forward hooks for a 14-token input:

```
input_ids (1, 14) → embed_tokens (1, 14, 1024)
conv layer 0:  in_proj → (1, 14, 3072) [= B, C, x]   out_proj → (1, 14, 1024)
attn layer 2:  q_proj → (1, 14, 1024) → q_layernorm on (1, 14, 16, 64)
               k_proj → (1, 14, 512)  → k_layernorm on (1, 14, 8, 64);  v_proj → (1, 14, 512)
MLP:           w1, w3 → (1, 14, 4608);  w2 → (1, 14, 1024)
embedding_norm (1, 14, 1024) → lm_head (1, 14, 65536)
```

**LoRA target names (used by every training script).** Linear layers by short
name: `q_proj` ×6, `k_proj` ×6, `v_proj` ×6, `out_proj` ×16 (6 attention + 10
conv), `in_proj` ×10, `w1`/`w2`/`w3` ×16, `lm_head` ×1. **There is no `o_proj`.**
The Liquid TRL guide lists `["q_proj", "k_proj", "v_proj", "o_proj"]` for text
models; on this checkpoint `o_proj` matches nothing, so that list would adapt
only q/k/v in the six attention layers. This lab targets
`q_proj k_proj v_proj out_proj in_proj w1 w2 w3`. Always inspect before choosing.

## 4. Interpretation: why the cache is small

During generation, attention layers must keep keys and values for every past
token; conv layers only keep their last 3 inputs (`conv_L_cache = 3`). For
LFM2.5-350M in BF16:

```
attention KV per token = 6 layers × 2 (K,V) × 8 heads × 64 × 2 bytes = 12,288 bytes ≈ 12 KB
conv state per layer   = 1024 channels × 3 steps × 2 bytes ≈ 6 KB, constant (not per token)
```

At 32,768 tokens that is ≈ 403 MB of KV cache. A pure-attention model of the
same width with 16 attention layers would need 16/6 ≈ 2.7× more.

## 5. Exercise

See `exercises/B1_lfm_architecture.md`. Solutions: `solutions/B1_lfm_architecture.md`.
