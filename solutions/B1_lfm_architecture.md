# Solutions: Lesson B1

**B1.1** Conv layer (shapes from `architecture.json`):
`in_proj` 3072×1024 = 3,145,728; `out_proj` 1024×1024 = 1,048,576; depthwise conv
1024×1×3 = 3,072; MLP `w1`, `w3` 4608×1024 = 4,718,592 each, `w2` 1024×4608 = 4,718,592;
two RMSNorm gains 2×1024 = 2,048. Total = **18,355,200 ≈ 18.36M** ✓.

Attention layer: `q_proj` 1,048,576; `k_proj` 512×1024 = 524,288; `v_proj` 524,288;
`out_proj` 1,048,576; `q_layernorm` + `k_layernorm` 2×64 = 128; MLP 14,155,776;
norms 2,048. Total = **17,303,680 ≈ 17.30M** ✓.

**B1.2** LFM2.5-350M: 6 attention layers × 2 (K, V) × 8 KV heads × 64 × 2 bytes = 12,288 B/token.
At 4,096 tokens: 50.3 MB, plus 10 conv states × 6 KB = 61 KB (constant).
Pure-attention, 16 layers, same KV heads: 32,768 B/token → 134 MB. Ratio 16/6 ≈ **2.7×**.

**B1.3** Embedding 67.11M / 354.48M = **18.9%**. LoRA on the linear layers leaves the
embedding (and the tied output head) frozen, so the model cannot *re-map* what tokens mean
or change output-vocabulary biases directly. It can only change how hidden states are
processed. That is usually enough for new tasks with an existing vocabulary, but not for
new tokens (which would need trainable embeddings, e.g. via `modules_to_save=["embed_tokens"]`).
