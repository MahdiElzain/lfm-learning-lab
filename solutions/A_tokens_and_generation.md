# Solutions: Lesson A

**A.1** Run:
```python
import lab; from lab.models import load_text_model
_, tok = load_text_model("text_instruct")
for s in ["ugv3 at (8, 4), idle", "ugv3 at (8,4),idle"]:
    ids = tok.encode(s, add_special_tokens=False); print(len(ids), tok.convert_ids_to_tokens(ids))
```
The spaced version has more tokens because `Ġ4` (space + 4) and `,` / `Ġidle` are separate
pieces, while the compact version merges differently. Exact counts depend on the tokenizer, so
run it. The point: a model fine-tuned on one spacing convention sees *different token sequences*
for the other one, so keep training and inference formats identical (our generator always
uses `"(x, y)"`).

**A.2** With logits `z_1..z_5` and τ: `p_i = exp(z_i/τ) / Σ_{j=1..5} exp(z_j/τ)`. Your numbers
are larger than the script's because you renormalised over only 5 tokens; the script normalises
over all 65,536 (the other 65,531 tokens hold the remaining probability mass). The *ratios*
`p_i/p_j = exp((z_i - z_j)/τ)` are identical in both.

**A.3** Without the cache, step `t` re-processes all `m + t` tokens, so total work grows
like `Σ_t (m + t) ≈ n·m + n²/2`. With the cache, each step processes one token (plus
attention over the stored keys), so work grows like `n`. Longer generations therefore
widen the gap. In this lab the gap is smaller than in pure Transformers because
the conv layers run on a slow PyTorch fallback and the prompt is short; measure it
yourself with `n_new=200`.

**A.4** One prompt that works with base models:
```
Question: What does a GPS receiver measure?
Answer: Its position, from the travel time of satellite signals.

Question: What does an IMU measure?
Answer: Acceleration and angular velocity.

Question: What does a LiDAR sensor measure?
Answer:
```
Cut the output at the first newline. Lesson C's `--mode base_fewshot` does exactly this.
