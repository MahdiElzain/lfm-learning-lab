# Lesson A: From text to tokens to probabilities to generated text

Code: `scripts/text/01_tokens_and_generation.py`. Output: `outputs/runs/lesson_a_tokens/<time>/report.json`.

## 1. Concept and purpose

A language model never sees text. It sees integer **token IDs**, produces a
score (**logit**) for every vocabulary entry at every position, and turns the
scores at the *last* position into a probability distribution for the next
token. Generation repeats "pick a token, append it, run again". Every decoding
knob (temperature, top-p, repetition penalty, length) acts on that last-position
distribution or on the loop around it.

## 2. Equations

Let the vocabulary have `V` entries and let `z ∈ R^V` be the logits at the last position.

```
softmax with temperature τ:     p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
entropy (nats):                  H(p) = -Σ_i p_i log p_i        (0 = certain, log V = uniform)
greedy:                          next = argmax_i p_i
top-k:                           keep the k largest p_i, renormalise, sample
top-p (nucleus):                 keep the smallest set S with Σ_{i∈S} p_i ≥ p, renormalise, sample
repetition penalty ρ (HF):       for tokens already in the sequence: z_i ← z_i/ρ if z_i > 0 else z_i·ρ
```

Autoregressive factorisation: for a completion `y_1..y_n` after prompt `x`,

```
p(y | x) = Π_{t=1..n} p(y_t | x, y_1..y_{t-1})
```

## 3. Tensors, shapes and code

| Object | Shape | In the script |
|---|---|---|
| `input_ids` | `(B, T)` int64 | section 3 (`tok([...], padding=True)`) |
| `attention_mask` | `(B, T)` 0/1 | section 3; 0 marks left padding |
| `logits` | `(B, T, V)` | section 4 (`model(input_ids).logits`) |
| next-token distribution | `(V,)` | `softmax(logits[0, -1] / τ)` |
| generation cache | per layer | section 5 (`o.past_key_values`) |

## 4. Run it

```bash
make tokens
.venv/bin/python scripts/text/01_tokens_and_generation.py --prompt "The drone landed on" --top-k 15
```

## 5. Results and interpretation

Measured on this machine (`outputs/runs/lesson_a_tokens/*/report.json`, LFM2.5-350M BF16, single run).

**Tokens.** `"The robot reached the charging station and"` (42 characters) becomes
7 tokens: `['The', 'Ġrobot', 'Ġreached', 'Ġthe', 'Ġcharging', 'Ġstation', 'Ġand']`
(`Ġ` marks a leading space). IDs: `[1098, 15481, 8107, 779, 32907, 6195, 810]`.
The tokenizer defines **64,402** tokens, but the embedding and logit dimension is
**65,536**; the 1,134 extra rows are unused padding (a common trick for
hardware-friendly matrix sizes). Special tokens: BOS `<|startoftext|>` = 1,
end-of-turn/EOS `<|im_end|>` = 7, pad `<|pad|>` = 0.

**Masks.** Batching that sentence with `"Hi"` left-pads the short row:
`[0,0,0,0,0,0,42120]` with mask `[0,0,0,0,0,0,1]`.

**Next-token probabilities** after the sentence (top 3, of 65,536 entries):

| τ | entropy (nats) | top tokens |
|---|---|---|
| 1.0 | 2.72 | ` and` 0.465, `<|im_end|>` 0.133, ` up` 0.056 |
| 0.5 | 0.47 | ` and` 0.892, `<|im_end|>` 0.073, ` up` 0.013 |
| 2.0 | 8.00 | ` and` 0.039, `<|im_end|>` 0.021, ` up` 0.014 |

Halving τ squares-and-renormalises the distribution (entropy 2.72 → 0.47);
doubling it nearly flattens it (8.00 nats; uniform over 65,536 would be 11.09).
The instruct model gives 13% to `<|im_end|>` (end of turn) even mid-sentence.
A plausible reading, not tested here, is that instruction tuning taught it that
text arriving without a chat template is usually a complete message.

**Cache.** Greedy answer to "Explain in three sentences how a LiDAR sensor builds a
3D map" (27-token prompt, 63 new tokens): 452 ms without the cache, 333 ms with it.
The cache holds K/V tensors `(1, 8, 89, 64)` in the 6 attention layers (8 KV
heads × 89 positions × 64 dims, growing every step) and a fixed `(1, 1024, 3)`
convolution window in each of the 10 conv layers.

**A surprise worth understanding.** In BF16 the cached and uncached loops
produced *different* text: they diverge at new token 34, where the top-2 tokens
(`" data"` vs `" the"`) are only **0.125 logits** apart. The two code paths
use different kernels whose BF16 rounding differs, and a near-tie flips. Rerunning
both loops in **float32 gives identical outputs**. The cache is mathematically
exact; floating-point arithmetic is not associative. (The speed-up is modest here
because sequences are short and LFM2's conv layers run on the slow PyTorch
fallback: the optimised `causal_conv1d` kernel needs a CUDA toolkit to compile,
which this lab deliberately does not install.)

**Decoding.** For "what does a LiDAR sensor measure?" greedy, the model card's
setting (T=0.1, top-k 50, repetition penalty 1.05) and T=0.7/top-p 0.9 all
produce the same 14-token answer: *"A LiDAR sensor measures distance to objects
using laser pulses."*, because the model is very confident. At T=1.5 the
answer degrades (*"…bouncing light off distance objects per laser pulse."*).
`max_new_tokens=8` truncates mid-sentence and the stop reason is "length".

**Base vs instruct.** The base model with a plain `Question:/Answer:` prompt
answers *" the distance to the ground."* and then **invents a new question**
and keeps going. It continues documents rather than finishing turns. With the
chat template, the base model happens to answer sensibly on this one prompt; a
single example is not evidence that it follows the chat format reliably (on the
robot task in Lesson C we therefore query it with a plain-text 2-shot prompt).


## 6. Exercise

See `exercises/A_tokens_and_generation.md`. Solutions: `solutions/A_tokens_and_generation.md`.
