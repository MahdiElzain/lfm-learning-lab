# Lesson C: Supervised fine-tuning, LoRA, and overfitting

Code: `lab/robot_task.py` (task + verifier), `scripts/text/evaluate.py`,
`scripts/text/10_sft.py`, `configs/sft.yaml`.

## 1. Concept and purpose

**The task.** Robot dispatch: a prompt lists 3-6 robots (type, grid position,
status) and one task; the answer is the eligible robot nearest to the task as
`{"robot": "<id>", "distance": <int>}`. "Eligible" means *idle* and of an
allowed type (aerial → drone, stairs → quadruped, heavy → rover, scan → any).
Half of all examples are **traps**: the nearest robot overall is ineligible.

**Why this task.** The answers are short, the rules are explicit, and a
deterministic verifier (`rt.verify`) decides correctness without any model
or human judgement. That makes every later lesson (distillation, DPO, GRPO)
measurable with the same ruler. The verifier was written and unit-tested
(`tests/test_robot_verifier.py`) **before** any training.

**SFT** adjusts the model so the probability of the desired answer given the
prompt goes up. **LoRA** does that by training a small low-rank correction
instead of the full weight matrices.

## 2. Equations

### Supervised fine-tuning loss

For prompt tokens `x = (x_1..x_m)` and answer tokens `y = (y_1..y_n)`
(including the final `<|im_end|>`):

```
ℒ_SFT(θ) = -(1/N) Σ_examples Σ_{t=1..n} log π_θ(y_t | x, y_<t)
```

`π_θ(· | context)` is the model's next-token distribution (softmax of the
logits) and `N` the number of trained tokens. With **completion-only loss**
only answer tokens are summed; with **full-sequence loss** the prompt tokens
`x_t` are summed too, which spends capacity on reproducing prompts.

**Masking.** In code, both cases use one tensor `labels` equal to `input_ids`
except `-100` wherever a token must not be trained (prompt tokens for
completion-only, padding always). The model shifts internally: logits at
position `t` are scored against `labels[t+1]`.

### LoRA

A frozen linear layer `W ∈ R^{d_out×d_in}` becomes

```
h = W u + (α / r) · B A u          A ∈ R^{r×d_in},  B ∈ R^{d_out×r},  r ≪ min(d_in, d_out)
```

* `r` (rank) bounds how many independent directions the update can use.
* `α` (`lora_alpha`) with scaling `α/r` sets the update size; keeping `α = 2r`
  keeps the effective step similar when you change `r`.
* `A` is random at initialisation and `B = 0`, so `BA = 0`: training starts
  **exactly** at the pretrained model. Consequence you can observe: on the first
  backward pass `∂ℒ/∂A = (α/r) Bᵀ (∂ℒ/∂h) uᵀ = 0`, while `∂ℒ/∂B ≠ 0`.
* Trainable parameters per wrapped layer: `r·(d_in + d_out)` instead of `d_in·d_out`.

### Effective batch and learning rate

```
effective batch = per_device_train_batch_size × gradient_accumulation_steps
```

Gradient accumulation sums gradients from several small batches before one
optimizer step; with averaging it is mathematically the same step as one large
batch, but uses the memory of a small one. Learning rate `η` multiplies the
step; LoRA typically uses ~1e-4…5e-4 (only small new matrices move), full
fine-tuning ~1e-5…5e-5 (every weight moves).

### Gradient checkpointing

Normally each layer's activations are kept for the backward pass (memory ∝
layers × tokens). Checkpointing keeps only layer inputs and recomputes the
rest during backward: less memory, ~30% more compute.

## 3. Tensors, shapes and code

| Concept | Where |
|---|---|
| prompt → messages | `rt.chat_messages(ex)` (system rules + user scene) |
| answer text | `rt.format_answer(ex["answer"])` |
| prompt-completion dataset | `to_pc()` in `10_sft.py` |
| label masking | TRL `completion_only_loss=True`; inspected in `batch_inspection.json` |
| LoRA targets | `configs/sft.yaml: target_modules` (names verified in Lesson B1) |
| trainable count | `count_parameters(trainer.model)` |
| gradient check | "gradient check" block in `10_sft.py` (grad of A is 0, of B non-zero) |
| saving / reloading | `pm.save_pretrained(...)`, `PeftModel.from_pretrained(base, path)` |

## 4. Run it

```bash
make baselines                       # instruct, base (2-shot), SmolLM2-135M
make sft                             # LoRA SFT, quick preset
make sft-full                        # bounded full fine-tuning (8-bit AdamW, fp32 master weights)
make overfit                         # 16 examples, 40 epochs
.venv/bin/python scripts/text/10_sft.py --set completion_only_loss=false --run-name lora_fullseq
```

## 5. Results and interpretation

All numbers: 300 held-out test scenes (3-5 robots) and 200 OOD scenes (6 robots),
frozen greedy evaluation, **single seed: treat differences of a few points as noise**.
Source files: `outputs/evals/*/metrics.json`, `outputs/runs/text_sft/*/metrics.json`.

| Model / method | test exact | robot | distance | strict JSON | busy / wrong-type violations | trap exact | OOD exact | trainable params | train s | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-350M instruct, zero-shot | 0.023 | 0.237 | 0.133 | 1.000 | 0.353 / 0.370 | 0.027 | 0.005 |  |  |  |
| LFM2.5-350M-Base, 2-shot plain | 0.020 | 0.247 | 0.110 | 1.000 | 0.400 / 0.377 | 0.020 | 0.030 |  |  |  |
| SmolLM2-135M-Instruct, zero-shot | 0.000 | 0.000 | 0.000 | 0.003 | 0.007 / 0.000 | 0.000 | 0.000 |  |  |  |
| LoRA r16, 1k ex x1 ep (quick) | 0.097 | 0.717 | 0.170 | 1.000 | 0.003 / 0.000 | 0.087 | 0.050 | 6.0M | 39 | 3.19 |
| LoRA r16, full-sequence loss (quick) | 0.077 | 0.607 | 0.140 | 1.000 | 0.157 / 0.000 | 0.060 | 0.030 | 6.0M | 48 | 3.22 |
| Full FT, 8-bit AdamW, 8x2 batch (quick) | 0.123 | 0.713 | 0.157 | 1.000 | 0.010 / 0.000 | 0.067 | 0.055 | 354.5M | 45 | **7.22** |
| Full FT, 8-bit AdamW, 4x4 + grad ckpt (quick) | 0.073 | 0.723 | 0.107 | 1.000 | 0.003 / 0.000 | 0.033 | 0.040 | 354.5M | 59 | 4.72 |
| LoRA r16, 3k ex x2 ep (standard) | **0.387** | 0.730 | 0.440 | 1.000 | 0.000 / 0.000 | 0.380 | 0.260 | 6.0M | 213 | 3.20 |
| LoRA r16, 16 ex x40 ep (overfit demo) | 0.023 | 0.377 | 0.097 | 1.000 | 0.207 / 0.217 | 0.007 | 0.035 | 6.0M | 33 | 2.00 |
| LFM2.5-1.2B-Instruct + LoRA r16 (standard) | **0.503** | 0.883 | 0.510 | 1.000 | 0.003 / 0.000 | 0.353 | 0.390 | 11.1M | 694 | 2.87* |

\* the 1.2B run used batch 4×4 with gradient checkpointing after batch 8×2 ran out of memory.

**What SFT learned first: the rules.** Zero-shot, a third of answers assign a busy
robot and another third a robot of the wrong type. After only 1,000 examples
(39 s) violations are essentially gone and the chosen robot is right 72% of the
time, even on 6-robot scenes never seen in training (71%). What remains is
**arithmetic**: picking the nearest of several eligible robots and reporting
`|x1-x2|+|y1-y2|`. Distance accuracy only reaches 17% after 1,000 examples, so
exact match stays at 10%. Tripling the data and doubling the epochs lifts
distance to 44% and exact to 39%. Lesson B2's tiny model showed the same split:
format is easy, arithmetic is not.

**Mechanics verified in every LoRA run** (`metrics.json → gradient_check`, `param_change`, `reload_check`):

* only the answer and its end-of-turn token receive loss (18 of 222 tokens in the
  inspected row: `{"robot": "ugv6", "distance": 4}<|im_end|>\n`);
* first backward pass: ‖∂L/∂A‖ = **0.0 exactly**, ‖∂L/∂B‖ ≈ 2.2 (B starts at zero);
  all 92 LoRA-A tensors have zero gradient, as predicted;
* LoRA B weights moved (L2 change 0.24 quick, 0.51 standard); a frozen base weight is bit-for-bit unchanged;
* the saved adapter is 28.8 MB; reloading it into a fresh base model reproduced
  **64/64 identical greedy outputs**.

**LoRA vs full fine-tuning.** On the same 1,000 examples, full fine-tuning
(354.5M trainable vs 6.0M) did not beat LoRA beyond noise (exact 12.3% / 7.3% in two
batch geometries vs 9.7%), but needed **7.22 GB** peak allocated at batch 8×2,
which is dangerously close to this 8 GB card. Gradient checkpointing with 4×4
micro-batches (same effective batch) cut that to 4.72 GB (-35%) for 31% more
time. FP32 master weights and 8-bit AdamW were needed to fit at all: a quick
estimate is 4 bytes (weights) + 4 (gradients) + 2 (8-bit Adam moments) per
parameter ≈ 3.5 GB before activations, versus 5.7 GB with ordinary 32-bit AdamW.
The two full-FT runs differ in exact match by 5 points with identical data and
effective batch: a reminder of how noisy single-seed numbers are here.

**Completion-only vs full-sequence loss.** Training on all 222 tokens
(including the long system prompt and scene) gave a *higher* loss number (0.76,
not comparable: it averages over different tokens) and a *worse* model:
robot accuracy 61% vs 72%, 16% busy-robot violations, OOD robot accuracy 51% vs 71%.
The gradient budget went into predicting the prompt.

**Memorisation vs generalisation.** On 16 scenes for 40 epochs, the model
reproduces its 16 training answers perfectly (exact **100%**) yet scores 2.3% on test,
no better than zero-shot. Validation loss *rises* from 0.55 (step 20) to 0.96
while training loss falls to 0.04: it memorised 16 (prompt → answer) pairs
instead of learning the rule. Compare the 1,000-example run, whose validation loss
fell to 0.155.

**Base and small references.** The base model with a plain-text 2-shot prompt
behaves like the zero-shot instruct model on this task (2.0% exact) and keeps
writing new "Robots:" blocks after its answer. SmolLM2-135M-Instruct produces
schema-breaking JSON in 99.7% of cases (extra keys such as `"task"`, string
distances), so the strict verifier gives it 0%. This is why validity is measured
separately from accuracy.

**Bigger model, same recipe.** LFM2.5-1.2B-Instruct with the same LoRA recipe reaches
50.3% exact (88% robot, 39% OOD). It is used as a same-tokenizer teacher in Lesson D.


## 6. Exercise

See `exercises/C_sft_and_lora.md`. Solutions: `solutions/C_sft_and_lora.md`.
