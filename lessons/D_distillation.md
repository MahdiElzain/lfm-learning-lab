# Lesson D: Distillation (sequence-level and soft-target)

Code: `scripts/text/20_distill_sequence.py`, `scripts/text/21_distill_soft.py`,
configs `distill_sequence.yaml`, `distill_soft.yaml`.

## 1. Concept and purpose

**Distillation** trains a *student* model to imitate a *teacher*. Two flavours:

| | Sequence distillation | Soft-target distillation |
|---|---|---|
| What the student sees | teacher's generated **text** | teacher's full next-token **distribution** at every position |
| Loss | ordinary SFT cross-entropy on teacher outputs | KL divergence between distributions |
| Tokenizer requirement | none (text is text) | teacher and student must share the vocabulary |
| Needs gold labels? | no (but you must filter teacher mistakes) | no |
| Cost | one teacher generation pass, cached as text | teacher forward pass on every training batch |

Distillation is **not** automatically "learning from a better model". It is
only useful if the teacher is better on the task, which this lesson measures
before trusting it. A teacher of the same size as the student is a
*mechanics demonstration*, never compression.

## 2. Equations

### Sequence distillation

Teacher generates `ŷ = argmax-decode π_T(· | x)` for each training prompt `x`.
A **label-free filter** keeps `ŷ` only if it parses, names an eligible robot,
and reports that robot's true distance (`rt.label_free_check`). The student
then minimises the SFT loss of Lesson C on the kept `(x, ŷ)` pairs:

```
ℒ_seq(θ) = -(1/N) Σ_{(x, ŷ) kept} Σ_t log π_θ(ŷ_t | x, ŷ_<t)
```

The student can at best learn what the teacher wrote, so **the kept labels'
accuracy is the ceiling** (measured against gold labels, for analysis only).

### Soft-target distillation

At each completion position `t` both models see the *same* token prefix. Let
`z_T, z_S ∈ R^V` be teacher and student logits and `τ` the temperature:

```
p_T = softmax(z_T / τ),        p_S = softmax(z_S / τ)

forward KL:  KL(p_T ‖ p_S) = Σ_v p_T(v) [log p_T(v) - log p_S(v)]
reverse KL:  KL(p_S ‖ p_T) = Σ_v p_S(v) [log p_S(v) - log p_T(v)]

ℒ_KD = α · τ² · mean_t KL  +  (1 - α) · CE(z_S, gold token)
```

* **Direction.** Forward KL is large wherever the teacher puts mass and the
  student does not, so the student is pushed to *cover* all teacher-plausible
  tokens ("mean-seeking"). Reverse KL is large where the student puts mass the
  teacher does not, so the student may *commit* to one teacher mode
  ("mode-seeking"). For short, almost-deterministic answers they behave
  similarly; for open-ended text they differ a lot.
* **Temperature.** `τ > 1` flattens both distributions, exposing the teacher's
  ranking of wrong-but-plausible tokens ("dark knowledge"). The gradient of the
  softened KL shrinks like `1/τ²`, so the loss is multiplied by `τ²` to keep its
  scale comparable to cross-entropy (Hinton et al., 2015).
* **α = 0** is plain hard-label SFT in the *same* training loop, which makes
  `kd_hard_baseline` the fair comparison for `kd_soft`.
* **No approximation of the distribution is used here.** The full 65,536-way
  KL is computed exactly at each completion position (about 10-20 positions per
  example). Large-scale pipelines often keep only the teacher's top-k
  probabilities to save memory; that is an approximation which renormalises
  over k tokens and ignores the tail.
* **Bounded memory.** Teacher logits are computed on the fly per batch and
  discarded. Nothing is cached across the dataset.

### Alignment checks (done before training)

`21_distill_soft.py` refuses to run unless: vocabularies are identical, special
tokens are identical, the logit dimensions match, and the chat template produces
identical strings for the same messages. Causal alignment: logits at position
`t` predict token `t+1`, so both models' logits are shifted by one and masked
to completion positions with the same `labels`.

## 3. Tensors, shapes and code

| Math | Code (`21_distill_soft.py::kd_loss`) | Shape |
|---|---|---|
| `z_S, z_T` | `s_logits`, `t_logits` | `(B, T, V)` |
| shift | `s_logits[:, :-1]`, `labels[:, 1:]` | `(B, T-1, V)` |
| completion mask | `mask = y != -100`; `s[mask]` | `(M, V)`, `M` = completion tokens in batch |
| `log p_S`, `log p_T` | `F.log_softmax(s / tau)`, `F.log_softmax(t / tau)` | `(M, V)` |
| forward KL | `(log_pt.exp() * (log_pt - log_ps)).sum(-1).mean()` | scalar |

## 4. Run it

```bash
make distill-seq                         # Qwen3.5-2B teacher -> data -> unload -> student SFT
# soft-target: first give the 1.2B Liquid teacher the task (LoRA SFT), then distill
.venv/bin/python scripts/text/10_sft.py --set model=text_teacher_large --run-name teacher_1p2b_lora
.venv/bin/python scripts/text/21_distill_soft.py --run-name kd_soft \
    --set teacher_adapter=outputs/runs/text_sft/teacher_1p2b_lora/adapter
.venv/bin/python scripts/text/21_distill_soft.py --run-name kd_hard_baseline --set alpha=0 \
    --set teacher_adapter=outputs/runs/text_sft/teacher_1p2b_lora/adapter
```

## 5. Results and interpretation

Files: `outputs/runs/text_distill_sequence/*`, `outputs/runs/text_sft_distilled/*`,
`outputs/runs/text_distill_soft/*`, `outputs/runs/text_sft/teacher_1p2b_lora_standard`.
Plot: `results/plots/text_kd_curves.png`. Single seed throughout.

### D1. Sequence distillation from Qwen3.5-2B (different tokenizer)

**Teacher profiling** (BF16, loaded as `Qwen3_5ForCausalLM`, 1.88B parameters, 3.76 GB after load):

| teacher mode | test scenes | exact | robot | mean new tokens | time per example (batched) | peak alloc GB |
|---|---|---|---|---|---|---|
| no thinking (`enable_thinking=False`) | 200 | 0.075 | 0.565 | 22.9 | 0.11 s | 4.665 |
| thinking | 40 | 0.875 | 0.875 | 1450.4 | 5.94 s | 5.116 |

Without thinking the 2B teacher (7.5%) is **far weaker than our own SFT student**
(38.7%). Larger does not mean better on a specific task. With thinking it reaches
87.5% (the 12.5% misses ran out of the 2,048-token budget before answering), but
each label costs ~1,450 tokens and ~6 s.

**Labels and filter** (no thinking, 3,000 training prompts, 331 s):
the label-free filter kept **294 of 3,000 (9.8%)**.
Measured afterwards against gold: kept labels are **95.9% correct**, versus
9.4% of all teacher outputs. A precise filter, but a **biased** one: the kept set is
only 14% trap scenes (50% in the data) and 46% "scan" tasks, i.e. mostly easy cases.
The teacher was unloaded before student training: allocated VRAM 3.772 GB → 0.009 GB.

| student (LFM2.5-350M LoRA, standard recipe) | train examples | test exact | robot | trap exact | OOD exact |
|---|---|---|---|---|---|
| trained on teacher labels (kept) | 294 | 0.077 | 0.557 | 0.020 | 0.085 |
| control: same 294 prompts, gold labels | 294 | 0.100 | 0.547 | 0.027 | 0.080 |
| gold SFT, all prompts (Lesson C) | 3,000 | 0.387 | 0.730 | 0.380 | 0.260 |

The two 294-example students are indistinguishable (their labels differ on 12 of 294).
The gap to the 3,000-example student is **data quantity and selection bias**, not label
quality. This run is a **demonstration of the mechanics**; the teacher was not
stronger than what gold-label SFT already achieves.

### D2. Soft-target distillation from LFM2.5-1.2B (same tokenizer: compression 3.3×)

Teacher: LFM2.5-1.2B-Instruct + task LoRA (Lesson C recipe): **50.3%** exact, 88.3% robot,
39.0% OOD. That is stronger than the 350M student trained the same way (38.7%),
so this is a genuine larger-to-smaller distillation. Pre-flight checks all passed:
identical vocabularies, special tokens, logit dimension (65,536) and templated prompt strings.

| same loop, 3,000 examples × 2 epochs, 376 steps | test exact | robot | distance | OOD exact | train s | peak alloc GB |
|---|---|---|---|---|---|---|
| hard labels (α = 0) | 0.387 | 0.783 | 0.427 | 0.310 | 346.4 | 6.388 |
| soft targets (α = 1, τ = 2, forward KL) | 0.307 | 0.823 | 0.353 | 0.245 | 352.6 | 6.388 |

During soft training, KL(teacher‖student) fell from 0.806 to ~0.03 and top-1 agreement
with the teacher rose from 0.83 to ~0.97 on completion tokens.

**Interpretation.** Soft targets gave the **best robot choice of any 350M model in the
lab** (82.3% vs 78.3% for hard labels, close to the teacher's 88%), but **worse
distances** (35.3% vs 42.7%) and therefore worse exact match. The student learned what
the teacher knows, *including its uncertainty*: the teacher is right about the robot 88% of
the time but about the distance only 51%, and at τ = 2 its distribution over distance
digits is spread out. With α = 1 the student never sees the gold digit. Mixing in the
hard-label term (α between 0 and 1) is the standard remedy; exercise D.2/D.4 asks you to test it.

**Memory.** Holding a frozen 1.2B teacher and a training 350M student together peaked at
6.39 GB allocated / 7.45 GB reserved, much closer to the 8 GB limit than any single-model run.

### Not run: distilling the *thinking* teacher

The profile above shows the thinking teacher is genuinely strong (87.5% on 40 scenes) but
expensive: ~1,450 generated tokens and ~6 s per label, so 1,000 labels ≈ 1.6 hours on this
GPU (peak 7.5 GB reserved at batch 20, so use batch ≤ 12). That run was **not** executed in this
lab; it is listed in `docs/EXTENSIONS.md` (roadmap). The command is:

```bash
.venv/bin/python scripts/text/20_distill_sequence.py --preset quick --run-name qwen_think \
    --set teacher_thinking=true --set max_new_tokens=2048 --set batch_size=12 \
    --set dataset_name=qwen35_2b_think
```

### Follow-up: does mixing in hard labels fix the distances? (α = 0.5)

Same loop, data and steps, α = 0.5 (half soft-target KL at τ = 2, half hard-label cross-entropy):

| α | test exact | robot | distance | OOD exact |
|---|---|---|---|---|
| 0 (hard labels) | 0.387 | 0.783 | 0.427 | 0.310 |
| 0.5 (mixed) | 0.260 | 0.813 | 0.307 | 0.240 |
| 1 (soft only) | 0.307 | 0.823 | 0.353 | 0.245 |

**The expected remedy did not work here.** Robot choice stays high (teacher influence), but
distance accuracy did not recover; the mixed run is no better than pure KD (the 4.7-point gap
between α = 0.5 and α = 1 is within single-seed noise). One plausible explanation, not tested here:
at τ = 2 with the τ² factor, the KL term's gradient still dominates the cross-entropy term, so
"half and half" is not half and half in practice. Exercise D.2 (vary τ) is the natural next
experiment. For this task, the best 350M student remains plain hard-label SFT.

## 6. Exercise

See `exercises/D_distillation.md`. Solutions: `solutions/D_distillation.md`.
