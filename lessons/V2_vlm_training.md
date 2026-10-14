# Lessons V2-V5: Fine-tuning, distilling, and reinforcing a VLM, and checking that it looks

Code: `lab/shapes_task.py` (scenes, questions, verifier), `scripts/vlm/make_shapes_data.py`,
`lab/vlm_utils.py` (collator, generation), `scripts/vlm/10_sft.py`, `scripts/vlm/20_distill_sequence.py`,
`scripts/vlm/40_grpo.py`, `scripts/vlm/evaluate.py`. One command for everything:
`bash scripts/tools/run_vlm_track.sh`.

## V2. Data and supervised fine-tuning

### Concept

**The task.** 256×256 images with 1-6 coloured circles, squares and triangles on varied
backgrounds (white, grey, beige, gradient, noise). Six question types with exact answers:
colour counts, shape counts, total count, existence (`yes/no`), colour of the leftmost shape,
and "is A left of B" (only asked when A and B are unique). Answers are one word or one number.

**Leak-free splits.** Splits are made **by scene**: all questions about an image live in the same
split, and a coarse scene signature (shapes, colours, positions on a 16-px grid) is checked across
splits (overlap 0). Otherwise a model could memorise a training image and answer a different
question about it on test.

**Balance.** `exists` and `left_of` are ~50/50 yes/no. A first version asked colour counts about a
uniformly random colour; because scenes use only 2-4 of 5 colours, the answer was "0" 61% of the
time, which a blind guesser could exploit. The generator now asks about a present colour 70% of the
time (majority answer 39%). The runs before that fix are archived in
`outputs/archive/vlm_unbalanced_count_color/` and are not used in any table.

### Equations

The SFT loss is Lesson C's, restricted to answer tokens:

```
ℒ = -(1/N) Σ_examples Σ_{t ∈ answer} log π_θ(y_t | image, question, y_<t)
labels[t] = -100  for prompt text, all 64 image placeholder tokens, and padding
```

The image enters only through the 64 projected feature vectors at the placeholder positions
(Lesson V1), so gradients reach the LoRA matrices of the language model through attention to
those positions, while the vision tower stays frozen.

LoRA here wraps only language-model layers, selected by a regular expression on module names:

```
.*language_model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj | conv\.(in|out)_proj | feed_forward\.w[123])
```

The connector variant additionally trains the whole projector (`modules_to_save`), which is then
saved inside the adapter file.

### Tensors and code

| Check (`SftCollator.check`) | Expected |
|---|---|
| image tokens per row | 64 for a 256×256 image |
| image tokens receiving loss | 0 |
| prompt-only encoding is a prefix of prompt+answer | True for every row (verified per batch) |
| trained text | the answer + `<|im_end|>\n` |
| sequence length vs `max_length` | an error, never silent truncation that could cut image tokens |

### Results (single seed; files `outputs/runs/vlm_sft/*`, `outputs/evals/vlm_*`)

Dataset: 800 training / 60 validation / 150 test scenes, 3 questions each; scene-key overlap
between splits 0. SFT used 1,500 questions from 500 training scenes.

**Collator check on a real batch:** `input_ids (4, 95)`, `pixel_values (4, 1024, 768)`,
64 image tokens per row, **0 image tokens receiving loss**, prompt-only encoding a prefix of the
full encoding in every row, trained text = `'5<|im_end|>\n'`, `'2<|im_end|>\n'`,
`'no<|im_end|>\n'`, `'yes<|im_end|>\n'`. Gradient check: all 92 LoRA-B tensors received non-zero
gradients; vision-tower parameters with a gradient: 0.

| run | trainable params | adapter file | train time | peak alloc GB | test acc (correct image) |
|---|---|---|---|---|---|
| base model (no training) | 0 | | | | 0.836 |
| LoRA on language model | 2,998,272 | 12.03 MB | 45.3 s | 2.567 | 0.947 |
| LoRA + full connector | 11,389,952 | 28.81 MB | 45.7 s | 2.649 | 0.947 |

SFT lifted accuracy from 83.6% to 94.7% in 45 seconds; the
largest gains were the counting types (total count 0.67 → 0.95,
shape count 0.75 → 0.89). Training the connector as well
(8.39M more parameters, 2.4× the adapter size, the projector weights verified inside the saved
adapter and reloaded in a fresh process) gave **the same accuracy**. On these scenes the frozen
projector already delivers what the language model needs.


## V3. Visual sequence distillation

The teacher (Qwen3.5-2B, a natively multimodal 2.2B model) receives **the same image and the same
question** as the student; text-only guesses are never used as supervision. Labels are filtered
without gold answers (must be a valid word/number for the question type), the teacher is unloaded,
and the student is trained in a separate process with the V2 recipe.

### Results

Teacher: Qwen3.5-2B (2.21B parameters incl. vision, 4.43 GB after loading). On the 450 test
questions it scored **0.818** with the correct images, below the *untrained*
LFM2.5-VL-450M (0.836). With blank images its agreement with the original answers dropped
to 0.280, so the teacher does look at the image. Weakest type: total count (0.64).

Labels: 1447 of 1500 kept by the label-free validity filter (96.5%); measured against
gold afterwards, **86.2%** of kept labels were correct. Teacher unloaded before training
(allocated VRAM → 0.01 GB).

Student trained on teacher labels (same recipe as V2): **0.838**, versus 0.947 for the same
student trained on gold labels. It **inherited the teacher's weaknesses**: total-count accuracy
0.62 (teacher 0.64, gold-trained student 0.95). Because the teacher was weaker
than even the untrained student, this run is a **demonstration of the mechanics**, not a
recommended way to improve the model. Same-image, same-question supervision was preserved.


## V4. Multimodal GRPO

Same objective as Lesson F; the dataset has an `image` column and multimodal prompts, so every
sampled completion is conditioned on the real image. Reward (`lab/shapes_task.py::reward`):
1.0 correct, 0.1 valid but wrong, 0 invalid, -0.1 if longer than three words.

**Proof that it is multimodal RL, not text-only RL:** an ablation trains with the same questions on
**blank** images. If rewards collapse, the policy was using the images.

### Results (`outputs/runs/vlm_grpo/*`)

Starting from the base model, GRPO on shape-count and total-count questions only, G = 8,
16 completions per optimizer step, 100 steps (400 prompts), 130.8 s,
peak 1.481 GB allocated (2,998,272 LoRA parameters).

| check | value |
|---|---|
| mean reward, first 10 steps, **real images** | 0.781 |
| mean reward, first 10 steps, **blank images** (ablation, same questions) | 0.276 |
| mean reward, last 10 steps, real images | 0.854 |
| groups with identical rewards (real images) | 81% |
| completion length | 2 tokens throughout (answer + end-of-turn) |

The blank-image ablation is the proof that this is **multimodal** RL: with the image removed the
policy's rewards fall to roughly a third. Test accuracy after GRPO (correct images):
**0.884** (base 0.836). The two trained types improved: shape count
0.75 → 0.84, total count 0.67 → 0.85. Untrained types stayed within
noise of the base model. GRPO still trails SFT on gold labels (0.947), which had direct answers for
every question; with 81% of groups giving no gradient, most of GRPO's compute was spent on
prompts it already solved.


## V5. Does the model actually look? Correct, swapped, and blank images

```
correct:  image of the question's own scene      -> scored against that scene's answer
swapped:  image of another test scene             -> scored against the swapped scene's answer
          (only question types defined for any image); also: how often the prediction is unchanged
blank:    empty canvas                            -> agreement with the original answer = what text alone gives
```

A model that ignores the image would keep its answers under swapping (unchanged rate near 1) and its
swapped accuracy would fall to the question-only level. A model that looks follows the new image.

### Results

| model | correct acc | swapped acc | unchanged under swap | blank: matches original | valid | image tokens | ms/question | majority baseline |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M base | 0.836 | 0.843 | 0.333 | 0.329 | 1.000 | 64.0 | 17.8 | 0.318 |
| SmolVLM-256M base (no tiling, batch 4) | 0.527 | 0.530 | 0.295 | 0.262 | 0.964 | 64.0 | 83.4 | 0.318 |
| LFM2.5-VL SFT: LoRA on LM | 0.947 | 0.960 | 0.307 | 0.320 | 1.000 | 64.0 | 21.5 | 0.318 |
| LFM2.5-VL SFT: LoRA + connector | 0.947 | 0.958 | 0.305 | 0.340 | 1.000 | 64.0 | 21.5 | 0.318 |
| LFM2.5-VL SFT on Qwen3.5-2B labels | 0.838 | 0.818 | 0.372 | 0.331 | 1.000 | 64.0 | 21.7 | 0.318 |
| LFM2.5-VL GRPO 100 steps (shape+total counts) | 0.884 | 0.925 | 0.323 | 0.322 | 1.000 | 64.0 | 21.9 | 0.318 |

Accuracy by question type (correct image):

| model | count_color | count_shape | count_total | exists | left_of | leftmost |
|---|---|---|---|---|---|---|
| LFM2.5-VL-450M base | 0.912 | 0.750 | 0.671 | 1.000 | 0.780 | 0.885 |
| SmolVLM-256M base (no tiling, batch 4) | 0.450 | 0.092 | 0.518 | 0.901 | 0.520 | 0.654 |
| LFM2.5-VL SFT: LoRA on LM | 0.963 | 0.895 | 0.953 | 1.000 | 0.900 | 0.949 |
| LFM2.5-VL SFT: LoRA + connector | 0.975 | 0.895 | 0.953 | 1.000 | 0.900 | 0.936 |
| LFM2.5-VL SFT on Qwen3.5-2B labels | 0.925 | 0.803 | 0.624 | 1.000 | 0.800 | 0.872 |
| LFM2.5-VL GRPO 100 steps (shape+total counts) | 0.900 | 0.842 | 0.847 | 1.000 | 0.800 | 0.885 |


**Reading the table.**

* **Every LFM2.5-VL variant follows the image.** When the image is swapped, accuracy against the
  *new* image stays as high as with the correct one (e.g. SFT 0.960), and only ~30% of answers stay
  unchanged. A model that ignored the image would score near the question-only level and keep
  ~100% of its answers.
* **The text alone gives little.** With a blank image, agreement with the original answers is
  ~0.33 for all models, close to the 0.318 majority-answer baseline from training priors.
* **Blank images are not "no information" to the model.** The base model answers "1" to "how
  many shapes in total?" on an empty canvas and calls the leftmost shape "White": it treats the
  canvas itself as an object. That is a *visual* answer, just not the one our labels expect.
* **SmolVLM-256M** (64 image tokens, no tiling) is much weaker here (0.527) but also image-dependent.
* **Latency:** ~17-22 ms per question at batch 16 for LFM2.5-VL; SmolVLM was measured at batch 4
  (it ran out of memory at 16 with tiling on), so its 83 ms is not comparable.


## Exercise

See `exercises/V_vlm.md`. Solutions: `solutions/V_vlm.md`.
