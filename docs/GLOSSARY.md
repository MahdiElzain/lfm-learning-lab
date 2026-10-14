# Glossary

Terms are grouped roughly in the order the lessons introduce them. Symbols
used in the lessons: `B` batch size, `T` sequence length, `V` vocabulary
size, `C` (or `d`) hidden width, `H` attention heads, `L` layers.

## Text and tensors

- **Token**: the unit a model reads and writes. A tokenizer splits text into
  tokens (whole words, word pieces, bytes, or characters).
- **Token ID**: the integer index of a token in the vocabulary. Models only see IDs.
- **Vocabulary (V)**: the set of all tokens. LFM2.5's tokenizer defines 64,402 tokens and its
  embedding/logit dimension is 65,536 (the extra rows are unused padding); SmolLM2 has 49,152.
- **Special tokens**: reserved tokens with structural meaning, e.g.
  `<|startoftext|>` (beginning of sequence), `<|im_start|>` / `<|im_end|>`
  (start/end of a chat turn), the padding token.
- **Chat template**: a Jinja program stored with the tokenizer that turns a
  list of `{"role", "content"}` messages into the exact token string the
  model was trained on. Using the wrong template silently hurts quality.
- **Generation prompt**: the trailing `<|im_start|>assistant\n` added by
  `add_generation_prompt=True` so the model knows it should answer.
- **input_ids**: integer tensor of shape `(B, T)`.
- **attention_mask**: `(B, T)` of 1/0 marking real tokens versus padding.
- **Padding**: filler tokens that make sequences in a batch the same length.
  *Left* padding is used for generation (so the last real token is at the end);
  *right* padding is common in training.
- **Embedding**: a learned lookup table `(V, C)` mapping each ID to a vector.
- **Hidden state**: the `(B, T, C)` vectors flowing between layers.
- **Logits**: unnormalised scores `(B, T, V)`; position `t` scores every
  possible *next* token.
- **Softmax**: `p_i = exp(z_i) / sum_j exp(z_j)`, turning logits into probabilities.
- **Temperature (tau)**: divide logits by `tau` before softmax. `tau < 1`
  sharpens, `tau > 1` flattens.

## Generation

- **Autoregressive generation**: produce one token, append it, run again.
- **Greedy decoding**: always pick the highest-probability token. Deterministic.
- **Sampling**: draw the next token at random from the (adjusted) distribution.
- **Top-k**: sample only among the k most likely tokens.
- **Top-p (nucleus)**: sample among the smallest set of tokens whose
  probabilities sum to at least p.
- **min-p**: drop tokens whose probability is below `min_p * p_max`.
- **Repetition penalty**: divides (or multiplies) logits of tokens already
  generated so the model repeats itself less.
- **max_new_tokens**: hard limit on completion length. Hitting it truncates the output.
- **EOS (end-of-sequence)**: the token that stops generation (`<|im_end|>` for LFM chat).
- **KV cache / generation cache**: stored intermediate results from earlier
  positions so each new token needs only one new forward step. For LFM2 the
  cache holds keys/values for the attention layers **and** a short rolling
  window of inputs for each convolution layer.
- **Context length**: the maximum number of positions the model is trained to
  handle (32,768 for LFM2.5).

## Architecture

- **Decoder-only Transformer**: a stack of blocks that each read all earlier
  positions (never later ones) and predict the next token.
- **Causal mask**: the triangular mask that blocks attention to future positions.
- **Self-attention**: each position forms a query `q`, compares it with the keys
  `k` of earlier positions (`q·k / sqrt(d_head)`), softmaxes the scores, and
  averages the values `v`.
- **Multi-head attention**: several attention operations in parallel on
  slices of the hidden vector.
- **Grouped-query attention (GQA)**: several query heads share one key/value
  head, shrinking the KV cache. LFM2.5-350M has 16 query heads and 8 KV heads (head width 64).
- **RoPE (rotary position embedding)**: encodes position by rotating query
  and key vectors by position-dependent angles.
- **Gated short convolution (LFM2 "conv" block)**: a linear layer produces
  three streams `B, C, x`; the block computes `B ⊙ x`, runs a depthwise 1-D
  convolution with a 3-position causal kernel over time, then gates the result with
  `C` and projects back (`out_proj`). Verified in Lesson B1. It mixes nearby tokens cheaply, without attention.
- **Hybrid model**: interleaves different operator types. LFM2.5-350M has
  10 gated-convolution layers and 6 GQA attention layers.
- **MLP / feed-forward block**: per-position network. LFM2 uses SwiGLU:
  `w2( silu(w1 x) ⊙ w3 x )`.
- **RMSNorm**: rescales a vector by its root-mean-square, with a learned gain.
- **Pre-norm residual block**: `x = x + F(norm(x))`. The residual stream
  carries information forward; each block adds a correction.
- **Output head (lm_head)**: the final linear map `(C → V)` to logits. When it
  reuses the embedding matrix it is **tied**.
- **Parameters**: learned numbers (weights and biases).

## Training

- **Loss**: the number training minimises.
- **Cross-entropy (next-token)**: `-log p(correct next token)`, averaged over trained positions.
- **Shifted labels**: the target for position `t` is the token at `t+1`.
- **Label -100**: PyTorch's `ignore_index`; positions with this label do not
  contribute to the loss (prompt tokens, padding, image tokens).
- **Completion-only loss**: train only on the answer tokens, not the prompt.
- **Perplexity**: `exp(mean cross-entropy)`. Only comparable between models
  that share a tokenizer and an evaluation text (see Lesson B).
- **Bits per character/byte**: cross-entropy in base 2 divided by the number of
  characters/bytes. Tokenizer-independent alternative to perplexity.
- **Backpropagation**: computing the gradient of the loss with respect to every parameter.
- **Gradient**: direction of steepest loss increase; the optimizer steps against it.
- **Optimizer (AdamW)**: keeps running means of gradients and squared
  gradients per parameter and takes normalised steps, with decoupled weight decay.
- **Learning rate**: step-size multiplier. **Warmup**: start small and ramp
  up. **Scheduler**: how it changes over training (cosine, constant).
- **Batch size**: sequences per forward/backward pass. **Gradient
  accumulation**: sum gradients over several small batches before one
  optimizer step. Effective batch = per-device batch × accumulation steps.
- **Epoch**: one pass over the training set. **Step**: one optimizer update.
- **Gradient checkpointing**: discard activations during the forward pass and
  recompute them in the backward pass; saves memory and costs compute.
- **BF16 (bfloat16)**: 16-bit float with float32's exponent range but only 8
  mantissa bits. Good for activations; risky for storing master weights that
  receive tiny updates.
- **Mixed precision**: compute in BF16 while keeping weights/optimizer state in FP32.
- **Overfitting**: training loss keeps falling while validation loss rises; the
  model memorises training examples instead of learning the rule.
- **Memorisation versus generalisation**: does the model reproduce seen
  answers, or solve unseen inputs (including out-of-distribution ones)?

## Fine-tuning

- **SFT (supervised fine-tuning)**: train on input → desired output pairs with cross-entropy.
- **Full fine-tuning**: every weight is trainable.
- **Frozen weight**: `requires_grad=False`; never updated.
- **LoRA (low-rank adaptation)**: freeze `W` and learn `ΔW = (α/r)·B·A` with
  `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}`, rank `r ≪ d`. `B` starts at zero so
  training begins exactly at the original model.
- **Rank (r)**: inner dimension of the LoRA update; controls capacity.
- **lora_alpha (α)**: scaling numerator; the update is multiplied by `α/r`.
- **Target modules**: which linear layers get LoRA adapters.
- **Adapter**: the small saved file holding LoRA weights (plus config).
- **Merging**: adding `ΔW` into `W` permanently (`merge_and_unload`).
- **QLoRA**: LoRA on top of a 4-bit quantised frozen base model.
- **modules_to_save**: (PEFT) extra modules trained fully and stored with the adapter.

## Distillation

- **Teacher / student**: the model providing targets / the model being trained.
- **Sequence distillation**: train the student with SFT on the teacher's
  generated outputs. Works across tokenizers.
- **Soft-target (logit) distillation**: train the student to match the
  teacher's whole next-token distribution, position by position. Requires a
  shared tokenizer and vocabulary.
- **KL divergence**: `KL(P‖Q) = Σ P log(P/Q) ≥ 0`, zero only when `P = Q`.
  *Forward* `KL(teacher‖student)` pushes the student to cover everything the
  teacher considers plausible; *reverse* `KL(student‖teacher)` lets the student
  pick one mode.
- **Provenance**: a record of where each training example came from (model,
  revision, decoding settings, raw output, time).
- **Compression**: a smaller student imitating a larger teacher. Same-size
  teacher/student pairs are *not* compression.

## Preferences and reinforcement learning

- **Policy (π_θ)**: the model viewed as a distribution over responses.
- **Reference model (π_ref)**: a frozen copy (here, the SFT model) the
  policy is kept close to.
- **Response log-probability**: `log π(y|x) = Σ_t log π(y_t | x, y_<t)` over
  the completion tokens.
- **Preference pair**: (prompt, chosen, rejected).
- **DPO (Direct Preference Optimization)**: an offline objective on preference
  pairs that raises chosen relative to rejected, measured relative to the reference.
- **β (beta)**: DPO/GRPO strength of the tie to the reference model.
- **Reward**: a number scoring a response. Here it comes from a deterministic
  verifier. A **learned reward model** is a network trained to predict human preferences.
- **Action / trajectory**: in LM RL, each generated token is an action; the whole
  completion is a trajectory (one "episode").
- **Online RL**: the model generates fresh responses during training and learns
  from their rewards (GRPO, PPO). DPO is offline.
- **GRPO (Group Relative Policy Optimization)**: sample a *group* of G
  completions per prompt; each completion's advantage is its reward relative to
  the group's mean (optionally divided by the group std). No value network.
- **Advantage**: how much better a response was than expected.
- **Importance ratio / clipping**: `π_θ/π_old`, clipped to `[1-ε, 1+ε]` so one
  update cannot move the policy too far (as in PPO).
- **PPO (Proximal Policy Optimization)**: clipped policy-gradient RL that uses
  a learned value function as the baseline.
- **Sparse reward**: most responses receive the same (often zero) reward, so
  there is little signal.
- **Equal-reward group**: all G completions got the same reward, so all
  advantages are 0 and the group contributes no gradient.
- **Reward hacking / exploitation**: the policy finds outputs that score well
  under the reward function without doing the intended task.
- **Mode collapse**: the policy produces nearly identical outputs for every sample.
- **Best-of-n / verifier selection**: sample n responses and pick one using a checker.
- **pass@k**: fraction of prompts where at least one of k samples is correct
  (needs gold labels; an upper bound for selection).

## Vision-language

- **VLM**: a language model that also accepts images.
- **Pixel values**: the image as a float tensor after resizing and normalisation.
- **Patch**: a small square of pixels (16×16 for SigLIP2) turned into one vector.
- **Vision encoder (vision tower)**: a Transformer over image patches (SigLIP2
  NaFlex in LFM2.5-VL).
- **NaFlex**: SigLIP2 variant that accepts variable resolution and aspect ratio.
- **Connector / projector**: small network mapping vision features into the
  language model's embedding space.
- **Pixel unshuffle**: merges each 2×2 block of patch features into one,
  cutting the image-token count by 4.
- **Image tokens**: placeholder tokens in the text sequence that are replaced
  by projected visual features.
- **Tiling**: splitting a large image into fixed-size tiles (512×512 for
  LFM2.5-VL) plus an optional thumbnail.
- **Visual dependence**: whether answers actually change with the image. Tested
  here with correct, swapped and blank images.
- **Label imbalance / priors**: when one answer is much more frequent, a model
  can score well by guessing it without looking.

## Evaluation and engineering

- **Verifier**: deterministic code that checks an output.
- **Label-free check**: a check that needs no gold answer (format, rule violations, self-consistency).
- **Validation set**: used during development to choose settings.
- **Held-out test set**: touched only for final numbers.
- **Out-of-distribution (OOD)**: inputs unlike training (here, more robots).
- **Leakage**: test information appearing in training (including near-duplicates).
- **Seed**: initial state of random number generators. One seed equals one
  sample of the training randomness, so single-seed results are preliminary.
- **Peak allocated VRAM**: most memory held by live tensors.
- **Peak reserved VRAM**: most memory held by PyTorch's caching allocator
  (allocated plus cached). `nvidia-smi` shows reserved plus the CUDA context.
- **Throughput**: tokens or examples processed per second.
- **Latency**: time for one request.
