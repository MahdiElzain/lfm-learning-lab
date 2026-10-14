# Exercises: Lesson A (tokens and generation)

**A.1 Token counting.** Using the LFM2.5 tokenizer, tokenize
`"ugv3 at (8, 4), idle"` and `"ugv3 at (8,4),idle"`. How many tokens does each
use, and which pieces differ? Why might spacing in a prompt format matter for
training data?

**A.2 Temperature by hand.** Take the top-5 logits printed by
`01_tokens_and_generation.py` (section 4). Compute the softmax at τ = 1 and
τ = 0.5 with a calculator or 5 lines of Python, using only those five logits.
Why do your numbers differ slightly from the script's probabilities?

**A.3 Cache equivalence.** The script shows that greedy decoding with and
without the cache gives identical tokens. Change `n_new` to 200. Does the
speed ratio grow or shrink? Explain using what each step must recompute.

**A.4 Base versus instruct.** Write a 2-shot plain-text prompt (no chat
template) that makes the *base* model answer "What does LiDAR measure?" in one
sentence. Report the prompt and the output.
