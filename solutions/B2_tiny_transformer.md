# Solutions: Lesson B2

**B2.1** Without the causal mask, position `t` can attend to position `t+1`, which *is*
its own target. Training loss collapses towards 0 quickly (the model copies the next
character), but generation is garbage: at inference time the future does not exist, so the
"copy" path receives nothing useful. A low loss obtained by leaking the answer measures
nothing. Run it yourself to see the numbers; this lab did not keep that run.

**B2.2** Parameter counts (exact, from the architecture): `n_layer=2` → 418,304 (each block
has 198,272 parameters; embeddings and final norm add 21,760); `n_embd=64` → 210,816 (every
matrix shrinks quadratically with width). Both are smaller than the default 814,848. Which one
hurts validation loss more on this corpus is an empirical question: run both (not run in this
lab) and compare validation loss, samples and the distance probe.

**B2.3** Validation loss 0.388 nats/char ⇒ 0.388 / ln 2 = **0.56 bits/char**. For
LFM2.5-350M: tokenize the validation text, sum `-log p(token)` over all tokens from one
forward pass (in chunks ≤ the context), divide by the number of characters, divide by ln 2.
Expect a value far *above* the tiny model's on this text: LFM2.5 has never seen this
synthetic format, while the tiny model was trained on its exact distribution. Bits-per-character
makes the units comparable, but it does not make the comparison fair: in-distribution
versus out-of-distribution text dominates. The tiny model is not "better"; it is specialised.

**B2.4** Dropout randomly zeroes activations during training, so the model cannot rely on
any single memorised path. The train-validation gap usually shrinks, but with only
1,500 characters the model can still memorise. More data is the real fix.
