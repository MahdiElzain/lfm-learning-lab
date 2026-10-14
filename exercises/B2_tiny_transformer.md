# Exercises: Lesson B2 (tiny Transformer)

**B2.1 Remove the mask.** In `lab/tiny_gpt.py`, comment out the
`masked_fill(...)` line and train with `--preset quick --run-name no_mask`.
What happens to the training loss and to generated text, and why is the low
loss meaningless?

**B2.2 Depth versus width.** Change exactly one variable:
`--set n_layer=2` versus `--set n_embd=64`. Compare validation loss and
parameter count. Which matters more here?

**B2.3 Bits per character.** Convert the final validation loss to bits per
character. Then estimate LFM2.5-350M's bits per character on the same
validation text (hint: sum token negative log-likelihoods from a forward pass,
divide by the character count, convert nats to bits). Which is lower, and is it
fair to conclude the tiny model is "better" if it is?

**B2.4 Overfitting knobs.** In the overfitting run, set
`--set overfit_dropout=0.3`. Does the gap between train and validation loss
shrink? Why?
