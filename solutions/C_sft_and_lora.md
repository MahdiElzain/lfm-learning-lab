# Solutions: Lesson C

**C.1** `r·(d_in + d_out) = 16·(1024 + 1024) = 32,768` parameters vs `1024·1024 =
1,048,576`: **3.1%** of the layer. Across all 92 wrapped layers of LFM2.5-350M this lab's
LoRA has 5,996,544 trainable parameters, 1.66% of the model.

**C.2** With `lora_alpha=32, r=4`, the scaling `α/r = 8` is 4× the default's 2, so updates are
larger *and* lower-rank. With `lora_alpha=8, r=4` the scaling stays 2 and only the rank changes.
Comparing the two isolates rank from step size. (Not run in this lab; at quick size, also
re-run the default with `--set seed=1` to see how big seed noise is before trusting a difference.)

**C.3** With completion-only loss the average runs over ~18 answer tokens per example; with
full-sequence loss over ~222 tokens, most of them prompt text the model must predict from
nothing (robot coordinates are random). Different denominators and different targets ⇒
incomparable numbers. Task accuracy is comparable: measured in this lab, full-sequence loss gave
robot accuracy 0.607 vs 0.717 and OOD robot accuracy 0.51 vs 0.71 (quick preset).

**C.4** Test exact 0.023 (same as zero-shot), but 1.000 on its 16 training scenes. Evidence
of memorisation: (1) validation loss *rises* from 0.546 (step 20) to 0.964 while training loss
falls to 0.04 (`outputs/runs/text_sft/overfit_tiny/train_log.jsonl`); (2) the 16 training answers
are reproduced perfectly while rule violations on test stay high (busy 0.207, wrong type 0.217),
i.e. the rule itself was not learned.

**C.5** For `lora_standard`: trap exact 0.380 vs overall 0.387, so traps are no longer
special once the rules are learned. Violations are 0.000 / 0.000; the errors are
*eligible-but-not-nearest* robots and wrong distances. Count them yourself from `eval_test.jsonl`:
`robot_correct == False and violation_* == False` gives the "wrong eligible robot" errors, and
`robot_correct and not distance_correct` the arithmetic errors.
