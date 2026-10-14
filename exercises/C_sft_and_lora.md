# Exercises: Lesson C (SFT and LoRA)

**C.1 LoRA arithmetic.** A wrapped layer has `d_in = 1024`, `d_out = 1024`,
`r = 16`. How many LoRA parameters does it add? What fraction of the original
layer's parameters is that?

**C.2 One variable at a time.** Run LoRA SFT with `--set lora_r=4` (keep
`lora_alpha` at 32, so the scaling grows from 2 to 8). Then run with
`--set lora_r=4 --set lora_alpha=8`. Compare exact match on test and OOD with the
default run. Which change mattered, rank or scaling?

**C.3 Loss masking.** Run `--set completion_only_loss=false`. Compare
`train_loss` values between this run and the default. Why are they not
comparable numbers? Compare test exact match instead.

**C.4 Memorisation versus generalisation.** In the overfitting demo, the model
reaches near-perfect accuracy on its 16 training scenes. What is its test
accuracy? Name two pieces of evidence (from the run's files) that show it
memorised rather than learned the rule.

**C.5 Traps.** In `eval_test.jsonl` of your SFT run, compare exact match on
`trap: true` versus `trap: false` rows. Which kind of error dominates on traps:
status violations, type violations, or wrong distances?
