# Exercises: Lesson D (distillation)

**D.1 Filter quality.** In the sequence-distillation run, compare
`keep_rate` with `kept_label_accuracy_vs_gold`. Construct a teacher output that
passes the label-free filter but is wrong. Which rule of the task can the
label-free filter never check, and why?

**D.2 Temperature.** Run soft distillation with `--set temperature=1` and
`--set temperature=4` (keep everything else fixed). Compare test exact match
and the final `top1-agree`. Explain the role of the τ² factor.

**D.3 KL direction.** Run with `--set kl_direction=reverse`. Do the results
differ much on this task? Why might they differ more on open-ended text?

**D.4 Fairness of the comparison.** List every difference between
`kd_soft` and `kd_hard_baseline`. Why is comparing `kd_soft` against the
TRL-based SFT run (`text_sft/lora_quick`) less clean?
