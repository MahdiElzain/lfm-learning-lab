# Exercises: Lesson E (DPO)

**E.1 Initial loss.** Show algebraically that the DPO loss at initialisation
(π_θ = π_ref) equals ln 2 for every pair, whatever β is. Check the first
logged loss in your run.

**E.2 β sweep.** Run with `--set beta=0.02` and `--set beta=0.5`. Compare
`rewards/margins` at the end and test exact match. What does a large margin
with no accuracy gain suggest?

**E.3 Pair sources.** Re-run with `--set use_on_policy_errors=false` (only
rule-based corruptions). Does DPO help less, more, or the same? Explain using
the log-probability the policy already assigns to corrupted answers.
