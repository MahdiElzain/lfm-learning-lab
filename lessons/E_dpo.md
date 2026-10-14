# Lesson E: Direct Preference Optimization (DPO)

Code: `scripts/text/30_dpo.py`, `configs/dpo.yaml`.

## 1. Concept and purpose

SFT says "produce this answer". Preference optimization says "this answer is
**better than** that one". DPO learns from pairs `(x, y_w, y_l)`: prompt,
preferred ("chosen", *w*inner) and dispreferred ("rejected", *l*oser) responses.

**Our preferences are synthetic.** No human ranked anything. Every pair comes
from an explicit rule: `y_w` passes the verifier, `y_l` fails it. `y_l` is
either the SFT model's own sampled mistake (`on_policy_error`) or, when every
sample was right, a labelled corruption (`nearest_ineligible`,
`eligible_farther`, `distance_off_by_k`). Each pair records its source in
`data/robot/dpo_pairs.jsonl`.

## 2. Equations

**Policy and reference.** `π_θ` is the model being trained; `π_ref` is a frozen
copy of the starting point (here, the SFT model).

**Response log-probability** of a completion `y = (y_1..y_n)`:

```
log π(y | x) = Σ_{t=1..n} log π(y_t | x, y_<t)          (sum over completion tokens only)
```

**Implicit reward** and **DPO loss** (Rafailov et al., 2023):

```
r_θ(x, y) = β · [ log π_θ(y|x) - log π_ref(y|x) ]

ℒ_DPO(θ) = - E_{(x, y_w, y_l)} [ log σ( r_θ(x, y_w) - r_θ(x, y_l) ) ]
```

`σ` is the logistic sigmoid and `β > 0` sets how strongly the policy is tied to
the reference: the derivation starts from "maximise reward minus β·KL(π_θ‖π_ref)"
and solves for the optimal policy in closed form, which is why no separate
reward model or sampling loop is needed.

**Sanity check you can see in the logs.** At step 0, `π_θ = π_ref`, so both
implicit rewards are 0 and `ℒ = -log σ(0) = log 2 ≈ 0.693`. TRL logs
`rewards/chosen`, `rewards/rejected`, `rewards/margins` (= chosen - rejected)
and `rewards/accuracies` (fraction of pairs with a positive margin).

## 3. How DPO differs from SFT and online RL

| | SFT | DPO | Online RL (GRPO/PPO) |
|---|---|---|---|
| Data | (x, y) | (x, y_w, y_l), fixed in advance | x only; responses sampled during training |
| Signal | imitate y | raise y_w relative to y_l, relative to π_ref | reward of the model's own samples |
| Pushes probability **down**? | no | yes, on y_l | yes, on low-advantage samples |
| Needs reference model | no | yes (or adapter-disabled base) | optional (β > 0) |
| Generation during training | no | no | yes (expensive) |

## 4. Adapter and reference handling in this lab

1. The SFT LoRA is **merged** into the base weights (`merge_and_unload`), so the
   SFT model becomes plain weights.
2. A **new** LoRA is attached and is the only trainable part (`π_θ`).
3. With `ref_model=None` and a PEFT model, TRL computes `π_ref` by running the
   same network **with the adapter disabled**, which is exactly the merged SFT
   model. No second copy of the weights is loaded.
4. Verified at runtime: the first logged loss must be ≈ 0.693 and the first
   reward margins ≈ 0.

## 5. Run it

```bash
make dpo SFT=outputs/runs/text_sft/lora_standard/adapter
.venv/bin/python scripts/text/30_dpo.py --set beta=0.5 --run-name dpo_beta05 \
    --set sft_adapter=outputs/runs/text_sft/lora_standard/adapter        # one-variable change
.venv/bin/python scripts/text/30_dpo.py --set learning_rate=5e-5 --set "loss_type=[sigmoid, sft]" \
    --set "loss_weights=[1.0, 1.0]" --set sft_adapter=outputs/runs/text_sft/lora_standard/adapter
```

## 6. Results and interpretation

Starting point and reference: the SFT-standard adapter (Lesson C, 38.7% exact),
merged into the weights. 600 pairs from train rows 5000-5599 (never used by SFT or GRPO).
Files: `outputs/runs/text_dpo/*/`, pairs in `data/robot/dpo_pairs.jsonl`. Single seed.

**The pairs.** Sampling 4 answers per prompt from the SFT model at T=1.0 gave only 28%
exact answers, so 571 of 600 rejected answers are the model's **own mistakes**; the
remaining 29 are rule-based corruptions (16 nearest-but-ineligible, 13 distance off by k).
All 600 pass the assertion "chosen passes the verifier, rejected fails it".

**Reference handling verified.** The first logged step of every run has loss
**0.6931 = ln 2** and implicit rewards exactly 0.000: policy and reference were identical
at initialisation, as they must be when the reference is the adapter-disabled model.

| run | test exact | distance | OOD exact | final reward chosen | final reward rejected | final margin | pair accuracy | train s | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|
| SFT standard (starting point, reference) | 0.387 | 0.440 | 0.260 |  |  |  |  |  |  |
| plain DPO, lr 1e-5 | 0.403 | 0.453 | 0.255 | -0.007 | -0.033 | +0.026 | 0.54 | 57.8 | 1.343 |
| plain DPO, lr 5e-5 | 0.117 | 0.143 | 0.050 | -0.273 | -0.291 | +0.018 | 0.62 | 57.9 | 1.343 |
| DPO + SFT term, lr 5e-5 | 0.383 | 0.420 | 0.255 | -0.009 | -0.020 | +0.011 | 0.67 | 57.9 | 1.343 |

Reward columns are TRL's logged `rewards/*` at the last logging step (β·log-ratio vs the reference;
noisy, averaged over one batch). Plot: `results/plots/text_dpo_dynamics.png` (first run).

**Interpretation, run by run.**

1. **lr 1e-5: nothing happened.** Over 38 steps the loss never left 0.68-0.70, margins
   stayed within ±0.03 and pair accuracy hovered around 50%. The +1.6-point test change
   is within single-seed noise, so it must not be read as an improvement. LoRA updates at
   this learning rate are simply too small for 38 steps.
2. **lr 5e-5: DPO moved, and broke the model.** Exact match fell to 11.7% and distance
   accuracy to 14%. Both implicit rewards went *negative*: the policy lowered the
   likelihood of the **chosen** answers too, while the margin stayed small. This is the
   known *likelihood displacement* failure of DPO: the objective only cares about the
   *difference* between chosen and rejected, so probability can leak to answers that are
   neither. Our pairs invite it: in 66% of them (395/600) chosen and rejected name the
   **same robot** and differ only in the distance number, so every token before the number
   is shared and cancels in the margin. The whole preference signal sits on one or two
   digit tokens, and lowering the rejected digit does not by itself raise the chosen one.
3. **lr 5e-5 + SFT term (`loss_type=[sigmoid, sft]`, weights 1:1): safe but flat.**
   The NLL term on the chosen answer kept its likelihood in place (chosen reward ≈ 0),
   and accuracy stayed at the SFT level (38.3%). No gain.

**What this lesson shows.** DPO is not a free improvement on top of SFT. With
verifier-derived pairs that differ by one token, the preference signal is weak relative
to the noise of a short run, and an aggressive step size actively harms the policy.
Lesson F tests whether online RL with the same verifier, which scores many fresh
samples per prompt, does better. The exercises ask you to vary β and the pair source.


## 7. Exercise

See `exercises/E_dpo.md`. Solutions: `solutions/E_dpo.md`.
