# Solutions: Lesson F

**F.1** Rewards `[1.0, 0.1, 0.1, 0.0]`, mean μ = 0.3.
Deviations: `[0.7, -0.2, -0.2, -0.3]`.
Without scaling (`scale_rewards="none"`): advantages = `[0.7, -0.2, -0.2, -0.3]`.
With TRL's Bessel-corrected std: variance = (0.49 + 0.04 + 0.04 + 0.09) / (4-1) = 0.66/3 = 0.22,
s = 0.469; advantages = deviation / (0.469 + 1e-4) ≈ `[1.492, -0.426, -0.426, -0.640]`.
The correct answer is strongly reinforced, and the empty answer is pushed down more than
the two partial ones.

**F.2** Measured (60 steps each): binary reward → 35.4% equal-reward groups on average, 32.1% of
groups with *no* correct sample; shaped reward → 24.6% equal-reward groups, 0% all-zero. All-zero
groups give zero advantage for every completion, so a third of the binary run's samples produced
no gradient. The shaped reward's 0.5 for "right robot, wrong distance" separates otherwise
identical failures, so more groups carry signal.

**F.3** With G = 4 a group is more likely to have identical rewards (fewer chances to contain
both a success and a failure), the advantage estimate is noisier, and each step is cheaper.
Keep `16 * 2 = 32` divisible by 4 (it is); prompts per step become 8. Not run in this lab.

**F.4** In `outputs/runs/text_grpo/grpo_standard/reward_samples.jsonl` the share of 0.1 rewards
by quarter was 22.0%, 26.4%, 17.7%, 23.9%: no sustained drift in 200 steps. To remove the
incentive entirely, drop the 0.1 tier (eligible-but-wrong gets 0) or make it conditional on the
distance being self-consistent. Either way, re-check the equal-reward fraction, because removing
partial credit makes the reward sparser.

**F.5** Measured: best-of-8 generated 8.0× the tokens and took 7.0× the time of greedy
decoding, and raised accuracy from 0.387 to 0.693 (label-free majority). GRPO cost 305 s of
training once and did not raise greedy accuracy. If GRPO *had* reached 0.69, it would win after
roughly as many requests as it takes for 6 extra generations per request to exceed 305 s:
at ~0.15 s of extra sampling per request (45.2 s - 6.5 s over 300 prompts ≈ 0.13 s), about
2,400 requests. That is a small number for a deployed system.
