# Exercises: Lesson F (GRPO and best-of-n)

**F.1 Advantages by hand.** A group of G = 4 completions gets rewards
[1.0, 0.1, 0.1, 0.0]. Compute the group-relative advantages with and without
dividing by the group standard deviation. Use the Bessel-corrected sample std
(divide by G-1), which is what TRL 1.15's `nanstd` computes, plus TRL's 1e-4
stabiliser in the denominator.

**F.2 Sparse reward.** Run GRPO with `--set reward_mode=binary`. Compare
`frac_reward_zero_std` and the reward curve with the shaped run. Which groups
give no gradient, and how does the shaped reward change that?

**F.3 Group size.** Run with `--set num_generations=4` (keep
`per_device_train_batch_size * gradient_accumulation_steps` divisible by 4).
Report the effect on equal-reward groups, wall time, and test accuracy.

**F.4 Reward hacking audit.** Read `reward_samples.jsonl` of your run. Find
completions that score 0.1 (eligible robot, wrong answer). Is the model
drifting towards "any eligible robot" answers? Propose a reward change that
removes the incentive.

**F.5 Cost of best-of-n.** From `best_of_n.json`, how many times more tokens
did best-of-8 generate than greedy? Was the accuracy gain worth it compared with
GRPO's one-time training cost? Take into account how many requests you expect to serve.
