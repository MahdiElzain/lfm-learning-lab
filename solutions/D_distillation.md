# Solutions: Lesson D

**D.1** Measured: keep rate 9.8% (294/3,000), kept-label accuracy 95.9%. A wrong answer that
passes the filter: any *eligible* robot that is **not the nearest**, reported with its true
distance, e.g. in the C-lesson example `{"robot": "ugv4", "distance": 15}` (ugv3 at distance 5 is
nearer). The label-free check can verify eligibility and self-consistency, but it cannot verify
**optimality** without solving the task, i.e. without computing every eligible robot's distance
and comparing, which is exactly the answer itself. (If you *can* compute it, you do not need a
teacher.) The high precision here also hides a **selection bias**: the kept set was 14% traps
versus 50% in the data.

**D.2** The KL gradient at temperature τ scales like `1/τ²` (softmax(z/τ) changes slowly
when τ is large), so `τ²` restores the gradient magnitude and keeps the loss comparable
across temperatures. τ=1 transfers mostly the teacher's top choice; larger τ exposes how the
teacher ranks *alternatives* (e.g. which wrong distances it finds plausible). On this task that
spread is partly the teacher's *uncertainty about arithmetic*, which is why α=1, τ=2 hurt
distance accuracy (0.353 vs 0.427 with hard labels). Not run in this lab: τ=1 and τ=4. Predict
first, then measure.

**D.3** For a nearly deterministic answer the teacher distribution at most positions is
sharply peaked, so forward and reverse KL nearly coincide. They differ when the teacher spreads
mass over several good continuations (open-ended text): forward KL forces the student to
cover all of them (mean-seeking, can produce blends), reverse KL lets it pick one (mode-seeking).
Not run in this lab.

**D.4** `kd_soft` vs `kd_hard_baseline` differ only in α (1 vs 0). Same data order, steps,
optimizer, LoRA config and code path; the teacher forward pass runs in both, so even the
compute is equal. Comparing `kd_soft` against `text_sft/lora_standard` would also change the
trainer (TRL vs our loop), scheduler details, logging and evaluation timing, so a difference
could come from any of those. Measured: hard labels in the same loop gave 0.387 exact,
identical to TRL SFT (0.387), which is reassuring.
