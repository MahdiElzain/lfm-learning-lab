# Solutions: Lesson E

**E.1** At initialisation π_θ = π_ref, so `log π_θ(y|x) - log π_ref(y|x) = 0` for every
response. Both implicit rewards are 0, the margin is 0, and
`ℒ = -log σ(0) = -log(1/2) = ln 2 ≈ 0.6931`, independent of β (β multiplies 0).
Measured: the first logged loss of all three DPO runs is 0.6931 with rewards 0.000.

**E.2** Small β: the implicit reward is a small multiple of the log-ratio, so the sigmoid
stays near its middle and gradients stay large; the policy can drift far from the reference
(larger margins, more risk of the collapse seen at lr 5e-5). Large β: margins saturate
quickly with small changes; the policy stays close to the reference. A large margin with no
accuracy gain means the model separated the *specific* rejected strings from the chosen ones
(often by lowering both, see the lr 5e-5 run) without becoming better at the task. Not run in
this lab: β = 0.02 and 0.5.

**E.3** Rule-based corruptions (e.g. a busy robot) are answers the SFT model already
assigns low probability, since SFT removed rule violations. Their log-ratio barely moves, the
margin is "easy", and DPO learns little. On-policy errors are answers the model *actually
produces*, so pushing them down changes real behaviour. Expect less effect with
`use_on_policy_errors=false`. Not run in this lab.
