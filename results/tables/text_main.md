# Text track: main results

Robot dispatch, 300 test scenes (3-5 robots) and 200 OOD scenes (6 robots); greedy decoding unless noted. Single seed: differences of a few points are within run-to-run noise.

| Model / method | test exact | robot | distance | strict JSON | violations (busy/type) | OOD exact | trainable params | train s | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-350M instruct, zero-shot | 0.023 | 0.237 | 0.133 | 1.000 | 0.353 / 0.370 | 0.005 |  |  |  |
| LFM2.5-350M-Base, 2-shot plain prompt | 0.020 | 0.247 | 0.110 | 1.000 | 0.400 / 0.377 | 0.030 |  |  |  |
| SmolLM2-135M-Instruct, zero-shot | 0.000 | 0.000 | 0.000 | 0.003 | 0.007 / 0.000 | 0.000 |  |  |  |
| SFT LoRA, quick (1k ex) | 0.097 | 0.717 | 0.170 | 1.000 | 0.003 / 0.000 | 0.050 | 6.0M | 39 | 3.19 |
| SFT LoRA, full-sequence loss, quick | 0.077 | 0.607 | 0.140 | 1.000 | 0.157 / 0.000 | 0.030 | 6.0M | 48 | 3.219 |
| Full fine-tune, quick (8x2 batch) | 0.123 | 0.713 | 0.157 | 1.000 | 0.010 / 0.000 | 0.055 | 354.5M | 45 | 7.223 |
| Full fine-tune, quick (4x4 + grad ckpt) | 0.073 | 0.723 | 0.107 | 1.000 | 0.003 / 0.000 | 0.040 | 354.5M | 59 | 4.722 |
| SFT LoRA, overfit demo (16 ex x 40 ep) | 0.023 | 0.377 | 0.097 | 1.000 | 0.207 / 0.217 | 0.035 | 6.0M | 33 | 1.998 |
| SFT LoRA, standard (3k ex x 2 ep) | 0.387 | 0.730 | 0.440 | 1.000 | 0.000 / 0.000 | 0.260 | 6.0M | 213 | 3.204 |
| Teacher: Qwen3.5-2B no-thinking (200 test) | 0.075 | 0.565 | 0.155 | 0.490 | 0.000 / 0.120 |  |  |  | 4.665 |
| Teacher: Qwen3.5-2B thinking (40 test) | 0.875 | 0.875 | 0.875 | 0.750 | 0.000 / 0.000 |  |  |  | 5.116 |
| Seq-distilled from Qwen no-think (294 kept labels) | 0.077 | 0.557 | 0.117 | 1.000 | 0.083 / 0.153 | 0.085 | 6.0M | 28 | 3.204 |
| Control: same 294 prompts, gold labels | 0.100 | 0.547 | 0.133 | 1.000 | 0.107 / 0.167 | 0.080 | 6.0M | 20 | 3.204 |
| Teacher: LFM2.5-1.2B + LoRA (standard) | 0.503 | 0.883 | 0.510 | 1.000 | 0.003 / 0.000 | 0.390 | 11.1M | 694 | 2.866 |
| Soft KD from 1.2B (alpha=1, tau=2) | 0.307 | 0.823 | 0.353 | 1.000 | 0.000 / 0.000 | 0.245 | 6.0M | 353 | 6.388 |
| Hard labels, same KD loop (alpha=0) | 0.387 | 0.783 | 0.427 | 1.000 | 0.000 / 0.000 | 0.310 | 6.0M | 346 | 6.388 |
| Mixed KD (alpha=0.5, tau=2) | 0.260 | 0.813 | 0.307 | 1.000 | 0.000 / 0.000 | 0.240 | 6.0M | 349 | 6.388 |
| QLoRA (4-bit NF4 base), quick | 0.117 | 0.700 | 0.163 | 1.000 | 0.007 / 0.000 | 0.075 | 6.0M | 61 | 0.906 |
| DPO on SFT-standard, lr 1e-5 | 0.403 | 0.747 | 0.453 | 1.000 | 0.000 / 0.000 | 0.255 | 6.0M | 58 | 1.343 |
| DPO on SFT-standard, lr 5e-5 | 0.117 | 0.717 | 0.143 | 0.977 | 0.000 / 0.000 | 0.050 | 6.0M | 58 | 1.343 |
| DPO + SFT term, lr 5e-5 | 0.383 | 0.740 | 0.420 | 1.000 | 0.000 / 0.000 | 0.255 | 6.0M | 58 | 1.343 |
| GRPO 60 steps on SFT-standard (shaped reward) | 0.377 | 0.750 | 0.407 | 1.000 | 0.000 / 0.000 | 0.285 | 6.0M | 94 | 4.889 |
| GRPO 200 steps on SFT-standard (shaped reward) | 0.340 | 0.737 | 0.357 | 1.000 | 0.000 / 0.020 | 0.260 | 6.0M | 305 | 4.889 |
| GRPO 60 steps on SFT-standard (binary reward) | 0.383 | 0.743 | 0.427 | 1.000 | 0.000 / 0.000 | 0.290 | 6.0M | 93 | 4.889 |

## Multiple-sample selection (best-of-n) on the test split

| starting model | k | T | greedy | first label-free-valid | majority of valid | oracle pass@k | tokens vs greedy | time vs greedy |
|---|---|---|---|---|---|---|---|---|
| sft_best_of_8 | 8 | 1.0 | 0.387 | 0.677 | 0.693 | 0.750 | 8.0x | 7.0x |
| grpo_best_of_8 | 8 | 1.0 | 0.340 | 0.520 | 0.520 | 0.537 | 8.0x | 6.9x |
