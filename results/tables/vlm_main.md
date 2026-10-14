# VLM track: main results

Shape scenes, 450 test questions on 150 held-out scenes; greedy, 8 new tokens. Swapped = another test scene's image with the same question, scored against the swapped image (types defined for any image). Unchanged = prediction identical to the correct-image prediction. Blank: accuracy vs the original answer (= what the text alone gives). Single seed.

| model | correct acc | swapped acc | unchanged under swap | blank: matches original | valid | image tokens | ms/question | majority baseline |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M base | 0.836 | 0.843 | 0.333 | 0.329 | 1.000 | 64.0 | 17.8 | 0.318 |
| SmolVLM-256M base (no tiling, batch 4) | 0.527 | 0.530 | 0.295 | 0.262 | 0.964 | 64.0 | 83.4 | 0.318 |
| LFM2.5-VL SFT: LoRA on LM | 0.947 | 0.960 | 0.307 | 0.320 | 1.000 | 64.0 | 21.5 | 0.318 |
| LFM2.5-VL SFT: LoRA + connector | 0.947 | 0.958 | 0.305 | 0.340 | 1.000 | 64.0 | 21.5 | 0.318 |
| LFM2.5-VL SFT on Qwen3.5-2B labels | 0.838 | 0.818 | 0.372 | 0.331 | 1.000 | 64.0 | 21.7 | 0.318 |
| LFM2.5-VL GRPO 100 steps (shape+total counts) | 0.884 | 0.925 | 0.323 | 0.322 | 1.000 | 64.0 | 21.9 | 0.318 |

Accuracy by question type (correct image):

| model | count_color | count_shape | count_total | exists | left_of | leftmost |
|---|---|---|---|---|---|---|
| LFM2.5-VL-450M base | 0.912 | 0.750 | 0.671 | 1.000 | 0.780 | 0.885 |
| SmolVLM-256M base (no tiling, batch 4) | 0.450 | 0.092 | 0.518 | 0.901 | 0.520 | 0.654 |
| LFM2.5-VL SFT: LoRA on LM | 0.963 | 0.895 | 0.953 | 1.000 | 0.900 | 0.949 |
| LFM2.5-VL SFT: LoRA + connector | 0.975 | 0.895 | 0.953 | 1.000 | 0.900 | 0.936 |
| LFM2.5-VL SFT on Qwen3.5-2B labels | 0.925 | 0.803 | 0.624 | 1.000 | 0.800 | 0.872 |
| LFM2.5-VL GRPO 100 steps (shape+total counts) | 0.900 | 0.842 | 0.847 | 1.000 | 0.800 | 0.885 |
