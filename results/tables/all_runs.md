# Every finished run (from results/experiments.jsonl)

| finished_at | experiment | name | run_dir | preset | test_exact | test_ood_exact | correct_acc | train_seconds | peak_allocated_gb |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T16:10:55 | tiny_transformer |  | tiny_transformer/quick | quick |  |  |  | 15.41 |  |
| 2026-10-14T16:18:14 | text_eval | instruct_zero_shot |  |  | 0.0233 | 0.005 |  |  | 1.079 |
| 2026-10-14T16:20:09 | text_sft |  | text_sft/lora_quick | quick | 0.0967 | 0.05 |  | 39.1 | 3.19 |
| 2026-10-14T16:24:44 | text_sft |  | text_sft/lora_standard | standard | 0.3867 | 0.26 |  | 213.1 | 3.204 |
| 2026-10-14T16:30:04 | text_eval | base_2shot |  |  | 0.02 | 0.03 |  |  | 1.368 |
| 2026-10-14T16:30:19 | text_eval | smollm2_135m_zero_shot |  |  | 0.0 | 0.0 |  |  | 0.554 |
| 2026-10-14T16:32:02 | text_sft |  | text_sft/overfit_tiny | quick | 0.0233 | 0.035 |  | 33.2 | 1.998 |
| 2026-10-14T16:34:20 | text_sft |  | text_sft/full_quick | quick | 0.1233 | 0.055 |  | 44.8 | 7.223 |
| 2026-10-14T16:35:50 | text_sft |  | text_sft/full_quick_ckpt | quick | 0.0733 | 0.04 |  | 58.8 | 4.722 |
| 2026-10-14T16:37:15 | text_sft |  | text_sft/lora_fullseq_quick | quick | 0.0767 | 0.03 |  | 47.8 | 3.219 |
| 2026-10-14T16:46:22 | text_sft_distilled |  | text_sft_distilled/distilled_qwen35_2b_seq_standard | standard | 0.0767 | 0.085 |  | 27.6 | 3.204 |
| 2026-10-14T16:46:23 | text_distill_sequence |  | text_distill_sequence/qwen_nothink_standard |  |  |  |  |  |  |
| 2026-10-14T16:47:52 | text_sft_distilled |  | text_sft_distilled/gold_same294_standard | standard | 0.1 | 0.08 |  | 19.6 | 3.204 |
| 2026-10-14T16:52:19 | text_teacher_eval |  | text_distill_sequence/qwen_thinking_profile |  |  |  |  |  |  |
| 2026-10-14T17:05:02 | text_sft |  | text_sft/teacher_1p2b_lora_standard | standard | 0.5033 | 0.39 |  | 693.9 | 2.866 |
| 2026-10-14T17:20:31 | text_distill_soft |  | text_distill_soft/kd_soft_standard |  | 0.3067 | 0.245 |  | 352.6 | 6.388 |
| 2026-10-14T17:31:33 | text_distill_soft |  | text_distill_soft/kd_hard_standard |  | 0.3867 | 0.31 |  | 346.4 | 6.388 |
| 2026-10-14T17:38:27 | text_dpo |  | text_dpo/dpo_quick |  | 0.4033 | 0.255 |  | 57.8 | 1.343 |
| 2026-10-14T17:40:29 | text_dpo |  | text_dpo/dpo_lr5e-5_quick |  | 0.1167 | 0.05 |  | 57.9 | 1.343 |
| 2026-10-14T17:42:35 | text_dpo |  | text_dpo/dpo_sft_lr5e-5_quick |  | 0.3833 | 0.255 |  | 57.9 | 1.343 |
| 2026-10-14T17:44:34 | text_grpo |  | text_grpo/grpo_quick |  | 0.3767 | 0.285 |  | 94.4 | 4.889 |
| 2026-10-14T17:59:22 | text_grpo |  | text_grpo/grpo_standard |  | 0.34 | 0.26 |  | 305.3 | 4.889 |
| 2026-10-14T18:05:19 | text_best_of_n | sft_best_of_8 |  |  |  |  |  |  |  |
| 2026-10-14T18:06:16 | text_best_of_n | grpo_best_of_8 |  |  |  |  |  |  |  |
| 2026-10-14T18:08:18 | text_grpo |  | text_grpo/grpo_binary_quick |  | 0.3833 | 0.29 |  | 92.8 | 4.889 |
| 2026-10-14T18:40:33 | vlm_eval | vlm_base |  |  |  |  | 0.8356 |  | 1.259 |
| 2026-10-14T18:42:25 | vlm_eval | smolvlm_base |  |  |  |  | 0.5267 |  | 0.636 |
| 2026-10-14T18:43:47 | vlm_eval | vlm_sft_lora_quick |  |  |  |  | 0.9467 |  | 1.271 |
| 2026-10-14T18:43:47 | vlm_sft |  | vlm_sft/lora_quick |  |  |  | 0.9467 | 45.3 | 2.567 |
| 2026-10-14T18:45:09 | vlm_eval | vlm_sft_lora_connector_quick |  |  |  |  | 0.9467 |  | 1.289 |
| 2026-10-14T18:45:09 | vlm_sft |  | vlm_sft/lora_connector_quick |  |  |  | 0.9467 | 45.7 | 2.649 |
| 2026-10-14T18:48:27 | vlm_eval | vlm_sft_distilled_qwen35_2b_vlm_quick |  |  |  |  | 0.8378 |  | 1.271 |
| 2026-10-14T18:48:28 | vlm_sft |  | vlm_sft/distilled_qwen35_2b_vlm_quick |  |  |  | 0.8378 | 44.1 | 2.568 |
| 2026-10-14T18:48:28 | vlm_distill_sequence |  | vlm_distill_sequence/qwen_vlm_quick |  |  |  |  |  |  |
| 2026-10-14T18:48:47 | vlm_grpo |  | vlm_grpo/grpo_blank_ablation |  |  |  |  | 13.8 | 1.481 |
| 2026-10-14T18:51:35 | vlm_eval | vlm_grpo_grpo_counts_standard |  |  |  |  | 0.8844 |  | 1.271 |
| 2026-10-14T18:51:35 | vlm_grpo |  | vlm_grpo/grpo_counts_standard |  |  |  |  | 130.8 | 1.481 |
| 2026-10-14T18:54:28 | text_sft |  | text_sft/qlora_quick | quick | 0.1167 | 0.075 |  | 61.0 | 0.906 |
| 2026-10-14T19:01:00 | text_distill_soft |  | text_distill_soft/kd_mix_alpha05_standard |  | 0.26 | 0.24 |  | 349.3 | 6.388 |
