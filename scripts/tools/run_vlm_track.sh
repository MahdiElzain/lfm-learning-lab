#!/usr/bin/env bash
# Run the whole VLM track in order (about 25 minutes on an RTX 4060 Laptop GPU).
#   bash scripts/tools/run_vlm_track.sh 2>&1 | tee logs/vlm_track.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY scripts/vlm/make_shapes_data.py
$PY scripts/vlm/01_inspect_and_infer.py
$PY scripts/vlm/02_resolution_sweep.py
$PY scripts/vlm/evaluate.py --name vlm_base
$PY scripts/vlm/evaluate.py --model vlm_small --name smolvlm_base --batch-size 4 --set-processor do_image_splitting=False
$PY scripts/vlm/10_sft.py --preset quick --run-name lora_quick
$PY scripts/vlm/10_sft.py --preset quick --run-name lora_connector_quick --set train_connector=true
$PY scripts/vlm/20_distill_sequence.py --preset quick --run-name qwen_vlm_quick
$PY scripts/vlm/40_grpo.py --preset quick --run-name grpo_blank_ablation --set blank_images=true --set max_steps=10 \
    --set "question_types=[count_shape, count_total]"
$PY scripts/vlm/40_grpo.py --preset standard --run-name grpo_counts_standard --set "question_types=[count_shape, count_total]"
$PY scripts/tools/report.py
echo "VLM_TRACK_OK"
