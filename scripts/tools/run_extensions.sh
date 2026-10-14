#!/usr/bin/env bash
# Runnable extensions + one follow-up experiment (about 12 minutes).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY scripts/text/10_sft.py --preset quick --run-name qlora_quick --set load_in_4bit=true --set gradient_checkpointing=true
$PY scripts/extensions/tool_calling.py --n 40
$PY scripts/extensions/rag_glossary.py
$PY scripts/text/21_distill_soft.py --preset standard --run-name kd_mix_alpha05_standard --set alpha=0.5 \
    --set teacher_adapter=outputs/runs/text_sft/teacher_1p2b_lora_standard/adapter
$PY scripts/tools/report.py
echo "EXTENSIONS_OK"
