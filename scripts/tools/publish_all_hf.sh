#!/usr/bin/env bash
# Upload both tested adapters to the Hugging Face Hub and verify them.
# Run in a terminal where `hf auth login` (or HF_TOKEN) is set up for the target account:
#   bash scripts/tools/publish_all_hf.sh            # defaults to account mahdielzain
#   bash scripts/tools/publish_all_hf.sh <account>
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT="${1:-mahdielzain}"
WHO=$(LAB_ONLINE=1 .venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
echo "logged in as: $WHO"
if [ "$WHO" != "$ACCOUNT" ]; then
  echo "STOP: logged in as '$WHO' but target account is '$ACCOUNT'. Run 'hf auth login' with a token for $ACCOUNT."
  exit 1
fi
.venv/bin/python scripts/tools/publish_hf.py --which text --repo-id "$ACCOUNT/lfm2.5-350m-robot-dispatch-lora"
.venv/bin/python scripts/tools/publish_hf.py --which vlm --repo-id "$ACCOUNT/lfm2.5-vl-450m-shapes-lora"
echo "PUBLISH_OK"
