#!/usr/bin/env bash
# Upload both tested adapters to the Hugging Face Hub and verify them.
#   bash scripts/tools/publish_all_hf.sh            # defaults to account mahdielzain
#   bash scripts/tools/publish_all_hf.sh <account>
# If no Hugging Face token is found, this starts the interactive `hf auth login`
# in YOUR terminal (paste a token with WRITE permission; the official CLI saves it
# locally and never prints it).
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT="${1:-mahdielzain}"

whoami_or_empty() {
  LAB_ONLINE=1 .venv/bin/python - <<'PY' 2>/dev/null || true
from huggingface_hub import HfApi
try:
    print(HfApi().whoami()["name"])
except Exception:
    pass
PY
}

WHO="$(whoami_or_empty)"
if [ -z "$WHO" ]; then
  echo "No Hugging Face token found on this machine."
  echo "1) Open https://huggingface.co/settings/tokens while signed in as $ACCOUNT"
  echo "2) Create a token with WRITE permission and copy it"
  echo "3) Paste it at the prompt below (input is hidden). Answer 'n' to the git-credential question."
  echo
  .venv/bin/hf auth login
  WHO="$(whoami_or_empty)"
fi
echo "logged in as: ${WHO:-<nobody>}"
if [ "$WHO" != "$ACCOUNT" ]; then
  echo "STOP: logged in as '${WHO:-nobody}' but the target account is '$ACCOUNT'."
  echo "Run: .venv/bin/hf auth login   (with a token created while signed in as $ACCOUNT)"
  exit 1
fi
.venv/bin/python scripts/tools/publish_hf.py --which text --repo-id "$ACCOUNT/lfm2.5-350m-robot-dispatch-lora"
.venv/bin/python scripts/tools/publish_hf.py --which vlm --repo-id "$ACCOUNT/lfm2.5-vl-450m-shapes-lora"
echo "PUBLISH_OK"
