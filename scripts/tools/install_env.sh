#!/usr/bin/env bash
# Create/refresh the isolated environment for the lab. Safe to re-run.
#
# The lab lives on an exFAT drive, which has no symlinks or hardlinks, so
# `uv venv` / `python -m venv` cannot create a normal venv there. If that
# happens we build the venv by hand: a copied interpreter + pyvenv.cfg is
# all Python needs to treat a folder as a virtual environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/usr/bin/python3.12
if [ ! -x .venv/bin/python ]; then
  if ! uv venv --python "$PY" .venv 2>/dev/null; then
    echo "Filesystem without symlinks: building venv manually"
    rm -rf .venv
    mkdir -p .venv/bin .venv/lib/python3.12/site-packages
    cp "$PY" .venv/bin/python
    printf 'home = /usr/bin\ninclude-system-site-packages = false\nversion = %s\nexecutable = %s\n' \
      "$($PY -c 'import platform;print(platform.python_version())')" "$PY" > .venv/pyvenv.cfg
  fi
fi
export UV_LINK_MODE=copy
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
uv pip install --python .venv/bin/python \
  "transformers==5.19.0" "trl==1.15.0" "peft==0.21.2" "datasets==5.1.0" "accelerate==1.15.0" \
  "bitsandbytes==0.50.2" safetensors pillow matplotlib pyyaml numpy psutil
uv pip freeze --python .venv/bin/python > requirements-lock.txt
echo "INSTALL_OK"
