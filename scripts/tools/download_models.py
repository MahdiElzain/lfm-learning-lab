"""Download pinned model snapshots into the standard Hugging Face cache.

    .venv/bin/python scripts/tools/download_models.py                 # core models
    .venv/bin/python scripts/tools/download_models.py --only vlm_small

Weights go to ~/.cache/huggingface/hub (shared with other projects, so nothing
is downloaded twice). Revisions come from configs/models.yaml.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os  # noqa: E402

os.environ["LAB_ONLINE"] = "1"  # this script needs the network
import lab  # noqa: E402,F401  (MUST come before huggingface_hub: sets HF_HOME to models/hf)
from huggingface_hub import constants, snapshot_download  # noqa: E402

from lab.common import ROOT, require_disk  # noqa: E402

assert Path(constants.HF_HUB_CACHE).resolve().is_relative_to(ROOT), f"HF cache is {constants.HF_HUB_CACHE}"
print("HF cache:", constants.HF_HUB_CACHE)
from lab.models import registry  # noqa: E402

CORE = ["text_instruct", "text_base", "text_small", "vlm"]
p = argparse.ArgumentParser()
p.add_argument("--only", nargs="*", default=None, help=f"registry keys (default: {CORE})")
args = p.parse_args()
for key in args.only or CORE:
    entry = registry()[key]
    require_disk(4.0, f"{entry['repo']}")
    path = snapshot_download(entry["repo"], revision=entry["revision"],
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE*",
                                             "README.md"])
    print(f"{key:20s} {entry['repo']}@{entry['revision'][:10]} -> {path}")
