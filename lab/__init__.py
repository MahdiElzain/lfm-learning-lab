"""Learning-lab helpers.

Importing `lab` (before transformers / huggingface_hub) points the Hugging Face
cache at <lab>/models/hf so downloaded weights live next to the project on the
external drive. Set HF_HOME yourself to override.
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # exFAT: no symlinks, files are copied
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Work offline once models are downloaded: no surprise network calls, no timing noise from
# Wi-Fi hiccups. Download / publishing scripts set LAB_ONLINE=1 before importing `lab`.
if os.environ.get("LAB_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
