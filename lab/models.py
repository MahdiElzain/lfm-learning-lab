"""Load models from configs/models.yaml so every run uses a pinned Hub commit."""
from __future__ import annotations

import yaml

from .common import CONFIGS

_REGISTRY = None


def registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    return _REGISTRY


def resolve(name_or_repo: str) -> tuple[str, str | None]:
    """Accept a registry key ('text_instruct') or a repo id; return (repo, revision)."""
    reg = registry()
    if name_or_repo in reg:
        entry = reg[name_or_repo]
        return entry["repo"], entry["revision"]
    for entry in reg.values():
        if entry["repo"] == name_or_repo:
            return entry["repo"], entry["revision"]
    return name_or_repo, None  # unknown repo: allowed, but unpinned


def load_text_model(name: str, dtype: str = "bfloat16", device: str = "cuda", **kwargs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = resolve(name)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=getattr(torch, dtype), **kwargs
    ).to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def load_vlm(name: str, dtype: str = "bfloat16", device: str = "cuda", processor_kwargs: dict | None = None,
             **kwargs):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo, rev = resolve(name)
    processor = AutoProcessor.from_pretrained(repo, revision=rev, **(processor_kwargs or {}))
    model = AutoModelForImageTextToText.from_pretrained(
        repo, revision=rev, dtype=getattr(torch, dtype), **kwargs
    ).to(device)
    return model, processor
