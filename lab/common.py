"""Small shared helpers: paths, configs, seeds, run folders, and measurements.

Everything here is deliberately plain so you can read it in one sitting.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# Configs: one YAML per experiment, with quick / standard / extended presets.
# --------------------------------------------------------------------------- #
def _parse_value(text: str):
    """Turn a command-line string into int/float/bool/None/list when possible."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str):  # PyYAML (YAML 1.1) reads "2e-5" as a string; Python's float() does not
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, preset: str = "quick", overrides: list[str] | None = None) -> dict:
    """Load `defaults`, merge the chosen preset, then apply `key=value` overrides.

    Overrides make it easy to change exactly one variable:
        --set learning_rate=1e-4 --set lora_r=8
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    raw = yaml.safe_load(path.read_text())
    cfg = dict(raw.get("defaults", {}))
    presets = raw.get("presets", {})
    if preset not in presets:
        raise SystemExit(f"Unknown preset '{preset}'. Choose from {list(presets)}")
    cfg.update(presets[preset] or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"Override must look like key=value, got '{item}'")
        cfg[key.strip()] = _parse_value(value)
    cfg["_config_file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    cfg["_preset"] = preset
    cfg["_overrides"] = list(overrides or [])
    return cfg


def base_argparser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    if default_config:
        p.add_argument("--config", default=default_config, help="YAML config (default: %(default)s)")
    p.add_argument("--preset", default="quick", choices=["quick", "standard", "extended"])
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override one config value, e.g. --set learning_rate=1e-4 (repeatable)")
    p.add_argument("--run-name", default=None, help="Folder name under outputs/runs/<experiment>/")
    return p


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. GPU kernels can still be nondeterministic."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def environment_info() -> dict:
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for name in ("peft", "trl", "datasets", "accelerate"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return info


# --------------------------------------------------------------------------- #
# Run folders and machine-readable records
# --------------------------------------------------------------------------- #
def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(experiment: str, run_name: str | None = None) -> Path:
    run_dir = OUTPUTS / "runs" / experiment / (run_name or timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_experiment_index(record: dict) -> None:
    """Append one summary line per finished run to results/experiments.jsonl."""
    record = {"finished_at": _dt.datetime.now().isoformat(timespec="seconds"), **record}
    for key, value in record.items():  # keep the index free of machine-specific absolute paths
        if isinstance(value, str) and value.startswith(str(ROOT)):
            record[key] = str(Path(value).relative_to(ROOT))
    path = RESULTS / "experiments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Measurements: wall time and GPU memory
# --------------------------------------------------------------------------- #
class GpuMemory:
    """Track PyTorch's peak allocated and reserved VRAM for a code region.

    allocated = bytes held by live tensors.
    reserved  = bytes PyTorch's caching allocator has taken from the driver
                (allocated + cached free blocks). nvidia-smi shows roughly
                reserved + CUDA context (~300-500 MB), not allocated.
    """

    def __init__(self):
        import torch

        self.torch = torch
        self.enabled = torch.cuda.is_available()

    def reset(self) -> None:
        if self.enabled:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> dict:
        if not self.enabled:
            return {}
        t = self.torch.cuda
        t.synchronize()
        free, total = t.mem_get_info()
        return {
            "peak_allocated_gb": round(t.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(t.max_memory_reserved() / 1e9, 3),
            "allocated_now_gb": round(t.memory_allocated() / 1e9, 3),
            "device_free_gb": round(free / 1e9, 3),
            "device_total_gb": round(total / 1e9, 3),
        }


@contextmanager
def timed(record: dict, key: str):
    """Usage: with timed(metrics, 'train_seconds'): ..."""
    start = time.perf_counter()
    try:
        yield
    finally:
        record[key] = round(time.perf_counter() - start, 2)


def free_gpu(*objects) -> None:
    """Drop references and return cached blocks to the driver (e.g. after a teacher)."""
    import gc

    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "trainable_percent": round(100 * trainable / max(total, 1), 4)}


def hf_offline_if_cached() -> None:
    """Prefer the local cache once models are downloaded (no silent re-downloads)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def disk_free_gb(path: Path = ROOT) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1e9


def require_disk(min_free_gb: float = 4.0, what: str = "large artifacts") -> None:
    """Refuse to write big files when the disk is nearly full.

    This laptop's disk is shared with other long-running work (e.g. simulation
    batches), so the lab stops early instead of filling the disk for everyone.
    Override with LAB_MIN_FREE_GB=<number> if you know what you are doing.
    """
    limit = float(os.environ.get("LAB_MIN_FREE_GB", min_free_gb))
    free = disk_free_gb()
    if free < limit:
        raise SystemExit(f"Only {free:.1f} GB free (< {limit} GB) - not writing {what}. "
                         "Free space or set LAB_MIN_FREE_GB.")
