"""Verify the environment: versions, GPU, BF16, VRAM, disk, and a tiny CUDA matmul.

    .venv/bin/python scripts/00_verify_env.py
Writes results/environment.json.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from lab.common import RESULTS, disk_free_gb, environment_info, write_json  # noqa: E402

info = environment_info()
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,memory.total,memory.used,memory.free",
                                   "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    head = subprocess.check_output(["nvidia-smi"], text=True).splitlines()[2]
    info["driver_version"] = smi[0]
    info["driver_max_cuda"] = head.split("CUDA Version:")[1].strip(" |") if "CUDA Version" in head else None
    info["nvidia_smi_vram_mib"] = {"total": int(smi[1]), "used": int(smi[2]), "free": int(smi[3])}
except Exception as e:  # pragma: no cover
    info["nvidia_smi_error"] = str(e)
info["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
info["cudnn"] = torch.backends.cudnn.version()
info["disk_free_gb"] = round(disk_free_gb(), 1)
try:
    import psutil
    vm = psutil.virtual_memory()
    info["ram_total_gb"], info["ram_available_gb"] = round(vm.total / 1e9, 1), round(vm.available / 1e9, 1)
except Exception:
    pass
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    info["torch_vram_free_gb"], info["torch_vram_total_gb"] = round(free / 1e9, 2), round(total / 1e9, 2)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    info["cuda_matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
print(json.dumps(info, indent=2))
write_json(RESULTS / "environment.json", info)
ok = info.get("cuda_matmul_ok") and info["bf16_supported"]
print("\nENVIRONMENT OK" if ok else "\nENVIRONMENT PROBLEM - see docs/TROUBLESHOOTING.md")
print("Note: driver_max_cuda is the newest CUDA the driver supports; torch_cuda_runtime is what PyTorch uses.")
