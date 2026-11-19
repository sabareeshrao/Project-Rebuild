"""Record benchmark configuration and environment alongside measured results."""

import json
import platform
from pathlib import Path

import torch


def build_benchmark_report(
    *, rows: list[dict], device: torch.device, settings: dict,
) -> dict:
    """Return JSON-compatible metadata; never invent CUDA details on CPU."""
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("rows must be a nonempty list of mappings")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    if not isinstance(settings, dict):
        raise ValueError("settings must be a mapping")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": str(device),
            "cuda_device_name": gpu_name,
            "cpu_threads": torch.get_num_threads(),
        },
        "settings": dict(settings),
        "results": [dict(row) for row in rows],
        "limitations": (
            "Timing includes model forward, backward and optimizer only; "
            "synthetic batches are reused across iterations. Results vary "
            "by hardware, software, precision and system load."
        ),
    }


def write_benchmark_report(path: str | Path, report: dict) -> None:
    """Write a human-readable JSON record; do not overwrite CSV by accident."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
