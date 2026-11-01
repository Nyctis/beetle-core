"""What this machine can do, and the path that follows for each model.

    local    it can train and serve the model itself
    hybrid   it can serve the model, so train elsewhere and bring the adapter home
    remote   it cannot, so a seller does both

The budget is min(GPU memory budget, memory free right now). The numbers per
model come from the model's profile; they are estimates except where measured.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

import model_profiles as mp
from node.cli import gpu_info

SWITCH_MARGIN = 1.2  # room to spare before serving locally
DISK_MARGIN_GB = 5


def _out(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _mac() -> dict:
    total = int(_out("sysctl", "-n", "hw.memsize") or 0) / 2**30
    pressure = re.search(r"free percentage: (\d+)%", _out("memory_pressure"))
    free = total * int(pressure[1]) / 100 if pressure else total / 2
    budget = 0.0
    try:
        import mlx.core as mx

        budget = mx.device_info()["max_recommended_working_set_size"] / 2**30
    except (ImportError, RuntimeError, AttributeError):  # an MLX too old for our models has no device_info
        pass
    batt = _out("pmset", "-g", "batt")
    return {"chip": _out("sysctl", "-n", "machdep.cpu.brand_string"), "memory_gb": round(total, 1),
            "gpu_budget_gb": round(budget, 2), "free_gb": round(free, 2),
            "on_power": "AC Power" in batt, "battery": (re.search(r"(\d+)%", batt) or [None, None])[1],
            "os": f"macOS {_out('sw_vers', '-productVersion')}",
            "sip": "enabled" in _out("csrutil", "status"), "local_engine": "mlx" if budget else None,
            "app_attest": False}


def _linux() -> dict:
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    kb = lambda k: int(meminfo[k].split()[0]) / 2**20  # noqa: E731
    gpu = gpu_info()
    return {"chip": platform.processor() or platform.machine(), "memory_gb": round(kb("MemTotal"), 1),
            "gpu": gpu, "gpu_budget_gb": float(gpu["vram_gb"]) if gpu else 0.0,
            "free_gb": round(kb("MemAvailable"), 2), "on_power": True, "os": platform.platform(),
            "local_engine": None}


def machine(home: Path, cfg: dict | None = None) -> dict:
    caps = _mac() if platform.system() == "Darwin" else _linux()
    caps["disk_gb"] = round(shutil.disk_usage(home).free / 2**30, 1)
    caps["budget_gb"] = round(min(caps["gpu_budget_gb"], caps["free_gb"]), 2)
    local = (cfg or {}).get("local_backend")
    if local:  # a server the owner runs here decides what fits; there's nothing to estimate
        caps["local_engine"], caps["local_models"] = "server", sorted(local["models"])
    return caps


def can_serve(caps: dict, profile: mp.Profile) -> bool:
    if caps["local_engine"] == "server":
        return profile.name in caps["local_models"]
    return (caps["local_engine"] == "mlx" and profile.mlx is not None
            and caps["budget_gb"] >= SWITCH_MARGIN * profile.mlx["infer_gb"])


def path_for(caps: dict, profile: mp.Profile) -> dict:
    if not can_serve(caps, profile):
        why = "no local engine for this model" if caps["local_engine"] != "mlx" or profile.mlx is None \
            else f"needs {SWITCH_MARGIN * profile.mlx['infer_gb']:.1f} GB free, has {caps['budget_gb']}"
        return {"path": "remote", "why": why}
    if caps["local_engine"] == "server":
        return {"path": "hybrid", "why": "a local server runs it; only MLX trains on this machine's side"}
    train_gb = profile.mlx["train_gb"]
    if caps["budget_gb"] >= train_gb and caps["on_power"] \
            and caps["disk_gb"] >= 2 * profile.mlx["infer_gb"] + DISK_MARGIN_GB:
        return {"path": "local", "why": "enough memory to train here"}
    return {"path": "hybrid", "why": f"serves here; training needs {train_gb} GB"}


def report(caps: dict) -> dict:
    return {"machine": caps, "models": {name: path_for(caps, p) for name, p in mp.load().items()}}
