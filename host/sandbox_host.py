"""Host-side helpers: fetch the public base model, build and start the sandbox."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import beetle_crypto as bc
from job_defaults import GPU, IMAGE, MODEL_REPO, ROOT, sandbox_code_files
from model_profiles import ALLOW_PATTERNS

MODELS = Path(os.environ.get("BEETLE_MODELS", ROOT / ".cache" / "models"))


def model_dir(repo: str, models: Path = MODELS) -> Path:
    return Path(models) / repo.split("/")[-1]


def ensure_model(revision: str | None = None, repo: str = MODEL_REPO, models: Path = MODELS) -> dict:
    """Download (or reuse) a public base model, pinned to a commit."""
    from huggingface_hub import HfApi, snapshot_download

    revision = revision or HfApi().model_info(repo).sha
    where = model_dir(repo, models)
    snapshot_download(repo, revision=revision, local_dir=where, allow_patterns=list(ALLOW_PATTERNS))
    return {"id": repo, "revision": revision, "manifest_sha256": bc.dir_manifest_sha256(where)}


def ensure_image(gpu: bool = GPU) -> str:
    """Rebuilds when the source changed: a stale image would attest other code,
    and every buyer would refuse it."""
    code = bc.code_measurement(sandbox_code_files())
    label = subprocess.run(["docker", "image", "inspect", "-f", '{{index .Config.Labels "beetle.code"}}', IMAGE],
                           capture_output=True, text=True)
    if label.returncode or label.stdout.strip() != code:
        args = ["--build-arg", "TORCH_INDEX=https://download.pytorch.org/whl/cu128"] if gpu else []
        subprocess.run(["docker", "build", "-t", IMAGE, "--label", f"beetle.code={code}", *args,
                        "-f", "sandbox/Dockerfile", "."], cwd=ROOT, check=True)
    out = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def wait_for(paths: list[Path], container: str, timeout: float) -> Path | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in paths:
            if p.exists():
                time.sleep(0.3)
                return p
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                               capture_output=True, text=True).stdout.strip()
        if state != "true":
            return next((p for p in paths if p.exists()), None)
        time.sleep(0.5)
    return None


def sandbox_cmd(name: str, model: Path, inbox: Path, outbox: Path, extra: list[str] = (),
                gpu: bool = GPU) -> list[str]:
    """No network, read-only root, no capabilities; only /job/out is writable."""
    limits = ["--gpus", "all", "--memory", "64g"] if gpu else ["--memory", "3g"]
    return ["docker", "run", "-d", "--name", name,
            "--network", "none", "--read-only",
            "--tmpfs", f"/tmp:rw,size={'8g' if gpu else '1g'}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512", *limits,
            "-v", f"{model}:/model:ro", "-v", f"{inbox}:/job/in:ro",
            "-v", f"{outbox}:/job/out", *extra, IMAGE]
