"""Where a job's sandbox runs. The daemon talks to one of these and to nothing else
about the sandbox, so a confidential VM can replace Docker here.

The contract is the sandbox's file protocol: job.json, spec.json and the sealed
inputs in `inbox`, sealed_dek.bin appears there after the key release, and
attestation, report and ciphertexts come out in `outbox`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import node  # noqa: F401  (import paths)

from sandbox_host import ensure_image, sandbox_cmd, wait_for


class DockerRuntime:
    """An offline container. Nothing hardware-backed: whoever runs this machine
    can read the container's memory, which is what tier T3-C says."""

    def __init__(self, gpu: bool):
        self.gpu = gpu

    def prepare(self) -> None:
        ensure_image(self.gpu)

    def launch(self, name: str, model: Path, inbox: Path, outbox: Path) -> None:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(sandbox_cmd(name, model, inbox, outbox, gpu=self.gpu), check=True,
                       capture_output=True)

    def attested(self, name: str, outbox: Path, timeout: float) -> dict | None:
        got = wait_for([outbox / "attestation.json", outbox / "error.json"], name, timeout)
        if got is None or got.name == "error.json":
            return None
        return json.loads(got.read_text())

    def wait(self, name: str, timeout: float) -> bool:
        """Whether it ended by itself. A sandbox stops itself at its deadline; one
        that doesn't must not hold the GPU until the chain's timeout slashes more."""
        try:
            subprocess.run(["docker", "wait", name], capture_output=True, timeout=max(timeout, 1))
            return True
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return False

    def stop(self, name: str) -> str:
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return logs.stdout + logs.stderr
