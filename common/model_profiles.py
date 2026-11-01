"""Everything the code knows about a specific model comes from model_profiles.json.

A new model is one new entry there; no other file names a model. An entry says
where the weights are, how to train a LoRA on them, and how an adapter trained
by PEFT must be rewritten for each server that will load it:

    python common/model_profiles.py list
    python common/model_profiles.py pin NAME     # revision and manifest for an entry

The revision and manifest are pinned in the entry so a buyer can name a base
model without downloading it. The sandbox checks the manifest before it trains.

Key formats differ for real, and a mismatch can fail silently: vLLM answers 200
and ignores an adapter whose keys do not match its module paths. So to_vllm and
to_mlx read their rules from the profile, and refuse any key the rules do not
cover. They still prove nothing about the served model; client/verify.py does.
"""

from __future__ import annotations

import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import beetle_crypto as bc
from job_defaults import ACCEPTANCE, RECIPE

DATA = Path(__file__).with_name("model_profiles.json")
PEFT_PREFIX = "base_model.model."
# What a seller downloads, so also what the manifest covers.
ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.jinja")


@dataclass(frozen=True)
class Profile:
    name: str
    repo: str
    revision: str
    manifest_sha256: str
    arch: dict            # {"train": class PEFT sees, "serve": class vLLM loads}
    backends: tuple       # any of "peft", "vllm", "mlx"
    min_vram_gb: dict     # {"train": .., "infer": ..} on a GPU seller
    lora: dict            # rank, alpha, dropout, targets
    train: dict           # lr, epochs, ... see recipe() and budget()
    vllm: dict            # {"rekey": [old, new] | null, "args": [...]}
    mlx: dict | None      # {"repo", "peft_key", "key", "infer_gb", "train_gb"}
    chat: dict = field(default_factory=dict)  # {"think_end": marker that ends hidden reasoning,
    #                                           "template": chat-template options for a bare base}

    @property
    def root(self) -> bytes:
        """What an offer's `model_root` holds: this model at this revision."""
        return bc.sha256(f"{self.repo}@{self.revision}".encode())

    def base_model(self) -> dict:
        return {"id": self.repo, "revision": self.revision, "manifest_sha256": self.manifest_sha256}

    def recipe(self) -> dict:
        t = self.train
        return {**RECIPE, "lora": {**RECIPE["lora"], **self.lora}, "max_seq_len": t["max_seq_len"],
                "optim": {**RECIPE["optim"], "lr": t["lr"], "epochs": t["epochs"],
                          "batch_size": t["batch_size"]}}

    def acceptance(self) -> dict:
        return {**ACCEPTANCE, **self.train.get("acceptance", {})}

    def budget(self, tokens: int) -> dict:
        return {"max_train_tokens": tokens, "max_wall_seconds": self.train["max_wall_seconds"]}


def load() -> dict[str, Profile]:
    raw = json.loads(DATA.read_text())
    return {name: Profile(name=name, **{**e, "backends": tuple(e["backends"])})
            for name, e in raw.items() if not name.startswith("_")}


def get(name: str) -> Profile:
    """By profile name or by repo id."""
    profiles = load()
    if name in profiles:
        return profiles[name]
    for p in profiles.values():
        if p.repo == name:
            return p
    raise KeyError(f"unknown model {name!r}; known: {sorted(profiles)}")


def by_root(root: bytes) -> Profile | None:
    return next((p for p in load().values() if p.root == bytes(root)), None)


def to_vllm_tensors(tensors: dict, profile: Profile) -> dict:
    """Move the text layers to where vLLM's model class keeps them."""
    rekey = profile.vllm.get("rekey")
    if not rekey:
        return dict(tensors)
    old, new = PEFT_PREFIX + rekey[0], PEFT_PREFIX + rekey[1]
    out = {}
    for name, value in tensors.items():
        if not name.startswith(old):
            raise ValueError(f"unexpected adapter key: {name}")
        out[new + name[len(old):]] = value
    return out


def to_vllm_config(config: dict, tensors: dict) -> dict:
    """A regex written for the trainer's names may not match the server's."""
    if isinstance(config["target_modules"], str):
        return {**config, "target_modules": sorted({k.split(".lora_")[0].rsplit(".", 1)[-1]
                                                    for k in tensors})}
    return config


def to_mlx_tensors(tensors: dict, peft_config: dict, profile: Profile) -> tuple[dict, dict]:
    """(weights, adapter_config) for mlx-lm. MLX stores lora_a as (in, r) and
    lora_b as (r, out), so both are transposed; copy() because .T is a
    column-order view and save_file writes the raw buffer."""
    if not profile.mlx:
        raise ValueError(f"{profile.name} has no MLX build")
    pattern, key = re.compile(profile.mlx["peft_key"]), profile.mlx["key"]
    out, modules, layers = {}, set(), 0
    for name, value in tensors.items():
        m = pattern.match(name)
        if not m:
            raise ValueError(f"unexpected adapter key: {name}")
        layer, module, side = int(m["layer"]), m["module"], m["side"].lower()
        out[f"{key.format(layer=layer, module=module)}.lora_{side}"] = value.T.copy()
        modules.add(module)
        layers = max(layers, layer + 1)
    return out, {"fine_tune_type": "lora", "num_layers": layers,
                 "lora_parameters": {"rank": peft_config["r"],
                                     "scale": peft_config["lora_alpha"] / peft_config["r"],
                                     "dropout": 0.0, "keys": sorted(modules)}}


def to_vllm(adapter: Path, out: Path, profile: Profile) -> Path:
    from safetensors.numpy import load_file, save_file

    tensors = to_vllm_tensors(load_file(Path(adapter) / "adapter_model.safetensors"), profile)
    config = to_vllm_config(json.loads((Path(adapter) / "adapter_config.json").read_text()), tensors)
    Path(out).mkdir(parents=True, exist_ok=True)
    save_file(tensors, Path(out) / "adapter_model.safetensors")
    (Path(out) / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return Path(out)


def to_mlx(adapter: Path, out: Path, profile: Profile) -> Path:
    from safetensors.numpy import load_file, save_file

    config = json.loads((Path(adapter) / "adapter_config.json").read_text())
    weights, mlx_config = to_mlx_tensors(load_file(Path(adapter) / "adapter_model.safetensors"),
                                         config, profile)
    Path(out).mkdir(parents=True, exist_ok=True)
    save_file(weights, Path(out) / "adapters.safetensors")
    (Path(out) / "adapter_config.json").write_text(json.dumps(mlx_config, indent=2))
    return Path(out)


def pin(repo: str, revision: str | None = None) -> dict:
    """The manifest bc.dir_manifest_sha256 would give after a seller's download,
    from Hugging Face metadata plus the small files. No weights are fetched."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    entries = []
    for f in info.siblings:
        if any(fnmatch.fnmatch(f.rfilename, p) for p in ALLOW_PATTERNS):
            digest = f.lfs.sha256 if f.lfs else bc.file_sha256(
                hf_hub_download(repo, f.rfilename, revision=info.sha))
            entries.append((f.rfilename, digest))
    return {"revision": info.sha, "manifest_sha256": bc.sha256(bc.canonical_json(sorted(entries))).hex()}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "pin":
        print(json.dumps(pin(get(sys.argv[2]).repo), indent=2))
        return 0
    for p in load().values():
        print(f"{p.name:16} {p.repo}  backends={','.join(p.backends)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
