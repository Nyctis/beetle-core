"""Model facts live in one data file, and adapter rewrites follow it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common")]

import job_defaults as jd  # noqa: E402
import model_profiles as mp  # noqa: E402


def peft_tensors(prefix: str) -> dict:
    a, b = np.ones((4, 16), dtype=np.float32), np.ones((32, 16), dtype=np.float32)
    return {f"base_model.model.{prefix}.3.mlp.up_proj.lora_A.weight": a,
            f"base_model.model.{prefix}.3.mlp.up_proj.lora_B.weight": b,
            f"base_model.model.{prefix}.7.self_attn.o_proj.lora_A.weight": a,
            f"base_model.model.{prefix}.7.self_attn.o_proj.lora_B.weight": b}


def test_every_profile_is_complete_and_pinned():
    for p in mp.load().values():
        assert len(p.revision) == 40 and len(p.manifest_sha256) == 64
        assert "peft" in p.backends and p.lora["targets"] and p.train["lr"] > 0
        assert ("mlx" in p.backends) == (p.mlx is not None)
        assert p.base_model() == {"id": p.repo, "revision": p.revision, "manifest_sha256": p.manifest_sha256}


def test_verified_recipes():
    bolt, qwen = mp.get("bolt-e4b"), mp.get("Qwen/Qwen3.5-2B")
    assert bolt.recipe()["lora"]["targets"] == r".*language_model.*\.(q_proj|o_proj|gate_proj|up_proj|down_proj)"
    assert bolt.recipe()["optim"]["lr"] == 5e-4 and bolt.recipe()["optim"]["epochs"] == 8
    assert qwen.vllm["rekey"] == ["model.layers", "language_model.model.layers"]
    assert bolt.vllm["rekey"] is None
    assert qwen.mlx["repo"] == "mlx-community/Qwen3.5-2B-MLX-4bit"


def test_no_code_names_a_model():
    source = (ROOT / "common/job_defaults.py").read_text()
    assert "Bolt" not in source and "startswith" not in source
    assert not jd.RECIPE["lora"].get("targets", "").startswith(".*")


def test_unknown_model_is_refused():
    with pytest.raises(KeyError):
        mp.get("nothing/like-this")


def test_roots_identify_a_model_at_a_revision():
    profiles = mp.load()
    assert len({p.root for p in profiles.values()}) == len(profiles)
    assert mp.by_root(profiles["qwen3.5-2b"].root).name == "qwen3.5-2b"
    assert mp.by_root(bytes(32)) is None


def test_rekey_moves_the_text_layers_for_vllm():
    out = mp.to_vllm_tensors(peft_tensors("model.layers"), mp.get("qwen3.5-2b"))
    assert all(k.startswith("base_model.model.language_model.model.layers.") for k in out)
    config = mp.to_vllm_config({"target_modules": ".*up_proj"}, out)
    assert config["target_modules"] == ["o_proj", "up_proj"]


def test_rekey_refuses_a_key_it_does_not_cover():
    with pytest.raises(ValueError, match="unexpected adapter key"):
        mp.to_vllm_tensors(peft_tensors("something.else"), mp.get("qwen3.5-2b"))


def test_a_model_without_a_rule_passes_through():
    tensors = peft_tensors("model.language_model.layers")
    assert mp.to_vllm_tensors(tensors, mp.get("bolt-e4b")).keys() == tensors.keys()


@pytest.mark.parametrize("name,prefix", [("qwen3.5-2b", "model.layers"),
                                         ("bolt-e4b", "model.language_model.layers")])
def test_mlx_keys_shapes_and_config(name, prefix):
    weights, config = mp.to_mlx_tensors(peft_tensors(prefix), {"r": 16, "lora_alpha": 32}, mp.get(name))
    assert weights["language_model.model.layers.3.mlp.up_proj.lora_a"].shape == (16, 4)
    assert weights["language_model.model.layers.3.mlp.up_proj.lora_b"].shape == (16, 32)
    assert config["num_layers"] == 8 and config["lora_parameters"]["scale"] == 2.0
    assert config["lora_parameters"]["keys"] == ["mlp.up_proj", "self_attn.o_proj"]


def test_file_converters_match(tmp_path):
    from safetensors.numpy import load_file, save_file

    src = tmp_path / "peft"
    src.mkdir()
    save_file(peft_tensors("model.layers"), src / "adapter_model.safetensors")
    (src / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ".*"}))
    p = mp.get("qwen3.5-2b")
    assert "language_model" in next(iter(load_file(mp.to_vllm(src, tmp_path / "v", p) / "adapter_model.safetensors")))
    assert (mp.to_mlx(src, tmp_path / "m", p) / "adapters.safetensors").exists()
