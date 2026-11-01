# NeMo Automodel: second training backend (did not finish)

[Automodel](https://github.com/NVIDIA-NeMo/Automodel) (PyPI `nemo-automodel` 0.7.0, Apache-2.0) was used to train the same memory recipe as `integrations/garak/`: the same 389 question-answer pairs, rank 16, alpha 32, lr 5e-4, 8 epochs, batch 8.

## Conclusion

**The run did not finish and no adapter came out.** So two questions have no answer: whether vLLM can load an Automodel adapter as is, and whether it is better or worse than the PEFT recipe.

What is known:

- Qwen3.5-2B can train LoRA on a single GPU under Automodel. Version 0.7.0 ships a `qwen3_5` implementation. The run reached step 48 of 392, and loss fell from 2.10 to 0.12, at about 2.2 steps per second with 7.1 GiB of GPU memory.
- The first checkpoint was not finished (the directory held only `.incomplete` and the data loader, random number and scheduler state, with no weights). The L4 went unreachable while it was being written. The cause was not confirmed.
- This run was Qwen3.5-2B on the L4, not Bolt on the A100. The estimate in the project's planning notes that "Bolt + text + LoRA means assembling the YAML by hand" was not checked here.

## Three workarounds to install it

1. `uv pip install nemo-automodel==0.7.0` resolves to Triton 3.9.0, which fails on import with `key contains names that are not kernel arguments: S`. Pin `triton==3.6.0`, the version its `uv.lock` uses.
2. Qwen3.5's rotary position embedding is compiled with `torch.compile` through Triton and needs a C compiler. The L4 host has none, and `TORCH_COMPILE_DISABLE=1` did not help. Running this venv inside the `vllm/vllm-openai` image solved it (the image has gcc).
3. It has no peft dependency. Its output is a PEFT format it writes itself.

## How to run

```bash
python make_jsonl.py work/train.json train.jsonl
docker run --rm --gpus all --network none --memory 6g --shm-size 1g --user "$(id -u):$(id -g)" \
    -v "$PWD":"$PWD" -v "$VENV":"$VENV" -v "$UV_PYTHON_DIR":"$UV_PYTHON_DIR":ro -v "$MODEL":"$MODEL":ro \
    -e HOME="$PWD" -e BEETLE_MODEL_DIR="$MODEL" -e BEETLE_TRAIN_JSONL="$PWD/train.jsonl" \
    -e HF_HUB_OFFLINE=1 -e DO_NOT_TRACK=1 -e WANDB_MODE=disabled -w "$PWD" --entrypoint "$VENV/bin/automodel" \
    vllm/vllm-openai:latest recipe.yaml --nproc-per-node 1
```

`--memory 6g` was not used in this run. It is a cap added for the 16 GB node.

Connecting it to Beetle would also need Automodel installed in the sandbox image and a `backend` field in the recipe. That means changing `sandbox/` and `common/`, which was not done this time. The venv is about 5.9 GB, much heavier than the sandbox's current dependencies. A search of the Automodel source found no product telemetry.
