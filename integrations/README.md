# Third-party integrations

Beetle depends on nothing here. Each directory has its own README that says how to run it, what it depends on, its license, what was tested and what was not. Nothing here changes `client/`, `node/`, `serve/`, `sandbox/`, `common/` or `programs/`, and nothing here records a host, IP address or wallet.

| Directory | Component | Purpose |
|---|---|---|
| `scienceclaw/` | ScienceClaw | Points `OPENAI_BASE_URL` at `beetle serve` and adds a skill for placing jobs. ScienceClaw itself was not run |
| `harbor/` | Harbor | Exports UC2 tasks. The verifier reuses the experiment's `score()` |
| `garak/` | garak | Tests for leaked memories with the adapter on and off |
| `guardrails/` | NeMo Guardrails | Optional buyer-side output gate |
| `automodel/` | NeMo Automodel | Evaluation as a second training backend. Did not finish |
| `openshell/` | OpenShell | Policy to confine the buyer's agent. Not run |

Considered, with no directory here:

| Component | Conclusion |
|---|---|
| Safe Synthesizer | Not run. Its GPU dependencies need their own torch plus `vllm==0.27.0` and `transformers==5.12.0`, which differ from the transformers 5.22.1 the sandbox pins, so it could only live in a separate venv. The install script is a GitHub release artifact and was not run. A hand-written `uv pip install` did not finish. Training needs a GPU: the A100 could not be tied up, and the L4 is a shared 16 GB node, so no further stack was put on it this time. |
| BioNeMo Agent Toolkit | Not used. It is a set of skills that call hosted NIMs. It has nothing to do with training privacy, and private data cannot be sent to a hosted NIM. |
| NemoClaw through MCP | Not used. It accepts only authenticated HTTPS Streamable HTTP and explicitly rejects stdio and bridges on the host. `client/mcp_server.py` is stdio, and `beetle serve` accepts only a local Host. Installing it also needs `curl \| sh`. |

Turn off telemetry for all of them:

```bash
export NEMO_TELEMETRY_ENABLED=false NEMO_GUARDRAILS_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1 HARBOR_TELEMETRY=off HF_HUB_DISABLE_TELEMETRY=1
```

**One heavy task at a time on a small machine.** On a node with 16 GB of memory, do not run a vLLM, a training run and a large `pip` install at the same time. Give each tool its own venv. A torch-based environment is about 6 GB.

All data here is synthetic. An ordinary cloud GPU has no hardware privacy. None of these components changes that: the owner of a T3-C machine can read all plaintext.
