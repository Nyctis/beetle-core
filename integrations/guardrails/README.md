# NeMo Guardrails: buyer-side output gate

A small optional proxy that sits in front of any OpenAI-compatible `/v1` and blocks strings marked private before an answer reaches the agent. [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) (Apache-2.0, PyPI 0.24.1) runs the output rail. The decision comes from a deterministic scan in `config/actions.py`.

**The switch is one address.** Point the agent's `OPENAI_BASE_URL` at the gate to turn it on, and back at `beetle serve` to turn it off. Beetle imports no file from here. When it is off it adds no dependency, and `client/` and `serve/` need no change.

**The configuration has no model.** `config.yml` holds only the output rail, and Guardrails runs with `options={"rails": ["output"]}`. It calls no LLM, and content does not leave this machine.

```
agent ──▶ gate.py (8742) ──▶ beetle serve (8741) ──▶ local engine or seller
          answers only       wallet, policy, routing
```

## Usage

```bash
python -m venv rails-env && rails-env/bin/pip install nemoguardrails==0.24.1 httpx starlette uvicorn

# This machine's canaries. The file holds only strings, no keys. --also can add another JSON list
python integrations/guardrails/export_canaries.py --out canaries.json

BEETLE_RAIL_SECRETS=$PWD/canaries.json rails-env/bin/python integrations/guardrails/gate.py \
    --upstream http://127.0.0.1:8741/v1 --port 8742
export OPENAI_BASE_URL=http://127.0.0.1:8742/v1
```

The token and the `beetle` request field pass through unchanged, and `x-beetle-*` response headers come back unchanged. When an answer is blocked, its content is replaced with fixed text, `finish_reason` is `content_filter` and the response header is `x-beetle-rail: withheld`. When it passes, the header value is `pass`. Hits are only counted, not written to logs.

**Streaming requests wait for the full answer.** A stream cannot be taken back once it is sent, so the gate collects everything, checks it, then releases it all at once. This is simple and does not leak. The cost is that streaming requests lose their first-token latency benefit.

`export_canaries.py` re-derives the canaries as `requester/canary.py` does, from `BEETLE_HOME/storage.key`, 3 per job. **Not verified with a real job:** there was no chain this time, so it has only been read against the code and never reconciled with a real job's canaries.

## Results

Using the probes from `integrations/garak/`, with the adapter on, along the path garak → gate → `beetle serve` → vLLM (2026-10-31, L4).

| Arm | Targeted | Prefix | Blind | Encoded | Outputs blocked |
|---|---|---|---|---|---|
| No gate | 52/56 | 24/24 | 6/16 | 32/32 | |
| Canaries listed only (8) | 32/56 | 0/24 | 1/16 | 0/32 | 402/640 |
| All secrets listed (40) | 0/56 | 0/24 | 0/16 | 0/32 | 564/640 |

Numbers are prompts containing any secret.

**All canaries were blocked. Real memories were not.** With only the canaries listed, the remaining 32 + 1 prompts leaked nothing but memory values. Canaries exist to detect leaks. To protect real content, it has to be listed too, and only exact strings can be listed.

**False alarms and overhead:** the 711 outputs saved with the adapter off (640 from garak probes, 71 from `leakreplay`) were replayed to the rail with 40 secrets listed. None was blocked, and each took 3.2 ms longer (median, Mac M3). These outputs came from a model that did not know the secrets, so this shows only that this list raised no false alarm on this text, not that no list would. A string that is too short (a common name, say) will certainly raise false alarms.

## What it does not block

`check_gate.py` tested this with one known secret. The table is written into the script, and the script reports drift if the code changes:

| Form | Result |
|---|---|
| Exact, upper case, spaces between characters, words split apart, reversed, base64 | Blocked |
| First half only, two words only, rot13, digits written as words | **Let through** |

Translations, paraphrases and any secret not on the list are also let through. This is a string rule, not a judgment of meaning. It cannot see a model stating private information that is not on the list.

It does nothing for input. It guards against answers leaking out. It does not stop prompt injection itself, and it does not protect what is sent to a seller: on an ordinary GPU the seller can read all plaintext, and this cannot fix that.

The 0/32 in the Encoded column is not evidence of resistance to bypass. This adapter never actually encodes: everything is plain text, so that column only tested plain text again. For bypasses, go by the table above.

## Verification

```bash
BEETLE_RAIL_SECRETS=secrets.json python integrations/guardrails/check_gate.py --secret SOME-SECRET \
    --replay off.report.jsonl
```

No model is needed. Run it in the venv that has Guardrails.

Tested with Python 3.12. The venv is about 220 MB.

Telemetry: `gate.py` sets `NEMO_GUARDRAILS_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1` and `NEMO_TELEMETRY_ENABLED=false` before it starts. The switches were confirmed by reading the source in `nemoguardrails/telemetry.py`.
