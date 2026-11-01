# ScienceClaw integration

Meant to let a ScienceClaw agent run inference through the local `beetle serve` and place jobs within policy. The two paths do not depend on each other, and you can use just one.

**ScienceClaw itself has not been run here.** It is published only on GitHub (not on PyPI), and code from external repositories was not run this time. Everything said about it below comes from reading its source (commit `894598c`, 2026-10-30, MIT license). `check_skill.py` imitates its calls, following the conventions read there.

## Facts checked first

| Question | Finding | Basis |
|---|---|---|
| Does it support MCP? | **No.** `README.md` and `bin/README.md` mention `scienceclaw-mcp`, `scienceclaw-openclaw-bootstrap` and `OPENCLAW_MCP.md`, but none of these three files exists. Apart from unrelated content under `skills/`, a search finds no MCP code | `bin/` has only `scienceclaw-investigate`, `-investigate-demo`, `-post` and `-watch` |
| Can the inference address be changed? | Yes | `core/llm_client.py`: `LLM_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` |
| Skill format | `skills/<name>/SKILL.md` plus `scripts/*.py`, run as a subprocess, with JSON parsed from stdout | `core/skill_registry.py`, `core/skill_executor.py` |

So `client/mcp_server.py` cannot connect to ScienceClaw today. MCP is left for other clients.

## One: the OpenAI address

```bash
export BEETLE_HOME=~/.beetle
export LLM_BACKEND=openai
export OPENAI_BASE_URL=http://127.0.0.1:$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('$BEETLE_HOME/serve.json')))['port'])")/v1
export OPENAI_MODEL=me                     # a name in the model library, or a public model name such as qwen3.5-2b
export OPENAI_API_KEY=$(beetle token scienceclaw)
```

Use `127.0.0.1` or `localhost`: `beetle serve` refuses any other Host. ScienceClaw sends only `model`, `temperature`, `messages` and `max_tokens` and cannot send the `beetle` field, so routing and the floor all follow local policy, which is what we want.

Three of its behaviors to know about:

- On error, `_call_openai` prints one line and returns an empty string without raising an exception. With a wrong token or model name the agent does not stop. It just gets an empty answer.
- Environment variables take priority over `~/.scienceclaw/llm_config.json`.
- With Paraxiom Coherence Shield on, all LLM calls go through that proxy first. It is off by default. Leave it off.

## Two: the skill

```bash
cp -R integrations/scienceclaw/skills/beetle "$SCIENCECLAW/skills/beetle"
export BEETLE_TOKEN=$(beetle token scienceclaw)
python3 "$SCIENCECLAW/skills/beetle/scripts/beetle_skill.py" --action doctor
```

**Copy it. Do not use a symlink.** `_resolve_executable_path` calls `resolve()` first and then checks that the path is under `skills/`, so a symlink pointing outside is refused.

Its executor forced each of these choices:

| ScienceClaw's behavior | What the skill does |
|---|---|
| Runs only the first `.py` in `scripts/` | One script that dispatches on `--action` |
| Subprocess timeout defaults to 30 seconds | `train` returns right after escrow and does not wait for training. HTTP waits 25 seconds and returns `still_running` on timeout. Call again with the same arguments |
| Uses its own interpreter, and installing dependencies automatically is best effort only | Standard library only |
| The planner often adds `--query` and `--format` | Both are accepted. `--query` is never sent out. Only `--message` spends money. The default action is the read-only `doctor` |
| Agents retry | The idempotency key is by default computed from the dataset, model, method, floor and the day's date, so a retry on the same day does not pay twice |
| `python`, `library`, `package` or `analysis` in the name plus description gets a skill classed as package type, which takes a different execution path | The description contains `fetch`, so it is classed as database type and takes the script path. Do not break this when editing the description |

## Privacy

ScienceClaw is designed to post results to a shared platform: `INFINITE_API_BASE` defaults to a public site.

- For private demos, use `scienceclaw-post --dry-run` or point it at a local Infinite.
- The skill's `train`, `status`, `fetch` and `proof` output only the stage, tier, pass or fail, amounts and hashes. They do not output scores, data or weights, so they can go into an artifact.
- `chat` answers go into the agent's context, which may be published. To block canaries from getting out, point `OPENAI_BASE_URL` at `integrations/guardrails/`.
- On an ordinary cloud GPU the seller can read all plaintext, so T3-C takes only public or synthetic data. The skill does not change that. The daemon enforces the floor.

## Verification

```bash
BEETLE_TOKEN=$(beetle token scienceclaw) python integrations/scienceclaw/check_skill.py --dataset NAME --chat MODEL
```

It runs the skill the way `SkillExecutor` does (arguments become flags, 30 seconds, stdout parsed), and calls `/v1` with the arguments `_call_openai` uses.

Run of `check_skill.py` (2026-10-31, L4 node, separate `BEETLE_HOME`, `--local-only`, `local_backend` pointing at a vLLM we started): 14/14 passed. In it, `/v1/chat/completions` returned `x-beetle-route: local`, `x-beetle-tier: T0` and a cost of 0; a wrong token got 401; an agent asking to change a limit got only a ticket (202); an agent calling the approval endpoint got 403 `owner_only`.

**Not tested:** the success paths of `train`, `status`, `fetch`, `proof` and `quote` on the paid route. These endpoints do exist (route table in `client/daemon.py`), but no chain was available this time: the integration chain was offline, and port 18899 on the L4 was another task's local validator. `quote` was verified only as far as correctly returning `no_route_meets_floor` under `--local-only`.

## Dependencies and license

Python 3.10 or later, standard library only, no third-party dependencies. ScienceClaw is MIT.
