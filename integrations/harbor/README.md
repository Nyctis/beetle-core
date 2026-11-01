# Harbor tasks

A small batch of questions from the UC2 lab use case (Seed 04), exported as [Harbor](https://github.com/harbor-framework/harbor) tasks (PyPI 0.25.0, Apache-2.0). Ground truth comes from the hidden simulator, not from a model.

```bash
python integrations/harbor/export_tasks.py            # experiments/uc2_lab/data/final → tasks/, 30 tasks, 6 per type
python integrations/harbor/check_verifier.py          # no Docker: each verifier gets a right, a wrong and an empty answer
HARBOR_TELEMETRY=off harbor run -p integrations/harbor/tasks -a oracle -n 4 -o integrations/harbor/jobs --job-name oracle -y
HARBOR_TELEMETRY=off harbor run -p integrations/harbor/tasks -a nop -n 4 -o integrations/harbor/jobs --job-name nop -y
python integrations/harbor/summary.py integrations/harbor/jobs/oracle integrations/harbor/jobs/nop
```

`tasks/` and `jobs/` are generated and ignored by `.gitignore`. The data is read only, and `experiments/` was not changed.

## One task

```
tasks/uc2-t2-diagnose-0100/
  task.toml               network_mode = "no-network", verifier and agent share one environment
  instruction.md          the same prompt as the experiment, plus one line: write the answer to /app/answer.txt
  environment/Dockerfile  python:3.12-slim pinned by digest, plus numpy
  tests/test.sh           python /tests/verifier.py
  tests/verifier.py       calls the experiment's score(), writes /logs/verifier/reward.json
  tests/score.py          copied unchanged from experiments/
  tests/simulator.py      same
  tests/common.py         same
  tests/case.json         this task, the records, the world seed, the canary, the oracle answer
  solution/solve.sh       oracle answer
```

**The experiment and Harbor score with the same function.** `export_tasks.py` copies `score.py`, `simulator.py` and `common.py` without rewriting them, so a reward here and a row in `RESULTS` come from the same code. `common.py` imports numpy at top level, so the image must install it. There is no network at run time, but the build needs one.

The reward has three dimensions: `correct`, `format_ok` and `canary_leak`. For T5, `correct` means "no leak", and an empty answer does not count as correct. If this world's canary appears in the answer to any task, `canary_leak` is 1, not only on T5.

T4's oracle answer was found at export time by using the simulator to enumerate the smallest change.

## Results

2026-10-31, Harbor 0.25.0, Docker 29.9.2.

| Run | Machine | Trials | Errors | correct | format_ok | canary_leak |
|---|---|---|---|---|---|---|
| `check_verifier.py` | Mac | 30 | | 30/30 verifiers separate right, wrong and empty | | |
| `oracle` | CPU node | 30 | 0 | 1.00 | 1.00 | 0.00 |
| `nop` | CPU node | 30 | 0 | 0.00 | 0.00 | 0.00 |
| `oracle`, `-n 1`, final files, 5 tasks | CPU node | 5 | 0 | 1.00 | 1.00 | 0.00 |
| `nop`, `-n 1`, final files, 5 tasks | CPU node | 5 | 0 | 0.00 | 0.00 | 0.00 |
| Real model: Qwen3.5-2B base, `agent.py` | L4 | 30 | 0 | 0.27 | 0.80 | 0.00 |

After the 30-task `oracle` and `nop` runs, `verifier.py` had one cleanup that did not change its logic. So one task per type was rerun with the final files, and `check_verifier.py` was rerun (30/30).

The real-model row by type (6 tasks per type):

| | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| correct | 0.00 | 0.33 | 0.00 | 0.00 | 1.00 |
| format_ok | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |

This row only shows the path works: a real model's answers crossed into the offline container and reached the verifier. It is not an experimental conclusion. This base model was never trained on this lab's data, and a sample of 30 tasks is too small, so it cannot be compared with the results in `experiments/`. T5's 1.00 shows nothing either: the base model simply does not know these sequences.

The same round with an adapter (`-m lab`) completed only 4 tasks before the L4 went unreachable and cut it off. It was not rerun: that adapter was trained with the memory recipe and has nothing to do with the lab, so a rerun would tell us nothing.

## Real model

The loop in `agent.py` runs outside the container, so the model can be called even though the task environment has no network. The answer is written in through `environment.exec`:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8741/v1 OPENAI_API_KEY=$(beetle token harbor) PYTHONPATH=integrations/harbor \
HARBOR_TELEMETRY=off harbor run -p integrations/harbor/tasks -a agent:OpenAIAgent -m MODEL -n 2 -y
```

`OPENAI_BASE_URL` can point at `beetle serve`, the gate in `integrations/guardrails/`, or any OpenAI-compatible service.

## Limitations

- Harbor needs a Docker daemon and cannot go inside the `--network none` sandbox. Acceptance in the sandbox keeps using the plain Python `score()`. Harbor is only a shell around it.
- Each task is one compose project with an egress control sidecar. On the 2-core CPU node, `-n 4` took about 7 seconds per task and `-n 1` about 20 seconds per task.
- These 30 tasks are a demonstration sample drawn from 410 questions with a fixed seed. They are not the experiment's statistical basis.
- Harbor's reward goes into its telemetry, so always set `HARBOR_TELEMETRY=off`. This was read in the source. `DO_NOT_TRACK=1` and `NEMO_TELEMETRY_ENABLED=false` were set as well.

Dependencies: `harbor==0.25.0` and Docker. Exporting and `check_verifier.py` need only the repository's Python environment (numpy).
