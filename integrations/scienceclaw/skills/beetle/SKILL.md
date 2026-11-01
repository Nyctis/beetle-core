---
name: beetle
description: Order adapter training and inference from Beetle sellers through the local daemon; quote, train, status, fetch, proof, chat
metadata:
---

# Beetle

Trains a LoRA adapter on a dataset a person has already registered, on a seller's
machine, and runs models through the local gateway. The money and the policy stay
in `beetle serve`; this skill holds an agent token and can only spend within that
policy. It cannot raise a limit, lower a floor, or name a file path.

## Before you start

On an ordinary GPU seller (T3-C) the machine's owner can read everything. Only
`synthetic` and `public` datasets are allowed there, and the daemon enforces it.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--action` | `doctor`, `datasets`, `market`, `quote`, `train`, `status`, `fetch`, `proof`, `limits` or `chat` | `doctor` |
| `--dataset` | Registered dataset name, never a path | required for `quote`, `train` |
| `--base` | Base model, such as `qwen3.5-2b`, or `auto` | `auto` |
| `--min-tier` | Privacy floor: `T0`, `T1`, `T2`, `T3-B`, `public` | required for `train` |
| `--max-usd` | Most this job may cost | required for `train` |
| `--idempotency-key` | Same key, same job; a retry never pays twice | derived from the job and the day |
| `--job` | Job id from `train` | required for `status`, `fetch`, `proof` |
| `--alias` | Name in the model library after `fetch` | `me` |
| `--model` | Model name for `chat` | `auto` |
| `--message` | The prompt for `chat` | required for `chat` |
| `--timeout` | Seconds to wait for the daemon | `25` |
| `--format` | `json` or `summary` | `json` |

## Examples

```bash
python3 {baseDir}/scripts/beetle_skill.py --action doctor
python3 {baseDir}/scripts/beetle_skill.py --action quote --dataset lab_notes --base qwen3.5-2b --min-tier public --max-usd 2
python3 {baseDir}/scripts/beetle_skill.py --action train --dataset lab_notes --base qwen3.5-2b --min-tier public --max-usd 2
python3 {baseDir}/scripts/beetle_skill.py --action status --job JOB
python3 {baseDir}/scripts/beetle_skill.py --action fetch --job JOB --alias lab
python3 {baseDir}/scripts/beetle_skill.py --action proof --job JOB
python3 {baseDir}/scripts/beetle_skill.py --action chat --model lab --message "What does QC fail on?"
```

`train` returns as soon as the money is in escrow. Poll `status` until the stage is
`delivered`, then `fetch`. A `403 approval_required` means a person must run the
`how` command; the skill cannot approve anything. `409 no_route_meets_floor` is
final: the daemon never lowers a floor.

## Output

JSON on stdout with `summary` and `results`. `train`, `status`, `fetch` and `proof`
return only stage, tier, pass or fail, hashes and cost, so they are safe to attach
to an artifact. They never contain data, scores or adapter weights. Errors are
`{"error": {"code", "message"}}` with exit code 0 so an agent can read them.
