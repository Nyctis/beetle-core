# Local API

`beetle serve` is the one process that holds the wallet, the policy and the jobs. The CLI, the SDK and the MCP server are its clients. To start it and get a token, see [usage.md](usage.md#local-api-sdk-and-mcp). Before you give a token to an agent, read [LIMITATIONS.md](../LIMITATIONS.md#agent-tokens).

[Access](#access) · [Chat](#chat) · [Endpoints](#endpoints) · [Errors](#errors) · [Python SDK](#python-sdk) · [MCP](#mcp)

## Access

The daemon binds `127.0.0.1` only. The port is `8741`, unless `beetle serve --port` names another or `8741` is taken, when it picks a free one. `~/.beetle/serve.json` holds the port in use.

Every call needs a local token in `Authorization: Bearer <token>`, including `GET /beetle/health`. Tokens start with `bk_local_` and are in `~/.beetle/tokens.json`. A request whose `Host` is not `127.0.0.1` or `localhost` is refused, so a web page cannot reach the API through DNS rebinding.

| Scope | Token | Reach |
|---|---|---|
| `owner` | Made by the daemon, for the person at this machine | Every endpoint |
| `agent` | `beetle token NAME` | Every endpoint except `/beetle/approvals`. It spends only within the policy |

No endpoint raises a limit, lowers a floor or changes a label, for either scope. A person does that at the machine, with `beetle policy set` and `beetle data add`.

## Chat

`POST /v1/chat/completions` is OpenAI compatible, with or without streaming.

```python
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8741/v1", api_key="<token>")
r = c.chat.completions.create(model="me", messages=[{"role": "user", "content": "..."}],
                              extra_body={"beetle": {"route": "auto", "min_tier": "public", "max_usd": 0.05}})
```

`model` is an alias such as `me`, a profile name, or `auto`. The gateway decides whether that name runs on this machine (mlx-lm, or a server set with `beetle init --local-backend`) or on a seller.

The `beetle` object is optional, and its fields can only tighten the policy.

| Field | Values | Effect |
|---|---|---|
| `route` | `auto` (default), `local`, `remote` | `auto` tries this machine first, then a seller. A failure before the first token tries the next route. After the first token the caller sees the error |
| `min_tier` | `public`, `T3-B`, `T1`, `T2`, `T2+`, `T0` | The strictest of policy, data label and request applies. Beetle never falls back to a weaker tier |
| `max_usd` | A number | Caps the deposit of a new session |
| `seller` | An offer address | Uses only that offer |

From the rest of the body, only `max_tokens`, `temperature` (default 0.7), `top_p` and `seed` are used. Parameters that multiply the work, such as `n`, are not forwarded.

### Response headers

| Header | Says |
|---|---|
| `x-beetle-route` | `local` or `remote` |
| `x-beetle-tier` | The tier of this response. `T0` on this machine |
| `x-beetle-model` | `sha256:` and the base model's manifest hash, then `+sha256:` and the adapter's hash if an adapter ran |
| `x-beetle-proof` | A `prf_` id for `GET /beetle/proof/{id}` |
| `x-beetle-cost-usd` | The cost. Non-streaming responses only |

A stream's headers go out before the cost is known, so a stream reports it in the last chunk, as `beetle.cost_usd`.

## Endpoints

`{id}` in a job path is a job's full id, its short id, or any prefix that matches one job. A leading `job_` is ignored.

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat, as above |
| `GET /v1/models` | Aliases in the model library, and every profile name |
| `GET /beetle/health` | `ok`, `pid`, API version and chain |
| `GET /beetle/doctor` | This machine's capabilities, each model's path, the network and the policy. With a chain, also sellers online per model, the floor and a wallet summary |
| `GET /beetle/market?kind=&model=&min_tier=` | Market listing, including `gpu`. `kind` is `train` or `infer`. A training row's `tier` is the seller's own claim. An inference row's `tier` is always T3-C, and the claim is in `tier_claimed`. The GPU model is the seller's own report, unchecked on every tier |
| `POST /beetle/quote` | A price. Spends nothing. Body: `dataset`, `base`, and optionally `method`, `min_tier`, `seller` |
| `GET /beetle/jobs` | Job list |
| `POST /beetle/jobs` | Place a job. Returns 201. Body: `dataset`, `min_tier`, `max_usd`, `idempotency_key`, and optionally `base` (default `auto`), `method`, `seller`, `alias`, `approved`. `approved` counts only with an owner token. A key already used returns that job |
| `GET /beetle/jobs/{id}` | Timeline and progress, plus `chain_state` |
| `POST /beetle/jobs/{id}/fetch` | Verify the result, open the adapter, test that it changes answers, then settle. Optional body: `alias` (default `me`) |
| `POST /beetle/jobs/{id}/reject` | Refuse a delivery and settle at the floor. Off T2 only, and only while the job is `Delivered` on chain |
| `POST /beetle/jobs/{id}/cancel` | Stop a job. Before the seller's `start`, the refund follows once its start window ends. After `start` and before the key is released, the refund is less a capped fee (1%, at most 0.01). After release it is too late: the job ends by the chain's rules, and `ended` says so |
| `GET /beetle/proof/{id}` | Proof card, by job id or `prf_` id. `on_the_way` and `on_their_machine` are kept apart. For a job, `receipt` lists the GPU, model version and hash, data hash, duration, cost, evaluation, settlement and transaction IDs. For a remote chat it lists the GPU, model, channel and settlement. A local chat has none |
| `GET /beetle/models` | Model library, with each alias's versions |
| `POST /beetle/models/{alias}/rollback` | Go back one version |
| `GET /beetle/datasets` | Registered datasets: name, label, floor, record count and hash. Never a path or contents |
| `GET /beetle/wallet` | Address, USDC and SOL balances, limits, and spending: reserved, in escrow, settled, used today |
| `GET /beetle/connection` | NAT type, plus the path and RTT of each link |
| `GET /beetle/chain/jobs` | Jobs whose accounts are open on chain: buyer, seller, amount and state. Anyone can read this on chain, and it holds no content |
| `POST /beetle/sessions/close` | End every inference session now, rather than when it has been idle a while. Returns how many |
| `GET /beetle/limits` | Read the limits |
| `POST /beetle/limits` | Ask for a change. Returns 202 with a ticket in `needs_approval`. It changes nothing, and approving the ticket changes nothing: a person runs `beetle policy set` |
| `GET /beetle/approvals` | What waits for a person. Owner token only |
| `POST /beetle/approvals/{id}` | Approve one. Owner token only |

## Errors

Every error has this shape, plus any extra fields listed below:

```json
{"error": {"code": "no_seller", "message": "...", "retry_after": 60}}
```

| Status | Code | When | Extra fields |
|---|---|---|---|
| 400 | `bad_request` | Bad or missing input, no policy set yet, nothing to fetch or reject, no earlier version, a dataset that changed since it was registered | |
| 401 | `unauthorized` | No token, or an unknown one | |
| 402 | `budget_exceeded` | Over this request's limit, over what is left today, or a session's deposit is used up | |
| 402 | `no_funds` | The wallet is short of USDC or of SOL for fees. Checked before a job or a channel opens | |
| 403 | `approval_required` | A person must approve: over the automatic amount, or the same data and model again within 24 hours | `approval`, `how` |
| 403 | `forbidden` | The `Host` header is not local | |
| 403 | `owner_only` | An agent token called `/beetle/approvals` | |
| 404 | `not_found` | No such job, model, dataset, offer, approval or proof | |
| 409 | `no_route_meets_floor` | Nothing meets the floor. It never downgrades | `floor`, `other_tiers` or `reason` where they apply |
| 409 | `local_only` | This machine has no chain configured | |
| 409 | `offer_not_training` | The named offer sells inference | |
| 409 | `offer_wrong_model` | The named offer does not train this model | |
| 429 | `rate_limited` | Over `max_concurrent` or `max_per_hour` in the policy (3 and 6 by default) | |
| 500 | `internal` | An unexpected failure | |
| 502 | `adapter_no_effect` | The adapter changed no answer. On `fetch` the delivery is refused | |
| 502 | `result_rejected` | The delivered result failed its checks. The delivery is refused | |
| 502 | `route_failed` | A route failed unexpectedly and no other was left | |
| 502 | `usage_mismatch` | A seller's receipt is far above this machine's count. The session stops | |
| 503 | `no_seller` | No seller could be used | `retry_after`, in seconds |

### Errors on a running job

A job runs in the background once `POST /beetle/jobs` has returned, so some failures never arrive as an HTTP error. They appear in the view from `GET /beetle/jobs/{id}`.

| Field | Holds |
|---|---|
| `error` | `{"code", "message"}`. Evidence that fails, or falls below the floor, gives `evidence_rejected`: the key is not released. Its status in `client/errors.py`, 502, is never sent. An automatic fetch reports its errors here too |
| `next_error` | Why the job did not move to another seller: `seller_pinned` if the job named its seller, otherwise a code from the table above |
| `ended` | In plain words, how the job ended and where the money went |

## Python SDK

[usage.md](usage.md#local-api-sdk-and-mcp) has the basic calls. `max_usd` and `min_tier` are required arguments. The SDK holds no keys: every request goes to the local daemon, and one starts if none is running. With no `token` argument it uses this machine's owner token, so pass an agent's token to an agent.

`hybrid` picks the path and brings the adapter here. A seller trains, and the model name stays the same whether the gateway then serves it here or from a seller.

```python
me = b.hybrid("ds_demo", base="auto")                   # picks the path, brings the adapter here
print(me.chat("Where do I go on Thursday?").text)
```

`base="auto"` means the largest model this machine can serve, or the smallest there is.

| Call | Result |
|---|---|
| `Client(max_usd, min_tier, route="auto", token=None)` | A client |
| `b.train(dataset, base="auto", method="memory", seller=None, alias=None, idempotency_key=None)` | A `Job`. A key is generated if you pass none |
| `b.hybrid(dataset, base="auto", alias="me")` | A finished `Job` |
| `b.chat(prompt, model="auto", route=None, **params)` | A `Reply`: `text`, `route`, `tier`, `cost_usd`, `proof` |
| `job.wait(alias="me")` | Waits for delivery, then fetches. Raises `RuntimeError` if the job failed |
| `job.status()` | The job's current view |
| `job.report`, `job.proof` | The result and the proof card |
| `job.chat(prompt)` | Chat with the job's model |

An API error raises `beetle.ApiError`, with `status`, `code` and `error`.

## MCP

The command is in [usage.md](usage.md#local-api-sdk-and-mcp). The server speaks newline-delimited JSON-RPC on stdio. It holds no keys and no wallet: it calls the local daemon with an agent's token, so it can do what the policy allows and nothing more.

| Tool | Required arguments | What it does |
|---|---|---|
| `doctor` | | This machine's capability and the path it suggests |
| `list_datasets` | | Names, labels, floors and sizes. Never contents |
| `quote_job` | `dataset`, `base` | Price, time and tier. Spends nothing |
| `train_adapter` | `dataset`, `base`, `min_tier`, `max_usd`, `idempotency_key` | Start a job |
| `job_status` | `job_id`, optional `wait_s` up to 60 | Stage, spending and evidence summary. With `wait_s` it answers when the stage changes |
| `fetch_adapter` | `job_id` | Verify a delivered job, store its adapter here, test that it changes answers, settle. Returns scores, never weights |
| `chat` | `messages`, `model` | Chat through the local gateway |
| `spending_limit` | `action`: `get` or `request_change` | Read the limits, or ask a person to change them |
| `proof` | `id` | What a job or a response proved |
| `cancel_job` | `job_id` | Stop a job, as `POST /beetle/jobs/{id}/cancel` does |

`dataset` is a registered name, never a path. `spending_limit` can only read or ask, so an agent cannot raise a limit. An API error comes back as a tool result with `isError` set, holding `status`, `code` and `message`.
