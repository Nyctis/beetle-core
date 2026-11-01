# Private inference: P2P direct link + end-to-end encryption

The buyer and the seller connect by direct link, and content is encrypted end to end up to the seller's runtime environment. The seller only needs an OpenAI-compatible inference service (vLLM, llama.cpp server and mlx_lm.server all use the same interface). The gateway sits in front of it, so changing the backend needs no code change.

```
Buyer serve/client.py                       Seller serve/gateway.py                             Backend
┌───────────────────────┐     iroh QUIC     ┌─────────────────┬──────────────────────────┐      ┌───────────────────┐
│ Inner encryption HPKE │ ── direct link ── │ Gateway         │ Environment              │ HTTP │ OpenAI-compatible │
│ Fixed-length frames   │     no relay      │ Ciphertext only │ Decrypts, meters, shapes │ ───▶ │ service           │
└───────────────────────┘                   └─────────────────┴──────────────────────────┘      └───────────────────┘
```

## What it protects

**On an ordinary machine, the machine's owner can read all plaintext.** On an ordinary GPU the gateway can only declare `T3-C`, which means "this is an ordinary cloud machine, and the cloud provider and anyone with root can see it". The gateway refuses to declare a higher tier, and the client refuses a higher tier that has no evidence. Until the Environment is placed in a confidential VM, the encryption here stops the network and people who only see traffic, not the seller.

| Blocks | Does not block |
|---|---|
| The network on the path reading or tampering with content | A machine owner who can read memory (the definition of T3-C) |
| Replay of the opening frame or later frames, and moving frames across sessions or directions | A leaked `enc_pk` private key: there is no forward secrecy within one key |
| Truncated responses: a response is not complete until the end message arrives | The host holding back or dropping traffic |
| Other buyers using an adapter you uploaded | The backend's own logs: remember to turn off request logging |
| Traffic shape: in `constant` mode, token timing and response length (to the nearest duration bucket) | Activity start time, duration bucket, request size tier; GPU and power side channels |

The inner protocol is our own composition in the shape of RFC 9458, with no external audit. `constant` mode has not been measured against the Whisper Leak classifier.

## From zero: one machine

Needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/). iroh ships wheels for macOS arm64 and for Linux x86_64 and aarch64.

```bash
git clone <this repository> && cd <repository>
uv sync --extra net --extra test
uv run pytest tests/test_net_channel.py tests/test_transfer.py tests/test_gateway.py
```

Start a backend. Any one of these:

```bash
# vLLM (with a GPU)
vllm serve MODEL --port 8000
# llama.cpp
llama-server -m MODEL.gguf --port 8000
# mlx_lm (Mac)
mlx_lm.server --model MODEL --port 8000
```

Then run the full path once. The gateway starts on this machine and the buyer connects to it through local iroh:

```bash
uv run python scripts/e2e_chat.py --backend http://127.0.0.1:8000 \
    chat --model MODEL
```

`MODEL` is a name from the backend's `/v1/models`.

## Two machines

Seller:

```bash
export BEETLE_BACKEND=http://127.0.0.1:8000 BEETLE_RENDEZVOUS=/some/shared/dir
uv run python -m serve.gateway --addr PUBLIC_IP:41710
```

On start it prints one line of JSON with `enc_pk`, `sign_pk`, `endpoint_id`, `tier` and the model list. `enc_pk` changes with each process and also reaches the buyer in the rendezvous reply, so `Session.open(rendezvous)` works without it. In that case it is only as trustworthy as the rendezvous. A tier with evidence must be bound by that evidence.

Buyer:

```python
from net.rendezvous import FileRendezvous
from serve.client import Session

rendezvous = FileRendezvous("shared/buyer.json", "shared/seller.json")
session = await Session.open(rendezvous, bytes.fromhex(ENC_PK), addrs=["MY_PUBLIC_IP:PORT"])
print(session.tier)                                   # T3-C
async for chunk in session.chat(messages, model="MODEL"):
    print(chunk, end="")
await session.close()
```

`Rendezvous` does one thing: it publishes its own `{id, addrs}` and waits for the other side's. This example uses files, and any way of sharing two small files will do. The on-chain implementation inherits the same interface.

`scripts/e2e_chat.py` can do this for you:

```bash
uv run python scripts/e2e_chat.py --ssh SELLER_HOST --remote-dir DIR --remote-python PY \
    --addr SELLER_IP:41710 --local-port 41711 --stun STUN_HOST:3478 \
    --backend http://127.0.0.1:8000 chat --model MODEL
```

ssh is used only to start the gateway and share the rendezvous files. The conversation itself goes over the iroh direct link. The other subcommands are `adapter`, `bench`, `plain`, `reconnect` and `scan-tap`. See `--help`.

### Hole punching and candidate addresses

There is no relay. Both sides must dial at the same time, so each must know the other's public address.

- The seller declares its public address with `--addr`.
- The buyer uses `net/echo.py` to ask public nodes "what address do you see for me?", which also gives the NAT type. **Echo nodes must have inbound UDP open.** Behind a cloud firewall the probe times out with the result `blocked`. In that case give the address directly with `--candidate`, or probe with `--stun`. STUN uses a different socket, so that address assumes the NAT preserves ports, and it must be used with `--local-port`.
- Telling the NAT type takes at least two echo nodes. With only one, it amounts to a central STUN.

## adapter

```python
name = await session.upload_adapter("path/to/peft-adapter")
async for chunk in session.chat(messages, model=name): ...
await session.close()   # unloads and deletes
```

The adapter is packed on the buyer's machine, padded to a size tier (32 / 128 / 512 MiB) and encrypted. The ciphertext goes to the host over P2P, and the key goes through the inner channel. The Environment decrypts it into tmpfs and loads it under a random name. Only this session can use it, and it is unloaded and deleted when the session ends. Only the two files `adapter_config.json` and `adapter_model.safetensors` are accepted, because other formats can execute code when loaded.

When `--scratch` is not tmpfs (on a Mac, for example), the gateway refuses to load the adapter rather than write it to disk. During development, `--allow-disk-scratch` lifts this. A buyer with an adapter in memory can use `session.upload_adapter_files(files)` and skip writing it to disk first.

This step uses the backend's dynamic loading endpoint `/v1/load_lora_adapter`, which only vLLM has today. Other backends return a clear error to the buyer, and chat is not affected. Start vLLM like this:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve MODEL --enable-lora --max-lora-rank 16
```

The documentation warns that this switch is only for isolated environments, so keep the backend listening on this machine only and do not expose it. **The backend must be able to read the gateway's `--scratch` directory.** When vLLM runs in a container, mount the same path into it, for example `-v /dev/shm/beetle:/dev/shm/beetle:ro`. Give each gateway its own scratch directory: on start a gateway clears any adapters a crash left there, so that they cannot be offered as public models.

**A success response from the backend does not mean the adapter took effect.** When key names do not match module paths, vLLM still returns 200 and then silently ignores the adapter, and answers are identical to the base model's. The common case is training with a text architecture (`...ForCausalLM`) while vLLM loads a multimodal architecture (`...ForConditionalGeneration`): `base_model.model.model.layers.` in the adapter has to become `base_model.model.language_model.model.layers.`. Before going live, test once with a question only the adapter can answer.

## Configuration

Every value has a flag. Where the table names an environment variable, that sets it too.

| Flag | Environment variable | Default | Description |
|---|---|---|---|
| `--backend` | `BEETLE_BACKEND` | `http://127.0.0.1:8000` | OpenAI-compatible address, with or without `/v1` |
| `--api-key` | `BEETLE_BACKEND_KEY` | None | For a backend that requires a token |
| `--models` | `BEETLE_MODELS` | All of the backend's `/v1/models` | Comma-separated list of models to offer |
| `--port` | `BEETLE_PORT` | `41710` | iroh's UDP port |
| `--addr` | None | Local network interface address | `ip:port` to declare publicly, repeatable |
| `--rendezvous` | `BEETLE_RENDEZVOUS` | `./rendezvous` | Directory shared with the buyer |
| `--tier` | `BEETLE_TIER` | `T3-C` | Accepts only tiers that need no evidence |
| `--scratch` | `BEETLE_SCRATCH` | `/dev/shm/beetle`, or the system temp directory if that is missing | Decrypted adapters. Must be tmpfs |
| `--allow-disk-scratch` | None | Off | Allows decrypted adapters on disk. Development only |
| `--spool` | `BEETLE_SPOOL` | `beetle-spool` in the system temp directory | Uploaded ciphertext |
| `--max-upload` | None | 1 GiB | Maximum size of a single upload |
| `--frame` | None | 4096 | Frame length. Must match on both ends |
| `--tap` | None | Off | Records every byte the host handles, for auditing |

The buyer chooses `mode` in `Session.open`:

- `constant` (default): sends one frame every `tick` (50ms), and a dummy frame when there is no content. After the end it pads to a whole multiple of `bucket` (5 seconds), and closing the session also waits until then. The first token waits at most one extra tick. Traffic is about 80 KB/s.
- `burst`: sends when there is content. Frame length is still fixed, but token timing is visible.

The stretch from gateway to backend is plaintext, so the gateway ignores any `HTTP_PROXY` in the environment. The backend should be on the same machine as the gateway or within the same trust boundary.

The only fields a buyer can pass to the backend are `max_tokens` (capped at 4096), `temperature`, `top_p`, `seed`, `stop`, `frequency_penalty` and `presence_penalty`. Other fields are not forwarded: `n` would multiply the cost, and the backend's own extension fields are not the gateway's to let through. For paid sessions see `node/metered.py`: an answer runs against vouchers, pauses when it goes past one and ends if the next does not arrive, and the `done` message carries `stopped`.

## Wire format

```
Open frame       OPEN | suite | session_id | enc | HPKE.Seal(hello)
Request key      Export("beetle/v1/request"), salt = enc
Response key     Export("beetle/v1/response"), salt = enc | random nonce
Frame i          AEAD(key, base XOR i, aad = domain | direction | session_id | i)
Frame plaintext  kind(1) ack(4) len(2) payload, zero-padded
Resume frame     RESUME | suite | session_id | enc | nonce | AEAD(receive count)
```

Every frame has the same length, in both directions, including the first. Messages are length-prefixed JSON carried on the byte stream the frames form.

**Resuming after a disconnect:** both ends keep frames that were sent but not acknowledged. On a new connection they exchange receive counts and each resends, unchanged, the frames the other is missing, so a response continues from where it broke, with nothing lost or duplicated. By default the seller keeps a session for 120 seconds and cleans up the adapter after that timeout. If recovery fails, the buyer gets `SessionLost` and never sees a partial result.

**Receipts:** after each response the Environment signs a `UsageReceipt` (`channel_id`, `seq`, cumulative tokens, `transcript_hash`). The buyer computes `transcript_hash` independently from what it sent and received, compares the two and raises `ReceiptError` on a mismatch. Token counts are reported by the backend. The buyer does not yet count them independently with a tokenizer.

## Public interfaces

| Module | Interface |
|---|---|
| `net/peer.py` | `Peer.bind`, `Peer.link`, `Link.write/read_exact/open_stream/accept_stream/close` |
| `net/channel.py` | `connect`, `accept`, `Channel`, `Pipe` |
| `net/transfer.py` | `send`, `receive`, `manifest`, `root_of`, `seal_tiered`, `tier` |
| `net/echo.py` | `answer`, `observe`, `classify`, `probe` |
| `net/rendezvous.py` | `Rendezvous`, `FileRendezvous`, `MemoryRendezvous` |
| `serve/gateway.py` | `Backend`, `Environment`, `Gateway`, `Spool` |
| `serve/client.py` | `Session.open/chat/upload_adapter/close` |
