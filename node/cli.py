"""`beetle host`: run a seller from the command line. A Linux server has nothing else."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import node  # noqa: F401  (import paths)

import model_profiles as mp
from beetle_chain import (KIND_INFERENCE, KIND_TRAINING, OFFER_ACTIVE, OFFER_PAUSED, REACH_ECHO,
                          REACH_PUBLIC_UDP, OfferTerms, net_keypair, public_endpoint)
from job_defaults import CHAIN_PARAMS, IMAGE, TIERS
from net.peer import Peer
from node import daemon, wire
from sandbox_host import ensure_image, ensure_model, model_dir
from serve.gateway import default_scratch

USDC = 1_000_000
# What each kind of evidence lets a seller honestly say. The tier never comes
# from a flag: an ordinary GPU server has nothing to show, so it is T3-C, and the
# client tells the buyer the owner can read it all.
DECLARED = {"none": ("T3-C", TIERS["t3c"]), "tpm": ("T3-B", TIERS["t3b"])}
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_CONTAINER = "beetle-vllm"


def gpu_info() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode or not out.stdout.strip():
        return None
    name, mem = out.stdout.splitlines()[0].rsplit(",", 1)
    return {"name": name.strip(), "vram_gb": int(mem) // 1024}


def context():
    where = wire.home()
    cfg = wire.load_config(where)
    if "host" not in cfg:
        raise SystemExit("this machine is not set up as a seller: run `beetle host init`")
    return where, cfg, wire.load_chain(cfg, where), wire.load_wallet(where)


def rendezvous_section(a, where: Path, cfg: dict) -> None:
    cfg["rendezvous"] = {"kind": "file", "dir": a.rendezvous_dir} if a.rendezvous_dir else {"kind": "chain"}


def endpoint_id(where: Path) -> bytes:
    """This node's iroh id, which stays the same across restarts."""
    async def read() -> bytes:
        peer = await Peer.bind(secret=wire.secret(where, "node.key"))
        try:
            return bytes.fromhex(peer.id)
        finally:
            await peer.close()

    return asyncio.run(read())


def cmd_init(a) -> int:
    where = wire.home()
    where.mkdir(parents=True, exist_ok=True)
    kp = wire.new_wallet(where)
    if a.chain:
        shutil.copy(a.chain, where / "chain.json")
    gpu = gpu_info()
    cfg = wire.read_json(where / "config.json", {})
    cfg.update({"chain": "chain.json", "stun": a.stun or cfg.get("stun", [])})
    rendezvous_section(a, where, cfg)
    cfg["host"] = {**cfg.get("host", {}), "port": a.port, "public_addr": a.public_addr,
                   "gpu": gpu is not None, "evidence": a.evidence, "attester": a.attester,
                   "backend": a.backend, "models": cfg.get("host", {}).get("models", [])}
    wire.write_json(where / "config.json", cfg)
    print(f"wallet   {kp.pubkey()}  (fund it with SOL for fees and USDC for the stake)")
    print(f"gpu      {gpu or 'none: CPU only'}")
    print(f"config   {where / 'config.json'}")
    print(f"tier     {DECLARED[a.evidence][0]}, decided by what this machine can prove")
    return 0


def checks(where: Path, cfg: dict) -> list[tuple[str, bool, str]]:
    host, out = cfg["host"], []
    tier = DECLARED[host.get("evidence", "none")][0]
    out.append(("tier", True, f"{tier}: no confidential hardware in use, the machine's owner can "
                              "read what runs here" if tier == "T3-C" else tier))
    try:
        chain, kp = wire.load_chain(cfg, where), wire.load_wallet(where)
        seller = chain.get_seller(kp.pubkey())
        sol = chain.sol_lamports(kp.pubkey()) / 1e9
        out.append(("wallet", sol > 0.01, f"{kp.pubkey()} {sol:.3f} SOL"))
        out.append(("chain", True, f"{chain.cfg['network']}, chain time {chain.chain_time()}"))
        try:
            usdc = chain.token_balance(kp.pubkey())
        except RuntimeError:  # no token account yet
            usdc = 0.0
        out.append(("usdc", True, f"{usdc:.2f}, stake {seller.stake_total / USDC:.2f}"
                    if seller else f"{usdc:.2f}, not registered yet"))
    except (OSError, RuntimeError) as e:
        out.append(("chain", False, str(e)[:120]))
    docker = subprocess.run(["docker", "info", "-f", "{{.ServerVersion}}"], capture_output=True, text=True)
    out.append(("docker", docker.returncode == 0, docker.stdout.strip() or "not reachable"))
    image = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    out.append(("sandbox image", image, IMAGE if image else "`beetle host list` builds it"))
    gpu = gpu_info()
    out.append(("gpu", gpu is not None or not host.get("gpu"), str(gpu or "none")))
    if gpu and image:
        probe = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--network", "none",
                                "--entrypoint", "nvidia-smi", IMAGE, "-L"], capture_output=True, text=True)
        out.append(("gpu in sandbox", probe.returncode == 0, probe.stdout.strip()[:80] or probe.stderr.strip()[:120]))
    free = shutil.disk_usage(where).free / 2**30
    out.append(("disk", free > 20, f"{free:.0f} GiB free"))
    nat = wire.nat_type(cfg)
    out.append(("network", nat["nat"] != "blocked", wire.nat_advice(nat)))
    proved = wire.inbound_open(cfg, host.get("port", 0))
    out.append(("inbound UDP", proved is not False, {True: "open, a STUN server tried it",
                                                     False: "closed, a STUN server tried it",
                                                     None: "not known: most STUN servers can't test it, "
                                                           "and a running daemon holds the port"}[proved]))
    for name in host.get("models", []):
        p, vram = mp.get(name), (gpu or {}).get("vram_gb", 0)
        need = max(p.min_vram_gb.values())
        out.append((f"model {name}", vram >= need,
                    f"needs {need} GiB of GPU memory, has {vram}; {'on disk' if model_dir(p.repo, where / 'models').exists() else 'not downloaded'}"))
    if host.get("backend"):
        import httpx

        try:
            models = httpx.get(host["backend"].rstrip("/") + "/v1/models", timeout=5, trust_env=False).json()
            out.append(("inference backend", True, str([m["id"] for m in models["data"]])))
        except (httpx.HTTPError, ValueError) as e:
            out.append(("inference backend", False, f"{host['backend']}: {type(e).__name__}"))
    nat = wire.nat_type(cfg)
    public = host.get("public_addr")
    out.append(("network", nat["nat"] != "blocked", f"{nat['nat']}, advertising {public or 'whatever STUN finds'}"))
    return out


def cmd_doctor(a) -> int:
    where = wire.home()
    results = checks(where, wire.load_config(where))
    for name, ok, detail in results:
        print(f"{'ok ' if ok else 'NO '} {name:18} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


def start_vllm(where: Path, host: dict, profile: mp.Profile) -> None:
    """One model per server; the backend URL then points at it."""
    scratch = Path(host.get("scratch") or default_scratch())
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", "-f", VLLM_CONTAINER], capture_output=True)
    fraction = host["vllm"]["gpu_fraction"]
    subprocess.run(["docker", "run", "-d", "--name", VLLM_CONTAINER, "--gpus", "all", "--restart", "unless-stopped",
                    "-p", f"127.0.0.1:{host['vllm']['port']}:8000",
                    "-e", "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True",
                    "-v", f"{model_dir(profile.repo, where / 'models')}:/model:ro",
                    # The parent, which outlives a reboot: Docker would recreate a
                    # missing tmpfs directory itself, as root, and then no one could write it.
                    "-v", f"{scratch.parent}:{scratch.parent}:ro", host["vllm"].get("image", VLLM_IMAGE),
                    "/model", "--served-model-name", profile.name,
                    "--gpu-memory-utilization", str(fraction), *profile.vllm["args"]], check=True)
    host["backend"] = f"http://127.0.0.1:{host['vllm']['port']}"


def check_memory(profiles: list, vram: float, infer_only: bool, served: list[str]) -> None:
    """A model the backend already serves may run on another machine altogether."""
    for p in profiles:
        need = p.min_vram_gb["infer" if infer_only else "train"]
        if vram < need and not (infer_only and p.name in served):
            raise SystemExit(f"{p.name} needs {need} GiB of GPU memory; this machine has {vram}")


def public_udp(a, cfg: dict, host: dict) -> bool:
    """Listed as reachable only if that isn't known to be false, and on its own only
    if it was proved. Buyers behind NAT count on such sellers to tell them their address."""
    if a.public_udp is False:
        return False
    if not host.get("public_addr"):
        if a.public_udp:
            raise SystemExit("--public-udp needs --public-addr: `beetle host init --public-addr IP:PORT`")
        return False
    proved = wire.inbound_open(cfg, host.get("port", 0))
    if proved is False and a.public_udp:
        raise SystemExit("inbound UDP on this port is closed, a STUN server tried it: open the port, "
                         "or leave out --public-udp")
    if proved is None:
        print("inbound UDP: not proved. Only a STUN server with a second address can test it, and a "
              "running daemon holds the port." + (" Listing it on your word." if a.public_udp else
                                                  " If the port is open, say so with --public-udp."))
    return bool(proved) or (proved is None and bool(a.public_udp))


def cmd_list(a) -> int:
    where, cfg, chain, kp = context()
    host = cfg["host"]
    me = kp.pubkey()
    profiles = [mp.get(m) for m in a.models.split(",")]
    gpu = gpu_info()
    vram = (gpu or {}).get("vram_gb", 0)
    check_memory(profiles, vram, a.infer_only, a.serves)
    reachable = public_udp(a, cfg, host)  # before anything is spent: it can refuse
    tier = DECLARED[host.get("evidence", "none")][1]
    root = daemon.measurement(host)
    host["infer_only"] = a.infer_only
    if not a.infer_only:  # a seller that only forwards to its server needs neither
        ensure_image(host.get("gpu", False))
    for p in profiles:
        if not a.infer_only or p.name == a.vllm_serve:
            print(f"model {p.name}: checking the download")
            ensure_model(p.revision, p.repo, where / "models")
    host["models"] = [p.name for p in profiles]
    host["serve_as"] = dict(m.split("=") for m in a.serve_as)
    if a.vllm_serve:
        host["vllm"] = {"port": a.vllm_port, "gpu_fraction": a.vllm_gpu_fraction}
        start_vllm(where, host, mp.get(a.vllm_serve))
    wire.write_json(where / "config.json", cfg)

    n_offers = (0 if a.infer_only else len(profiles)) + (1 if host.get("backend") and a.infer_in is not None else 0)
    params = CHAIN_PARAMS
    stake = a.stake * USDC if a.stake else params["min_stake"][tier] + (n_offers + 1) * params["offer_stake"]
    chain.create_ata(kp, me)
    net_pk = net_keypair(wire.secret(where, "net.key")).public_bytes
    seller = chain.get_seller(me)
    if seller is None:
        chain.register_seller(kp, net_pk=net_pk)
        seller = chain.get_seller(me)
    elif seller.net_pk != net_pk:
        chain.rotate_net_pk(kp, me, net_pk)
    if seller.free_stake < stake:
        chain.stake(kp, stake - seller.free_stake)
    now = chain.chain_time()
    common = dict(tier=tier, measurement=root, valid_until=now + daemon.OFFER_LIFETIME,
                  slots_total=a.slots, hw_class=wire.hardware_class(gpu), mem_gb=vram)
    if reachable:  # others can then learn their own address from this node
        print("public UDP: this machine's address goes on chain for good: fine for a server, not a home")
        common.update(reach=REACH_PUBLIC_UDP | REACH_ECHO,
                      endpoint=public_endpoint(endpoint_id(where), host["public_addr"]))
    wanted = []
    for i, p in enumerate(profiles):
        if not a.infer_only:
            wanted.append((2 * i, f"train {p.name}", OfferTerms(
                KIND_TRAINING, price_a=int(a.train_usd_per_mtok * USDC), tokens_per_s=a.tokens_per_s,
                grace_s=a.grace_s, start_window_s=a.start_window_s, model_root=p.root, **common)))
        if host.get("backend") and a.infer_in is not None and p.name in (a.vllm_serve, *a.serves):
            wanted.append((2 * i + 1, f"infer {p.name}", OfferTerms(
                KIND_INFERENCE, price_a=int(a.infer_in * USDC), price_b=int(a.infer_out * USDC),
                model_root=p.root, **common)))
    for idx, label, terms in wanted:
        existing = chain.get_offer(chain.offer_pda(me, idx))
        if existing:
            chain.update_offer(kp, existing, terms)
        else:
            chain.post_offer(kp, me, idx, terms)
        print(f"offer {idx}: {label} at tier {DECLARED[host.get('evidence', 'mock')][0]}, {chain.offer_pda(me, idx)}")
    return 0


def cmd_run(a) -> int:
    where, cfg, chain, kp = context()
    try:
        asyncio.run(daemon.serve(where, cfg, chain, kp))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_status(a) -> int:
    where, cfg, chain, kp = context()
    me, seller = kp.pubkey(), None
    seller = chain.get_seller(me)
    out = {"wallet": str(me), "sol": chain.sol_lamports(me) / 1e9,
           "tier": DECLARED[cfg["host"].get("evidence", "none")][0],
           "stake": seller and {"total": seller.stake_total / USDC, "locked": seller.stake_locked / USDC,
                                "settled_jobs": seller.n_settled, "volume": seller.volume / USDC,
                                "failed": seller.n_nostart + seller.n_timeout + seller.n_rejected,
                                "buyer_no_release": seller.n_norelease},
           "offers": [{"address": str(o.address), "kind": ["train", "infer", "hosting"][o.kind],
                       "status": "active" if o.status == OFFER_ACTIVE else "paused",
                       "model": (mp.by_root(o.model_root) or o).name if mp.by_root(o.model_root) else o.model_root.hex()[:12],
                       "price_a": o.price_a / USDC, "price_b": o.price_b / USDC,
                       "slots": f"{o.slots_used}/{o.slots_total}"}
                      for o in daemon.offers_of(chain, me)],
           "jobs": [{"job": j.job_id.hex()[:16], "state": j.state, "price": j.price / USDC}
                    for j in chain.jobs_for_host(me)]}
    print(json.dumps(out, indent=2))
    return 0


def cmd_pause(a) -> int:
    where, cfg, chain, kp = context()
    for offer in daemon.offers_of(chain, kp.pubkey()):
        if offer.status == OFFER_ACTIVE:
            chain.pause_offer(kp, offer)
            print(f"paused {offer.address}")
    return 0


def cmd_resume(a) -> int:
    where, cfg, chain, kp = context()
    until = chain.chain_time() + daemon.OFFER_LIFETIME
    for offer in daemon.offers_of(chain, kp.pubkey()):
        if offer.status == OFFER_PAUSED:
            chain.update_offer(kp, offer, daemon.terms_of(offer, valid_until=until))
            print(f"resumed {offer.address}")
    return 0


def add_parser(sub) -> None:
    host = sub.add_parser("host", help="run a seller").add_subparsers(dest="host_command", required=True)
    p = host.add_parser("init", help="wallet and config")
    p.add_argument("--chain", type=Path, help="chain.json of the network to join")
    p.add_argument("--rendezvous-dir", help="a directory shared with buyers, in place of the chain")
    p.add_argument("--port", type=int, default=41710, help="UDP port for jobs; sessions use the next one")
    p.add_argument("--public-addr", help="ip:port to advertise, if this machine has one")
    p.add_argument("--backend", help="an OpenAI-compatible server to sell inference from")
    p.add_argument("--evidence", choices=sorted(DECLARED), default="none")
    p.add_argument("--attester", default="http://127.0.0.1:8700", help="verifier for TPM enrollment")
    p.add_argument("--stun", action="append", help="host:port of a STUN server")
    p.set_defaults(fn=cmd_init)
    host.add_parser("doctor", help="what this machine can sell").set_defaults(fn=cmd_doctor)
    p = host.add_parser("list", help="register, stake and post offers")
    p.add_argument("--models", required=True, help="comma-separated profile names")
    p.add_argument("--infer-only", action="store_true", help="sell inference and no training")
    p.add_argument("--train-usd-per-mtok", type=float, default=0.0)
    p.add_argument("--infer-in", type=float, help="USD per million input tokens")
    p.add_argument("--infer-out", type=float, default=0.0, help="USD per million output tokens")
    p.add_argument("--serves", action="append", default=[], help="models the backend already serves")
    p.add_argument("--serve-as", action="append", default=[], metavar="NAME=SERVED",
                   help="the backend's name for a model, if it isn't the profile's")
    p.add_argument("--vllm-serve", help="start vLLM in Docker with this model")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-gpu-fraction", type=float, default=0.4)
    p.add_argument("--stake", type=int, help="USDC; default: the tier's minimum plus the offers' stake")
    p.add_argument("--slots", type=int, default=1, help="jobs at once")
    p.add_argument("--public-udp", action=argparse.BooleanOptionalAction, default=None,
                   help="inbound UDP on the peer port is open, so this seller can be an echo node; "
                        "by default that is turned on only if a STUN server proves it")
    p.add_argument("--tokens-per-s", type=int, default=500, help="training speed you promise")
    p.add_argument("--grace-s", type=int, default=900, help="setup time on top of the run budget")
    p.add_argument("--start-window-s", type=int, default=900)
    p.set_defaults(fn=cmd_list)
    host.add_parser("run", help="the daemon; systemd runs this").set_defaults(fn=cmd_run)
    host.add_parser("status", help="offers, stake, jobs").set_defaults(fn=cmd_status)
    host.add_parser("pause", help="stop taking jobs").set_defaults(fn=cmd_pause)
    host.add_parser("resume", help="take jobs again").set_defaults(fn=cmd_resume)
