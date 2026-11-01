"""Client for the beetle Solana program (programs/beetle).

Every chain interaction goes through this file. Instructions are encoded by
hand (Anchor discriminator + Borsh args), so the account order in each builder
must match the program's `#[derive(Accounts)]` structs exactly.

The usual calling order:

    seller:  register_seller -> stake -> post_offer -> (touch_offer ...)
    buyer:   list_offers -> current_slot / new_job_id -> open_job
    seller:  start (declares the sandbox keys and the evidence hash)
    buyer:   verifies the evidence locally -> release_key or reject_evidence
    seller:  submit_result
    buyer:   accept or reject
    anyone:  finalize once the settle delay is over -> claim_due

Only on T2 does the sandbox's signature release the full price. On every other
tier the seller holds the signing key, so the buyer checks the delivery itself
and accepts it; reject, or saying nothing, pays the floor (Job.expected_dues).

If a side stops: timeout_start, timeout_run or claim_no_release, then claim_due.
A seller can also back out: decline_job any time before it could be slashed for
not starting, which is also what it does with a job it won't take (see
job_defaults.spec_problem), or abort_job once running. Either side can end an
unstarted job with abort_unstarted after a rendezvous. Each is followed by
claim_due.

The two sides find each other through a Rendezvous account: rdv_open, rdv_answer,
rdv_reoffer, rdv_fail, rdv_close here, driven by ChainRendezvous in
net/rendezvous.py. Addresses are sealed to the other side (rdv_seal, rdv_unseal).

Inference doesn't go through jobs. A session is a Solana Foundation payment
channel (the second half of this file): the buyer calls open_channel with a
one-time session key, signs cumulative vouchers off chain with sign_voucher, and
the seller ends it with settle_and_seal then distribute. Keypair arguments sign; the first signer of a transaction pays its fee. Amounts
are in the smallest token unit (1 USDC = 1_000_000).
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import beetle_crypto as bc
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.token.associated import get_associated_token_address
from solders.transaction import Transaction

SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ATA_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
INSTRUCTIONS_SYSVAR = Pubkey.from_string("Sysvar1nstructions1111111111111111111111111")
CLOCK_SYSVAR = Pubkey.from_string("SysvarC1ock11111111111111111111111111111111")
ED25519_PROGRAM = Pubkey.from_string("Ed25519SigVerify111111111111111111111111111")
UPGRADEABLE_LOADER = Pubkey.from_string("BPFLoaderUpgradeab1e11111111111111111111111")
RENT_SYSVAR = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
MEMO_PROGRAM = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
NO_VERIFIER = Pubkey.default()

STATES = ["Opened", "Started", "Running", "Delivered", "Settled", "NoStart", "Rejected",
          "NoRelease", "TimedOut", "Declined", "Aborted", "Unstarted"]
FINISHED = {"Settled", "NoStart", "Rejected", "NoRelease", "TimedOut", "Declined", "Aborted",
            "Unstarted"}
PAYEE_SELLER, PAYEE_BUYER, PAYEE_SLASH = 0, 1, 2
KIND_TRAINING, KIND_INFERENCE, KIND_HOSTING = 0, 1, 2
PRICE_BY_TOKEN, PRICE_BY_TIME = 0, 1
OUTCOME_PASSED, OUTCOME_FAILED, OUTCOME_REQUESTER_FAULT, OUTCOME_INCONCLUSIVE = 0, 1, 2, 3
TIER_T2 = 2  # the one tier whose sandbox signature is more than the seller's word
OFFER_PAUSED, OFFER_ACTIVE = 0, 1
AWAY = 2**63 - 1  # Seller.outage_until after a no-start, until the operator is back
REACH_PUBLIC_UDP, REACH_ECHO, REACH_RELAY = 1, 2, 4  # Offer.reach bits
RDV_STATES = ["Offered", "Answered", "Failed"]
RDV_DIRECT_ONLY, RDV_RELAY_ALLOWED, RDV_RELAY_PREFERRED = 0, 1, 2  # rdv_open policy
ROLE_BUYER, ROLE_SELLER = 0, 1
RDV_PLAIN_LEN, RDV_BLOB_LEN, RDV_MAX_ROUNDS, RDV_MAX_CANDIDATES = 176, 224, 2, 5
RDV_TAG = b"beetle/v1/rdv"
BPS = 10_000

OFFER_SIZE = 272
OFFER_STATUS_OFFSET, OFFER_KIND_OFFSET, OFFER_TIER_OFFSET = 9, 10, 11
OFFER_LAYOUT = struct.Struct("<8s6B5H32s32s4QH3I32s64s2qIQHB11s")
JOB_HOST_OFFSET = 8 + 32  # discriminator + requester
TOKEN_STATE_OFFSET = 108
TOKEN_FROZEN = 2
CHANNEL_STATES = ["Open", "Sealed", "Closing", "Distributed"]
VOUCHER_MAGIC = bytes([0x56, 0x01])


class Rpc:
    """The handful of JSON-RPC calls this client needs, over plain HTTP."""

    def __init__(self, url: str):
        self.url = url

    def call(self, method: str, *params):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    out = json.loads(r.read())
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        if "error" in out:
            raise RuntimeError(f"{method}: {out['error']}")
        return out["result"]


def load_keypair(path: Path) -> Keypair:
    return Keypair.from_bytes(bytes(json.loads(Path(path).read_text())))


def ata(owner: Pubkey, mint: Pubkey) -> Pubkey:
    return get_associated_token_address(owner, mint)


def _disc(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode()).digest()[:8]


def _account_disc(name: str) -> bytes:
    return hashlib.sha256(f"account:{name}".encode()).digest()[:8]


def ed25519_ix(pk: bytes, sig: bytes, msg: bytes) -> Instruction:
    """One-signature Ed25519 precompile call; offsets point into itself."""
    pk_off, sig_off = 16, 48
    msg_off = sig_off + 64
    header = bytes([1, 0]) + struct.pack(
        "<7H", sig_off, 0xFFFF, pk_off, 0xFFFF, msg_off, len(msg), 0xFFFF
    )
    return Instruction(ED25519_PROGRAM, header + pk + sig + msg, [])


def new_job_id(requester: Pubkey, nonce: bytes, open_slot: int) -> bytes:
    """job_id = SHA256(requester || nonce || open_slot). The program rejects any
    other id, and an `open_slot` more than 1,500 slots old."""
    return bc.sha256(bytes(requester) + nonce + struct.pack("<Q", open_slot))


def evidence_hash(evidence: dict) -> bytes:
    """What `start` commits to and `release_key` confirms."""
    return bc.sha256(bc.canonical_json(evidence))


def chain_ref_for(genesis_hash: str) -> bytes:
    """SHA256 of this network's CAIP-2 id. Everything signed binds it, so a
    signature made for one network means nothing on another."""
    return bc.sha256(f"solana:{genesis_hash[:32]}".encode())


def pack_addr(addr: str) -> bytes:
    """`ip:port` or `[ip6]:port` as family(1) ip(16) port(2 LE)."""
    host, port = addr.rsplit(":", 1)
    ip = ipaddress.ip_address(host.strip("[]"))
    return bytes([ip.version]) + ip.packed.ljust(16, b"\0") + struct.pack("<H", int(port))


def unpack_addr(data: bytes) -> str | None:
    if data[0] == 4:
        return f"{ipaddress.IPv4Address(data[1:5])}:{struct.unpack('<H', data[17:19])[0]}"
    if data[0] == 6:
        return f"[{ipaddress.IPv6Address(data[1:17])}]:{struct.unpack('<H', data[17:19])[0]}"
    return None


def public_endpoint(endpoint_id: bytes, addr: str) -> bytes:
    """OfferTerms.endpoint of a seller with a public entry point: the iroh
    EndpointId it accepts unsealed connections on, and that address."""
    return endpoint_id + pack_addr(addr)


def rdv_plaintext(role: int, round_: int, endpoint_id: bytes, not_after: int, addrs: list[str],
                  reply_pk: bytes = bytes(32), nat_hint: int = 0, flags: int = 0) -> bytes:
    """The fixed 176 bytes of a rendezvous blob (net.md 5.7). At most five
    addresses fit; the fixed size hides how many there are. `endpoint_id` is an
    iroh id made for this job, never a wallet key. An offer carries `reply_pk`,
    a one-time X25519 key the answer is sealed to."""
    cands = [pack_addr(a) for a in addrs[:RDV_MAX_CANDIDATES]]
    return (bytes([1, role, round_, nat_hint, flags, len(cands)]) + bytes(2) + endpoint_id + reply_pk
            + struct.pack("<q", not_after) + b"".join(cands).ljust(95, b"\0") + bytes(1))


def rdv_parse(plain: bytes) -> dict:
    """Reverses rdv_plaintext. Raises ValueError if this isn't one."""
    if len(plain) != RDV_PLAIN_LEN or plain[0] != 1 or plain[5] > RDV_MAX_CANDIDATES:
        raise ValueError("not a rendezvous blob")
    addrs = [unpack_addr(plain[80 + 19 * i:99 + 19 * i]) for i in range(plain[5])]
    return {"role": plain[1], "round": plain[2], "nat_hint": plain[3], "flags": plain[4],
            "id": plain[8:40].hex(), "reply_pk": plain[40:72],
            "not_after": struct.unpack("<q", plain[72:80])[0], "addrs": [a for a in addrs if a]}


def net_keypair(seed: bytes) -> bc.HpkeKeyPair:
    """The seller's rendezvous key from its 32-byte secret; `public_bytes` is
    what goes on chain as `net_pk`. Keep the secret on the machine."""
    kp = bc.HpkeKeyPair()
    kp._kp = bc._SUITE.kem.derive_key_pair(seed)
    return kp


def session_salt(terms: dict) -> int:
    """First 8 bytes of SHA256(session terms), so a channel's address commits to
    the terms both sides agreed on and no extra account is needed."""
    return int.from_bytes(bc.sha256(bc.canonical_json(terms))[:8], "little")


def voucher_message(channel: Pubkey, cumulative: int, expires_at: int = 0) -> bytes:
    """The 50 bytes a session key signs. `expires_at` 0 means no expiry."""
    return VOUCHER_MAGIC + bytes(channel) + struct.pack("<Qq", cumulative, expires_at)


def sign_voucher(session_key: Keypair, channel: Pubkey, cumulative: int, expires_at: int = 0) -> tuple[bytes, bytes]:
    """Authorizes the seller to take `cumulative` in total from the channel (not
    this much more). Returns (message, signature) for settle_channel or
    settle_and_seal; a newer voucher supersedes the older ones."""
    msg = voucher_message(channel, cumulative, expires_at)
    return msg, bytes(session_key.sign_message(msg))


@dataclass
class OfferTerms:
    """What a seller publishes. See Offer in programs/beetle/src/lib.rs.

    For training, `price_a` is per million tokens and `price_b` per hour, chosen
    by `price_mode`. The run budget of a job is its tokens / `tokens_per_s` (or
    its seconds) plus `grace_s` for setup. `bond_bps` of the price is locked from
    the seller's stake for each job. `reach` holds REACH_* bits; a seller with
    a public address fills `endpoint` from public_endpoint().
    """

    kind: int
    tier: int
    measurement: bytes
    price_a: int = 0
    price_b: int = 0
    min_order: int = 0
    max_order: int = 2**64 - 1
    bond_bps: int = 5000
    start_window_s: int = 900
    grace_s: int = 600
    tokens_per_s: int = 1000
    valid_until: int = 0  # unix seconds; at most 7 days ahead
    slots_total: int = 1
    reach: int = 0
    price_mode: int = PRICE_BY_TOKEN
    hw_class: int = 0
    mem_gb: int = 0
    region: int = 0
    model_root: bytes = bytes(32)
    endpoint: bytes = bytes(64)

    def pack(self) -> bytes:
        return (struct.pack("<4B4H", self.kind, self.tier, self.reach, self.price_mode,
                            self.hw_class, self.mem_gb, self.slots_total, self.region)
                + self.measurement
                + struct.pack("<4QH3I", self.price_a, self.price_b, self.min_order, self.max_order,
                              self.bond_bps, self.start_window_s, self.grace_s, self.tokens_per_s)
                + self.model_root + self.endpoint.ljust(64, b"\0")
                + struct.pack("<q", self.valid_until))


@dataclass
class Offer:
    address: Pubkey
    schema: int
    status: int
    kind: int
    tier: int
    reach: int
    price_mode: int
    hw_class: int
    mem_gb: int
    slots_total: int
    slots_used: int
    region: int
    seller: Pubkey        # the Seller account, not the wallet
    measurement: bytes
    price_a: int
    price_b: int
    min_order: int
    max_order: int
    bond_bps: int
    start_window_s: int
    grace_s: int
    tokens_per_s: int
    model_root: bytes
    endpoint: bytes
    valid_until: int
    updated_at: int
    rev: int
    offer_stake: int
    idx: int
    bump: int

    @classmethod
    def decode(cls, address: Pubkey, data: bytes) -> "Offer":
        f = OFFER_LAYOUT.unpack(data[:OFFER_SIZE])
        return cls(address, *f[1:12], Pubkey.from_bytes(f[12]), *f[13:-1])

    def quote(self, units: int) -> tuple[int, int]:
        """(price, run budget in seconds) for `units`: train tokens for a
        by-token offer, seconds for a by-time one. Same as the program."""
        if self.price_mode == PRICE_BY_TOKEN:
            price = -(-units * self.price_a // 1_000_000)
            run = -(-units // max(self.tokens_per_s, 1))
        else:
            price, run = -(-units * self.price_b // 3600), units
        return price, run + self.grace_s

    @property
    def public_contact(self) -> dict | None:
        """{id, addrs} of a seller reachable without a rendezvous, or None."""
        addr = unpack_addr(self.endpoint[32:51])
        return {"id": self.endpoint[:32].hex(), "addrs": [addr]} if addr else None

    def usage_cost(self, in_tokens: int, out_tokens: int) -> int:
        """What this much inference costs: `price_a` and `price_b` per million
        input and output tokens. Both sides count; pay the smaller number."""
        return -(-(in_tokens * self.price_a + out_tokens * self.price_b) // 1_000_000)


@dataclass
class Seller:
    address: Pubkey
    authority: Pubkey
    operator: Pubkey
    net_pk: bytes
    net_pk_epoch: int
    stake_total: int
    stake_locked: int
    unstake_pending: int
    unstake_ready_at: int
    registered_at: int
    stake_usd_days: int
    stake_touched_at: int
    offers: int
    active_jobs: int
    n_settled: int
    n_failed_accept: int
    n_nostart: int
    n_timeout: int
    n_rejected: int
    n_norelease: int
    n_declined: int
    n_abort: int
    n_connect_fail: int
    volume: int
    last_fail_at: int
    outage_until: int

    LAYOUT = struct.Struct("<8s32s32s32sI3Qqq Q q BH 9I Q q q")

    @classmethod
    def decode(cls, address: Pubkey, data: bytes) -> "Seller":
        f = cls.LAYOUT.unpack(data[:cls.LAYOUT.size])
        return cls(address, Pubkey.from_bytes(f[1]), Pubkey.from_bytes(f[2]), *f[3:])

    @property
    def free_stake(self) -> int:
        return max(self.stake_total - self.stake_locked - self.unstake_pending, 0)

    @property
    def away(self) -> bool:
        """No offer of this seller takes orders until its operator is back."""
        return self.outage_until == AWAY


@dataclass
class Job:
    address: Pubkey
    requester: Pubkey
    host: Pubkey          # the seller's authority, which is also who gets paid
    payer: Pubkey
    offer: Pubkey
    verifier: Pubkey
    job_id: bytes
    spec_hash: bytes
    price: int
    bond: int
    state: str
    opened_at: int
    start_by: int
    release_by: int
    run_secs: int
    deadline: int
    sign_pk: bytes
    enc_pk: bytes
    measurement: bytes
    evidence_hash: bytes
    has_dek: bool
    sealed_dek: bytes
    outcome: int
    work_bps: int
    adapter_ct_hash: bytes
    has_adapter_key: bool
    sealed_adapter_key: bytes
    report_hash: bytes
    challenge_end: int
    offer_rev: int
    floor_bps: int
    fault_bps: int
    tier: int
    seller_due: int
    buyer_due: int
    slash_due: int
    rdv_opened: bool
    units: int            # what was paid for: train tokens or seconds, by price_mode
    grace_s: int
    price_mode: int
    rdv_answered: bool

    @classmethod
    def decode(cls, address: Pubkey, data: bytes) -> "Job":
        c = 8
        def take(n):
            nonlocal c
            v = data[c:c + n]
            c += n
            return v
        pk = lambda: Pubkey.from_bytes(take(32))
        u8 = lambda: take(1)[0]
        u16 = lambda: struct.unpack("<H", take(2))[0]
        u32 = lambda: struct.unpack("<I", take(4))[0]
        u64 = lambda: struct.unpack("<Q", take(8))[0]
        i64 = lambda: struct.unpack("<q", take(8))[0]
        b = lambda: take(1) == b"\x01"
        return cls(address, pk(), pk(), pk(), pk(), pk(), take(32), take(32), u64(), u64(),
                   STATES[u8()], i64(), i64(), i64(), i64(), i64(), take(32), take(32), take(32),
                   take(32), b(), take(80), u8(), u16(), take(32), b(), take(80), take(32), i64(),
                   u32(), u16(), u16(), u8(), u64(), u64(), u64(), b(), u64(), u32(), u8(), b())

    @property
    def terms(self) -> dict:
        """What the sandbox runs under, for job.json. It signs these, and the
        program compares them with this job."""
        return {"units": self.units, "price_mode": self.price_mode, "run_secs": self.run_secs,
                "grace_s": self.grace_s}

    def expected_dues(self, accepted: bool = False) -> tuple[int, int]:
        """(seller, buyer) this job owes. Delivered jobs: what settling will
        write, same as `payout` in the program, with `accepted` saying whether
        that's the buyer's accept. Finished jobs: what's left."""
        if self.state != "Delivered":
            return self.seller_due, self.buyer_due
        proven = self.tier == TIER_T2
        floor = self.price * self.floor_bps * self.work_bps // (BPS * BPS)
        fault = self.price * self.fault_bps // BPS
        if self.outcome == OUTCOME_PASSED:
            seller = self.price if proven or accepted else floor
        elif self.outcome == OUTCOME_FAILED:
            seller = floor
        elif self.outcome == OUTCOME_REQUESTER_FAULT:
            seller = max(fault, self.price * self.work_bps // BPS) if proven else fault
        else:
            seller = 0
        return seller, self.price - seller


@dataclass
class RendezvousAccount:
    """Where a job's two sides swap sealed addresses. `round` counts offers
    within one connection and `opened_at` changes with each, so a client can
    tell every publication apart."""

    address: Pubkey
    job_id: bytes
    requester: Pubkey
    host: Pubkey
    payer: Pubkey
    opened_at: int
    expires_at: int
    policy: int
    round: int
    state: str
    offer: bytes
    answer: bytes

    LAYOUT = struct.Struct("<8s32s32s32s32sqq4B224s224s")

    @classmethod
    def decode(cls, address: Pubkey, data: bytes) -> "RendezvousAccount":
        f = RendezvousAccount.LAYOUT.unpack(data[:RendezvousAccount.LAYOUT.size])
        return cls(address, f[1], *[Pubkey.from_bytes(k) for k in f[2:5]], f[5], f[6], f[7], f[8],
                   RDV_STATES[f[9]], f[11], f[12])


@dataclass
class Channel:
    """A payment channel, from the Solana Foundation's program. `settled` is the
    seller's watermark; `deposit` is the most the buyer can ever spend."""

    address: Pubkey
    status: str
    salt: int
    deposit: int
    settled: int
    payout_watermark: int
    closure_started_at: int
    payer_withdrawn_at: int
    grace_period: int
    payer: Pubkey
    payee: Pubkey
    authorized_signer: Pubkey
    mint: Pubkey
    rent_payer: Pubkey
    open_slot: int

    LAYOUT = struct.Struct("<4B4Q2qI32s32s32s32s32s32sQ")
    PAYEE_OFFSET = struct.calcsize("<4B4Q2qI32s32s")

    @classmethod
    def decode(cls, address: Pubkey, data: bytes) -> "Channel":
        f = Channel.LAYOUT.unpack(data[:Channel.LAYOUT.size])
        keys = [Pubkey.from_bytes(k) for k in f[12:17]]
        return cls(address, CHANNEL_STATES[f[3]], *f[4:11], *keys, f[17])


class Chain:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rpc = Rpc(cfg["rpc_url"])
        self.program_id = Pubkey.from_string(cfg["program_id"])
        self.mint = Pubkey.from_string(cfg["mint"])
        self.chain_ref = bytes.fromhex(cfg["chain_ref"])
        self.config_pda = self.pda(b"config")
        # Empty on networks where the channel program isn't configured.
        self.channel_program = Pubkey.from_string(cfg["channel_program"]) if cfg.get("channel_program") else None

    @classmethod
    def from_file(cls, path: Path) -> "Chain":
        return cls(json.loads(Path(path).read_text()))

    def spec_chain(self) -> dict:
        """The chain fields bound into the job spec (and so into spec_hash). The
        sandbox takes `cluster` as the digest's chain tag: it's the chain id in hex."""
        return {"program_id": str(self.program_id), "program_id_hex": bytes(self.program_id).hex(),
                "cluster": self.chain_ref.hex()}

    def pda(self, *seeds: bytes) -> Pubkey:
        return Pubkey.find_program_address(list(seeds), self.program_id)[0]

    def job_pda(self, job_id: bytes) -> Pubkey:
        return self.pda(b"job", job_id)

    def vault_pda(self, job_id: bytes) -> Pubkey:
        return self.pda(b"vault", job_id)

    def rdv_pda(self, job_id: bytes) -> Pubkey:
        return self.pda(b"rdv", job_id)

    def seller_pda(self, authority: Pubkey) -> Pubkey:
        return self.pda(b"seller", bytes(authority))

    def seller_vault_pda(self, authority: Pubkey) -> Pubkey:
        return self.pda(b"seller_vault", bytes(self.seller_pda(authority)))

    def offer_pda(self, authority: Pubkey, idx: int) -> Pubkey:
        return self.pda(b"offer", bytes(self.seller_pda(authority)), struct.pack("<H", idx))

    def _ix(self, name: str, args: bytes, accounts: list[tuple[Pubkey, bool, bool]]) -> Instruction:
        metas = [AccountMeta(k, is_signer=s, is_writable=w) for k, s, w in accounts]
        return Instruction(self.program_id, _disc(name) + args, metas)

    def send(self, ixs: list[Instruction], signers: list[Keypair]) -> str:
        bh = self.rpc.call("getLatestBlockhash", {"commitment": "confirmed"})["value"]["blockhash"]
        blockhash = Hash.from_string(bh)
        msg = Message.new_with_blockhash(ixs, signers[0].pubkey(), blockhash)
        tx = Transaction(signers, msg, blockhash)
        sig = self.rpc.call("sendTransaction", base64.b64encode(bytes(tx)).decode(),
                            {"encoding": "base64", "preflightCommitment": "confirmed"})
        for _ in range(300):
            st = self.rpc.call("getSignatureStatuses", [sig])["value"][0]
            if st and st.get("err"):
                raise RuntimeError(f"transaction failed: {st['err']}")
            if st and st.get("confirmationStatus") in ("confirmed", "finalized"):
                return sig
            time.sleep(0.1)
        raise TimeoutError(f"transaction not confirmed: {sig}")

    @staticmethod
    def _signers(*kps: Keypair) -> list[Keypair]:
        """Drop repeats: one key can play several roles in a transaction."""
        return list({kp.pubkey(): kp for kp in kps}.values())

    # --- reads ---------------------------------------------------------------

    def _account(self, addr: Pubkey) -> bytes | None:
        acc = self.rpc.call("getAccountInfo", str(addr),
                            {"encoding": "base64", "commitment": "confirmed"})["value"]
        return base64.b64decode(acc["data"][0]) if acc else None

    def _program_accounts(self, name: str, filters: list[dict]) -> list[tuple[Pubkey, bytes]]:
        disc = base64.b64encode(_account_disc(name)).decode()
        res = self.rpc.call("getProgramAccounts", str(self.program_id), {
            "encoding": "base64", "commitment": "confirmed",
            "filters": [{"memcmp": {"offset": 0, "bytes": disc, "encoding": "base64"}}, *filters]})
        return [(Pubkey.from_string(a["pubkey"]), base64.b64decode(a["account"]["data"][0]))
                for a in res]

    def current_slot(self) -> int:
        """The slot to pass as `open_slot` when opening a job now."""
        return self.rpc.call("getSlot", {"commitment": "confirmed"})

    def chain_time(self) -> int:
        """The clock the program sees; it can differ from the wall clock."""
        return struct.unpack("<q", self._account(CLOCK_SYSVAR)[32:40])[0]

    def is_mainnet(self) -> bool:
        """By the network this config is pinned to, not by what the RPC node says."""
        return self.chain_ref == chain_ref_for(MAINNET_GENESIS)

    def get_config(self) -> dict:
        """The parameters this deployment was initialized with, and `chain_ref`."""
        data = self._account(self.config_pda)
        names = ["floor_bps", "fault_bps", "min_bond_bps", "reject_fee_bps", "no_release_fee_bps",
                 "nostart_slash_bps", "timeout_slash_bps", "reject_fee_cap", "no_release_fee_cap",
                 "nostart_slash_min", "offer_stake", "settle_delay", "release_window",
                 "unstake_cooldown", "clock_margin", "max_start_window", "allowed_tiers",
                 "decline_window", "rdv_abort_after", "abort_slash_bps"]
        f = struct.unpack_from("<7H4Q5Q4qIBqqH", data, 8 + 32 + 32)
        return {"chain_ref": data[40:72], "min_stake": list(f[11:16]),
                **dict(zip(names, f[:11] + f[16:]))}

    def get_seller(self, authority: Pubkey) -> Seller | None:
        addr = self.seller_pda(authority)
        data = self._account(addr)
        return Seller.decode(addr, data) if data else None

    def get_offer(self, address: Pubkey) -> Offer | None:
        data = self._account(address)
        return Offer.decode(address, data) if data else None

    def list_offers(self, kind: int | None = None, tier: int | None = None,
                    active_only: bool = True, fresh_within: int | None = 86_400) -> list[Offer]:
        """Offers on chain, filtered by the RPC node with memcmp on fixed offsets.

        `active_only` drops paused and expired offers and those with no free
        slot. `fresh_within` drops offers not touched for that many seconds
        (None keeps them), so dead sellers fall off the list. `tier` is what the
        seller claims, not a proof: evidence decides, see release_key. Results
        are unsorted; a client shouldn't rank by price alone.
        """
        filters: list[dict] = []  # RPC nodes take 4 at most, one being the discriminator
        for offset, value in ((OFFER_KIND_OFFSET, kind), (OFFER_TIER_OFFSET, tier),
                              (OFFER_STATUS_OFFSET, OFFER_ACTIVE if active_only else None)):
            if value is not None:
                filters.append({"memcmp": {"offset": offset, "encoding": "base64",
                                           "bytes": base64.b64encode(bytes([value])).decode()}})
        offers = [Offer.decode(a, d) for a, d in self._program_accounts("Offer", filters)]
        now = self.chain_time()
        if active_only:
            offers = [o for o in offers if o.valid_until >= now and o.slots_used < o.slots_total]
        if fresh_within is not None:
            offers = [o for o in offers if o.updated_at >= now - fresh_within]
        return offers

    def get_rendezvous(self, job_id: bytes) -> RendezvousAccount | None:
        addr = self.rdv_pda(job_id)
        data = self._account(addr)
        return RendezvousAccount.decode(addr, data) if data else None

    def get_job(self, job_id: bytes) -> Job | None:
        """None once the last claim has closed the account (or before open_job)."""
        addr = self.job_pda(job_id)
        data = self._account(addr)
        return Job.decode(addr, data) if data else None

    def jobs_for_host(self, host: Pubkey) -> list[Job]:
        """Open jobs of the seller whose authority is `host`."""
        rows = self._program_accounts("Job", [{"memcmp": {"offset": JOB_HOST_OFFSET, "bytes": str(host)}}])
        return [Job.decode(a, d) for a, d in rows]

    def open_jobs(self) -> list[Job]:
        """Every job whose account is still open. A settled job's account is closed,
        so history isn't here."""
        return [Job.decode(a, d) for a, d in self._program_accounts("Job", [])]

    def token_amount(self, owner: Pubkey) -> int:
        r = self.rpc.call("getTokenAccountBalance", str(ata(owner, self.mint)),
                          {"commitment": "confirmed"})["value"]
        return int(r["amount"])

    def token_balance(self, owner: Pubkey) -> float:
        return self.token_amount(owner) / 1e6

    def sol_lamports(self, owner: Pubkey) -> int:
        return self.rpc.call("getBalance", str(owner), {"commitment": "confirmed"})["value"]

    def wait_job(self, job_id: bytes, pred, timeout: float, every: float = 2.0) -> Job | None:
        """Polls until pred(job) holds. Returns None if the account is gone."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            job = self.get_job(job_id)
            if job is None or pred(job):
                return job
            time.sleep(every)
        raise TimeoutError("job did not reach the expected state")

    # --- test network helpers ----------------------------------------------------

    def airdrop(self, owner: Pubkey, lamports: int = 2_000_000_000) -> None:
        sig = self.rpc.call("requestAirdrop", str(owner), lamports)
        for _ in range(60):
            st = self.rpc.call("getSignatureStatuses", [sig])["value"][0]
            if st and st.get("confirmationStatus") in ("confirmed", "finalized"):
                return
            time.sleep(0.5)
        raise TimeoutError("airdrop not confirmed")

    def create_ata(self, payer: Keypair, owner: Pubkey) -> str:
        """Creates the owner's token account for the mint; no-op if it exists."""
        return self.send([Instruction(ATA_PROGRAM, bytes([1]), [
            AccountMeta(payer.pubkey(), True, True), AccountMeta(ata(owner, self.mint), False, True),
            AccountMeta(owner, False, False), AccountMeta(self.mint, False, False),
            AccountMeta(SYSTEM_PROGRAM, False, False), AccountMeta(TOKEN_PROGRAM, False, False),
        ])], [payer])

    def mint_test_usdc(self, mint_authority: Keypair, owner: Pubkey, amount: int) -> str:
        """Test networks only: mints our own worthless test-USDC."""
        self.create_ata(mint_authority, owner)
        return self.send([Instruction(TOKEN_PROGRAM, bytes([7]) + struct.pack("<Q", amount), [
            AccountMeta(self.mint, False, True), AccountMeta(ata(owner, self.mint), False, True),
            AccountMeta(mint_authority.pubkey(), True, False)])], [mint_authority])

    # --- config --------------------------------------------------------------------

    def init_config(self, authority: Keypair, params: dict) -> str:
        """Only the program's upgrade authority can call this. Params: see
        CHAIN_PARAMS in common/job_defaults.py."""
        if chain_ref_for(self.rpc.call("getGenesisHash")) != self.chain_ref:
            raise ValueError("chain_ref in chain.json is not this network's")
        p = params
        args = (struct.pack("<7H", p["floor_bps"], p["fault_bps"], p["min_bond_bps"],
                            p["reject_fee_bps"], p["no_release_fee_bps"], p["nostart_slash_bps"],
                            p["timeout_slash_bps"])
                + struct.pack("<9Q", p["reject_fee_cap"], p["no_release_fee_cap"],
                              p["nostart_slash_min"], p["offer_stake"], *p["min_stake"])
                + struct.pack("<4qIB", p["settle_delay"], p["release_window"],
                              p["unstake_cooldown"], p["clock_margin"], p["max_start_window"],
                              p["allowed_tiers"])
                + struct.pack("<qqH", p["decline_window"], p["rdv_abort_after"], p["abort_slash_bps"])
                + self.chain_ref)
        program_data = Pubkey.find_program_address([bytes(self.program_id)], UPGRADEABLE_LOADER)[0]
        return self.send([self._ix("init_config", args, [
            (authority.pubkey(), True, True), (self.config_pda, False, True),
            (self.mint, False, False), (self.program_id, False, False),
            (program_data, False, False), (SYSTEM_PROGRAM, False, False)])], [authority])

    # --- sellers and stake -----------------------------------------------------------

    def register_seller(self, authority: Keypair, operator: Pubkey | None = None,
                        net_pk: bytes = bytes(32), payer: Keypair | None = None) -> str:
        """Creates the Seller account and its stake vault. `authority` is the
        cold key that controls stake and gets paid; `operator` (default: the
        same key) is the machine's hot key that signs offers and job steps.
        `net_pk` is the X25519 key buyers seal rendezvous offers to, see
        net_keypair; a seller without one can't be reached by rendezvous."""
        payer = payer or authority
        seller = self.seller_pda(authority.pubkey())
        args = bytes(operator or authority.pubkey()) + net_pk
        return self.send([self._ix("register_seller", args, [
            (payer.pubkey(), True, True), (authority.pubkey(), True, False),
            (self.config_pda, False, False), (seller, False, True),
            (self.seller_vault_pda(authority.pubkey()), False, True), (self.mint, False, False),
            (TOKEN_PROGRAM, False, False), (SYSTEM_PROGRAM, False, False)])],
            self._signers(payer, authority))

    def set_operator(self, authority: Keypair, operator: Pubkey) -> str:
        return self._seller_admin("set_operator", bytes(operator), authority)

    def rotate_net_pk(self, operator: Keypair, authority: Pubkey, net_pk: bytes) -> str:
        """New rendezvous key; bumps `net_pk_epoch`. Destroy the old secret: what
        was sealed to it stays on the public ledger forever."""
        return self.send([self._ix("rotate_net_pk", net_pk, [
            (operator.pubkey(), True, False), (self.seller_pda(authority), False, True)])], [operator])

    def _seller_admin(self, name: str, args: bytes, authority: Keypair) -> str:
        return self.send([self._ix(name, args, [
            (authority.pubkey(), True, False), (self.config_pda, False, False),
            (self.seller_pda(authority.pubkey()), False, True)])], [authority])

    def _stake_ix(self, name: str, args: bytes, authority: Pubkey, token: Pubkey | None = None) -> Instruction:
        return self._ix(name, args, [
            (authority, True, False), (self.config_pda, False, False),
            (self.seller_pda(authority), False, True), (self.seller_vault_pda(authority), False, True),
            (self.mint, False, False), (token or ata(authority, self.mint), False, True),
            (TOKEN_PROGRAM, False, False)])

    def stake(self, authority: Keypair, amount: int) -> str:
        """Moves `amount` from the authority's token account into the stake vault."""
        return self.send([self._stake_ix("stake", struct.pack("<Q", amount), authority.pubkey())],
                         [authority])

    def request_unstake(self, authority: Keypair, amount: int) -> str:
        """Starts the cooldown for `amount` of free stake. Offers the rest can't
        back stay listed but can't be taken."""
        return self._seller_admin("request_unstake", struct.pack("<Q", amount), authority)

    def cancel_unstake(self, authority: Keypair) -> str:
        return self._seller_admin("cancel_unstake", b"", authority)

    def withdraw_stake(self, authority: Keypair) -> str:
        """After the cooldown, sends the requested stake back to the authority."""
        return self.send([self._stake_ix("withdraw_stake", b"", authority.pubkey())], [authority])

    # --- offers ----------------------------------------------------------------------

    def post_offer(self, operator: Keypair, authority: Pubkey, idx: int, terms: OfferTerms,
                   payer: Keypair | None = None) -> str:
        """Posts offer number `idx` (0-65535, unique per seller) of the seller
        whose authority is `authority`. Needs free stake of the tier's minimum
        plus the per-offer stake, which stays locked until close_offer. A seller
        can have 16 offers. Find its address with `offer_pda(authority, idx)`."""
        payer = payer or operator
        return self.send([self._ix("post_offer", struct.pack("<H", idx) + terms.pack(), [
            (payer.pubkey(), True, True), (operator.pubkey(), True, False),
            (self.config_pda, False, False), (self.seller_pda(authority), False, True),
            (self.offer_pda(authority, idx), False, True), (SYSTEM_PROGRAM, False, False)])],
            self._signers(payer, operator))

    def _offer_ix(self, name: str, args: bytes, operator: Keypair, offer: Offer) -> str:
        return self.send([self._ix(name, args, [
            (operator.pubkey(), True, False), (self.config_pda, False, False),
            (offer.seller, False, True), (offer.address, False, True)])], [operator])

    def update_offer(self, operator: Keypair, offer: Offer, terms: OfferTerms) -> str:
        """Replaces the terms (kind and tier can't change), bumps `rev` and
        reactivates a paused offer, which also brings the seller back from a
        no-start. Jobs already open keep their terms."""
        return self._offer_ix("update_offer", terms.pack(), operator, offer)

    def touch_offer(self, operator: Keypair, offer: Offer, valid_until: int) -> str:
        """Heartbeat: moves `valid_until` (at most 7 days ahead); `rev` stays."""
        return self._offer_ix("touch_offer", struct.pack("<q", valid_until), operator, offer)

    def pause_offer(self, operator: Keypair, offer: Offer) -> str:
        return self._offer_ix("pause_offer", b"", operator, offer)

    def close_offer(self, operator: Keypair, offer: Offer) -> str:
        """Closes an offer with no job in progress: unlocks its stake and sends
        the rent to the seller's authority."""
        seller = Seller.decode(offer.seller, self._account(offer.seller))
        return self.send([self._ix("close_offer", b"", [
            (operator.pubkey(), True, False), (seller.address, False, True),
            (offer.address, False, True), (seller.authority, False, True)])], [operator])

    # --- jobs ------------------------------------------------------------------------

    def open_job(self, payer: Keypair, requester: Keypair, offer: Offer, job_id: bytes,
                 nonce: bytes, open_slot: int, spec_hash: bytes, units: int,
                 max_price: int | None = None, verifier: Pubkey = NO_VERIFIER) -> str:
        """Takes an offer: escrows the price and locks the seller's bond.

        `units` is train tokens or seconds, see Offer.quote. `job_id` comes from
        new_job_id; pass `open_slot` from current_slot just before. The offer's
        `rev` as you read it is checked, so a price change in between fails the
        transaction, and `max_price` (default: the quote) caps what you pay.
        `verifier` optionally names a third party whose signature `start` then
        requires.
        """
        max_price = offer.quote(units)[0] if max_price is None else max_price
        args = (job_id + nonce + struct.pack("<Q", open_slot) + spec_hash
                + struct.pack("<QQI", units, max_price, offer.rev) + bytes(verifier))
        ix = self._ix("open_job", args, [
            (payer.pubkey(), True, True), (requester.pubkey(), True, False),
            (self.config_pda, False, False), (offer.address, False, True),
            (offer.seller, False, True), (self.job_pda(job_id), False, True),
            (self.vault_pda(job_id), False, True), (self.mint, False, False),
            (ata(requester.pubkey(), self.mint), False, True), (TOKEN_PROGRAM, False, False),
            (SYSTEM_PROGRAM, False, False)])
        return self.send([ix], self._signers(payer, requester))

    def start(self, operator: Keypair, job: Job, sign_pk: bytes, enc_pk: bytes,
              measurement: bytes, evidence_hash: bytes,
              registration: tuple[bytes, bytes, bytes, int] | None = None) -> str:
        """Declares the sandbox keys and commits to the hash of its evidence,
        which goes to the buyer off chain. `measurement` must equal the offer's.

        `registration` is needed only when the job names a verifier: (verifier
        public key, its signature, the signed message, expiry). The signature
        check goes in the Ed25519 precompile instruction just before `start`.
        """
        expiry = registration[3] if registration else 0
        args = sign_pk + enc_pk + measurement + evidence_hash + struct.pack("<q", expiry)
        ix = self._ix("start", args, [
            (operator.pubkey(), True, False), (self.config_pda, False, False),
            (self.seller_pda(job.host), False, False), (job.address, False, True),
            (INSTRUCTIONS_SYSVAR, False, False)])
        pre = [ed25519_ix(*registration[:3])] if registration else []
        return self.send([*pre, ix], [operator])

    def release_key(self, requester: Keypair, job_id: bytes, sealed_dek: bytes,
                    evidence_hash: bytes) -> str:
        """Stores the DEK sealed to the sandbox. Call it only after verifying
        the evidence locally (Requester.check_evidence): this is what endorses
        the sandbox keys. Must happen before `release_by`; the seller's run
        budget starts here."""
        assert len(sealed_dek) == 80
        return self.send([self._ix("release_key", sealed_dek + evidence_hash, [
            (requester.pubkey(), True, False), (self.job_pda(job_id), False, True)])], [requester])

    def _end_ix(self, name: str, caller: Pubkey, job: Job) -> Instruction:
        return self._ix(name, b"", [
            (caller, True, False), (self.config_pda, False, False), (job.address, False, True),
            (self.seller_pda(job.host), False, True), (job.offer, False, True)])

    def reject_evidence(self, requester: Keypair, job: Job) -> str:
        """The evidence failed local checks: ends the job with a refund less a
        small capped fee. Follow with claim_due."""
        return self.send([self._end_ix("reject_evidence", requester.pubkey(), job)], [requester])

    def claim_no_release(self, caller: Keypair, job: Job) -> str:
        """Anyone, after `release_by`: the buyer never released. Follow with claim_due."""
        return self.send([self._end_ix("claim_no_release", caller.pubkey(), job)], [caller])

    def decline_job(self, operator: Keypair, job: Job) -> str:
        """The seller turns down an open job, any time until `start_by` plus the
        clock margin: nothing arrived, the spec asks for more than was paid, or it
        can't serve it. Refund, no slash. Follow with claim_due."""
        return self.send([self._end_ix("decline_job", operator.pubkey(), job)], [operator])

    def abort_job(self, operator: Keypair, job: Job) -> str:
        """The seller gives up on a running job. Refund, and a smaller slash than
        timeout_run would cost. Follow with claim_due."""
        return self.send([self._end_ix("abort_job", operator.pubkey(), job)], [operator])

    def abort_unstarted(self, caller: Keypair, job: Job) -> str:
        """Either side, once the buyer has opened a rendezvous, the abort delay has
        passed and the job still hasn't started. Refund, no slash. A seller that
        never answered gets a failed connection on its record; a buyer leaving
        after an answer pays a flat fee, the reject fee cap. Follow with claim_due."""
        return self.send([self._end_ix("abort_unstarted", caller.pubkey(), job)], [caller])

    def settle_message(self, job_id: bytes, report: dict, report_hash: bytes) -> bytes:
        """What the sandbox must sign for submit_result, the format the program
        re-derives. Report fields: `outcome`, `work_bps`, `sealed_dek_hash`,
        `adapter_ct_hash` (hex), `sealed_adapter_key` (hex or None), and the job's
        `terms`, which the program takes from the job."""
        sealed = report["sealed_adapter_key"]
        body = (job_id + bytes([report["outcome"]]) + struct.pack("<H", report["work_bps"])
                + bytes.fromhex(report["sealed_dek_hash"]) + bytes.fromhex(report["adapter_ct_hash"])
                + bytes([sealed is not None]) + (bytes.fromhex(sealed) if sealed else bytes(80))
                + report_hash + struct.pack("<QBqI", report["units"], report["price_mode"],
                                            report["run_secs"], report["grace_s"]))
        return bc.sha256(bc.DOMAIN + b"settle" + bytes(self.program_id)
                         + self.chain_ref.hex().encode() + body)

    def submit_result_ix(self, operator: Pubkey, job: Job, report: dict, report_hash: bytes) -> Instruction:
        sealed = report["sealed_adapter_key"]
        args = (bytes([report["outcome"]]) + struct.pack("<H", report["work_bps"])
                + bytes.fromhex(report["sealed_dek_hash"]) + bytes.fromhex(report["adapter_ct_hash"])
                + bytes([sealed is not None]) + (bytes.fromhex(sealed) if sealed else bytes(80))
                + report_hash)
        # The offer goes last and is optional: with it the job's slot is free at
        # once. Without it, as from an older client, the slot waits for settling.
        return self._ix("submit_result", args, [
            (operator, True, False), (self.config_pda, False, False),
            (self.seller_pda(job.host), False, False), (job.address, False, True),
            (INSTRUCTIONS_SYSVAR, False, False), (job.offer, False, True)])

    def submit_result(self, operator: Keypair, job: Job, report: dict, report_hash: bytes,
                      settle_sig: bytes) -> str:
        """Posts the sandbox-signed result. `settle_sig` is the sandbox key's
        signature over settle_message(); the settle delay starts here."""
        msg = self.settle_message(job.job_id, report, report_hash)
        return self.send([ed25519_ix(job.sign_pk, settle_sig, msg),
                          self.submit_result_ix(operator.pubkey(), job, report, report_hash)],
                         [operator])

    def _payable(self, owner: Pubkey) -> bool:
        data = self._account(ata(owner, self.mint))
        return data is not None and data[TOKEN_STATE_OFFSET] != TOKEN_FROZEN

    def claim_ix(self, job: Job, who: int, to: Pubkey | None = None) -> Instruction:
        owner = job.host if who == PAYEE_SELLER else job.requester
        return self._ix("claim", bytes([who]), [
            (self.config_pda, False, False), (job.address, False, True),
            (self.vault_pda(job.job_id), False, True), (self.seller_pda(job.host), False, False),
            (self.seller_vault_pda(job.host), False, True), (self.mint, False, False),
            (to or ata(owner, self.mint), False, True), (job.payer, False, True),
            (TOKEN_PROGRAM, False, False)])

    def _claim_ixs(self, job: Job, accepted: bool = False) -> list[Instruction]:
        """Claims for every payee whose account can receive now. A frozen or
        missing account is left out, so it can't hold up the other side."""
        seller_due, buyer_due = job.expected_dues(accepted)
        ixs = []
        if seller_due and self._payable(job.host):
            ixs.append(self.claim_ix(job, PAYEE_SELLER))
        if buyer_due and self._payable(job.requester):
            ixs.append(self.claim_ix(job, PAYEE_BUYER))
        if job.slash_due and self._payable(job.requester):
            ixs.append(self.claim_ix(job, PAYEE_SLASH))
        return ixs

    def accept(self, requester: Keypair, job: Job) -> str:
        """The buyer takes a delivered job. Off T2 this is the only thing that
        pays the full price. Pays out in the same transaction."""
        return self.send([self._end_ix("accept", requester.pubkey(), job),
                          *self._claim_ixs(job, accepted=True)], [requester])

    def reject(self, requester: Keypair, job: Job) -> str:
        """Off T2, the buyer's hidden questions failed on the delivery: settles now
        at what finalize would pay, the floor scaled by work_bps. Pays out in the
        same transaction. A T2 result can't be refused."""
        return self.send([self._end_ix("reject", requester.pubkey(), job), *self._claim_ixs(job)],
                         [requester])

    def finalize(self, caller: Keypair, job: Job) -> str:
        """Anyone, after the settle delay; pays out in the same transaction. Off
        T2 a buyer that said nothing hasn't accepted, so this pays the floor."""
        return self.send([self._end_ix("finalize", caller.pubkey(), job), *self._claim_ixs(job)],
                         [caller])

    def timeout_start(self, caller: Keypair, job: Job) -> str:
        """Anyone, after `start_by`: the seller never started. Refunds the buyer,
        slashes the seller, pauses the offer. Follow with claim_due."""
        return self.send([self._end_ix("timeout_start", caller.pubkey(), job)], [caller])

    def timeout_run(self, caller: Keypair, job: Job) -> str:
        """Anyone, after `deadline`: no result in time. Refunds the buyer,
        slashes the seller. Follow with claim_due."""
        return self.send([self._end_ix("timeout_run", caller.pubkey(), job)], [caller])

    def claim(self, caller: Keypair, job: Job, who: int) -> str:
        """Pays one payee (PAYEE_SELLER, PAYEE_BUYER or PAYEE_SLASH) of a
        finished job. The last claim closes the job and refunds its rent."""
        return self.send([self.claim_ix(job, who)], [caller])

    def claim_due(self, caller: Keypair, job_id: bytes) -> str | None:
        """Pays everything a finished job owes to accounts that can receive it.
        Returns the signature, or None if nothing can be paid now."""
        job = self.get_job(job_id)
        ixs = self._claim_ixs(job) if job else []
        return self.send(ixs, [caller]) if ixs else None

    # --- rendezvous -------------------------------------------------------------------

    def rdv_info(self, job_id: bytes, role: int, round_: int) -> bytes:
        """HPKE context: a blob opens only for this deployment, job, sender and round."""
        return RDV_TAG + bytes(self.program_id) + self.chain_ref + job_id + bytes([role, round_])

    def rdv_seal(self, recipient_pk: bytes, job_id: bytes, role: int, round_: int, plain: bytes) -> bytes:
        """`role` is the sender's. The chain is public and permanent, so an address
        never goes there in the clear: it would tie a wallet to an IP for good."""
        return bc.hpke_seal(recipient_pk, plain, self.rdv_info(job_id, role, round_))

    def rdv_unseal(self, key: bc.HpkeKeyPair, job_id: bytes, role: int, round_: int, blob: bytes) -> dict:
        """Opens what `role` sealed to `key`. Raises if it's not for us or is garbage."""
        fields = rdv_parse(key.open(blob, self.rdv_info(job_id, role, round_)))
        if (fields["role"], fields["round"]) != (role, round_):
            raise ValueError("rendezvous blob is for another round")
        return fields

    def rdv_open(self, requester: Keypair, job_id: bytes, offer: bytes, ttl_secs: int = 600,
                 policy: int = RDV_DIRECT_ONLY, payer: Keypair | None = None) -> str:
        """The buyer starts a connection with its first offer. A job in any state
        before settlement qualifies, and calling it again starts over at round 1.
        The account's rent is a deposit that rdv_close returns to `payer`."""
        payer = payer or requester
        args = struct.pack("<Bq", policy, ttl_secs) + offer
        return self.send([self._ix("rdv_open", args, [
            (payer.pubkey(), True, True), (requester.pubkey(), True, False),
            (self.job_pda(job_id), False, True), (self.rdv_pda(job_id), False, True),
            (SYSTEM_PROGRAM, False, False)])], self._signers(payer, requester))

    def rdv_answer(self, operator: Keypair, rdv: RendezvousAccount, round_: int, answer: bytes) -> str:
        """The seller's reply to the current offer. Start dialing before this
        lands: the buyer's first packet then finds the seller's NAT already open."""
        return self.send([self._ix("rdv_answer", bytes([round_]) + answer, [
            (operator.pubkey(), True, False), (self.seller_pda(rdv.host), False, False),
            (rdv.address, False, True), (self.job_pda(rdv.job_id), False, True)])], [operator])

    def rdv_reoffer(self, requester: Keypair, rdv: RendezvousAccount, round_: int, offer: bytes) -> str:
        """Round `round_` (the current one plus one) with fresh addresses."""
        return self.send([self._ix("rdv_reoffer", bytes([round_]) + offer, [
            (requester.pubkey(), True, False), (rdv.address, False, True)])], [requester])

    def rdv_fail(self, caller: Keypair, rdv: RendezvousAccount, reason: int = 0) -> str:
        """Either side gives up on this connection; anyone can then close it."""
        return self.send([self._ix("rdv_fail", bytes([reason]), [
            (caller.pubkey(), True, False), (self.seller_pda(rdv.host), False, False),
            (rdv.address, False, True)])], [caller])

    def rdv_close(self, caller: Keypair, rdv: RendezvousAccount) -> str:
        """Returns the deposit to whoever paid it. The buyer can always do this;
        anyone can once the rendezvous has failed or expired."""
        return self.send([self._ix("rdv_close", b"", [
            (caller.pubkey(), True, False), (rdv.address, False, True),
            (rdv.payer, False, True)])], [caller])

    # --- inference: Solana Foundation payment channels -------------------------------
    #
    # Beetle's program stays out of this money path. A seller who stops serving
    # costs the buyer at most one step, so inference needs no stake.

    def channel_pda(self, payer: Pubkey, payee: Pubkey, session_key: Pubkey, salt: int,
                    open_slot: int) -> Pubkey:
        return Pubkey.find_program_address(
            [b"channel", bytes(payer), bytes(payee), bytes(self.mint), bytes(session_key),
             struct.pack("<Q", salt), struct.pack("<Q", open_slot)], self.channel_program)[0]

    def get_channel(self, address: Pubkey) -> Channel | None:
        """None once distribute or reclaim has deleted the account."""
        data = self._account(address)
        return Channel.decode(address, data) if data else None

    def _channel_ix(self, disc: int, args: bytes, accounts: list[tuple[Pubkey, bool, bool]]) -> Instruction:
        metas = [AccountMeta(k, is_signer=s, is_writable=w) for k, s, w in accounts]
        return Instruction(self.channel_program, bytes([disc]) + args, metas)

    def open_channel(self, payer: Keypair, payee: Pubkey, session_key: Pubkey, deposit: int,
                     grace_period: int, salt: int, rent_payer: Keypair | None = None) -> Pubkey:
        """Opens a session: escrows `deposit`, the hard ceiling on what the session
        can spend. `session_key` is a one-time key that signs vouchers and can do
        nothing else. `grace_period` is how many seconds the seller gets to settle
        after a request_close. `salt` comes from session_salt. Returns the channel."""
        rent_payer = rent_payer or payer
        open_slot = self.current_slot()
        channel = self.channel_pda(payer.pubkey(), payee, session_key, salt, open_slot)
        # An empty distribution list: everything goes to the payee.
        args = struct.pack("<QQIQI", salt, deposit, grace_period, open_slot, 0)
        event_authority = Pubkey.find_program_address([b"event_authority"], self.channel_program)[0]
        self.send([self._channel_ix(1, args, [
            (payer.pubkey(), True, True), (rent_payer.pubkey(), True, True), (payee, False, False),
            (self.mint, False, False), (session_key, False, False), (channel, False, True),
            (ata(payer.pubkey(), self.mint), False, True), (ata(channel, self.mint), False, True),
            (TOKEN_PROGRAM, False, False), (SYSTEM_PROGRAM, False, False), (RENT_SYSVAR, False, False),
            (ATA_PROGRAM, False, False), (event_authority, False, False),
            (self.channel_program, False, False)])], self._signers(payer, rent_payer))
        return channel

    def top_up_channel(self, payer: Keypair, channel: Channel, amount: int) -> str:
        """Raises the session's ceiling by `amount`."""
        return self.send([self._channel_ix(3, struct.pack("<Q", amount), [
            (payer.pubkey(), True, True), (channel.address, False, True),
            (ata(payer.pubkey(), self.mint), False, True), (ata(channel.address, self.mint), False, True),
            (self.mint, False, False), (TOKEN_PROGRAM, False, False)])], [payer])

    def settle_channel(self, caller: Keypair, channel: Channel, voucher: tuple[bytes, bytes]) -> str:
        """Anyone: moves the watermark up to the voucher, with the session open."""
        msg, sig = voucher
        return self.send([ed25519_ix(bytes(channel.authorized_signer), sig, msg),
                          self._channel_ix(2, b"", [(channel.address, False, True),
                                                    (INSTRUCTIONS_SYSVAR, False, False)])], [caller])

    def settle_and_seal(self, payee: Keypair, channel: Channel,
                        voucher: tuple[bytes, bytes] | None = None) -> str:
        """The seller ends the session, applying its last voucher if given. Works
        during the grace period of a request_close too. Follow with distribute."""
        pre = [ed25519_ix(bytes(channel.authorized_signer), voucher[1], voucher[0])] if voucher else []
        return self.send([*pre, self._channel_ix(4, bytes([voucher is not None]), [
            (payee.pubkey(), True, False), (channel.address, False, True),
            (INSTRUCTIONS_SYSVAR, False, False)])], [payee])

    def request_close(self, payer: Keypair, channel: Channel) -> str:
        """The buyer's way out of a session the seller left: starts the grace period."""
        return self.send([self._channel_ix(5, b"", [
            (payer.pubkey(), True, False), (channel.address, False, True)])], [payer])

    def seal_channel(self, caller: Keypair, channel: Channel) -> str:
        """Anyone, after the grace period. Follow with distribute."""
        return self.send([self._channel_ix(6, b"", [(channel.address, False, True)])], [caller])

    def distribute(self, caller: Keypair, channel: Channel) -> str:
        """Anyone, on a sealed channel: pays the seller what was settled, refunds
        the buyer the rest and closes the escrow. Check first that both token
        accounts exist: the program sends a share whose account is unusable to
        its treasury for good."""
        if not self.cfg.get("channel_treasury"):
            raise RuntimeError("channel_treasury is not set in chain.json")
        treasury = Pubkey.from_string(self.cfg["channel_treasury"])
        event_authority = Pubkey.find_program_address([b"event_authority"], self.channel_program)[0]
        return self.send([self._channel_ix(7, struct.pack("<I", 0), [
            (channel.address, False, True), (channel.payer, False, True), (channel.rent_payer, False, True),
            (ata(channel.address, self.mint), False, True), (ata(channel.payer, self.mint), False, True),
            (ata(channel.payee, self.mint), False, True), (ata(treasury, self.mint), False, True),
            (self.mint, False, False), (TOKEN_PROGRAM, False, False), (event_authority, False, False),
            (self.channel_program, False, False)])], [caller])

    def channels_for(self, payee: Pubkey) -> list[Channel]:
        """Channels that pay `payee`: how a seller learns a session has opened."""
        res = self.rpc.call("getProgramAccounts", str(self.channel_program), {
            "encoding": "base64", "commitment": "confirmed",
            "filters": [{"memcmp": {"offset": Channel.PAYEE_OFFSET, "bytes": str(payee)}}]})
        return [Channel.decode(Pubkey.from_string(a["pubkey"]), base64.b64decode(a["account"]["data"][0]))
                for a in res]

    def post_memo(self, signer: Keypair, about: Pubkey, data: bytes) -> str:
        """Writes `data` to the ledger so that a search for `about` finds it. The
        memo program accepts no account but signers, so a zero transfer does the
        mentioning. A transaction carries about 300 bytes this way."""
        mention = Instruction(SYSTEM_PROGRAM, struct.pack("<IQ", 2, 0), [
            AccountMeta(signer.pubkey(), True, True), AccountMeta(about, False, True)])
        return self.send([mention, Instruction(MEMO_PROGRAM, base64.b64encode(data), [])], [signer])

    def memos(self, about: Pubkey, signer: Pubkey, seen: set[str], limit: int = 20) -> list[bytes]:
        """New memos from `signer` on transactions that mention `about`, oldest
        first. `seen` holds the signatures already read and grows."""
        found = []
        for entry in reversed(self.rpc.call("getSignaturesForAddress", str(about),
                                            {"limit": limit, "commitment": "confirmed"})):
            if entry["signature"] in seen or entry["err"]:
                continue
            raw = self.rpc.call("getTransaction", entry["signature"], {
                "encoding": "base64", "commitment": "confirmed", "maxSupportedTransactionVersion": 0})
            if raw is None:  # not served yet; it comes up on the next pass
                continue
            seen.add(entry["signature"])
            message = Transaction.from_bytes(base64.b64decode(raw["transaction"][0])).message
            keys = message.account_keys
            if keys[0] == signer:
                found += [base64.b64decode(bytes(ix.data)) for ix in message.instructions
                          if keys[ix.program_id_index] == MEMO_PROGRAM]
        return found

    def reclaim_channel(self, caller: Keypair, channel: Channel) -> str:
        """Anyone: returns a distributed channel's rent to whoever paid it, once
        1,500 slots have passed since it opened."""
        return self.send([self._channel_ix(9, b"", [
            (channel.address, False, True), (channel.rent_payer, False, True)])], [caller])
