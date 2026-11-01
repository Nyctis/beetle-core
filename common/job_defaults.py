"""Job and chain parameters shared by the offline runner and the chain clients."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("BEETLE_IMAGE", "beetle-sandbox:dev")
GPU = os.environ.get("BEETLE_GPU") == "1"  # a GPU host built the image with a CUDA torch
MODEL_REPO = os.environ.get("BEETLE_MODEL_REPO", "HuggingFaceTB/SmolLM2-135M-Instruct")
MODEL_DIR = ROOT / ".cache" / "models" / MODEL_REPO.split("/")[-1]
# Everything copied into the sandbox image's /app. The measurement the sandbox
# reports must equal the one computed from these public files.
APP_FILES = [ROOT / "common/beetle_crypto.py", ROOT / "common/beetle_archive.py",
             ROOT / "common/job_defaults.py", ROOT / "sandbox/enclave.py",
             ROOT / "sandbox/train_eval.py",
             ROOT / "sandbox/recipe_data.py", ROOT / "sandbox/program_runner.py",
             ROOT / "sandbox/launcher.py", ROOT / "sandbox/requirements.txt"]
SDK_FILES = [ROOT / "sandbox/sdk/beetle_sdk.py"]


def sandbox_code_files() -> dict[str, bytes]:
    files = {p.name: p.read_bytes() for p in APP_FILES}
    files.update({f"sdk/{p.name}": p.read_bytes() for p in SDK_FILES})
    return files


# Privacy tiers as numbered on chain; a job accepts a bitmask of them.
TIERS = {"mock": 0, "t1": 1, "t2": 2, "t3b": 3, "t3c": 4}
TIER_LABELS = {0: "MOCK (development)", 1: "T1 Mac App Attest", 2: "T2 TDX", 3: "T3-B TPM",
               4: "T3-C plain GPU, no evidence"}

PRICE = 2_000_000          # 2.00 USDC (6 decimals)
PROTOCOL_FEE_BPS = 0       # no fee inside the program
COMPUTE_FLOOR_BPS = 2000   # 20%, scaled by the share of the work done
REQUESTER_FAULT_BPS = 500  # 5%
BOND_RATIO_BPS = 5000      # 0.5x price
LATE_SLASH_BPS = 5000      # 50% of bond

# Report outcomes as numbered on chain.
OUTCOME_PASSED, OUTCOME_FAILED, OUTCOME_REQUESTER_FAULT, OUTCOME_INCONCLUSIVE = 0, 1, 2, 3

# By-token and by-time offers, as numbered on chain.
PRICE_BY_TOKEN, PRICE_BY_TIME = 0, 1

# The chain's release window is at most MAX_RELEASE_WINDOW (lib.rs). The sandbox
# attests before `start` lands, so it waits that long and the time to land it.
MAX_RELEASE_WINDOW = 600
START_LATENCY_S = 300
DEK_WAIT_SECONDS = MAX_RELEASE_WINDOW + START_LATENCY_S
# Of a job's run budget, what training can't use: the sandbox stops itself
# SUBMIT_RESERVE_S before the deadline so the seller can still submit, and
# EVAL_RESERVE_S is kept for the last evaluation and for sealing.
SUBMIT_RESERVE_S = 120
EVAL_RESERVE_S = 120
RUN_RESERVE_S = SUBMIT_RESERVE_S + EVAL_RESERVE_S

# What a job's sealed inputs can ask of the sandbox. Every limit is here, and
# none in the job, because a limit the buyer writes is no limit.
MAX_GAIN_REQUIRED = 0.9     # also the share of what the baseline left unanswered
MAX_EVAL_ITEMS = 256
MAX_FIELD_CHARS = 64 << 10
MAX_TRAIN_BYTES = 64 << 20
BYTES_PER_TOKEN = 16        # far more than a token takes: data past this can't be paid for
MAX_PROGRAM_BYTES = 1 << 20
MAX_PROGRAM_FILES = 64
RECIPE_LIMITS = {"max_seq_len": 4096, "rank": 256, "batch_size": 64, "epochs": 1000,
                 "checkpoint_every": 1 << 20}

# Sandbox errors the host cannot fake. The sandbox has no network, so everything
# it reads was carried in by the host; only what happens after a ciphertext's
# hash matched the spec (and the sealed key's hash matched the chain) is proof
# that the requester sent something bad.
REQUESTER_FAULTS = {"dek_open_failed", "data_decrypt_failed", "malformed_inputs",
                    "program_hash_mismatch", "unknown_mode", "threshold_above_cap",
                    "baseline_already_passes"}
# Never reported on chain: the host can cause the first by holding the file back,
# and the chain already knows whether the key was released (release_by). The
# others stop the sandbox before it attests, so the seller declines the job.
UNREPORTABLE_ERRORS = {"dek_timeout", "job_terms_missing", "spec_malformed", "budget_not_paid_for"}


def outcome_for_error(reason: str) -> int | None:
    """The outcome a sandbox error maps to. Anything that isn't provably the
    requester's fault is Inconclusive (a swapped input file, for example), so
    no error can earn the host money it didn't work for. None means the
    sandbox signs no result and the job ends through the timeouts."""
    if reason in UNREPORTABLE_ERRORS:
        return None
    return OUTCOME_REQUESTER_FAULT if reason in REQUESTER_FAULTS else OUTCOME_INCONCLUSIVE


def work_bps(trained_tokens: int, planned_tokens: int) -> int:
    """Share of the planned tokens actually trained, in basis points."""
    return min(10_000, trained_tokens * 10_000 // max(planned_tokens, 1))


def outcome_for_run(passed: bool, trained_tokens: int, planned_tokens: int, wall_seconds: float,
                    max_wall_seconds: float, requester_fault: bool = False) -> tuple[int, int]:
    """(outcome, work_bps) of a run that reached evaluation. Hitting the wall
    clock proves nothing about the model: the host controls the machine's
    speed, so that's Inconclusive rather than a failure that pays the floor. A
    requester's program that hung or crashed is its own doing, whoever's clock
    ran out, and pays for what it used."""
    done = work_bps(trained_tokens, planned_tokens)
    if passed:
        return OUTCOME_PASSED, done
    if requester_fault:
        return OUTCOME_REQUESTER_FAULT, done
    if wall_seconds > max_wall_seconds:
        return OUTCOME_INCONCLUSIVE, done
    return OUTCOME_FAILED, done


def fit_budget(tokens: int, run_secs: int) -> dict:
    """The budget for an order of `tokens` with this run budget. The wall limit is
    the longest a seller takes: past it the sandbox couldn't report in time."""
    return {"max_train_tokens": tokens, "max_wall_seconds": run_secs - RUN_RESERVE_S}


def train_bytes_limit(terms: dict) -> int:
    """Reading data costs the seller too, and an order for one token must not
    buy minutes of tokenizing."""
    if terms["price_mode"] != PRICE_BY_TOKEN:
        return MAX_TRAIN_BYTES
    return min(MAX_TRAIN_BYTES, (64 << 10) + BYTES_PER_TOKEN * terms["units"])


def _int(x, low: int, high: float = float("inf")) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and low <= x <= high


def _num(x, low: float, high: float) -> bool:  # false for NaN too
    return isinstance(x, (int, float)) and not isinstance(x, bool) and low <= x <= high


def _text(x) -> bool:
    return isinstance(x, str) and 0 < len(x) <= MAX_FIELD_CHARS


def spec_problem(spec: dict, terms: dict) -> str | None:
    """Why this public spec can't run on what the chain says was paid, or None.
    Both the seller and the sandbox ask, before anything starts: `terms` is
    Job.terms, and the sandbox must sign the same numbers to settle. The answer
    means decline, which costs the seller nothing."""
    budget = spec.get("budget")
    inputs = spec.get("inputs")
    if not (isinstance(budget, dict) and isinstance(inputs, dict)
            and set(inputs) == {"body", "train", "eval", "program"}):
        return "spec_malformed"
    tokens, wall = budget.get("max_train_tokens"), budget.get("max_wall_seconds")
    if not (_int(tokens, 1) and _int(wall, 1)):
        return "spec_malformed"
    if terms["price_mode"] == PRICE_BY_TOKEN and tokens > terms["units"]:
        return "budget_not_paid_for"
    if wall > terms["run_secs"] - RUN_RESERVE_S:
        return "budget_not_paid_for"
    return None


def _recipe_ok(r) -> bool:
    if not isinstance(r, dict):
        return False
    lora, optim, cap = r.get("lora"), r.get("optim"), RECIPE_LIMITS
    if not (isinstance(lora, dict) and isinstance(optim, dict)):
        return False
    targets = lora.get("targets", "all-linear")
    return (_int(r.get("max_seq_len"), 8, cap["max_seq_len"]) and _int(r.get("seed"), 0, 2**32 - 1)
            and _int(r.get("checkpoint_every"), 1, cap["checkpoint_every"])
            and _int(lora.get("rank"), 1, cap["rank"]) and _num(lora.get("alpha"), 0, 1e6)
            and _num(lora.get("dropout"), 0, 0.99)
            and (_text(targets) or (isinstance(targets, list) and bool(targets)
                                    and all(_text(t) for t in targets)))
            and _num(optim.get("lr"), 1e-12, 1) and _num(optim.get("weight_decay"), 0, 1)
            and _num(optim.get("warmup_ratio"), 0, 1) and _int(optim.get("epochs"), 1, cap["epochs"])
            and _int(optim.get("batch_size"), 1, cap["batch_size"]))


def inputs_problem(body, memories, eval_items) -> str | None:
    """The error a job's decrypted inputs earn, or None if they can run. This
    runs before anything is trained, so a job the requester got wrong ends as
    its fault with the seller paid, not as a crash that pays nobody."""
    if not (isinstance(body, dict) and isinstance(memories, list) and isinstance(eval_items, list)):
        return "malformed_inputs"
    mode, rule = body.get("mode"), body.get("acceptance")
    if mode not in ("recipe", "program"):
        return "unknown_mode"
    if not _recipe_ok(body.get("recipe")) or not isinstance(rule, dict):
        return "malformed_inputs"
    if not (_num(rule.get("personal_qa_min_gain"), -1, float("inf"))
            and _num(rule.get("regression_max_loss_increase"), 0, 1e6)):
        return "malformed_inputs"
    if rule["personal_qa_min_gain"] > MAX_GAIN_REQUIRED:
        return "threshold_above_cap"
    if not 1 <= len(eval_items) <= MAX_EVAL_ITEMS or not all(
            isinstance(e, dict) and _text(e.get("question")) and _text(e.get("answer"))
            for e in eval_items):
        return "malformed_inputs"
    if mode == "recipe":
        if not memories or not all(isinstance(m, dict) and _text(m.get("topic")) and _text(m.get("answer"))
                                   for m in memories):
            return "malformed_inputs"
    else:
        program = body.get("program")
        if not (isinstance(program, dict) and _text(program.get("entry"))
                and "/" not in program["entry"] and isinstance(program.get("bundle_sha256"), str)):
            return "malformed_inputs"
    return None


def required_gain(rule: dict, base_acc: float) -> float:
    """The gain a job has to show. The accuracy is capped at 1, so what the rule
    asks must fit in what the baseline left: otherwise the requester could pick
    questions the base model already answers, and no training could pass."""
    return min(rule["personal_qa_min_gain"], MAX_GAIN_REQUIRED * (1 - base_acc))


# init_config parameters (design notes v0.2, section 6.4). USDC has 6 decimals.
CHAIN_PARAMS = {
    "floor_bps": COMPUTE_FLOOR_BPS, "fault_bps": REQUESTER_FAULT_BPS,
    "min_bond_bps": BOND_RATIO_BPS,
    # Sellers can't tell these from income, so they only cover transaction fees.
    "reject_fee_bps": 100, "reject_fee_cap": 10_000,
    "no_release_fee_bps": 200, "no_release_fee_cap": 10_000,
    "nostart_slash_bps": 1000, "nostart_slash_min": 100_000,
    "timeout_slash_bps": LATE_SLASH_BPS,
    "offer_stake": 5_000_000,
    # MOCK, T1, T2, T3-B, T3-C
    "min_stake": [10_000_000, 20_000_000, 200_000_000, 10_000_000, 10_000_000],
    "settle_delay": 3600, "release_window": 600, "unstake_cooldown": 7 * 86_400,
    "clock_margin": 60, "max_start_window": 1800, "allowed_tiers": 0b11111,
    "decline_window": 60, "rdv_abort_after": 60, "abort_slash_bps": 1000,
}
# The local validator: short windows, so tests and demos can wait them out.
LOCAL_CHAIN_PARAMS = {**CHAIN_PARAMS, "settle_delay": 3, "release_window": 12,
                      "unstake_cooldown": 3, "clock_margin": 0, "decline_window": 6,
                      "rdv_abort_after": 4}
# MOCK evidence proves nothing, so mainnet doesn't allow that tier at all.
MAINNET_CHAIN_PARAMS = {**CHAIN_PARAMS, "allowed_tiers": 0b11110}
NETWORK_PARAMS = {"local": LOCAL_CHAIN_PARAMS, "devnet": CHAIN_PARAMS,
                  "mainnet": MAINNET_CHAIN_PARAMS}

# Defaults; each model overrides them in model_profiles.json.
RECIPE = {
    "method": "memory",
    "lora": {"rank": 16, "alpha": 32, "targets": "all-linear", "dropout": 0.0},
    "quant": "none",
    "optim": {"lr": 2e-3, "warmup_ratio": 0.05, "epochs": 4, "batch_size": 8,
              "weight_decay": 0.0},
    "max_seq_len": 128,
    "seed": 42,
    "checkpoint_every": 20,
}

ACCEPTANCE = {"personal_qa_min_gain": 0.5, "regression_max_loss_increase": 1.0}
BUDGET = {"max_train_tokens": 2_000_000, "max_wall_seconds": 1800}
