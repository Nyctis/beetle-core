"""What the sandbox does with a job it can't trust, without a GPU or a container:
torch and the trainer are stood in for, the rules around them are the real ones.
The real thing is `host/run_local.py`.

Run from the beetle directory:  uv run --with pytest pytest -q tests/test_sandbox_rules.py
"""

from __future__ import annotations

import gzip
import importlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "sandbox")]

import beetle_crypto as bc  # noqa: E402
import job_defaults as jd  # noqa: E402
from beetle_archive import read_tar, tar_bytes  # noqa: E402


class Adapter:
    """The trained model, as far as this file cares."""

    device = "cpu"

    def __init__(self, accuracy=0.9, loss=1.0):
        self.accuracy, self.loss = accuracy, loss

    def eval(self):
        return self


class FakeTrainer(ModuleType):
    """train_eval, with a model that scores what the test says it scores."""

    REGRESSION_SET = [("q", "a")]
    BadRecipe = type("BadRecipe", (Exception,), {})

    def __init__(self):
        super().__init__("train_eval")
        self.base = Adapter(accuracy=0.1)
        self.trained = Adapter()
        self.trained_tokens = 1000
        self.calls = []
        self.fail_add_lora = False

    def load_base(self, _):
        return SimpleNamespace(eos_token="</s>"), self.base

    def qa_accuracy(self, model, tok, items):
        return model.accuracy

    def answer_loss(self, model, tok, pairs, max_len):
        return model.loss

    def augment(self, memories):
        return [("q", m["answer"]) for m in memories]

    def encode(self, tok, q, a, max_len):
        return [0] * 50, [0] * 50

    def train_lora(self, model, tok, pairs, recipe, budget, log):
        if self.fail_add_lora:
            raise self.BadRecipe("ValueError")
        self.calls.append(budget)
        log("epoch 1")
        return self.trained, {"steps": 1, "train_tokens": min(self.trained_tokens, budget["max_train_tokens"]),
                              "loss_log": [1.0], "checkpoint_root": bytes(32), "wall_seconds": 1.0}

    def adapter_bundle(self, model):
        return tar_bytes({"adapter_model.safetensors": b"w"})


class Sealed:
    """A sandbox on a temporary directory, and the buyer who wrote the job."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.trainer = FakeTrainer()
        for name, module in (("torch", SimpleNamespace(set_num_threads=lambda n: None, get_num_threads=lambda: 1)),
                             ("train_eval", self.trainer)):
            monkeypatch.setitem(sys.modules, name, module)
        self.en = importlib.import_module("enclave")
        self.inbox, self.outbox = tmp_path / "in", tmp_path / "out"
        self.inbox.mkdir()
        self.outbox.mkdir()
        (tmp_path / "app" / "sdk").mkdir(parents=True)
        (tmp_path / "app" / "x.py").write_text("x")
        (tmp_path / "app" / "requirements.txt").write_text("x")
        (tmp_path / "model").mkdir()
        (tmp_path / "model" / "w").write_text("w")
        for name, value in (("APP", tmp_path / "app"), ("INBOX", self.inbox), ("OUTBOX", self.outbox),
                            ("MODEL", tmp_path / "model")):
            monkeypatch.setattr(self.en, name, value)
        monkeypatch.setattr(self.en, "network_check", lambda: {"isolated": True})
        self.buyer = bc.HpkeKeyPair()
        self.dek, self.job_id = bc.new_key(), os.urandom(32)
        self.terms = {"units": 1000, "price_mode": jd.PRICE_BY_TOKEN, "run_secs": 2000, "grace_s": 600}

    def body(self, **over) -> dict:
        return {"mode": "recipe", "acceptance": {"personal_qa_min_gain": 0.5, "regression_max_loss_increase": 1.0},
                "recipe": {**jd.RECIPE}, "program": None, "parent_adapter": None, **over}

    def job(self, body=None, memories=None, eval_items=None, program=b"", budget=None, raw=None, enc_pk=None):
        plain = {"body": bc.canonical_json(self.body() if body is None else body),
                 "train": bc.canonical_json([{"topic": "t", "answer": "a"}] if memories is None else memories),
                 "eval": bc.canonical_json([{"question": "q", "answer": "a"}] if eval_items is None else eval_items),
                 "program": program or tar_bytes({})}
        plain.update(raw or {})
        hashes = {}
        for name, data in plain.items():
            path = self.inbox / f"{name}.enc"
            path.write_bytes(bc.seal_blob(self.dek, data, self.job_id + name.encode()))
            hashes[name] = bc.file_sha256(path)
        spec = {"budget": budget or jd.fit_budget(self.terms["units"], self.terms["run_secs"]), "inputs": hashes,
                "base_model": {"manifest_sha256": bc.dir_manifest_sha256(self.tmp / "model")},
                "requester_enc_pk": (enc_pk or self.buyer.public_bytes).hex(),
                "chain": {"program_id_hex": "00" * 32, "cluster": "local"}}
        (self.inbox / "spec.json").write_text(json.dumps(spec))
        spec_hash = bc.sha256(bc.canonical_json(spec)).hex()
        (self.inbox / "job.json").write_text(json.dumps(
            {"job_id": self.job_id.hex(), "spec_hash": spec_hash, **self.terms}))

    def run(self, deliver_key=True) -> int:
        self.mp.setattr(sys, "exit", lambda *_: None)
        if deliver_key:
            import threading

            def put():
                while not (self.outbox / "attestation.json").exists():
                    time.sleep(0.01)
                att = json.loads((self.outbox / "attestation.json").read_text())
                (self.inbox / "sealed_dek.bin").write_bytes(bc.hpke_seal(
                    bytes.fromhex(att["enc_pk"]), self.dek, info=b"dek" + self.job_id))
            threading.Thread(target=put, daemon=True).start()
        return self.en.main()

    @property
    def report(self) -> dict:
        return json.loads((self.outbox / "report.json").read_text())

    @property
    def error(self) -> str:
        return json.loads((self.outbox / "error.json").read_text())["reason"]

    @property
    def private(self) -> dict:
        return json.loads(self.buyer.open_blob((self.outbox / "private_report.enc").read_bytes(),
                                               b"private-report" + self.job_id))


@pytest.fixture
def box(tmp_path, monkeypatch):
    bc.configure(bc.PROGRAM_ID, bc.CLUSTER)
    yield Sealed(tmp_path, monkeypatch)
    sys.modules.pop("enclave", None)
    sys.modules.pop("program_runner", None)
    bc.configure(b"Beetle111111111111111111111111111111111111", b"local")


# --- F1: what was paid for -----------------------------------------------------------


def test_an_honest_job_passes_and_signs_the_terms(box):
    box.job()
    assert box.run() == 0
    r = box.report
    assert r["outcome"] == jd.OUTCOME_PASSED and r["sealed_adapter_key"]
    assert {k: r[k] for k in box.terms} == box.terms


def test_outputs_are_padded_to_a_tier(box):
    box.job()
    box.run()
    assert (box.outbox / "adapter.enc").stat().st_size == bc.ADAPTER_TIERS[0] + 28
    assert (box.outbox / "private_report.enc").stat().st_size == bc.REPORT_BUCKET + 48


def test_a_spec_above_the_order_never_reaches_the_key(box):
    box.job(budget={"max_train_tokens": 2_000_000, "max_wall_seconds": 800})
    assert box.run(deliver_key=False) == 3
    assert box.error == "budget_not_paid_for"
    assert not (box.outbox / "attestation.json").exists() and not (box.outbox / "report.json").exists()


def test_a_wall_clock_the_run_budget_cant_cover_is_refused(box):
    box.job(budget={"max_train_tokens": 1000, "max_wall_seconds": box.terms["run_secs"]})
    assert box.run(deliver_key=False) == 3
    assert box.error == "budget_not_paid_for"


def test_a_chain_job_needs_the_chains_terms(box):
    box.job()
    job = json.loads((box.inbox / "job.json").read_text())
    (box.inbox / "job.json").write_text(json.dumps({k: job[k] for k in ("job_id", "spec_hash")}))
    assert box.run(deliver_key=False) == 3
    assert box.error == "job_terms_missing"


def test_a_key_that_cant_be_sealed_to_is_refused_before_any_work(box):
    box.job(enc_pk=bytes(32))
    assert box.run(deliver_key=False) == 3
    assert box.error == "spec_malformed"


def test_training_never_goes_past_the_tokens_in_the_budget(box):
    box.trainer.trained_tokens = 10**9
    box.job(budget={"max_train_tokens": 700, "max_wall_seconds": 1000})
    assert box.run() == 0
    assert box.trainer.calls[0]["max_train_tokens"] == 700
    assert box.private["training"]["train_tokens"] == 700


def test_out_of_time_is_reported_before_the_deadline_not_left_to_the_slash(box):
    box.terms["run_secs"] = jd.SUBMIT_RESERVE_S + 1
    box.terms["grace_s"] = 0
    def slow(*_):
        time.sleep(30)
    box.trainer.qa_accuracy = slow
    box.job(budget={"max_train_tokens": 1000, "max_wall_seconds": 1})
    box.mp.setattr(jd, "RUN_RESERVE_S", 0)
    t0 = time.monotonic()
    assert box.run() == 3
    assert time.monotonic() - t0 < 10
    assert box.error == "run_budget_exhausted"
    assert box.report["outcome"] == jd.OUTCOME_INCONCLUSIVE and box.report["sealed_adapter_key"] is None
    assert box.report["sealed_dek_hash"]


# --- H4: the buyer doesn't set the bar for itself ------------------------------------


@pytest.mark.parametrize("gain", [0.91, 1, 9e9, float("inf")])
def test_the_cap_on_the_threshold_is_not_the_buyers(box, gain):
    rule = {"personal_qa_min_gain": gain, "regression_max_loss_increase": 1.0, "max_gain_required": 9e9}
    box.job(body=box.body(acceptance=rule))
    assert box.run() == 3
    assert box.error == "threshold_above_cap"
    assert box.report["outcome"] == jd.OUTCOME_REQUESTER_FAULT


@pytest.mark.parametrize("over", [
    {"acceptance": {"personal_qa_min_gain": -1}},                                    # found only after training
    {"acceptance": {"personal_qa_min_gain": 0.5, "regression_max_loss_increase": -1}},
    {"acceptance": {"personal_qa_min_gain": float("nan"), "regression_max_loss_increase": 1}},
    {"acceptance": None},
    {"recipe": None},
    {"recipe": {**jd.RECIPE, "max_seq_len": 10**7}},
    {"recipe": {**jd.RECIPE, "optim": {**jd.RECIPE["optim"], "batch_size": 10**6}}},
    {"recipe": {**jd.RECIPE, "lora": {**jd.RECIPE["lora"], "rank": True}}},
    {"recipe": {k: v for k, v in jd.RECIPE.items() if k != "seed"}},
    {"mode": "program", "program": None},
    {"mode": "program", "program": {"entry": "../x.py", "bundle_sha256": "00"}},
])
def test_anything_malformed_is_the_buyers_fault_before_any_training(box, over):
    box.job(body=box.body(**over))
    assert box.run() == 3
    assert box.error == "malformed_inputs"
    assert box.report["outcome"] == jd.OUTCOME_REQUESTER_FAULT and not box.trainer.calls


@pytest.mark.parametrize("raw", [
    {"body": b"[" * 200_000},   # a RecursionError, not a ValueError
    {"body": b"not json"},
    {"body": b"[1]"},
    {"train": b'{"a": 1}'},
    {"train": b'[{"topic": 1}]'},
    {"eval": b"[]"},
    {"eval": b'[{"question": "q", "answer": ""}]'},
    {"eval": bc.canonical_json([{"question": "q", "answer": "a"}] * (jd.MAX_EVAL_ITEMS + 1))},
    {"train": b"[" + b" " * (jd.MAX_TRAIN_BYTES + 1) + b"]"},
])
def test_inputs_that_dont_parse_or_fit_are_the_buyers_fault(box, raw):
    box.job(raw=raw)
    assert box.run() == 3
    assert box.error == "malformed_inputs"
    assert box.report["outcome"] == jd.OUTCOME_REQUESTER_FAULT


def test_more_data_than_the_order_could_pay_for(box):
    box.job(memories=[{"topic": "t", "answer": "a" * 60_000}] * 2)
    assert box.run() == 3
    assert box.error == "malformed_inputs"
    assert jd.train_bytes_limit({"price_mode": jd.PRICE_BY_TOKEN, "units": 10**12}) == jd.MAX_TRAIN_BYTES
    assert jd.train_bytes_limit({"price_mode": jd.PRICE_BY_TIME, "units": 1}) == jd.MAX_TRAIN_BYTES


def test_a_mode_nobody_knows(box):
    box.job(body=box.body(mode="magic"))
    assert box.run() == 3
    assert box.error == "unknown_mode"


def test_a_recipe_that_doesnt_fit_the_model_is_the_buyers_fault(box):
    box.trainer.fail_add_lora = True
    box.job()
    assert box.run() == 3
    assert box.error == "malformed_inputs"


def test_questions_the_base_model_already_answers(box):
    box.trainer.base = Adapter(accuracy=1.0)
    box.job()
    assert box.run() == 3
    assert box.error == "baseline_already_passes"
    assert box.report["outcome"] == jd.OUTCOME_REQUESTER_FAULT and not box.trainer.calls


def test_the_gain_asked_for_fits_what_the_baseline_left():
    rule = {"personal_qa_min_gain": 0.9}
    assert jd.required_gain(rule, 0.0) == 0.9
    assert jd.required_gain(rule, 0.5) == pytest.approx(0.45)
    assert jd.required_gain({"personal_qa_min_gain": -1}, 0.5) == -1


def test_a_baseline_that_leaves_little_can_still_pass(box):
    box.trainer.base = Adapter(accuracy=0.8)
    box.trainer.trained = Adapter(accuracy=1.0)
    rule = {"personal_qa_min_gain": 0.9, "regression_max_loss_increase": 1.0}
    box.job(body=box.body(acceptance=rule))
    assert box.run() == 0
    assert box.report["outcome"] == jd.OUTCOME_PASSED


def test_a_failed_run_pays_the_floor_for_the_work(box):
    box.trainer.trained = Adapter(accuracy=0.2)
    box.trainer.trained_tokens = 500
    box.job(body=box.body())
    assert box.run() == 0
    r = box.report
    # planned is the tokens in the data, here 50, and the budget caps it
    assert r["outcome"] == jd.OUTCOME_FAILED and r["work_bps"] == 10_000 and r["sealed_adapter_key"] is None


def test_a_wall_limit_below_what_was_paid_for_is_the_buyers(box):
    box.trainer.trained = Adapter(accuracy=0.2)
    def stopped(model, tok, pairs, recipe, budget, log):
        return box.trainer.trained, {"steps": 1, "train_tokens": 25, "loss_log": [1.0],
                                     "checkpoint_root": bytes(32), "wall_seconds": budget["max_wall_seconds"] + 0.1}
    box.trainer.train_lora = stopped
    box.job(budget={"max_train_tokens": 1000, "max_wall_seconds": 1})
    assert box.run() == 0
    # 25 of the 200 planned: 50 tokens of data over the recipe's 4 epochs
    assert (box.report["outcome"], box.report["work_bps"]) == (jd.OUTCOME_REQUESTER_FAULT, 1250)


def test_a_slow_machine_inside_the_paid_time_is_not_the_buyers(box):
    box.trainer.trained = Adapter(accuracy=0.2)
    def stopped(model, tok, pairs, recipe, budget, log):
        return box.trainer.trained, {"steps": 1, "train_tokens": 25, "loss_log": [1.0],
                                     "checkpoint_root": bytes(32), "wall_seconds": budget["max_wall_seconds"] + 0.1}
    box.trainer.train_lora = stopped
    box.job()
    assert box.run() == 0
    assert box.report["outcome"] == jd.OUTCOME_INCONCLUSIVE


# --- H5: a program's outputs are paid for --------------------------------------------


class FakeRunner(ModuleType):
    PROGRAM_FAULTS = {"wall_budget", "request_too_large", "child_stopped_responding"}

    def __init__(self, trainer, **stats):
        super().__init__("program_runner")
        self.trainer, self.stats = trainer, stats

    def Session(self, model, tok, recipe, memories, max_tokens):
        self.max_tokens = max_tokens
        return SimpleNamespace(model=self.trainer.trained)

    def run_program(self, session, files, entry, wall):
        self.files, self.wall = files, wall
        return {"exit_code": 0, "stop_reason": None, "output": "SECRET OUTPUT", "log": ["SECRET LOG"],
                "steps": 2, "train_tokens": 100, "loss_log": [0.123], "checkpoint_root": bytes(32),
                "wall_seconds": 5.0, **self.stats}


def program_job(box, trained=0.2, entry="train.py", **stats):
    runner = FakeRunner(box.trainer, **stats)
    box.mp.setitem(sys.modules, "program_runner", runner)
    box.trainer.trained = Adapter(accuracy=trained)
    blob = tar_bytes({"train.py": b"print(1)"})
    box.job(body=box.body(mode="program", program={"entry": entry, "bundle_sha256": bc.sha256(blob).hex()}),
            program=blob)
    return runner


def test_a_program_that_passes_gets_its_outputs(box):
    program_job(box, trained=0.9)
    assert box.run() == 0
    p = box.private
    assert p["program"]["output"] == "SECRET OUTPUT" and p["log"] == ["SECRET LOG"]
    assert p["training"]["loss_log"] == [0.123]


@pytest.mark.parametrize("stats", [{}, {"stop_reason": "wall_budget", "exit_code": -9}, {"exit_code": 1},
                                   {"stop_reason": "token_budget", "exit_code": -9}])
def test_a_program_that_doesnt_pass_gets_counters_not_content(box, stats):
    program_job(box, **stats)
    assert box.run() == 0
    assert b"SECRET" not in bc.canonical_json(box.private)
    assert box.private["training"]["loss_log"] == []
    assert box.report["sealed_adapter_key"] is None


def test_the_program_pays_against_what_was_paid_not_its_own_budget(box):
    box.terms.update(units=1000)
    program_job(box)
    assert box.run() == 0
    assert box.report["work_bps"] == 1000  # 100 of the 1,000 tokens paid for


def test_a_program_cant_shrink_its_bill_with_a_huge_budget_on_a_by_time_job(box):
    box.terms.update(units=100, price_mode=jd.PRICE_BY_TIME)
    program_job(box, wall_seconds=50.0, train_tokens=3)
    box.job(body=json.loads(box.en.bc.canonical_json(box.body(
        mode="program", program={"entry": "train.py", "bundle_sha256": bc.sha256(tar_bytes({"train.py": b"print(1)"})).hex()}))),
        program=tar_bytes({"train.py": b"print(1)"}),
        budget={"max_train_tokens": 10**12, "max_wall_seconds": 1000})
    assert box.run() == 0
    assert box.report["work_bps"] == 5000


@pytest.mark.parametrize("stats,outcome", [
    ({"stop_reason": "wall_budget", "exit_code": -9}, jd.OUTCOME_REQUESTER_FAULT),
    ({"stop_reason": "child_stopped_responding", "exit_code": -9}, jd.OUTCOME_REQUESTER_FAULT),
    ({"stop_reason": "request_too_large", "exit_code": -9}, jd.OUTCOME_REQUESTER_FAULT),
    ({"exit_code": 1}, jd.OUTCOME_REQUESTER_FAULT),
    ({"stop_reason": "token_budget", "exit_code": -9}, jd.OUTCOME_FAILED),
    ({}, jd.OUTCOME_FAILED),
])
def test_who_stopped_a_program_decides_the_outcome(box, stats, outcome):
    program_job(box, **stats)
    assert box.run() == 0
    assert box.report["outcome"] == outcome


def test_a_bundle_without_its_entry(box):
    program_job(box, entry="missing.py")
    assert box.run() == 3
    assert box.error == "malformed_inputs"


def test_a_file_name_no_disk_takes(box):
    blob = tar_bytes({"train.py": b"1", "n" * 300: b"2"})
    box.job(body=box.body(mode="program", program={"entry": "train.py", "bundle_sha256": bc.sha256(blob).hex()}),
            program=blob)
    assert box.run() == 3
    assert box.error == "malformed_inputs"


def test_a_compression_bomb_is_the_buyers_fault(box):
    bomb = gzip.compress(tar_bytes({"train.py": bytes(64 << 20)}))
    assert len(bomb) < jd.MAX_PROGRAM_BYTES
    box.job(body=box.body(mode="program", program={"entry": "train.py", "bundle_sha256": bc.sha256(bomb).hex()}),
            program=bomb)
    assert box.run() == 3
    assert box.error == "malformed_inputs"
    assert box.report["outcome"] == jd.OUTCOME_REQUESTER_FAULT


def test_a_bundle_too_big_to_be_one(box):
    blob = tar_bytes({"train.py": bytes(jd.MAX_PROGRAM_BYTES)})
    box.job(body=box.body(mode="program", program={"entry": "train.py", "bundle_sha256": bc.sha256(blob).hex()}),
            program=blob)
    assert box.run() == 3
    assert box.error == "malformed_inputs"


# --- H1: a program can't hold the sandbox ---------------------------------------------


class Model(SimpleNamespace):
    device = "cpu"
    calls = 0

    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, **_):
        Model.calls += 1
        return SimpleNamespace(loss=SimpleNamespace(item=lambda: 1.0, backward=lambda: None))


LAUNCHER = """import os, sys
print("launcher: seccomp on (test)", flush=True)
exec(compile(open(sys.argv[2]).read(), sys.argv[2], "exec"))
"""
CHILD = """import json, os, sys, time
w, r = (int(x) for x in os.environ["BEETLE_RPC"].split(","))
def call(method, **params):
    os.write(w, json.dumps({"method": method, "params": params}).encode() + b"\\n")
"""


@pytest.fixture
def runner(tmp_path, monkeypatch):
    trainer = FakeTrainer()
    trainer.add_lora = lambda model, recipe: model
    trainer.encode = lambda tok, p, r, n: ([0] * 10, [0] * 10)
    trainer.collate = lambda tok, batch, device: (0, 0, SimpleNamespace(sum=lambda: 10 * len(batch)))
    trainer.lora_state_hash = lambda model: bytes(32)
    trainer.merkle_root = lambda leaves: bytes(32)
    torch = SimpleNamespace(manual_seed=lambda s: None, optim=SimpleNamespace(AdamW=lambda *a, **k: SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "train_eval", trainer)
    pr = importlib.import_module("program_runner")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(LAUNCHER)
    monkeypatch.setattr(pr, "LAUNCHER", launcher)
    monkeypatch.setattr(pr, "tempfile", SimpleNamespace(mkdtemp=lambda **_: str(tmp_path / "work")))
    (tmp_path / "work").mkdir()
    Model.calls = 0

    def run(program: str, wall: float = 1, memories=(), max_tokens: int = 1000) -> dict:
        session = pr.Session(Model(), None, {**jd.RECIPE}, list(memories), max_tokens)
        t0 = time.monotonic()
        stats = pr.run_program(session, {"train.py": (CHILD + program).encode()}, "train.py", wall)
        return {**stats, "took": time.monotonic() - t0, "session": session}

    yield SimpleNamespace(run=run, pr=pr)
    sys.modules.pop("program_runner", None)


def test_a_program_that_stops_reading_cant_outlast_the_wall_clock(runner):
    """The answer is bigger than a pipe, so a blocking write would wait forever."""
    out = runner.run('call("dataset")\ntime.sleep(60)\n', memories=["x" * 1000] * 1000)
    assert out["stop_reason"] == "child_stopped_responding" and out["took"] < 5


def test_a_program_that_just_sleeps_is_stopped(runner):
    out = runner.run("time.sleep(60)\n")
    assert out["stop_reason"] == "wall_budget" and out["took"] < 5
    assert out["stop_reason"] in runner.pr.PROGRAM_FAULTS


def test_a_queue_of_requests_cant_outlast_the_wall_clock(runner):
    out = runner.run('os.write(w, b\'{"method": "log", "params": {"msg": "x"}}\\n\' * 100000)\ntime.sleep(60)\n', wall=0.2)
    assert out["stop_reason"] in ("wall_budget", "child_stopped_responding") and out["took"] < 5


def test_tokens_are_charged_before_the_work(runner):
    out = runner.run('import itertools\nfor _ in itertools.count():\n'
                     '    call("forward_backward", pairs=[["p", "r"]])\n    os.read(r, 4096)\n',
                     wall=10, max_tokens=25)
    assert out["stop_reason"] == "token_budget"
    assert out["train_tokens"] == 20 and Model.calls == 2  # the third batch never ran


def test_a_crash_is_not_a_stop_reason_but_counts_as_one(runner):
    out = runner.run("sys.exit(3)\n")
    assert out["stop_reason"] is None and out["exit_code"] == 3


def test_a_failed_lockdown_is_not_the_requesters_fault(runner, tmp_path):
    (tmp_path / "launcher.py").write_text("import os\nos._exit(97)\n")
    with pytest.raises(RuntimeError):
        runner.run("pass\n")


def test_a_program_cant_fake_a_failed_lockdown(runner):
    out = runner.run("os._exit(97)\n")
    assert out["exit_code"] == 97


# --- H1: archives ------------------------------------------------------------------


def test_read_tar_takes_a_plain_flat_archive():
    assert read_tar(tar_bytes({"a": b"1", "b": b"2"})) == {"a": b"1", "b": b"2"}
    assert read_tar(tar_bytes({})) == {}


def test_read_tar_refuses_compression():
    with pytest.raises(tarfile.ReadError):
        read_tar(gzip.compress(tar_bytes({"a": bytes(1 << 20)})))


def test_read_tar_limits_size_and_count_before_reading():
    with pytest.raises(ValueError, match="too large"):
        read_tar(tar_bytes({"a": bytes(2000)}), max_bytes=1999)
    with pytest.raises(ValueError, match="too large"):
        read_tar(tar_bytes({str(i): b"" for i in range(5)}), max_files=4)


def test_read_tar_refuses_sparse_members():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        info = tarfile.TarInfo("big")
        info.type = tarfile.GNUTYPE_SPARSE
        info.size = 0
        tar.addfile(info)
    with pytest.raises((ValueError, tarfile.TarError)):
        read_tar(buf.getvalue())
