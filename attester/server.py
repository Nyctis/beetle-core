"""HTTP front end for the attester (localhost only; hosts reach it over a tunnel).

A buyer can run this too: a TPM host then enrolls with the buyer's own instance,
and the buyer verifies that host's quotes locally from the same state file.

    POST /register             {job_id, spec_hash, sign_pk, enc_pk, host, evidence}
    POST /tpm/enroll-begin     {ek_cert, ak_public}            -> {enrollment_id, credential}
    POST /tpm/enroll-finish    {enrollment_id, secret}
    POST /app-attest/enroll    {key_id, attestation_object, challenge}
    GET  /info                 attester public key and accepted evidence types

Hex for keys and hashes, base64 for binary blobs.

Usage (from the beetle directory):
    uv run python attester/server.py --chain chain.json --tpm-ca-dir .keys/test-tpm-ca

The attester serves one network: every digest it checks or signs includes
that network's program id and cluster tag, exactly as the program computes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
import tpm_attest as tpm  # noqa: E402
from attester import Attester, Rejected, b64d, b64e  # noqa: E402
from job_defaults import sandbox_code_files  # noqa: E402

MAX_BODY = 1 << 20


def dev_policy(tpm_ca_dir: Path | None) -> dict:
    """Development policy: measurements come from the public sandbox source.

    The mock sandbox's measurement is the source hash. A TPM host proves the
    same image by extending PCR 23 with it during its locked boot; PCRs 0-7
    are zero on swtpm (a real policy lists the golden firmware values).
    """
    code = bytes.fromhex(bc.code_measurement(sandbox_code_files()))
    pcr23 = tpm.pcr_extend(bytes(32), code)
    return {
        "registration_ttl_seconds": 3600,
        "mock": {"allow": True, "allowed_measurements": [code.hex()]},
        "tdx": {"allowed_tcb_status": ["UpToDate"], "allowed_measurements": []},
        "app_attest": {"team_id": "TEAMID0000", "bundle_id": "ai.nyctis.beetle.worker",
                       "production": True, "require_sip_policy": True,
                       "allowed_measurements": []},
        "tpm": {"trusted_ek_ca_files": sorted(str(p) for p in tpm_ca_dir.glob("*.pem"))
                if tpm_ca_dir else [],
                "expected_pcrs": {**{str(i): bytes(32).hex() for i in range(8)}, "23": pcr23.hex()},
                "allowed_measurements": [pcr23.hex()]},
    }


def make_handler(attester: Attester):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # one line per request on stderr
            sys.stderr.write(f"[attester] {self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

        def do_GET(self):
            if self.path == "/info":
                return self._reply(200, {"attester_pk": attester.public_key.hex(),
                                         "mock_allowed": attester.policy["mock"]["allow"]})
            self._reply(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._reply(413, {"error": "request too large"})
            try:
                req = json.loads(self.rfile.read(n))
                if self.path == "/register":
                    reg = attester.register(
                        bytes.fromhex(req["job_id"]), bytes.fromhex(req["spec_hash"]),
                        bytes.fromhex(req["sign_pk"]), bytes.fromhex(req["enc_pk"]),
                        bytes.fromhex(req["host"]), req["evidence"])
                    return self._reply(200, reg.to_json())
                if self.path == "/tpm/enroll-begin":
                    eid, cred = attester.tpm_enroll_begin(b64d(req["ek_cert"]), b64d(req["ak_public"]))
                    return self._reply(200, {"enrollment_id": eid, "credential": b64e(cred)})
                if self.path == "/tpm/enroll-finish":
                    attester.tpm_enroll_finish(req["enrollment_id"], b64d(req["secret"]))
                    return self._reply(200, {"enrolled": req["enrollment_id"]})
                if self.path == "/app-attest/enroll":
                    did = attester.app_attest_enroll(b64d(req["key_id"]), b64d(req["attestation_object"]),
                                                     b64d(req["challenge"]))
                    return self._reply(200, {"enrolled": did})
                return self._reply(404, {"error": "not found"})
            except Rejected as e:
                return self._reply(403, {"error": str(e)})
            except (KeyError, ValueError, TypeError) as e:
                return self._reply(400, {"error": f"bad request: {type(e).__name__}"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--key", default=str(ROOT / ".keys/attester.key"))
    ap.add_argument("--state", default=str(ROOT / ".keys/attester_state.json"))
    ap.add_argument("--tpm-ca-dir", type=Path, default=None)
    ap.add_argument("--chain", type=Path, required=True, help="chain.json of the network served")
    args = ap.parse_args()
    chain = json.loads(args.chain.read_text())
    from solders.pubkey import Pubkey

    bc.configure(bytes(Pubkey.from_string(chain["program_id"])), chain["cluster"].encode())
    attester = Attester(Path(args.key), dev_policy(args.tpm_ca_dir), Path(args.state))
    print(f"[attester] public key {attester.public_key.hex()}  program {chain['program_id']}  "
          f"listening on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(attester)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
