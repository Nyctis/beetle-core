"""The seller's node: `beetle host`. See node/daemon.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common"), str(ROOT / "host"), str(ROOT / "requester"),
                str(ROOT / "attester"), str(ROOT / "scripts"), str(ROOT / "sandbox")]
