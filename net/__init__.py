"""Beetle's network layer: direct links, the inner channel and file transfer."""

import sys
from pathlib import Path

# The shared crypto lives in common/ and is imported as a top-level module
# everywhere else in the repo, so it must not load twice under another name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
