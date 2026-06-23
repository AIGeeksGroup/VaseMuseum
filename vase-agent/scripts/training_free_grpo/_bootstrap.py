"""Put the vase-agent package root on ``sys.path`` so ``agent_run``, ``metrics``, etc. resolve."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_vase_agent_on_path() -> Path:
    """Walk upward from this file until ``agent_run.py`` is found."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "agent_run.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise RuntimeError(
        "Cannot locate vase-agent root (expected agent_run.py). "
        "Run from the vase-agent directory or add it to PYTHONPATH."
    )


ensure_vase_agent_on_path()
