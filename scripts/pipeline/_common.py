"""Common helpers shared by pipeline stage scripts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"

# Make sure `dtmb` is importable when stages are launched directly as
# scripts from `make`.
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def child_env() -> dict[str, str]:
    """Return a PYTHONPATH-augmented env for launching subcommands."""

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join((str(PYTHON_DIR), existing))
    else:
        env["PYTHONPATH"] = str(PYTHON_DIR)
    return env


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_capture_sidecar(ci8_path: Path) -> dict[str, Any]:
    """Return the CI8 sidecar metadata for the given capture.

    Stages treat the sidecar JSON as authoritative for sample rate and
    capture identity. Missing sidecars are a hard error because the entire
    pipeline assumes bounded captures recorded with `dtmb.capture`.
    """

    sidecar = ci8_path.with_suffix(ci8_path.suffix + ".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"capture sidecar missing: {sidecar}; record the CI8 with "
            "`python -m dtmb.capture` or `scripts/pipeline/capture_hackrf.py`"
        )
    metadata = read_json(sidecar)
    if "byte_count" not in metadata:
        if "bytes" in metadata:
            metadata["byte_count"] = int(metadata["bytes"])
        else:
            acquire_complete = metadata.get("acquire_complete")
            if isinstance(acquire_complete, dict) and "bytes_written" in acquire_complete:
                metadata["byte_count"] = int(acquire_complete["bytes_written"])
    return metadata
