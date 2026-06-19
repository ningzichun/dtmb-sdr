"""Helpers for HackRF-style signed interleaved 8-bit IQ files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Ci8FileInfo:
    """Basic facts about a CI8 file."""

    path: Path
    byte_count: int
    sample_count: int
    has_trailing_byte: bool


def inspect_ci8_file(path: str | Path) -> Ci8FileInfo:
    """Inspect a headerless signed interleaved 8-bit IQ file."""

    capture_path = Path(path)
    byte_count = capture_path.stat().st_size
    return Ci8FileInfo(
        path=capture_path,
        byte_count=byte_count,
        sample_count=byte_count // 2,
        has_trailing_byte=(byte_count % 2) == 1,
    )


def read_ci8(
    path: str | Path,
    max_samples: int | None = None,
    *,
    skip_samples: int = 0,
) -> np.ndarray:
    """Read CI8 data as normalized complex64 samples.

    HackRF raw receive files are signed interleaved 8-bit quadrature samples:
    I0, Q0, I1, Q1, ...

    Pass ``path='-'`` to read from ``sys.stdin`` so the receiver can operate
    on a live byte stream (for example, the output of a piped transmitter).
    """

    if skip_samples < 0:
        raise ValueError("skip_samples must be non-negative")
    max_bytes = -1 if max_samples is None else max_samples * 2
    target = str(path)
    if target == "-":
        import sys

        stdin_buffer = sys.stdin.buffer
        if skip_samples:
            remaining = skip_samples * 2
            while remaining > 0:
                chunk = stdin_buffer.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
        if max_bytes < 0:
            raw_bytes = stdin_buffer.read()
        else:
            raw_bytes = stdin_buffer.read(max_bytes)
        raw = np.frombuffer(raw_bytes, dtype=np.int8)
    else:
        capture_path = Path(target)
        with capture_path.open("rb") as handle:
            if skip_samples:
                handle.seek(skip_samples * 2)
            raw = np.fromfile(handle, dtype=np.int8, count=max_bytes)
    if raw.size % 2:
        raw = raw[:-1]

    iq = raw.reshape((-1, 2)).astype(np.float32) / 128.0
    return iq[:, 0] + 1j * iq[:, 1]
