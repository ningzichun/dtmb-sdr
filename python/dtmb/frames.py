"""DTMB signal-frame slicing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .pn import PN_DEFINITIONS, PnMode


@dataclass(frozen=True)
class SignalFrame:
    """One DTMB signal frame split into PN header and 3780-symbol body."""

    mode: PnMode
    start: int
    header: np.ndarray
    body: np.ndarray


def iter_signal_frames(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    max_frames: int | None = None,
) -> Iterator[SignalFrame]:
    """Yield complete DTMB signal frames from symbol-rate samples."""

    if phase_offset < 0:
        raise ValueError("phase_offset must be non-negative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    emitted = 0
    for start in range(
        phase_offset,
        signal.size - definition.frame_symbols + 1,
        definition.frame_symbols,
    ):
        header_start = start
        header_stop = header_start + definition.header_symbols
        body_stop = start + definition.frame_symbols
        yield SignalFrame(
            mode=mode,
            start=start,
            header=signal[header_start:header_stop],
            body=signal[header_stop:body_stop],
        )
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            break
