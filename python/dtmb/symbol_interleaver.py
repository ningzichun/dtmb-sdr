"""DTMB convolutional symbol interleaver helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


InterleaverMode = Literal["mode1", "mode2"]


@dataclass(frozen=True)
class SymbolInterleaverSpec:
    """Convolutional interleaver parameters."""

    mode: InterleaverMode
    branch_count: int
    delay_step: int

    @property
    def max_branch_delay(self) -> int:
        return (self.branch_count - 1) * self.delay_step

    @property
    def full_stream_latency_symbols(self) -> int:
        return self.max_branch_delay * self.branch_count


SYMBOL_INTERLEAVERS: dict[InterleaverMode, SymbolInterleaverSpec] = {
    "mode1": SymbolInterleaverSpec(mode="mode1", branch_count=52, delay_step=240),
    "mode2": SymbolInterleaverSpec(mode="mode2", branch_count=52, delay_step=720),
}


def convolutional_interleave(
    symbols: np.ndarray,
    *,
    mode: InterleaverMode,
    fill: complex = 0j,
    flush: bool = False,
    phase: int = 0,
) -> np.ndarray:
    """Apply the DTMB convolutional symbol interleaver."""

    spec = SYMBOL_INTERLEAVERS[mode]
    return _apply_convolutional_delays(
        symbols,
        branch_count=spec.branch_count,
        delay_step=spec.delay_step,
        complement=False,
        fill=fill,
        flush=flush,
        phase=phase,
    )


def convolutional_deinterleave(
    symbols: np.ndarray,
    *,
    mode: InterleaverMode,
    fill: complex = 0j,
    flush: bool = False,
    phase: int = 0,
) -> np.ndarray:
    """Apply the matching DTMB convolutional symbol deinterleaver."""

    spec = SYMBOL_INTERLEAVERS[mode]
    return _apply_convolutional_delays(
        symbols,
        branch_count=spec.branch_count,
        delay_step=spec.delay_step,
        complement=True,
        fill=fill,
        flush=flush,
        phase=phase,
    )


def convolutional_deinterleave_source_indices(
    symbol_count: int,
    *,
    mode: InterleaverMode,
    phase: int = 0,
    output_start: int = 0,
    after_latency_discard: bool = True,
) -> np.ndarray:
    """Map post-deinterleaver symbols to pre-deinterleaver input indices.

    GB 20600-2006 section 4.4.4 synchronizes the first symbol of each
    3744-symbol basic data block to interleaver branch 0. Since 3744 is an
    exact multiple of the 52 interleaver branches, receiver output symbol
    indices can be traced back to their physical pre-deinterleaver branch and
    frame by inverting the complement delays used by
    :func:`convolutional_deinterleave`.

    Returned indices are relative to the receiver input passed to the
    deinterleaver, not to the original transmitter-side LDPC symbol stream.
    Invalid leading symbols are reported as ``-1`` when
    ``after_latency_discard`` is false.
    """

    spec = SYMBOL_INTERLEAVERS[mode]
    return convolutional_deinterleave_source_indices_custom(
        symbol_count,
        branch_count=spec.branch_count,
        delay_step=spec.delay_step,
        phase=phase,
        output_start=output_start,
        after_latency_discard=after_latency_discard,
    )


def source_symbol_for_output_symbol(
    output_symbol_index: int,
    *,
    mode: InterleaverMode,
    phase: int = 0,
    after_latency_discard: bool = True,
) -> int:
    """Return the pre-deinterleaver input symbol for one receiver output symbol.

    By default ``output_symbol_index`` is relative to the post-latency payload
    stream, matching the symbols that feed QAM demapping and LDPC. Set
    ``after_latency_discard=False`` to trace raw deinterleaver output positions,
    where leading fill symbols can map to ``-1``.
    """

    index = int(output_symbol_index)
    if index < 0:
        raise ValueError("output_symbol_index must be non-negative")
    return int(
        convolutional_deinterleave_source_indices(
            1,
            mode=mode,
            phase=phase,
            output_start=index,
            after_latency_discard=after_latency_discard,
        )[0]
    )


def convolutional_interleave_custom(
    symbols: np.ndarray,
    *,
    branch_count: int,
    delay_step: int,
    fill: complex = 0j,
    flush: bool = False,
    phase: int = 0,
) -> np.ndarray:
    """Apply a custom convolutional interleaver for tests and experiments."""

    return _apply_convolutional_delays(
        symbols,
        branch_count=branch_count,
        delay_step=delay_step,
        complement=False,
        fill=fill,
        flush=flush,
        phase=phase,
    )


def convolutional_deinterleave_custom(
    symbols: np.ndarray,
    *,
    branch_count: int,
    delay_step: int,
    fill: complex = 0j,
    flush: bool = False,
    phase: int = 0,
) -> np.ndarray:
    """Apply a custom matching convolutional deinterleaver."""

    return _apply_convolutional_delays(
        symbols,
        branch_count=branch_count,
        delay_step=delay_step,
        complement=True,
        fill=fill,
        flush=flush,
        phase=phase,
    )


def convolutional_deinterleave_source_indices_custom(
    symbol_count: int,
    *,
    branch_count: int,
    delay_step: int,
    phase: int = 0,
    output_start: int = 0,
    after_latency_discard: bool = True,
) -> np.ndarray:
    """Custom-parameter variant of :func:`convolutional_deinterleave_source_indices`."""

    if branch_count <= 0:
        raise ValueError("branch_count must be positive")
    if delay_step < 0:
        raise ValueError("delay_step must be non-negative")
    if phase < 0 or phase >= branch_count:
        raise ValueError("phase must be in 0..branch_count-1")
    count = max(0, int(symbol_count))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    start_index = max(0, int(output_start))
    max_delay = (branch_count - 1) * delay_step
    latency = max_delay * branch_count
    output_indices = start_index + np.arange(count, dtype=np.int64)
    if after_latency_discard:
        output_indices = output_indices + int(latency)

    starts = output_indices % int(branch_count)
    branches = (starts + int(phase)) % int(branch_count)
    branch_delays = (int(branch_count) - 1 - branches) * int(delay_step)
    branch_offsets = (output_indices - starts) // int(branch_count)
    source_offsets = branch_offsets - branch_delays
    source_indices = starts + int(branch_count) * source_offsets
    source_indices[source_offsets < 0] = -1
    return source_indices.astype(np.int64, copy=False)


def _apply_convolutional_delays(
    symbols: np.ndarray,
    *,
    branch_count: int,
    delay_step: int,
    complement: bool,
    fill: complex,
    flush: bool,
    phase: int,
) -> np.ndarray:
    if branch_count <= 0:
        raise ValueError("branch_count must be positive")
    if delay_step < 0:
        raise ValueError("delay_step must be non-negative")
    if phase < 0 or phase >= branch_count:
        raise ValueError("phase must be in 0..branch_count-1")
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    max_delay = (branch_count - 1) * delay_step
    output_length = values.size + (max_delay * branch_count if flush else 0)
    output = np.empty(output_length, dtype=np.complex64)
    output[:] = np.complex64(fill)
    for branch in range(branch_count):
        start = (branch - phase) % branch_count
        branch_delay = (
            (branch_count - 1 - branch) * delay_step
            if complement
            else branch * delay_step
        )
        source = values[start::branch_count]
        target = output[start::branch_count]
        _delay_branch(source, target, branch_delay, fill=fill)
    return output


def _delay_branch(
    source: np.ndarray,
    target: np.ndarray,
    delay: int,
    *,
    fill: complex,
) -> None:
    if delay == 0:
        count = min(source.size, target.size)
        target[:count] = source[:count]
        if target.size > count:
            target[count:] = np.complex64(fill)
        return
    leading = min(delay, target.size)
    target[:leading] = np.complex64(fill)
    available = min(source.size, max(0, target.size - delay))
    if available:
        target[delay : delay + available] = source[:available]
    tail_start = delay + available
    if tail_start < target.size:
        target[tail_start:] = np.complex64(fill)
