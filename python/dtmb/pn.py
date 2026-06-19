"""DTMB PN frame-header sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

import numpy as np


PnMode = Literal["pn420", "pn595", "pn945"]


@dataclass(frozen=True)
class PnDefinition:
    """Definition of one DTMB frame-header PN mode."""

    mode: PnMode
    header_symbols: int
    degree: int
    phase0_seed_d_high_to_low: str
    recurrence_taps: tuple[int, ...]
    signal_frames_per_superframe: int
    header_to_body_power_ratio: float

    @property
    def frame_symbols(self) -> int:
        return self.header_symbols + 3780


PN_DEFINITIONS: dict[PnMode, PnDefinition] = {
    # Taps are expressed against the last `degree` generated chips. They are the
    # reciprocal-polynomial convention that matches GB 20600-2006 appendices D/E.
    "pn420": PnDefinition(
        mode="pn420",
        header_symbols=420,
        degree=8,
        phase0_seed_d_high_to_low="10110000",
        recurrence_taps=(0, 2, 3, 7),
        signal_frames_per_superframe=225,
        header_to_body_power_ratio=2.0,
    ),
    "pn595": PnDefinition(
        mode="pn595",
        header_symbols=595,
        degree=10,
        phase0_seed_d_high_to_low="0000000001",
        recurrence_taps=(0, 7),
        signal_frames_per_superframe=216,
        header_to_body_power_ratio=1.0,
    ),
    "pn945": PnDefinition(
        mode="pn945",
        header_symbols=945,
        degree=9,
        phase0_seed_d_high_to_low="111110111",
        recurrence_taps=(0, 1, 2, 7),
        signal_frames_per_superframe=200,
        header_to_body_power_ratio=2.0,
    ),
}


def pn_bits(mode: PnMode, *, seed_d_high_to_low: str | None = None) -> np.ndarray:
    """Return PN header bits for a DTMB frame-header mode.

    Bits use the standard binary convention before non-return-to-zero mapping:
    `0` maps to `+1`, and `1` maps to `-1`.
    """

    definition = PN_DEFINITIONS[mode]
    seed = seed_d_high_to_low or definition.phase0_seed_d_high_to_low
    return lfsr_bits(
        seed_d_high_to_low=seed,
        degree=definition.degree,
        recurrence_taps=definition.recurrence_taps,
        length=definition.header_symbols,
    )


def pn_bipolar(mode: PnMode, *, seed_d_high_to_low: str | None = None) -> np.ndarray:
    """Return PN header chips mapped as `0 -> +1` and `1 -> -1`."""

    bits = pn_bits(mode, seed_d_high_to_low=seed_d_high_to_low)
    return bits_to_bipolar(bits)


def pn_header_symbols(
    mode: PnMode,
    *,
    seed_d_high_to_low: str | None = None,
) -> np.ndarray:
    """Return PN chips with the transmitted equal-I/Q 4QAM mapping.

    GB 20600-2006 section 4.5.2 specifies that the frame header uses identical
    I and Q branches. Each bipolar PN chip is therefore transmitted as
    ``chip * (1 + 1j)``.
    """

    chips = pn_bipolar(mode, seed_d_high_to_low=seed_d_high_to_low)
    return (chips.astype(np.complex64) * np.complex64(1.0 + 1.0j)).astype(
        np.complex64,
        copy=False,
    )


def pn_header_symbols_for_body_power(mode: PnMode, *, body_power: float) -> np.ndarray:
    """Return a PN header scaled to the normative frame-header/body power ratio.

    GB 20600-2006 sections 4.6.2.1-4.6.2.3 specify header/body average-power
    ratios of 2, 1, and 2 for PN420, PN595, and PN945 respectively.
    """

    if body_power < 0.0:
        raise ValueError("body_power must be non-negative")
    symbols = pn_header_symbols(mode)
    symbol_power = float(np.mean(np.abs(symbols) ** 2))
    if symbol_power <= 0.0 or body_power == 0.0:
        return np.zeros_like(symbols)
    target_power = float(body_power) * PN_DEFINITIONS[mode].header_to_body_power_ratio
    scale = np.sqrt(target_power / symbol_power)
    return (symbols * np.float32(scale)).astype(np.complex64, copy=False)


@lru_cache(maxsize=None)
def pn_cyclic_family_bipolar(mode: PnMode) -> np.ndarray:
    """Return all cyclic PN420/PN945 header phases as bipolar rows."""

    if mode == "pn420":
        prefix_symbols = 82
        body_symbols = 255
        suffix_symbols = 83
    elif mode == "pn945":
        prefix_symbols = 217
        body_symbols = 511
        suffix_symbols = 217
    else:
        raise ValueError("cyclic PN phase families are only defined for PN420 and PN945")

    phase0 = pn_bits(mode)
    body = phase0[prefix_symbols : prefix_symbols + body_symbols]
    rows = []
    for phase in range(body_symbols):
        shifted = np.roll(body, -phase)
        bits = np.concatenate(
            (
                shifted[-prefix_symbols:],
                shifted,
                shifted[:suffix_symbols],
            )
        )
        rows.append(bits_to_bipolar(bits))
    return np.vstack(rows)


@lru_cache(maxsize=None)
def pn_cyclic_family_symbols(mode: PnMode) -> np.ndarray:
    """Return cyclic PN phases with the transmitted equal-I/Q mapping."""

    family = pn_cyclic_family_bipolar(mode).astype(np.complex64)
    return (family * np.complex64(1.0 + 1.0j)).astype(np.complex64, copy=False)


def bits_to_bipolar(bits: Iterable[int] | np.ndarray) -> np.ndarray:
    """Map binary PN bits to float32 bipolar chips."""

    array = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.uint8)
    return np.where(array == 0, 1.0, -1.0).astype(np.float32)


def lfsr_bits(
    *,
    seed_d_high_to_low: str,
    degree: int,
    recurrence_taps: tuple[int, ...],
    length: int,
) -> np.ndarray:
    """Generate bits from the DTMB appendix-compatible LFSR recurrence."""

    if len(seed_d_high_to_low) != degree:
        raise ValueError("seed length must match LFSR degree")
    if any(char not in "01" for char in seed_d_high_to_low):
        raise ValueError("seed must contain only '0' and '1'")
    if set(seed_d_high_to_low) == {"0"}:
        raise ValueError("all-zero LFSR seed is invalid")
    if length < 0:
        raise ValueError("length must be non-negative")
    if any(tap < 0 or tap >= degree for tap in recurrence_taps):
        raise ValueError("recurrence taps must be within the LFSR degree")

    sequence = [int(char) for char in seed_d_high_to_low]
    while len(sequence) < length:
        window_start = len(sequence) - degree
        next_bit = 0
        for tap in recurrence_taps:
            next_bit ^= sequence[window_start + tap]
        sequence.append(next_bit)
    return np.asarray(sequence[:length], dtype=np.uint8)


def bits_from_hex(hex_string: str, bit_count: int | None = None) -> np.ndarray:
    """Convert a hexadecimal appendix vector to bits, most significant bit first."""

    compact = "".join(hex_string.split())
    bits: list[int] = []
    for char in compact:
        value = int(char, 16)
        bits.extend((value >> shift) & 1 for shift in (3, 2, 1, 0))
    if bit_count is not None:
        bits = bits[:bit_count]
    return np.asarray(bits, dtype=np.uint8)
