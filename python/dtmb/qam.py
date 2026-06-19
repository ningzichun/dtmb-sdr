"""DTMB QAM constellation mapping and hard/soft demapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


QamMode = Literal["4qam", "16qam", "32qam", "64qam"]


@dataclass(frozen=True)
class QamDefinition:
    """DTMB QAM definition.

    Rectangular modes use ``levels`` for identical I/Q axes. 32QAM uses the
    explicit cross-constellation labels from GB 20600-2006 Figure 4.
    """

    mode: QamMode
    bits_per_symbol: int
    levels: np.ndarray | None = None
    points_by_index: np.ndarray | None = None

    @property
    def axis_bits(self) -> int:
        if self.levels is None or self.bits_per_symbol % 2:
            raise ValueError(f"{self.mode} does not have rectangular QAM axes")
        return self.bits_per_symbol // 2

    @property
    def average_power(self) -> float:
        if self.points_by_index is not None:
            points = self.points_by_index.astype(np.complex128, copy=False)
            return float(np.mean(np.abs(points) ** 2))
        if self.levels is None:
            raise ValueError(f"{self.mode} does not define constellation points")
        return float(2.0 * np.mean(self.levels.astype(np.float64) ** 2))


def _make_32qam_points_by_index() -> np.ndarray:
    """Return GB 20600-2006 Figure 4 32QAM points indexed by b0-first code."""

    figure_points = (
        ("10111", -4.5, 7.5),
        ("10011", -1.5, 7.5),
        ("11011", 1.5, 7.5),
        ("11111", 4.5, 7.5),
        ("10010", -7.5, 4.5),
        ("00111", -4.5, 4.5),
        ("00011", -1.5, 4.5),
        ("01011", 1.5, 4.5),
        ("01111", 4.5, 4.5),
        ("11010", 7.5, 4.5),
        ("10110", -7.5, 1.5),
        ("00110", -4.5, 1.5),
        ("00010", -1.5, 1.5),
        ("01010", 1.5, 1.5),
        ("01110", 4.5, 1.5),
        ("11110", 7.5, 1.5),
        ("10100", -7.5, -1.5),
        ("00100", -4.5, -1.5),
        ("00000", -1.5, -1.5),
        ("01000", 1.5, -1.5),
        ("01100", 4.5, -1.5),
        ("11100", 7.5, -1.5),
        ("10000", -7.5, -4.5),
        ("00101", -4.5, -4.5),
        ("00001", -1.5, -4.5),
        ("01001", 1.5, -4.5),
        ("01101", 4.5, -4.5),
        ("11000", 7.5, -4.5),
        ("10101", -4.5, -7.5),
        ("10001", -1.5, -7.5),
        ("11001", 1.5, -7.5),
        ("11101", 4.5, -7.5),
    )
    points = np.empty(32, dtype=np.complex64)
    seen = np.zeros(32, dtype=np.bool_)
    for label, i_value, q_value in figure_points:
        index = sum(int(bit) << offset for offset, bit in enumerate(label))
        points[index] = np.complex64(i_value + 1j * q_value)
        seen[index] = True
    if not bool(np.all(seen)):  # pragma: no cover - import-time table guard
        raise RuntimeError("incomplete 32QAM Figure 4 mapping table")
    return points


QAM_DEFINITIONS: dict[QamMode, QamDefinition] = {
    "4qam": QamDefinition(
        mode="4qam",
        bits_per_symbol=2,
        levels=np.asarray([-4.5, 4.5], dtype=np.float32),
    ),
    "16qam": QamDefinition(
        mode="16qam",
        bits_per_symbol=4,
        levels=np.asarray([-6.0, -2.0, 2.0, 6.0], dtype=np.float32),
    ),
    "32qam": QamDefinition(
        mode="32qam",
        bits_per_symbol=5,
        points_by_index=_make_32qam_points_by_index(),
    ),
    "64qam": QamDefinition(
        mode="64qam",
        bits_per_symbol=6,
        levels=np.asarray([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32),
    ),
}


def qam_modulate(bits: np.ndarray, *, mode: QamMode) -> np.ndarray:
    """Map bits to DTMB rectangular QAM symbols.

    Per GB 20600-2006 section 4.4.3 the first FEC-encoded bit to enter the
    mapper becomes the LSB (``b0``) of the symbol codeword. The mapping figures
    then define the constellation labels. For 64QAM the I-axis labels from
    ``-7`` to ``+7`` are ``000,001,011,010,110,111,101,100``; 16QAM uses the
    same Gray sequence truncated to two bits. 32QAM is the explicit Figure 4
    cross constellation rather than a rectangular axis product.
    """

    definition = QAM_DEFINITIONS[mode]
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    if values.size % definition.bits_per_symbol:
        raise ValueError("bit count is not a whole number of QAM symbols")
    grouped = values.reshape(-1, definition.bits_per_symbol)
    if definition.points_by_index is not None:
        indices = _bits_to_indices(grouped)
        return definition.points_by_index[indices].astype(np.complex64, copy=False)
    if definition.levels is None:
        raise ValueError(f"{mode} does not define rectangular QAM levels")
    axis_bits = definition.axis_bits
    # axis_bits=1 (4QAM): bits per symbol are [b0, b1]; I=b0, Q=b1.
    # axis_bits=2/3: each axis code is carried LSB-first, but the constellation
    # figure labels the physical levels in reflected Gray-code order. Convert
    # the axis code to the binary level index before indexing the level table.
    i_index = _axis_bits_to_level_index(grouped[:, :axis_bits])
    q_index = _axis_bits_to_level_index(grouped[:, axis_bits : 2 * axis_bits])
    return (
        definition.levels[i_index] + 1j * definition.levels[q_index]
    ).astype(np.complex64)


def qam_hard_demodulate(
    symbols: np.ndarray,
    *,
    mode: QamMode,
    normalize: bool = True,
) -> np.ndarray:
    """Slice QAM symbols and return hard bits in spec bit order (b0 first).

    Output layout mirrors :func:`qam_modulate`: each output group contains
    ``(b0, b1, ..., bN)`` where ``b0`` is the first bit emitted per symbol.
    """

    definition = QAM_DEFINITIONS[mode]
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if normalize:
        values = normalize_qam_symbols(values, mode=mode)
    if definition.points_by_index is not None:
        indices = _nearest_point_indices(values, definition.points_by_index)
        return (
            _index_to_spec_label(indices, definition.bits_per_symbol)
            .reshape(-1)
            .astype(np.uint8)
        )
    if definition.levels is None:
        raise ValueError(f"{mode} does not define rectangular QAM levels")
    i_index = _nearest_level_indices(values.real, definition.levels)
    q_index = _nearest_level_indices(values.imag, definition.levels)
    i_bits = _level_index_to_axis_bits(i_index, definition.axis_bits)
    q_bits = _level_index_to_axis_bits(q_index, definition.axis_bits)
    return np.concatenate((i_bits, q_bits), axis=1).reshape(-1).astype(np.uint8)


def qam_soft_demodulate(
    symbols: np.ndarray,
    *,
    mode: QamMode,
    noise_variance: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Return max-log LLR values for each bit.

    Positive LLR means bit `0` is closer than bit `1`.
    """

    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    definition = QAM_DEFINITIONS[mode]
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if normalize:
        values = normalize_qam_symbols(values, mode=mode)
    points, point_bits = constellation_points(mode=mode)
    distances = np.abs(values[:, None] - points[None, :]) ** 2
    llrs = []
    for bit_index in range(definition.bits_per_symbol):
        zero = distances[:, point_bits[:, bit_index] == 0]
        one = distances[:, point_bits[:, bit_index] == 1]
        llrs.append((np.min(one, axis=1) - np.min(zero, axis=1)) / noise_variance)
    return np.stack(llrs, axis=1).reshape(-1).astype(np.float32)


def normalize_qam_symbols(symbols: np.ndarray, *, mode: QamMode) -> np.ndarray:
    """Normalize gain and residual common phase against a rectangular QAM grid."""

    definition = QAM_DEFINITIONS[mode]
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if values.size == 0:
        return values.astype(np.complex64, copy=False)
    observed_power = float(np.mean(np.abs(values) ** 2))
    if observed_power <= 1e-24:
        return values.astype(np.complex64, copy=False)
    scale = np.sqrt(observed_power / definition.average_power)
    corrected = values / max(scale, 1e-12)
    nearest = qam_nearest(corrected, mode=mode)
    denominator = float(np.vdot(nearest, nearest).real)
    if denominator > 1e-12:
        gain = np.vdot(nearest, values) / denominator
        if abs(gain) > 1e-12:
            corrected = values / gain
    return corrected.astype(np.complex64, copy=False)


def qam_nearest(symbols: np.ndarray, *, mode: QamMode) -> np.ndarray:
    """Slice symbols to nearest constellation points."""

    definition = QAM_DEFINITIONS[mode]
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if definition.points_by_index is not None:
        indices = _nearest_point_indices(values, definition.points_by_index)
        return definition.points_by_index[indices].astype(np.complex64, copy=False)
    if definition.levels is None:
        raise ValueError(f"{mode} does not define rectangular QAM levels")
    real = definition.levels[_nearest_level_indices(values.real, definition.levels)]
    imag = definition.levels[_nearest_level_indices(values.imag, definition.levels)]
    return (real + 1j * imag).astype(np.complex64)


def qam_symbol_quality(
    symbols: np.ndarray,
    *,
    mode: QamMode,
    normalize: bool = True,
) -> dict[str, Any]:
    """Summarize whether sliced QAM symbols look like a plausible payload.

    This is a diagnostic helper, not a receiver decision rule. Scrambled DTMB
    payloads should exercise QAM labels fairly evenly over long streams; a low
    nearest-grid EVM with most decisions collapsed onto inner axis levels is a
    strong sign that channel/equalizer output is structurally wrong before FEC.
    """

    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if values.size == 0:
        return {
            "symbol_count": 0,
            "power": 0.0,
            "normalized_power": 0.0,
            "grid_evm_rms": None,
            "hard_bit_balance_per_plane": [],
            "max_abs_hard_bit_bias": None,
        }
    normalized = normalize_qam_symbols(values, mode=mode) if normalize else values
    nearest = qam_nearest(normalized, mode=mode)
    nearest_power = float(np.mean(np.abs(nearest) ** 2)) if nearest.size else 0.0
    grid_evm = (
        None
        if nearest_power <= 1e-24
        else float(np.sqrt(np.mean(np.abs(normalized - nearest) ** 2) / nearest_power))
    )
    hard_bits = qam_hard_demodulate(normalized, mode=mode, normalize=False)
    definition = QAM_DEFINITIONS[mode]
    bit_balance = _hard_bit_balance_per_plane(
        hard_bits,
        bits_per_symbol=definition.bits_per_symbol,
    )
    summary: dict[str, Any] = {
        "symbol_count": int(values.size),
        "power": float(np.mean(np.abs(values) ** 2)),
        "normalized_power": float(np.mean(np.abs(normalized) ** 2)),
        "grid_evm_rms": grid_evm,
        "hard_bit_balance_per_plane": bit_balance,
        "max_abs_hard_bit_bias": _max_abs_hard_bit_bias(bit_balance),
    }
    if definition.levels is not None:
        levels = definition.levels.astype(np.float64, copy=False)
        i_axis = _axis_level_occupancy(nearest.real, levels)
        q_axis = _axis_level_occupancy(nearest.imag, levels)
        summary["axis_level_occupancy"] = {
            "levels": [float(level) for level in levels],
            "uniform_fraction": 1.0 / float(levels.size),
            "i": i_axis,
            "q": q_axis,
            "max_abs_fraction_error_from_uniform": float(
                max(
                    i_axis["max_abs_fraction_error_from_uniform"],
                    q_axis["max_abs_fraction_error_from_uniform"],
                )
            ),
            "mean_abs_fraction_error_from_uniform": float(
                np.mean(
                    [
                        i_axis["mean_abs_fraction_error_from_uniform"],
                        q_axis["mean_abs_fraction_error_from_uniform"],
                    ]
                )
            ),
        }
    return summary


def constellation_points(*, mode: QamMode) -> tuple[np.ndarray, np.ndarray]:
    """Return all points and their bit labels in spec bit order.

    ``labels[k]`` carries the per-symbol bit vector ``(b0, b1, ..., bN)`` in
    the same order :func:`qam_modulate` consumes and :func:`qam_hard_demodulate`
    emits: the first element is the first FEC-encoded bit (the LSB of the
    symbol codeword) per GB 20600-2006 section 4.4.3.
    """

    definition = QAM_DEFINITIONS[mode]
    bit_count = definition.bits_per_symbol
    labels = _index_to_spec_label(
        np.arange(1 << bit_count, dtype=np.int32),
        bit_count,
    )
    if definition.points_by_index is not None:
        return definition.points_by_index.astype(np.complex64, copy=True), labels
    return qam_modulate(labels.reshape(-1), mode=mode), labels


def _axis_bits_to_level_index(axis_bits: np.ndarray) -> np.ndarray:
    """Return rectangular-QAM level index from per-axis Gray bits.

    Per GB 20600-2006 the first bit emitted per axis group is the LSB, while
    Figures 3, 5, and 6 place each axis in Gray-code order. ``axis_bits`` is
    therefore interpreted as an LSB-first Gray code and converted to the binary
    level index used by ``QamDefinition.levels``.
    """

    if axis_bits.ndim != 2:
        raise ValueError("axis_bits must be a two-dimensional grouped array")
    gray = _bits_to_indices(axis_bits)
    return _gray_to_binary(gray)


def _bits_to_indices(bits: np.ndarray) -> np.ndarray:
    """Convert grouped b0-first bit rows into integer codeword indices."""

    if bits.ndim != 2:
        raise ValueError("bits must be a two-dimensional grouped array")
    weights = (1 << np.arange(bits.shape[1], dtype=np.int32)).astype(np.int32)
    return np.sum(bits.astype(np.int32) * weights[None, :], axis=1)


def _level_index_to_axis_bits(indices: np.ndarray, width: int) -> np.ndarray:
    """Invert :func:`_axis_bits_to_level_index`.

    Returns an array shaped ``(len(indices), width)`` where position 0 is the
    LSB of the reflected Gray code used by the DTMB constellation figures.
    """

    values = np.asarray(indices, dtype=np.int32).reshape(-1)
    values = values ^ (values >> 1)
    shifts = np.arange(width, dtype=np.int32)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def _index_to_spec_label(indices: np.ndarray, width: int) -> np.ndarray:
    """Convert 0..(2^width - 1) into DTMB spec bit order (b0 first).

    Used to enumerate all constellation points with labels matching the
    order :func:`qam_modulate` consumes.
    """

    values = np.asarray(indices, dtype=np.int32).reshape(-1)
    shifts = np.arange(width, dtype=np.int32)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def _nearest_level_indices(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(values[:, None] - levels[None, :]), axis=1)


def _nearest_point_indices(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(values[:, None] - points[None, :]) ** 2, axis=1)


def _hard_bit_balance_per_plane(
    hard_bits: np.ndarray,
    *,
    bits_per_symbol: int,
) -> list[float]:
    values = np.asarray(hard_bits, dtype=np.uint8).reshape(-1)
    usable = (values.size // bits_per_symbol) * bits_per_symbol
    if usable == 0:
        return []
    grouped = values[:usable].reshape(-1, bits_per_symbol)
    return [float(np.mean(grouped[:, index])) for index in range(bits_per_symbol)]


def _max_abs_hard_bit_bias(bit_balance: list[float]) -> float | None:
    if not bit_balance:
        return None
    return float(max(abs(float(value) - 0.5) for value in bit_balance))


def _axis_level_occupancy(values: np.ndarray, levels: np.ndarray) -> dict[str, Any]:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    level_values = np.asarray(levels, dtype=np.float64).reshape(-1)
    counts = np.asarray(
        [np.count_nonzero(axis == level) for level in level_values],
        dtype=np.int64,
    )
    total = int(np.sum(counts))
    fractions = (
        counts.astype(np.float64) / float(total)
        if total > 0
        else np.zeros(level_values.size, dtype=np.float64)
    )
    uniform = 1.0 / float(level_values.size)
    abs_error = np.abs(fractions - uniform)
    nonzero = fractions[fractions > 0.0]
    entropy = (
        0.0
        if nonzero.size == 0
        else float(-np.sum(nonzero * np.log(nonzero)) / np.log(level_values.size))
    )
    abs_levels = np.abs(level_values)
    inner_mask = abs_levels == np.min(abs_levels)
    outer_mask = abs_levels == np.max(abs_levels)
    return {
        "counts": [int(value) for value in counts],
        "fractions": [float(value) for value in fractions],
        "inner_fraction": float(np.sum(fractions[inner_mask])),
        "outer_fraction": float(np.sum(fractions[outer_mask])),
        "entropy_fraction": entropy,
        "max_abs_fraction_error_from_uniform": float(np.max(abs_error)),
        "mean_abs_fraction_error_from_uniform": float(np.mean(abs_error)),
    }


def _gray_to_binary(values: np.ndarray) -> np.ndarray:
    binary = np.asarray(values, dtype=np.int32).copy()
    shift = 1
    while shift < 32:
        binary ^= binary >> shift
        shift <<= 1
    return binary
