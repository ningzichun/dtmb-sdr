"""C=3780 logical carrier-axis permutation helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .frequency import FRAME_BODY_SYMBOLS, SYSTEM_INFO_POSITIONS


CARRIER_AXIS_SIZES = {
    "i": 3,
    "j": 3,
    "k": 3,
    "l": 2,
    "m": 2,
    "n": 5,
    "o": 7,
}
CARRIER_AXIS_ORDER = ("i", "j", "k", "l", "m", "n", "o")


def logical_position_to_coords(position: int) -> tuple[int, int, int, int, int, int, int]:
    i, rem = divmod(int(position), 1260)
    j, rem = divmod(rem, 420)
    k, rem = divmod(rem, 140)
    l, rem = divmod(rem, 70)
    m, rem = divmod(rem, 35)
    n, o = divmod(rem, 7)
    return int(i), int(j), int(k), int(l), int(m), int(n), int(o)


def logical_position_from_coords(coords: Sequence[int]) -> int:
    i, j, k, l, m, n, o = (int(value) for value in coords)
    return i * 1260 + j * 420 + k * 140 + l * 70 + m * 35 + n * 7 + o


def carrier_permutation_operations(name: str) -> tuple[str, ...]:
    operations = tuple(
        operation.strip().lower()
        for operation in str(name).split("+")
        if operation.strip()
    )
    return operations or ("identity",)


def apply_carrier_axis_operation(
    coords: Sequence[int],
    operation: str,
) -> tuple[int, int, int, int, int, int, int]:
    values = [int(value) for value in coords]
    op = str(operation).strip().lower()
    if op == "identity":
        return tuple(values)  # type: ignore[return-value]
    if op.startswith("rev_"):
        axis = op[4:]
        if axis not in CARRIER_AXIS_SIZES:
            raise ValueError(f"unknown carrier permutation axis: {operation}")
        index = CARRIER_AXIS_ORDER.index(axis)
        values[index] = CARRIER_AXIS_SIZES[axis] - 1 - values[index]
        return tuple(values)  # type: ignore[return-value]
    if op.startswith("swap_"):
        lhs, rhs = carrier_swap_axes(op)
        if CARRIER_AXIS_SIZES[lhs] != CARRIER_AXIS_SIZES[rhs]:
            raise ValueError(f"carrier axes {lhs!r} and {rhs!r} have different sizes")
        lhs_index = CARRIER_AXIS_ORDER.index(lhs)
        rhs_index = CARRIER_AXIS_ORDER.index(rhs)
        values[lhs_index], values[rhs_index] = values[rhs_index], values[lhs_index]
        return tuple(values)  # type: ignore[return-value]
    raise ValueError(f"unknown carrier permutation: {operation}")


def carrier_swap_axes(operation: str) -> tuple[str, str]:
    raw = str(operation)[5:]
    if "_" in raw:
        parts = [part for part in raw.split("_") if part]
    else:
        parts = list(raw)
    if len(parts) != 2 or parts[0] not in CARRIER_AXIS_SIZES or parts[1] not in CARRIER_AXIS_SIZES:
        raise ValueError(f"unknown carrier swap permutation: {operation}")
    return parts[0], parts[1]


@lru_cache(maxsize=None)
def logical_inserted_permutation_indices(name: str) -> tuple[int, ...]:
    normalized = str(name).strip().lower() or "identity"
    indices = np.empty(FRAME_BODY_SYMBOLS, dtype=np.int32)
    for destination_position in range(FRAME_BODY_SYMBOLS):
        coords = logical_position_to_coords(destination_position)
        for operation in carrier_permutation_operations(normalized):
            coords = apply_carrier_axis_operation(coords, operation)
        indices[destination_position] = logical_position_from_coords(coords)
    if np.unique(indices).size != FRAME_BODY_SYMBOLS:
        raise ValueError(f"carrier permutation {name!r} is not bijective")
    return tuple(int(value) for value in indices)


def apply_carrier_permutation_to_inserted(
    symbols: np.ndarray,
    *,
    permutation_name: str = "identity",
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body requires 3780 symbols")
    normalized = str(permutation_name).strip().lower() or "identity"
    if normalized == "identity":
        return values
    return values[np.asarray(logical_inserted_permutation_indices(normalized), dtype=np.int32)]


@lru_cache(maxsize=None)
def system_info_positions_for_permutation(name: str) -> tuple[int, ...]:
    return system_info_positions_for_extraction(
        name,
        logical_position_shift_symbols=0,
    )


@lru_cache(maxsize=None)
def system_info_positions_for_extraction(
    name: str,
    logical_position_shift_symbols: int = 0,
) -> tuple[int, ...]:
    normalized = str(name).strip().lower() or "identity"
    positions = []
    shifted = (
        SYSTEM_INFO_POSITIONS.astype(np.int64) + int(logical_position_shift_symbols)
    ) % FRAME_BODY_SYMBOLS
    for position in shifted:
        coords = logical_position_to_coords(int(position))
        for operation in carrier_permutation_operations(normalized):
            coords = apply_carrier_axis_operation(coords, operation)
        positions.append(logical_position_from_coords(coords))
    return tuple(int(value) for value in positions)
