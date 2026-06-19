"""Bit and byte packing helpers."""

from __future__ import annotations

import numpy as np


def unpack_bytes_msb(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Return bits from bytes, most-significant bit first."""

    array = np.frombuffer(bytes(data), dtype=np.uint8)
    if array.size == 0:
        return np.empty(0, dtype=np.uint8)
    shifts = np.arange(7, -1, -1, dtype=np.uint8)
    return ((array[:, None] >> shifts[None, :]) & 1).reshape(-1).astype(np.uint8)


def pack_bits_msb(bits: np.ndarray, *, truncate: bool = False) -> bytes:
    """Pack bits into bytes, most-significant bit first.

    If `truncate` is false, the bit count must be a multiple of 8. If it is
    true, trailing incomplete bytes are ignored.
    """

    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    remainder = values.size % 8
    if remainder:
        if not truncate:
            raise ValueError("bit count must be a multiple of 8")
        values = values[: values.size - remainder]
    if values.size == 0:
        return b""
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8)).astype(np.uint8)
    packed = np.sum(values.reshape(-1, 8) * weights[None, :], axis=1, dtype=np.uint16)
    return packed.astype(np.uint8).tobytes()
