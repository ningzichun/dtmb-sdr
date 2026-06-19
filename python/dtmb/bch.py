"""Shortened DTMB BCH(762, 752) helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


BCH_MESSAGE_BITS = 752
BCH_CODE_BITS = 762
BCH_PARITY_BITS = BCH_CODE_BITS - BCH_MESSAGE_BITS

# GB 20600-2006: GBCH(x) = 1 + x^3 + x^10.
BCH_GENERATOR_POLY = (1 << 10) | (1 << 3) | 1


def bch_encode(message_bits: np.ndarray) -> np.ndarray:
    """Encode one shortened systematic BCH(762, 752) block."""

    message = _as_binary_vector(message_bits)
    if message.size != BCH_MESSAGE_BITS:
        raise ValueError("BCH message requires 752 bits")
    work = np.concatenate((message, np.zeros(BCH_PARITY_BITS, dtype=np.uint8)))
    remainder = _poly_remainder(work, BCH_GENERATOR_POLY, BCH_PARITY_BITS)
    encoded = work.copy()
    encoded[-BCH_PARITY_BITS:] = remainder
    return encoded


def bch_syndrome(codeword_bits: np.ndarray) -> np.ndarray:
    """Return the 10-bit BCH syndrome/remainder."""

    codeword = _as_binary_vector(codeword_bits)
    if codeword.size != BCH_CODE_BITS:
        raise ValueError("BCH codeword requires 762 bits")
    return _poly_remainder(codeword, BCH_GENERATOR_POLY, BCH_PARITY_BITS)


def bch_decode(
    codeword_bits: np.ndarray,
    *,
    correct: bool = True,
) -> tuple[np.ndarray, int, bool]:
    """Decode one BCH block.

    Returns `(message_bits, corrected_errors, clean)`. This shortened code has
    ten parity bits; this implementation corrects single-bit errors by syndrome
    lookup and reports uncorrectable non-zero syndromes as `clean=False`.
    """

    codeword = _as_binary_vector(codeword_bits).copy()
    if codeword.size != BCH_CODE_BITS:
        raise ValueError("BCH codeword requires 762 bits")
    syndrome = bch_syndrome(codeword)
    if not np.any(syndrome):
        return codeword[:BCH_MESSAGE_BITS].copy(), 0, True
    if correct:
        key = _syndrome_key(syndrome)
        position = _single_error_syndrome_table().get(key)
        if position is not None:
            codeword[position] ^= 1
            return codeword[:BCH_MESSAGE_BITS].copy(), 1, True
    return codeword[:BCH_MESSAGE_BITS].copy(), 0, False


def _as_binary_vector(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return values


def _poly_remainder(bits: np.ndarray, generator: int, degree: int) -> np.ndarray:
    work = np.asarray(bits, dtype=np.uint8).copy()
    generator_bits = np.asarray(
        [(generator >> shift) & 1 for shift in range(degree, -1, -1)],
        dtype=np.uint8,
    )
    for index in range(work.size - degree):
        if work[index]:
            work[index : index + degree + 1] ^= generator_bits
    return work[-degree:].astype(np.uint8)


def _syndrome_key(bits: np.ndarray) -> int:
    key = 0
    for bit in bits:
        key = (key << 1) | int(bit)
    return key


@lru_cache(maxsize=1)
def _single_error_syndrome_table() -> dict[int, int]:
    table: dict[int, int] = {}
    for position in range(BCH_CODE_BITS):
        error = np.zeros(BCH_CODE_BITS, dtype=np.uint8)
        error[position] = 1
        table[_syndrome_key(bch_syndrome(error))] = position
    return table
