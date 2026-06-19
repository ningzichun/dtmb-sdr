"""DTMB C=3780 frequency-domain interleaver helpers."""

from __future__ import annotations

import numpy as np


FRAME_BODY_SYMBOLS = 3780
SYSTEM_INFO_SYMBOLS = 36
DATA_SYMBOLS_PER_FRAME = 3744
SYSTEM_INFO_POSITIONS = np.asarray(
    [
        0,
        140,
        279,
        419,
        420,
        560,
        699,
        839,
        840,
        980,
        1119,
        1259,
        1260,
        1400,
        1539,
        1679,
        1680,
        1820,
        1959,
        2099,
        2100,
        2240,
        2379,
        2519,
        2520,
        2660,
        2799,
        2939,
        2940,
        3080,
        3219,
        3359,
        3360,
        3500,
        3639,
        3779,
    ],
    dtype=np.int32,
)


def frequency_interleave_index_map() -> np.ndarray:
    """Return logical C=3780 carrier index -> physical FFT-bin index."""

    return _LOGICAL_TO_PHYSICAL.copy()


def frequency_deinterleave_index_map() -> np.ndarray:
    """Return physical FFT-bin index -> logical C=3780 carrier index."""

    return _PHYSICAL_TO_LOGICAL.copy()


def frequency_interleave(symbols: np.ndarray) -> np.ndarray:
    """Apply the DTMB C=3780 frequency interleaver."""

    z = _system_info_inserted(symbols)
    return z[_PHYSICAL_TO_LOGICAL]


def frequency_deinterleave_inserted(symbols: np.ndarray) -> np.ndarray:
    """Undo the DTMB C=3780 permutation, keeping system symbols inserted."""

    y = np.asarray(symbols)
    if y.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frequency deinterleaver requires 3780 symbols")
    return y[_LOGICAL_TO_PHYSICAL]


def _build_frequency_interleaver_maps() -> tuple[np.ndarray, np.ndarray]:
    logical_to_physical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.int32)
    physical_to_logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.int32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(2):
                    for m in range(2):
                        for n in range(5):
                            for o in range(7):
                                physical = (
                                    o * 540
                                    + n * 108
                                    + m * 54
                                    + l * 27
                                    + k * 9
                                    + j * 3
                                    + i
                                )
                                logical = (
                                    i * 1260
                                    + j * 420
                                    + k * 140
                                    + l * 70
                                    + m * 35
                                    + n * 7
                                    + o
                                )
                                logical_to_physical[logical] = physical
                                physical_to_logical[physical] = logical
    return logical_to_physical, physical_to_logical


_LOGICAL_TO_PHYSICAL, _PHYSICAL_TO_LOGICAL = _build_frequency_interleaver_maps()


def frequency_deinterleave(symbols: np.ndarray) -> np.ndarray:
    """Undo the DTMB C=3780 frequency interleaver to logical frame-body order."""

    return _system_info_removed(frequency_deinterleave_inserted(symbols))


def split_system_info_and_data(deinterleaved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(system_info, data)` from a deinterleaved C=3780 frame body."""

    values = np.asarray(deinterleaved)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body requires 3780 symbols")
    return values[:SYSTEM_INFO_SYMBOLS], values[SYSTEM_INFO_SYMBOLS:]


def frame_body_fft(body: np.ndarray) -> np.ndarray:
    """FFT one 3780-sample DTMB multicarrier frame body."""

    samples = np.asarray(body, dtype=np.complex64)
    if samples.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body FFT requires 3780 symbols")
    return np.fft.fft(samples).astype(np.complex64)


def _system_info_inserted(symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frequency interleaver requires 3780 symbols")
    z = np.empty(FRAME_BODY_SYMBOLS, dtype=values.dtype)
    z[SYSTEM_INFO_POSITIONS] = values[:SYSTEM_INFO_SYMBOLS]
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[SYSTEM_INFO_POSITIONS] = False
    z[mask] = values[SYSTEM_INFO_SYMBOLS:]
    return z


def _system_info_removed(symbols: np.ndarray) -> np.ndarray:
    z = np.asarray(symbols)
    if z.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body requires 3780 symbols")
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[SYSTEM_INFO_POSITIONS] = False
    return np.concatenate((z[SYSTEM_INFO_POSITIONS], z[mask]))
