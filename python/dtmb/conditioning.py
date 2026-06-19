"""Signal-conditioning helpers for offline captures."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .frame_sync import DTMB_SYMBOL_RATE_SPS


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """Remove the complex mean from a sample vector."""

    signal = np.asarray(samples, dtype=np.complex64)
    if signal.size == 0:
        return signal.astype(np.complex64, copy=False)
    return (signal - np.mean(signal, dtype=np.complex128)).astype(np.complex64, copy=False)


def frequency_shift(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    shift_hz: float,
) -> np.ndarray:
    """Mix complex samples by `shift_hz`."""

    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    signal = np.asarray(samples, dtype=np.complex64)
    if signal.size == 0 or shift_hz == 0:
        return signal.astype(np.complex64, copy=False)
    time_index = np.arange(signal.size, dtype=np.float64)
    rotation = np.exp(2j * np.pi * float(shift_hz) * time_index / sample_rate_sps)
    return (signal * rotation).astype(np.complex64, copy=False)


def resample_to_symbol_rate(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_denominator: int = 10_000,
    matched_filter: bool = True,
    matched_filter_roll_off: float = 0.05,
    matched_filter_span: int = 8,
) -> tuple[np.ndarray, int, int]:
    """Resample complex samples to the DTMB symbol-rate grid.

    When ``matched_filter`` is true the SRRC matched filter (alpha=0.05,
    GB 20600-2006 section 4.8) is used as the prototype filter inside
    :func:`scipy.signal.resample_poly`. That performs one-shot
    polyphase matched filtering + rational resampling, recovering clean
    symbols from SRRC-shaped captures (real HackRF DTMB, for example).
    Set ``matched_filter=False`` to fall back to scipy's default Kaiser
    anti-alias window (useful for non-DTMB captures).

    Returns ``(resampled, up, down)``.
    """

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "resampling requires scipy; install with `uv pip install -e .[dsp]`"
        ) from exc

    if sample_rate_sps <= 0 or symbol_rate_sps <= 0:
        raise ValueError("sample rates must be positive")
    ratio = Fraction(symbol_rate_sps, sample_rate_sps).limit_denominator(
        max_denominator
    )
    up = ratio.numerator
    down = ratio.denominator
    if sample_rate_sps == symbol_rate_sps:
        return (
            np.asarray(samples, dtype=np.complex64),
            up,
            down,
        )
    window = None
    if matched_filter:
        from .pulse_shape import square_root_raised_cosine

        # resample_poly internally uses a filter of length
        # ``2 * max(up, down) * half_len + 1`` with the given window
        # applied to a sinc kernel. To supply an explicit SRRC kernel we
        # pass ``window=taps`` of the expected length. The design rate
        # for the inner kernel is ``max(up, down)`` samples per symbol.
        spb = max(up, down)
        taps = square_root_raised_cosine(
            symbol_span=matched_filter_span,
            samples_per_symbol=spb,
            roll_off=matched_filter_roll_off,
        ).astype(np.float32)
        # resample_poly's ``window`` argument expects a length-matching
        # window array that is multiplied point-wise with an internal
        # sinc. We instead pass the final filter as ``window`` by
        # providing a flat sinc (all-ones) shape via half_len=symbol_span
        # and applying the SRRC as the window coefficients.
        half_len = matched_filter_span
        window = taps
        resampled = resample_poly(
            samples,
            up,
            down,
            window=window,
            padtype="constant",
        )
    else:
        resampled = resample_poly(samples, up, down)
    return (
        resampled.astype(np.complex64, copy=False),
        up,
        down,
    )
