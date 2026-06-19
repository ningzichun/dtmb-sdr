"""DTMB baseband pulse-shaping helpers.

GB 20600-2006 section 4.8 defines the baseband SRRC pulse-shape filter
with roll-off ``alpha = 0.05``. This module provides a small square-root
raised-cosine filter builder and an up-sample + pulse-shape helper we
use to turn discrete symbol-rate IQ into a realistic baseband CI8 stream
at any chosen SDR sample rate (20 Msps for HackRF, 10 Msps, etc.).
"""

from __future__ import annotations

import numpy as np


def square_root_raised_cosine(
    *,
    symbol_span: int,
    samples_per_symbol: int,
    roll_off: float = 0.05,
) -> np.ndarray:
    """Return a normalised SRRC impulse response.

    Parameters mirror the DTMB baseband pulse-shape definition. ``symbol_span``
    is the one-sided filter length in symbols, so the returned response has
    ``2 * symbol_span * samples_per_symbol + 1`` taps. ``roll_off`` defaults
    to ``alpha = 0.05`` per GB 20600-2006.
    """

    if symbol_span <= 0:
        raise ValueError("symbol_span must be positive")
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if not 0.0 < roll_off <= 1.0:
        raise ValueError("roll_off must be within (0, 1]")

    taps = 2 * symbol_span * samples_per_symbol + 1
    t = (np.arange(taps) - (taps - 1) / 2) / samples_per_symbol
    response = np.empty_like(t)
    for index, value in enumerate(t):
        response[index] = _srrc_value(value, roll_off)
    response /= np.sqrt(np.sum(response ** 2))
    return response


def _srrc_value(t: float, alpha: float) -> float:
    """Closed-form SRRC tap value with analytic handling of singularities."""

    if t == 0.0:
        return 1.0 - alpha + 4.0 * alpha / np.pi
    denom_zero = abs(abs(4.0 * alpha * t) - 1.0) < 1e-12
    if denom_zero:
        scale = alpha / np.sqrt(2.0)
        return scale * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
        )
    numerator = np.sin(np.pi * t * (1.0 - alpha)) + 4.0 * alpha * t * np.cos(
        np.pi * t * (1.0 + alpha)
    )
    denominator = np.pi * t * (1.0 - (4.0 * alpha * t) ** 2)
    return numerator / denominator


def pulse_shape_to_rate(
    symbols: np.ndarray,
    *,
    symbol_rate_sps: int,
    sample_rate_sps: int,
    roll_off: float = 0.05,
    symbol_span: int = 16,
    peak_amplitude: float | None = None,
) -> np.ndarray:
    """Up-sample a discrete symbol stream to ``sample_rate_sps`` via SRRC.

    The symbol stream is expected at one sample per symbol. The function
    returns the pulse-shaped complex baseband signal at ``sample_rate_sps``.

    Internally uses :func:`scipy.signal.upfirdn` with a polyphase SRRC filter
    designed for the requested rational resample ratio. For integer ratios
    this collapses to pure upsample + SRRC convolution; for rational ratios
    it is a single polyphase step and stays cheap even at DTMB 7.56 -> 20
    Msps (500/189).
    """

    from fractions import Fraction

    if sample_rate_sps <= 0 or symbol_rate_sps <= 0:
        raise ValueError("rates must be positive")
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)

    if sample_rate_sps == symbol_rate_sps:
        shaped = values.astype(np.complex64, copy=False)
    else:
        ratio = Fraction(sample_rate_sps, symbol_rate_sps).limit_denominator(10_000)
        up = ratio.numerator
        down = ratio.denominator
        try:
            from scipy.signal import upfirdn
        except ImportError as exc:
            raise RuntimeError(
                "pulse_shape_to_rate requires scipy; install with .[dsp]"
            ) from exc
        taps = square_root_raised_cosine(
            symbol_span=symbol_span,
            samples_per_symbol=up,
            roll_off=roll_off,
        ).astype(np.float32)
        # Apply gain for the upsampler so the passband stays at unit
        # amplitude after the polyphase decimation.
        taps = taps * up
        shaped = upfirdn(taps, values, up=up, down=down).astype(np.complex64)
    if peak_amplitude is not None and peak_amplitude > 0:
        peak = float(np.max(np.abs(shaped)))
        if peak > 0:
            shaped = shaped * (peak_amplitude / peak)
    return shaped.astype(np.complex64, copy=False)


def _pulse_shape_integer(
    values: np.ndarray,
    *,
    samples_per_symbol: int,
    roll_off: float,
    symbol_span: int,
) -> np.ndarray:
    if samples_per_symbol <= 1:
        return values.astype(np.complex64, copy=False)
    upsampled = np.zeros(values.size * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = values
    taps = square_root_raised_cosine(
        symbol_span=symbol_span,
        samples_per_symbol=samples_per_symbol,
        roll_off=roll_off,
    ).astype(np.float32)
    return np.convolve(upsampled, taps, mode="same").astype(np.complex64)
