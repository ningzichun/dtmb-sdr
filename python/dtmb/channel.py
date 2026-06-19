"""PN-header channel-estimation helpers for DTMB diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .frame_sync import PnFamilyDelayMatch, pn_family_delay_match
from .pn import PN_DEFINITIONS, PnMode, pn_cyclic_family_symbols


@dataclass(frozen=True)
class PnChannelEstimate:
    """Frequency response estimated from one cyclic PN header."""

    mode: PnMode
    pn_phase: int
    delay_symbols: int
    metric: float
    taps: np.ndarray
    response: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pn_phase": self.pn_phase,
            "delay_symbols": self.delay_symbols,
            "metric": self.metric,
            "tap_count": int(self.taps.size),
            "response_bins": int(self.response.size),
            "tap_power": float(np.sum(np.abs(self.taps) ** 2)),
        }


def estimate_pn_channel(
    header: np.ndarray,
    *,
    mode: PnMode,
    pn_phase: int | None = None,
    fft_size: int = 3780,
    channel_taps: int | None = None,
    max_delay_symbols: int = 256,
    regularization: float = 1e-3,
) -> PnChannelEstimate:
    """Estimate a channel response from the cyclic portion of a PN420/PN945 header."""

    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    channel_taps = _resolve_channel_taps(mode, channel_taps)
    if channel_taps <= 0:
        raise ValueError("channel_taps must be positive")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if pn_phase is None:
        match = pn_family_delay_match(
            header,
            mode=mode,
            max_delay_symbols=max_delay_symbols,
        )
    else:
        match = PnFamilyDelayMatch(
            mode=mode,
            frame_start=0,
            pn_phase=pn_phase,
            delay_symbols=0,
            metric=1.0,
        )
    return estimate_pn_channel_from_match(
        header,
        mode=mode,
        match=match,
        fft_size=fft_size,
        channel_taps=channel_taps,
        regularization=regularization,
    )


def estimate_pn_channel_from_match(
    header: np.ndarray,
    *,
    mode: PnMode,
    match: PnFamilyDelayMatch,
    fft_size: int = 3780,
    channel_taps: int | None = None,
    regularization: float = 1e-3,
) -> PnChannelEstimate:
    """Estimate a channel response using an existing PN-family match."""

    signal = np.asarray(header, dtype=np.complex64).reshape(-1)
    channel_taps = _resolve_channel_taps(mode, channel_taps)
    if channel_taps <= 0:
        raise ValueError("channel_taps must be positive")
    taps = _pn_core_taps(
        signal,
        mode=mode,
        pn_phase=match.pn_phase,
        regularization=regularization,
    )
    if channel_taps < taps.size:
        taps = taps.copy()
        taps[channel_taps:] = 0
    response = np.fft.fft(taps, fft_size).astype(np.complex64)
    return PnChannelEstimate(
        mode=mode,
        pn_phase=match.pn_phase,
        delay_symbols=match.delay_symbols,
        metric=match.metric,
        taps=taps[: min(channel_taps, taps.size)].astype(np.complex64, copy=False),
        response=response,
    )


def _pn_core_taps(
    signal: np.ndarray,
    *,
    mode: PnMode,
    pn_phase: int,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Deconvolve one header's cyclic core into the full circular tap vector.

    The returned vector has one tap per cyclic-core symbol (511 for PN945,
    255 for PN420). Channel delays are represented modulo the core length:
    paths arriving before the chosen frame reference wrap to the high indices.
    """

    prefix, body_length, _suffix = _cyclic_layout(mode)
    family = pn_cyclic_family_symbols(mode)
    if not 0 <= pn_phase < family.shape[0]:
        raise ValueError("pn_phase is out of range for the PN family")
    reference = family[pn_phase]
    observed_core = np.asarray(signal, dtype=np.complex64).reshape(-1)[
        prefix : prefix + body_length
    ]
    reference_core = reference[prefix : prefix + body_length]
    if observed_core.size != body_length:
        raise ValueError("header is shorter than the PN cyclic core")
    ref_fft = np.fft.fft(reference_core)
    obs_fft = np.fft.fft(observed_core)
    response_core = obs_fft * np.conj(ref_fft) / (
        np.abs(ref_fft) ** 2 + regularization
    )
    response_core = _repair_low_energy_pn_dc_response(response_core, ref_fft)
    return np.fft.ifft(response_core).astype(np.complex64)


def canonicalize_pn_channel_estimate(
    estimate: PnChannelEstimate,
    *,
    relative_threshold: float = 0.05,
) -> PnChannelEstimate:
    """Move a PN phase/tap alias into the earliest-tap frame coordinate."""

    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive")
    taps = np.asarray(estimate.taps, dtype=np.complex64).reshape(-1)
    if taps.size <= 1:
        return estimate
    magnitudes = np.abs(taps)
    peak = float(np.max(magnitudes))
    if peak <= 0.0:
        return estimate
    significant = np.flatnonzero(magnitudes >= peak * relative_threshold)
    if significant.size == 0:
        return estimate
    shift = int(significant[0])
    if shift == 0:
        return estimate

    shifted = np.zeros_like(taps)
    shifted[: taps.size - shift] = taps[shift:]
    phase_count = (1 << PN_DEFINITIONS[estimate.mode].degree) - 1
    response = np.fft.fft(shifted, estimate.response.size).astype(np.complex64)
    return PnChannelEstimate(
        mode=estimate.mode,
        pn_phase=(int(estimate.pn_phase) - shift) % phase_count,
        delay_symbols=int(estimate.delay_symbols) - shift,
        metric=estimate.metric,
        taps=shifted,
        response=response,
    )


def estimate_pn_channel_compact(
    header: np.ndarray,
    *,
    mode: PnMode,
    fft_size: int = 3780,
    channel_taps: int | None = None,
    max_delay_symbols: int = 256,
    regularization: float = 1e-3,
) -> PnChannelEstimate:
    """Estimate the PN channel, choosing the phase with the most compact taps.

    The delay-tolerant family matcher (``estimate_pn_channel`` with
    ``pn_phase=None``) can return a phase/lag *alias* of the transmitted PN
    sequence. Because a PN cyclic-shift is indistinguishable from a channel
    delay, two different ``(phase, delay)`` pairs score almost equally, and the
    matcher sometimes selects the alias whose deconvolution reference
    (``family[alias_phase]``) is wrong. That yields a corrupted, energy-spread
    tap response even after :func:`canonicalize_pn_channel_estimate` relabels it
    to phase 0 -- which silently breaks both PN-ISI restoration and the
    equalizer response (observed on real 482 MHz captures and the synthetic
    multipath fixture; see AGENTS.md 2026-05-29).

    Real radio channels are compact: their impulse response is concentrated in a
    few early taps. This helper computes the channel for both the family-matched
    phase and the zero-delay phase (:func:`detect_pn_phase`), canonicalizes
    each, and returns whichever has the smaller normalized tap spread. The
    family match wins when the dominant tap is delayed (so zero-delay phase
    detection aliases), and the zero-delay phase wins when the family matcher
    picked a wraparound alias.
    """

    candidates: list[PnChannelEstimate] = []
    family_estimate = canonicalize_pn_channel_estimate(
        estimate_pn_channel(
            header,
            mode=mode,
            fft_size=fft_size,
            channel_taps=channel_taps,
            max_delay_symbols=max_delay_symbols,
            regularization=regularization,
        )
    )
    candidates.append(family_estimate)
    # Always evaluate the zero-delay-phase candidate as well. The family
    # matcher may select a phase/delay alias whose canonicalized phase equals
    # the zero-delay phase yet whose taps were deconvolved against the wrong
    # PN reference (``family[alias_phase]``), leaving a subtly less compact
    # response. Comparing canonicalized phases alone would hide that, so we
    # build the pinned estimate unconditionally and let the tap-spread metric
    # choose.
    try:
        zero_delay_phase = detect_pn_phase(header, mode=mode)
    except ValueError:
        zero_delay_phase = None
    if zero_delay_phase is not None:
        candidates.append(
            canonicalize_pn_channel_estimate(
                estimate_pn_channel(
                    header,
                    mode=mode,
                    pn_phase=zero_delay_phase,
                    fft_size=fft_size,
                    channel_taps=channel_taps,
                    regularization=regularization,
                )
            )
        )
    return min(candidates, key=_pn_tap_spread)


def _pn_tap_spread(estimate: PnChannelEstimate) -> float:
    """Energy-weighted mean tap index; lower means a more compact channel."""

    taps = np.abs(np.asarray(estimate.taps, dtype=np.complex128).reshape(-1))
    energy = float(np.sum(taps ** 2))
    if energy <= 0.0:
        return float("inf")
    indices = np.arange(taps.size, dtype=np.float64)
    return float(np.sum(indices * taps ** 2) / energy)


@dataclass(frozen=True)
class WidebandPnChannelModel:
    """Cross-frame averaged multi-cluster PN channel model.

    Built once per capture from many PN headers (the broadcast channel is
    static over a capture window), then instantiated per frame with a single
    complex scale that carries the frame's common phase/gain. Unlike the
    compact estimator this supports echo clusters anywhere within the PN
    cyclic core, including pre-cursor paths that wrap to high circular-tap
    indices at the acquisition's nominal timing.
    """

    mode: PnMode
    fft_size: int
    base_pn_phase: int
    pn_phase: int
    rotation_symbols: int
    template_taps: np.ndarray
    support_mask: np.ndarray
    dominant_tap_index: int
    frame_count: int
    phase_agreement: float
    threshold_factor: float
    per_frame_noise_tap_power: float
    averaged_noise_tap_power: float
    noise_variance: float
    truncated_energy_fraction: float

    def signed_rotation(self, core_length: int) -> int:
        shift = int(self.rotation_symbols) % core_length
        if shift > core_length // 2:
            return shift - core_length
        return shift

    def to_dict(self) -> dict[str, Any]:
        magnitudes = np.abs(np.asarray(self.template_taps, dtype=np.complex128))
        peak = float(np.max(magnitudes)) if magnitudes.size else 0.0
        significant = np.flatnonzero(np.asarray(self.support_mask, dtype=bool))
        tap_profile = []
        if peak > 0.0:
            for index in significant.tolist():
                tap_profile.append(
                    {
                        "tap": int(index),
                        "relative_db": float(
                            20.0 * np.log10(max(magnitudes[index], 1e-12) / peak)
                        ),
                    }
                )
        return {
            "mode": self.mode,
            "fft_size": int(self.fft_size),
            "base_pn_phase": int(self.base_pn_phase),
            "pn_phase": int(self.pn_phase),
            "rotation_symbols": int(self.rotation_symbols),
            "span_symbols": int(self.template_taps.size),
            "significant_taps": int(significant.size),
            "dominant_tap_index": int(self.dominant_tap_index),
            "frame_count": int(self.frame_count),
            "phase_agreement": float(self.phase_agreement),
            "threshold_factor": float(self.threshold_factor),
            "per_frame_noise_tap_power": float(self.per_frame_noise_tap_power),
            "averaged_noise_tap_power": float(self.averaged_noise_tap_power),
            "noise_variance": float(self.noise_variance),
            "truncated_energy_fraction": float(self.truncated_energy_fraction),
            "tap_power": float(np.sum(magnitudes ** 2)),
            "tap_profile": tap_profile,
        }


def build_wideband_pn_channel_model(
    headers: Sequence[np.ndarray],
    *,
    mode: PnMode,
    fft_size: int = 3780,
    threshold_factor: float = 3.0,
    guard_taps: int = 2,
    max_span_symbols: int | None = None,
    min_relative_power: float = 1e-5,
    regularization: float = 1e-3,
) -> WidebandPnChannelModel:
    """Build a multi-cluster PN channel model from many frame headers.

    Per header the full circular cyclic-core deconvolution is computed against
    that header's detected PN phase. Real PN945 broadcasts rotate this phase
    across frames, while the resulting channel taps remain in one common delay
    coordinate. The snapshots are coherently averaged after de-rotation by each
    frame's dominant-tap phase (the same per-frame common phase anchor
    :func:`average_pn_channel_estimates` uses), which lowers the per-tap
    estimation noise by roughly ``sqrt(frame_count)``. Taps are then selected
    against the averaged noise floor, so echo clusters of any spread survive
    while pure-noise taps are zeroed -- the failure mode that made simply
    widening ``channel_taps`` on the compact estimator worse, not better, on
    real multi-cluster channels (trial-log 2026-06-11).

    The longest run of insignificant taps in the circular profile defines the
    channel's causal window: the tap vector is rotated so the earliest cluster
    sits at tap zero, and ``pn_phase`` is re-referenced by the same rotation
    (the :func:`canonicalize_pn_channel_estimate` convention, applied
    circularly so wrapped pre-cursor energy is preserved instead of dropped).
    Guard taps are added only after this rotation and do not wrap around tap
    zero, avoiding an artificial timing shift on an already-causal channel.
    """

    if threshold_factor <= 0.0:
        raise ValueError("threshold_factor must be positive")
    if guard_taps < 0:
        raise ValueError("guard_taps must be non-negative")
    prefix, core_length, _suffix = _cyclic_layout(mode)
    if max_span_symbols is None:
        max_span_symbols = prefix
    if not 0 < max_span_symbols <= core_length:
        raise ValueError("max_span_symbols must be in (0, core_length]")
    header_arrays = [
        np.asarray(header, dtype=np.complex64).reshape(-1) for header in headers
    ]
    if not header_arrays:
        raise ValueError("headers must be non-empty")

    phases = np.asarray(
        [detect_pn_phase(header, mode=mode) for header in header_arrays],
        dtype=np.int64,
    )
    base_pn_phase = int(np.bincount(phases).argmax())
    phase_agreement = float(np.mean(phases == base_pn_phase))

    rows = np.stack(
        [
            _pn_core_taps(
                header,
                mode=mode,
                pn_phase=int(pn_phase),
                regularization=regularization,
            )
            for header, pn_phase in zip(header_arrays, phases)
        ]
    ).astype(np.complex128)

    raw_power = np.mean(np.abs(rows) ** 2, axis=0)
    dominant_raw = int(np.argmax(raw_power))
    anchors = rows[:, dominant_raw]
    anchor_magnitudes = np.abs(anchors)
    rotations = np.where(
        anchor_magnitudes > 0.0,
        np.conj(anchors) / np.maximum(anchor_magnitudes, 1e-30),
        1.0,
    )
    mean_taps = np.mean(rows * rotations[:, None], axis=0)

    averaged_power = np.abs(mean_taps) ** 2
    noise_power = float(np.median(averaged_power))
    mask = _threshold_support_mask(
        averaged_power,
        noise_power=noise_power,
        threshold_factor=threshold_factor,
        min_relative_power=min_relative_power,
    )
    if bool(np.any(~mask)):
        # Weak but coherent multipath can occupy a sizeable fraction of the
        # cyclic tap vector. A mean over the provisional "noise" set is then
        # biased upward by exactly the paths the second pass should retain.
        # The median remains a robust estimate while fewer than half the taps
        # contain channel energy.
        refined_noise = float(np.median(averaged_power[~mask]))
        if refined_noise > 0.0:
            noise_power = refined_noise
            mask = _threshold_support_mask(
                averaged_power,
                noise_power=noise_power,
                threshold_factor=threshold_factor,
                min_relative_power=min_relative_power,
            )
    if not bool(np.any(mask)):
        mask = np.zeros(core_length, dtype=bool)
        mask[dominant_raw] = True

    if bool(np.any(~mask)):
        per_frame_noise = float(np.median(np.abs(rows[:, ~mask]) ** 2))
    else:
        per_frame_noise = 0.0
    averaged_noise = float(noise_power)

    gap_start, gap_length = _longest_circular_false_run(mask)
    arc_start = (gap_start + gap_length) % core_length
    rotation = arc_start
    rotated_taps = np.roll(mean_taps, -rotation)
    rotated_mask = _dilate_linear_mask(np.roll(mask, -rotation), guard_taps)
    significant = np.flatnonzero(rotated_mask)
    span = int(significant[-1]) + 1 if significant.size else 1

    truncated_energy_fraction = 0.0
    if span > max_span_symbols:
        kept_energy = float(
            np.sum(averaged_power[mask])
        )
        dropped = rotated_mask.copy()
        dropped[:max_span_symbols] = False
        dropped_energy = float(
            np.sum(np.abs(rotated_taps[dropped]) ** 2)
        )
        if kept_energy > 0.0:
            truncated_energy_fraction = dropped_energy / kept_energy
        span = max_span_symbols
        rotated_mask = rotated_mask.copy()
        rotated_mask[max_span_symbols:] = False

    template = np.where(rotated_mask, rotated_taps, 0.0)[:span].astype(np.complex64)
    support = rotated_mask[:span].copy()
    dominant_tap_index = int(np.argmax(np.abs(template)))
    pn_phase = (base_pn_phase - rotation) % core_length

    return WidebandPnChannelModel(
        mode=mode,
        fft_size=int(fft_size),
        base_pn_phase=base_pn_phase,
        pn_phase=int(pn_phase),
        rotation_symbols=int(rotation),
        template_taps=template,
        support_mask=support,
        dominant_tap_index=dominant_tap_index,
        frame_count=len(header_arrays),
        phase_agreement=phase_agreement,
        threshold_factor=float(threshold_factor),
        per_frame_noise_tap_power=per_frame_noise,
        averaged_noise_tap_power=averaged_noise,
        noise_variance=per_frame_noise * float(fft_size),
        truncated_energy_fraction=truncated_energy_fraction,
    )


def wideband_pn_channel_estimate_for_header(
    header: np.ndarray,
    model: WidebandPnChannelModel,
    *,
    regularization: float = 1e-3,
) -> PnChannelEstimate:
    """Instantiate the wideband channel model for one frame's header.

    The frame-specific information is a single complex scale measured on the
    dominant tap: it carries the frame's common phase (so the PN equalizer
    keeps its per-frame absolute phase lock, see receiver CPE notes) and any
    slow gain variation, while the multi-tap structure comes from the
    low-noise cross-frame template.
    """

    _prefix, core_length, _suffix = _cyclic_layout(model.mode)
    header_phase = detect_pn_phase(header, mode=model.mode)
    taps = _pn_core_taps(
        header,
        mode=model.mode,
        pn_phase=header_phase,
        regularization=regularization,
    ).astype(np.complex128)
    rotated = np.roll(taps, -int(model.rotation_symbols))
    anchor = rotated[int(model.dominant_tap_index)]
    reference = np.complex128(
        model.template_taps[int(model.dominant_tap_index)]
    )
    metric = 1.0
    scale = np.complex128(1.0)
    anchor_power = float(np.abs(anchor)) ** 2
    noise_floor = float(model.per_frame_noise_tap_power)
    if abs(reference) > 0.0 and anchor_power > 4.0 * noise_floor:
        scale = anchor / reference
        magnitude = abs(scale)
        if magnitude > 4.0 or magnitude < 0.25:
            scale = scale / magnitude
        if noise_floor > 0.0:
            metric = float(min(anchor_power / (anchor_power + noise_floor), 1.0))
    else:
        metric = 0.0
    frame_taps = (
        np.asarray(model.template_taps, dtype=np.complex128) * scale
    ).astype(np.complex64)
    response = np.fft.fft(frame_taps, model.fft_size).astype(np.complex64)
    estimate = PnChannelEstimate(
        mode=model.mode,
        pn_phase=int((header_phase - int(model.rotation_symbols)) % core_length),
        delay_symbols=-model.signed_rotation(core_length),
        metric=metric,
        taps=frame_taps,
        response=response,
    )
    # Re-referencing pn_phase by the causal rotation makes the restored body a
    # circular shift of the true transmitted body (the restore/equalize chain
    # operates in the rotated frame coordinate). Fold the compensating FFT
    # phase ramp into the response so the equalized spectrum is aligned with
    # the unrotated body the downstream deinterleaver expects.
    rotation = model.signed_rotation(core_length)
    if rotation != 0:
        estimate = apply_body_window_offset_to_pn_channel_response(
            estimate,
            -rotation,
        )
    return estimate


def _threshold_support_mask(
    averaged_power: np.ndarray,
    *,
    noise_power: float,
    threshold_factor: float,
    min_relative_power: float,
) -> np.ndarray:
    peak = float(np.max(averaged_power))
    threshold = max(
        threshold_factor ** 2 * max(noise_power, 0.0),
        peak * min_relative_power,
    )
    return averaged_power >= threshold


def _dilate_linear_mask(mask: np.ndarray, guard_taps: int) -> np.ndarray:
    """Expand selected taps without wrapping guard noise across tap zero."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if guard_taps <= 0 or not bool(np.any(values)):
        return values.copy()
    selected = np.flatnonzero(values)
    dilated = values.copy()
    for index in selected:
        start = max(0, int(index) - guard_taps)
        stop = min(values.size, int(index) + guard_taps + 1)
        dilated[start:stop] = True
    return dilated


def _longest_circular_false_run(mask: np.ndarray) -> tuple[int, int]:
    """Return (start, length) of the longest circular run of False values."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    size = values.size
    if size == 0:
        return 0, 0
    if not bool(np.any(~values)):
        return 0, 0
    if not bool(np.any(values)):
        return 0, size
    doubled = np.concatenate((values, values))
    best_start = 0
    best_length = 0
    run_start = None
    for index in range(2 * size):
        if not doubled[index]:
            if run_start is None:
                run_start = index
            length = index - run_start + 1
            if length > best_length and run_start < size:
                best_length = length
                best_start = run_start
        else:
            run_start = None
    return best_start % size, min(best_length, size)


def equalize_spectrum_with_channel(
    spectrum: np.ndarray,
    estimate: PnChannelEstimate,
    *,
    floor: float = 1e-6,
    noise_variance: float | None = None,
) -> np.ndarray:
    """Equalize one frame-body spectrum with a PN-derived channel estimate.

    With ``noise_variance is None`` this is the historical pure zero-forcing
    (ZF) inverse ``X / H`` with a magnitude floor. ZF is exact on a noise-free
    channel but amplifies noise wherever ``|H|`` is small (spectral nulls from
    multipath), which is the dominant real-capture failure documented in
    AGENTS.md.

    With a positive ``noise_variance`` (``N``) this becomes the MMSE inverse
    ``X * conj(H) / (|H|^2 + N)``. MMSE limits null amplification by shrinking
    low-SNR carriers toward zero, trading a small high-SNR bias for a large
    reduction in deep-null noise gain. ``N`` is the per-carrier noise power
    relative to the channel/body scale; see
    :func:`estimate_frequency_domain_noise_variance`.
    """

    values = np.asarray(spectrum, dtype=np.complex64)
    if values.size != estimate.response.size:
        raise ValueError("spectrum and channel response sizes differ")
    response = np.asarray(estimate.response, dtype=np.complex64)
    if noise_variance is None:
        return (values / np.where(np.abs(response) >= floor, response, 1.0)).astype(
            np.complex64
        )
    if noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative")
    denom = np.abs(response) ** 2 + np.float64(noise_variance)
    return (values * np.conj(response) / denom).astype(np.complex64)


def average_pn_channel_estimates(
    estimates: Sequence[PnChannelEstimate],
    *,
    align_common_phase: bool = True,
    fft_size: int | None = None,
) -> PnChannelEstimate:
    """Average several PN-header channel estimates into one low-noise estimate.

    Each PN header gives an independent noisy snapshot of the same (slowly
    varying) channel. Averaging ``K`` snapshots reduces the channel-estimate
    noise by roughly ``sqrt(K)``, which is the field-SNR robustness item called
    out in AGENTS.md.

    Real captures carry residual CFO/CPE, so each snapshot's response is rotated
    by an unknown per-frame common phase. Averaging the raw complex responses
    would let those phases cancel destructively. With
    ``align_common_phase=True`` each estimate is first de-rotated by the phase of
    its dominant time-domain tap (a clean per-frame phase anchor) before
    averaging; the averaged channel then carries the dominant-tap-real-positive
    convention and downstream per-frame CPE correction restores absolute phase.
    """

    if not estimates:
        raise ValueError("estimates must be non-empty")
    modes = {est.mode for est in estimates}
    if len(modes) != 1:
        raise ValueError("all estimates must share one PN mode")
    response_size = fft_size if fft_size is not None else estimates[0].response.size
    tap_len = max(int(est.taps.size) for est in estimates)
    aligned_taps = np.zeros((len(estimates), tap_len), dtype=np.complex128)
    for row, est in enumerate(estimates):
        taps = np.asarray(est.taps, dtype=np.complex128).reshape(-1)
        rotation = np.complex128(1.0)
        if align_common_phase and taps.size:
            dominant = int(np.argmax(np.abs(taps)))
            peak = taps[dominant]
            if abs(peak) > 0.0:
                rotation = np.conj(peak) / abs(peak)
        aligned_taps[row, : taps.size] = taps * rotation
    mean_taps = aligned_taps.mean(axis=0).astype(np.complex64)
    response = np.fft.fft(mean_taps, response_size).astype(np.complex64)
    metric = float(np.mean([float(est.metric) for est in estimates]))
    return PnChannelEstimate(
        mode=estimates[0].mode,
        pn_phase=int(estimates[0].pn_phase),
        delay_symbols=int(estimates[0].delay_symbols),
        metric=metric,
        taps=mean_taps,
        response=response,
    )


def estimate_frequency_domain_noise_variance(
    estimate: PnChannelEstimate,
    *,
    noise_tap_fraction: float = 0.5,
) -> float:
    """Estimate per-carrier MMSE noise variance from PN-channel tap noise.

    DTMB multipath energy is concentrated in the first taps of the PN channel
    impulse response. The trailing taps are dominated by estimation noise, so
    their mean power is a usable proxy for the per-carrier noise variance ``N``
    used by the MMSE branch of :func:`equalize_spectrum_with_channel`. The
    returned value is scaled by the FFT length so it matches the
    frequency-domain ``|H|^2`` scale.

    ``noise_tap_fraction`` selects the trailing fraction of taps treated as the
    noise floor.
    """

    if not 0.0 < noise_tap_fraction < 1.0:
        raise ValueError("noise_tap_fraction must be in (0, 1)")
    taps = np.asarray(estimate.taps, dtype=np.complex128).reshape(-1)
    if taps.size < 4:
        return 0.0
    split = int(taps.size * (1.0 - noise_tap_fraction))
    split = min(max(split, 1), taps.size - 1)
    noise_taps = taps[split:]
    if noise_taps.size == 0:
        return 0.0
    tap_noise_power = float(np.mean(np.abs(noise_taps) ** 2))
    return tap_noise_power * float(estimate.response.size)


def shift_pn_channel_response(
    estimate: PnChannelEstimate,
    fft_bin_shift: int,
) -> PnChannelEstimate:
    """Align a PN response with a spectrum that has been rolled by bin shift."""

    shift = int(fft_bin_shift)
    if shift == 0:
        return estimate
    return replace(
        estimate,
        response=np.roll(np.asarray(estimate.response, dtype=np.complex64), shift).astype(
            np.complex64,
            copy=False,
        ),
    )


def apply_body_window_offset_to_pn_channel_response(
    estimate: PnChannelEstimate,
    body_window_offset_symbols: int,
) -> PnChannelEstimate:
    """Apply the FFT phase ramp induced by a circular body-window shift."""

    offset = int(body_window_offset_symbols)
    if offset == 0:
        return estimate
    response = np.asarray(estimate.response, dtype=np.complex64)
    bins = np.arange(response.size, dtype=np.float64)
    ramp = np.exp((2j * np.pi * float(offset) / float(response.size)) * bins)
    return replace(
        estimate,
        response=(response * ramp.astype(np.complex64)).astype(
            np.complex64,
            copy=False,
        ),
    )


def _repair_low_energy_pn_dc_response(
    response_core: np.ndarray,
    ref_fft: np.ndarray,
) -> np.ndarray:
    """Interpolate the PN-core DC channel bin when capture DC removal corrupts it."""

    response = np.asarray(response_core, dtype=np.complex64)
    reference = np.asarray(ref_fft, dtype=np.complex64)
    if response.size < 3 or reference.size != response.size:
        return response
    ref_power = np.abs(reference) ** 2
    typical_ref_power = float(np.median(ref_power[1:]))
    if typical_ref_power <= 0.0 or float(ref_power[0]) > typical_ref_power * 0.01:
        return response
    provisional_taps = np.fft.ifft(response)
    tap_magnitudes = np.abs(provisional_taps)
    peak = float(np.max(tap_magnitudes))
    if peak <= 0.0:
        return response
    significant = np.flatnonzero(tap_magnitudes >= peak * 0.05)
    if significant.size and int(significant[-1]) > 64:
        return response
    interpolated = (response[1] + response[-1]) * np.complex64(0.5)
    scale = max(float(abs(interpolated)), 1e-12)
    if float(abs(response[0] - interpolated)) <= scale * 0.25:
        return response
    repaired = response.copy()
    repaired[0] = interpolated
    return repaired


def detect_pn_phase(header: np.ndarray, *, mode: PnMode) -> int:
    """Return the zero-delay PN cyclic phase present in ``header``.

    The delay-tolerant matcher in :mod:`dtmb.frame_sync` can return phase/lag
    pairs that are aliases of the same physical sequence. For PN-tail
    cancellation we want the *physical* phase that was transmitted at the
    sample boundaries we are currently using, so we run a zero-delay family
    correlation. This is appropriate when the frame boundaries are already
    aligned via cyclic-extension acquisition or manual ``phase_offset``.
    """

    signal = np.asarray(header, dtype=np.complex64).reshape(-1)
    family = pn_cyclic_family_symbols(mode)
    if signal.size != family.shape[1]:
        raise ValueError("header length does not match PN mode")
    correlations = family @ signal
    return int(np.argmax(np.abs(correlations)))


def pn_tail_cancel_body(
    body: np.ndarray,
    *,
    pn_phase: int,
    mode: PnMode,
    taps: np.ndarray,
) -> np.ndarray:
    """Subtract the PN header's linear-convolution tail from a received body.

    In DTMB TDS-OFDM the PN header is transmitted immediately before the 3780-
    sample multicarrier body. After passing through a multipath channel with
    impulse response ``h`` of length ``L``, the first ``L-1`` samples of the
    received body are contaminated by ``h * pn`` tails. Subtracting that tail
    leaves only the channel's response to the body signal itself.
    """

    body_values = np.asarray(body, dtype=np.complex64).reshape(-1)
    if body_values.size == 0:
        return body_values.copy()
    taps_values = np.asarray(taps, dtype=np.complex64).reshape(-1)
    if taps_values.size <= 1:
        return body_values.copy()
    definition = PN_DEFINITIONS[mode]
    family = pn_cyclic_family_symbols(mode)
    if not 0 <= pn_phase < family.shape[0]:
        raise ValueError("pn_phase is out of range for the PN family")
    reference = family[pn_phase]
    full = np.convolve(reference, taps_values)
    tail_len = min(taps_values.size - 1, body_values.size)
    if tail_len <= 0:
        return body_values.copy()
    header_symbols = definition.header_symbols
    tail = full[header_symbols : header_symbols + tail_len]
    cleaned = body_values.copy()
    cleaned[:tail_len] = cleaned[:tail_len] - tail
    return cleaned


def pn_circular_restore_body(
    cleaned_body: np.ndarray,
    *,
    pn_phase: int,
    mode: PnMode,
    taps: np.ndarray,
    next_header: np.ndarray,
) -> np.ndarray:
    """Reconstruct the cyclic-convolution body from a PN-tail-cancelled body.

    After :func:`pn_tail_cancel_body` removes the previous PN header's ISI, the
    received body still contains the *linear* convolution of the channel with
    the body samples. To make a 3780-point FFT see the true channel response
    (which assumes circular convolution) the missing wrap-around tail must be
    added back. That tail is exactly the body's ISI leaking into the next PN
    header. It can be recovered from the next PN header's received samples by
    subtracting the known ``h * pn_next`` portion.
    """

    cleaned = np.asarray(cleaned_body, dtype=np.complex64).reshape(-1)
    taps_values = np.asarray(taps, dtype=np.complex64).reshape(-1)
    if cleaned.size == 0 or taps_values.size <= 1:
        return cleaned.copy()
    next_samples = np.asarray(next_header, dtype=np.complex64).reshape(-1)
    family = pn_cyclic_family_symbols(mode)
    if not 0 <= pn_phase < family.shape[0]:
        raise ValueError("pn_phase is out of range for the PN family")
    pn_next = family[pn_phase]
    tail_len = min(taps_values.size - 1, cleaned.size, next_samples.size)
    if tail_len <= 0:
        return cleaned.copy()
    pn_next_clean = np.convolve(pn_next, taps_values)[:tail_len]
    body_tail = next_samples[:tail_len] - pn_next_clean
    restored = cleaned.copy()
    restored[:tail_len] = restored[:tail_len] + body_tail
    return restored


def pn_restore_circular_body_window(
    body: np.ndarray,
    *,
    body_window_offset_symbols: int = 0,
    pn_phase: int,
    next_pn_phase: int,
    mode: PnMode,
    taps: np.ndarray,
    next_header: np.ndarray,
) -> np.ndarray:
    """Return a PN-ISI-cancelled circular body at a diagnostic FFT offset.

    ``body`` must be the received C=3780 body that starts immediately after the
    current PN header. The PN-tail subtraction and next-header circular restore
    are defined at that physical boundary. A nonzero body-window offset is then
    represented as a circular time shift of the restored body, avoiding the
    invalid alternative of cutting samples from the next/previous PN header into
    the FFT window.
    """

    restored = pn_circular_restore_body(
        pn_tail_cancel_body(
            body,
            pn_phase=pn_phase,
            mode=mode,
            taps=taps,
        ),
        pn_phase=next_pn_phase,
        mode=mode,
        taps=taps,
        next_header=next_header,
    )
    offset = int(body_window_offset_symbols)
    if offset == 0:
        return restored
    return np.roll(restored, -offset).astype(np.complex64, copy=False)


def _cyclic_layout(mode: PnMode) -> tuple[int, int, int]:
    if mode == "pn420":
        return 82, 255, 83
    if mode == "pn945":
        return 217, 511, 217
    raise ValueError("PN channel estimation requires PN420 or PN945")


def _resolve_channel_taps(mode: PnMode, channel_taps: int | None) -> int:
    if channel_taps is not None:
        return int(channel_taps)
    _prefix, body_length, _suffix = _cyclic_layout(mode)
    return body_length
