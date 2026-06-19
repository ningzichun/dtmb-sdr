"""PN-based frame synchronization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .pn import PN_DEFINITIONS, PnMode, pn_bipolar, pn_cyclic_family_bipolar


DTMB_SYMBOL_RATE_SPS = 7_560_000


@dataclass(frozen=True)
class PnCorrelationPeak:
    """A detected PN correlation peak."""

    mode: PnMode
    offset: int
    metric: float
    frame_symbols: int
    header_symbols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "offset": self.offset,
            "metric": self.metric,
            "frame_symbols": self.frame_symbols,
            "header_symbols": self.header_symbols,
        }


@dataclass(frozen=True)
class PnFrameTrain:
    """Correlation score for a repeated DTMB frame cadence."""

    mode: PnMode
    phase_offset: int
    frame_symbols: int
    header_symbols: int
    mean_metric: float
    max_metric: float
    hit_count: int
    observed_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "phase_offset": self.phase_offset,
            "frame_symbols": self.frame_symbols,
            "header_symbols": self.header_symbols,
            "mean_metric": self.mean_metric,
            "max_metric": self.max_metric,
            "hit_count": self.hit_count,
            "observed_frames": self.observed_frames,
        }


@dataclass(frozen=True)
class PnFamilyDelayMatch:
    """Best PN phase/delay match for one candidate frame header."""

    mode: PnMode
    frame_start: int
    pn_phase: int
    delay_symbols: int
    metric: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frame_start": self.frame_start,
            "pn_phase": self.pn_phase,
            "delay_symbols": self.delay_symbols,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class PnFamilyDelayTrain:
    """Delay-tolerant PN-family score at one repeated frame phase."""

    mode: PnMode
    phase_offset: int
    frame_symbols: int
    header_symbols: int
    mean_metric: float
    max_metric: float
    hit_count: int
    observed_frames: int
    median_delay_symbols: float
    delay_mad_symbols: float
    dominant_pn_phase: int | None
    dominant_pn_phase_count: int
    frame_matches: tuple[PnFamilyDelayMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "phase_offset": self.phase_offset,
            "frame_symbols": self.frame_symbols,
            "header_symbols": self.header_symbols,
            "mean_metric": self.mean_metric,
            "max_metric": self.max_metric,
            "hit_count": self.hit_count,
            "observed_frames": self.observed_frames,
            "median_delay_symbols": self.median_delay_symbols,
            "delay_mad_symbols": self.delay_mad_symbols,
            "dominant_pn_phase": self.dominant_pn_phase,
            "dominant_pn_phase_count": self.dominant_pn_phase_count,
            "frame_matches": [match.to_dict() for match in self.frame_matches],
        }


def should_apply_pn_family_delay(
    train: PnFamilyDelayTrain,
    *,
    family_hit_threshold: float = 0.45,
    min_observed_frames: int = 8,
) -> bool:
    """Return whether a PN-family delay train is stable enough to seed timing."""

    if train.observed_frames < min_observed_frames:
        return False
    required_hits = max(2, int(np.ceil(train.observed_frames * 0.5)))
    if train.hit_count < required_hits:
        return False
    if train.mean_metric < family_hit_threshold:
        return False
    if abs(train.median_delay_symbols) < 0.5:
        return False
    # A real timing delay should be stable across adjacent frames. Scattered
    # best lags are usually a strong cyclic-only artifact or an unlocked frame.
    return train.delay_mad_symbols <= 8.0


def delay_corrected_phase_offset(
    phase_offset: int,
    *,
    median_delay_symbols: float,
    frame_symbols: int,
) -> int:
    """Apply a stable PN-family delay to a repeated frame phase."""

    corrected = phase_offset - int(round(median_delay_symbols))
    return corrected % frame_symbols


def normalized_sliding_correlation(
    samples: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Return magnitude-normalized sliding correlation metrics."""

    signal = np.asarray(samples, dtype=np.complex64)
    ref = np.asarray(reference, dtype=np.complex64)
    if signal.ndim != 1 or ref.ndim != 1:
        raise ValueError("samples and reference must be one-dimensional")
    if ref.size == 0:
        raise ValueError("reference must not be empty")
    if signal.size < ref.size:
        return np.asarray([], dtype=np.float32)

    correlation = np.correlate(signal, ref, mode="valid")
    sample_power = np.abs(signal) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power, dtype=np.float64)))
    window_power = cumulative[ref.size :] - cumulative[: -ref.size]
    ref_power = float(np.sum(np.abs(ref) ** 2))
    denominator = np.sqrt(np.maximum(window_power * ref_power, 1e-30))
    return (np.abs(correlation) / denominator).astype(np.float32)


def strongest_peaks(
    metric: np.ndarray, *, max_peaks: int = 5, min_distance: int = 1
) -> list[tuple[int, float]]:
    """Return strongest local peaks with simple distance suppression."""

    values = np.asarray(metric)
    if values.size == 0 or max_peaks <= 0:
        return []

    order = np.argsort(values)[::-1]
    chosen: list[tuple[int, float]] = []
    suppressed = np.zeros(values.size, dtype=bool)
    radius = max(0, min_distance)
    for raw_index in order:
        index = int(raw_index)
        if suppressed[index]:
            continue
        chosen.append((index, float(values[index])))
        if len(chosen) >= max_peaks:
            break
        low = max(0, index - radius)
        high = min(values.size, index + radius + 1)
        suppressed[low:high] = True
    return chosen


def detect_pn_headers(
    symbols: np.ndarray,
    *,
    modes: Iterable[PnMode] = ("pn420", "pn595", "pn945"),
    max_peaks_per_mode: int = 3,
    min_peak_distance_symbols: int | None = None,
) -> list[PnCorrelationPeak]:
    """Detect PN header candidates in symbol-rate complex samples."""

    signal = np.asarray(symbols, dtype=np.complex64)
    peaks: list[PnCorrelationPeak] = []
    for mode in modes:
        definition = PN_DEFINITIONS[mode]
        reference = pn_bipolar(mode).astype(np.complex64)
        metric = normalized_sliding_correlation(signal, reference)
        min_distance = (
            min_peak_distance_symbols
            if min_peak_distance_symbols is not None
            else max(1, definition.header_symbols // 2)
        )
        for offset, value in strongest_peaks(
            metric, max_peaks=max_peaks_per_mode, min_distance=min_distance
        ):
            peaks.append(
                PnCorrelationPeak(
                    mode=mode,
                    offset=offset,
                    metric=value,
                    frame_symbols=definition.frame_symbols,
                    header_symbols=definition.header_symbols,
                )
            )
    return sorted(peaks, key=lambda peak: peak.metric, reverse=True)


def detect_pn_frame_trains(
    symbols: np.ndarray,
    *,
    modes: Iterable[PnMode] = ("pn420", "pn595", "pn945"),
    hit_threshold: float = 0.35,
    max_trains_per_mode: int = 3,
) -> list[PnFrameTrain]:
    """Score PN correlation peaks that repeat at valid DTMB frame intervals."""

    signal = np.asarray(symbols, dtype=np.complex64)
    trains: list[PnFrameTrain] = []
    for mode in modes:
        definition = PN_DEFINITIONS[mode]
        reference = pn_bipolar(mode).astype(np.complex64)
        metric = normalized_sliding_correlation(signal, reference)
        if metric.size == 0:
            continue
        scores = _score_metric_by_frame_phase(
            metric,
            frame_symbols=definition.frame_symbols,
            hit_threshold=hit_threshold,
        )
        for phase_offset, mean_metric, max_metric, hit_count, observed_frames in scores[
            :max_trains_per_mode
        ]:
            trains.append(
                PnFrameTrain(
                    mode=mode,
                    phase_offset=phase_offset,
                    frame_symbols=definition.frame_symbols,
                    header_symbols=definition.header_symbols,
                    mean_metric=mean_metric,
                    max_metric=max_metric,
                    hit_count=hit_count,
                    observed_frames=observed_frames,
                )
            )
    return sorted(
        trains,
        key=lambda train: (train.hit_count, train.mean_metric, train.max_metric),
        reverse=True,
    )


def pn_cyclic_extension_metric(symbols: np.ndarray, *, mode: PnMode) -> np.ndarray:
    """Return phase-independent PN cyclic-extension similarity metrics.

    PN420 and PN945 repeat the start/end of the underlying m-sequence inside the
    header. This detector does not require knowing the frame's PN phase.
    """

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    if mode == "pn420":
        pairs = ((0, 255, 82), (82, 337, 83))
    elif mode == "pn945":
        pairs = ((0, 511, 217), (217, 728, 217))
    else:
        raise ValueError("cyclic-extension metrics are only defined for PN420 and PN945")

    if signal.size < definition.header_symbols:
        return np.asarray([], dtype=np.float32)

    pair_metrics = [
        _shifted_window_similarity(
            signal,
            first_offset=first_offset,
            second_offset=second_offset,
            length=length,
            header_symbols=definition.header_symbols,
        )
        for first_offset, second_offset, length in pairs
    ]
    return np.mean(np.vstack(pair_metrics), axis=0).astype(np.float32)


def detect_pn_cyclic_extension_trains(
    symbols: np.ndarray,
    *,
    modes: Iterable[PnMode] = ("pn420", "pn945"),
    hit_threshold: float = 0.35,
    max_trains_per_mode: int = 3,
) -> list[PnFrameTrain]:
    """Score repeated PN420/PN945 headers without requiring a fixed PN phase."""

    trains: list[PnFrameTrain] = []
    for mode in modes:
        definition = PN_DEFINITIONS[mode]
        metric = pn_cyclic_extension_metric(symbols, mode=mode)
        if metric.size == 0:
            continue
        scores = _score_metric_by_frame_phase(
            metric,
            frame_symbols=definition.frame_symbols,
            hit_threshold=hit_threshold,
        )
        for phase_offset, mean_metric, max_metric, hit_count, observed_frames in scores[
            :max_trains_per_mode
        ]:
            trains.append(
                PnFrameTrain(
                    mode=mode,
                    phase_offset=phase_offset,
                    frame_symbols=definition.frame_symbols,
                    header_symbols=definition.header_symbols,
                    mean_metric=mean_metric,
                    max_metric=max_metric,
                    hit_count=hit_count,
                    observed_frames=observed_frames,
                )
            )
    return sorted(
        trains,
        key=lambda train: (train.hit_count, train.mean_metric, train.max_metric),
        reverse=True,
    )


def detect_pn_phase_family_trains(
    symbols: np.ndarray,
    *,
    modes: Iterable[PnMode] = ("pn420", "pn945"),
    extension_hit_threshold: float = 0.25,
    family_hit_threshold: float = 0.45,
    candidate_phases_per_mode: int = 8,
    max_trains_per_mode: int = 3,
) -> list[PnFrameTrain]:
    """Score PN trains against all cyclic PN phases for PN420/PN945."""

    signal = np.asarray(symbols, dtype=np.complex64)
    trains: list[PnFrameTrain] = []
    mode_list = tuple(modes)
    for mode in mode_list:
        definition = PN_DEFINITIONS[mode]
        extension_metric = pn_cyclic_extension_metric(signal, mode=mode)
        if extension_metric.size == 0:
            continue
        candidate_scores = _score_metric_by_frame_phase(
            extension_metric,
            frame_symbols=definition.frame_symbols,
            hit_threshold=extension_hit_threshold,
        )[:candidate_phases_per_mode]
        mode_trains: list[PnFrameTrain] = []
        for phase_offset, *_ in candidate_scores:
            values = _pn_family_metrics_at_frame_phase(
                signal,
                mode=mode,
                phase_offset=phase_offset,
            )
            if values.size == 0:
                continue
            hit_count = int(np.count_nonzero(values >= family_hit_threshold))
            hits = values[values >= family_hit_threshold]
            strongest = hits if hits.size else np.sort(values)[-min(8, values.size) :]
            mode_trains.append(
                PnFrameTrain(
                    mode=mode,
                    phase_offset=phase_offset,
                    frame_symbols=definition.frame_symbols,
                    header_symbols=definition.header_symbols,
                    mean_metric=float(np.mean(strongest)),
                    max_metric=float(np.max(values)),
                    hit_count=hit_count,
                    observed_frames=int(values.size),
                )
            )
        trains.extend(
            sorted(
                mode_trains,
                key=lambda train: (train.hit_count, train.mean_metric, train.max_metric),
                reverse=True,
            )[:max_trains_per_mode]
        )
    return sorted(
        trains,
        key=lambda train: (train.hit_count, train.mean_metric, train.max_metric),
        reverse=True,
    )


def pn_family_delay_match(
    header: np.ndarray,
    *,
    mode: PnMode,
    frame_start: int = 0,
    max_delay_symbols: int = 128,
) -> PnFamilyDelayMatch:
    """Return the best cyclic PN-family match allowing timing/channel delay.

    A direct zero-delay PN correlation is fragile on TDS-OFDM captures because
    the PN header is filtered by the channel. This score correlates every cyclic
    PN phase against the header and permits a bounded integer-symbol delay.
    """

    definition = PN_DEFINITIONS[mode]
    signal = np.asarray(header, dtype=np.complex64)
    if signal.ndim != 1:
        raise ValueError("header must be one-dimensional")
    if signal.size != definition.header_symbols:
        raise ValueError("header length does not match PN mode")
    if max_delay_symbols < 0:
        raise ValueError("max_delay_symbols must be non-negative")
    lags, metrics, phases = _pn_family_delay_profile(
        signal,
        mode=mode,
        max_delay_symbols=max_delay_symbols,
    )
    lag_index = _select_best_delay_lag(metrics, lags)
    return PnFamilyDelayMatch(
        mode=mode,
        frame_start=frame_start,
        pn_phase=int(phases[lag_index]),
        delay_symbols=int(lags[lag_index]),
        metric=float(metrics[lag_index]),
    )


def score_pn_family_delay_train(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    max_frames: int = 16,
    max_delay_symbols: int = 128,
    hit_threshold: float = 0.45,
) -> PnFamilyDelayTrain:
    """Score repeated headers with delay-tolerant PN-family matching."""

    if phase_offset < 0:
        raise ValueError("phase_offset must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    definition = PN_DEFINITIONS[mode]
    signal = np.asarray(symbols, dtype=np.complex64)
    starts: list[int] = []
    lag_values = None
    metric_rows = []
    phase_rows = []
    for start in range(
        phase_offset,
        signal.size - definition.header_symbols + 1,
        definition.frame_symbols,
    ):
        header = signal[start : start + definition.header_symbols]
        lags, metrics, phases = _pn_family_delay_profile(
            header,
            mode=mode,
            max_delay_symbols=max_delay_symbols,
        )
        if lag_values is None:
            lag_values = lags
        starts.append(start)
        metric_rows.append(metrics)
        phase_rows.append(phases)
        if len(starts) >= max_frames:
            break
    if not metric_rows or lag_values is None:
        return PnFamilyDelayTrain(
            mode=mode,
            phase_offset=phase_offset,
            frame_symbols=definition.frame_symbols,
            header_symbols=definition.header_symbols,
            mean_metric=0.0,
            max_metric=0.0,
            hit_count=0,
            observed_frames=0,
            median_delay_symbols=0.0,
            delay_mad_symbols=0.0,
            dominant_pn_phase=None,
            dominant_pn_phase_count=0,
            frame_matches=(),
        )

    metric_matrix = np.vstack(metric_rows)
    phase_matrix = np.vstack(phase_rows)
    selected_lag_indices = [
        _select_best_delay_lag(row, lag_values) for row in metric_matrix
    ]
    matches = tuple(
        PnFamilyDelayMatch(
            mode=mode,
            frame_start=start,
            pn_phase=int(phase_matrix[row_index, selected_lag_indices[row_index]]),
            delay_symbols=int(lag_values[selected_lag_indices[row_index]]),
            metric=float(metric_matrix[row_index, selected_lag_indices[row_index]]),
        )
        for row_index, start in enumerate(starts)
    )
    metrics = np.asarray([match.metric for match in matches], dtype=np.float32)
    delays = np.asarray([match.delay_symbols for match in matches], dtype=np.float32)
    phases = np.asarray([match.pn_phase for match in matches], dtype=np.int32)
    hit_count = int(np.count_nonzero(metrics >= hit_threshold))
    unique_phases, phase_counts = np.unique(phases, return_counts=True)
    dominant_index = int(np.argmax(phase_counts))
    dominant_phase = int(unique_phases[dominant_index])
    dominant_phase_count = int(phase_counts[dominant_index])
    median_delay = float(np.median(delays))
    return PnFamilyDelayTrain(
        mode=mode,
        phase_offset=phase_offset,
        frame_symbols=definition.frame_symbols,
        header_symbols=definition.header_symbols,
        mean_metric=float(np.mean(metrics)),
        max_metric=float(np.max(metrics)),
        hit_count=hit_count,
        observed_frames=len(matches),
        median_delay_symbols=median_delay,
        delay_mad_symbols=float(np.median(np.abs(delays - median_delay))),
        dominant_pn_phase=dominant_phase,
        dominant_pn_phase_count=dominant_phase_count,
        frame_matches=matches,
    )


def estimate_cfo_from_pn_cyclic_extension(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_frames: int = 16,
) -> float | None:
    """Estimate coarse CFO from repeated PN cyclic-extension segments."""

    if symbol_rate_sps <= 0:
        raise ValueError("symbol_rate_sps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    if mode == "pn420":
        pairs = ((0, 255, 82), (82, 337, 83))
        delay_symbols = 255
    elif mode == "pn945":
        pairs = ((0, 511, 217), (217, 728, 217))
        delay_symbols = 511
    else:
        raise ValueError("cyclic-extension CFO is only defined for PN420 and PN945")

    accumulator = 0j
    used_frames = 0
    for start in range(
        phase_offset,
        signal.size - definition.header_symbols + 1,
        definition.frame_symbols,
    ):
        for first_offset, second_offset, length in pairs:
            first = signal[start + first_offset : start + first_offset + length]
            second = signal[start + second_offset : start + second_offset + length]
            accumulator += complex(np.sum(np.conj(first) * second))
        used_frames += 1
        if used_frames >= max_frames:
            break
    if used_frames == 0 or abs(accumulator) == 0:
        return None
    phase = float(np.angle(accumulator))
    return phase * symbol_rate_sps / (2 * np.pi * delay_symbols)


def estimate_residual_cfo_from_frame_headers(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_frames: int = 600,
    pn_phase: int | None = None,
    min_fit_r_squared: float = 0.5,
) -> float | None:
    """Estimate fine residual CFO from the PN-header phase trend across frames.

    ``estimate_cfo_from_pn_cyclic_extension`` measures CFO from the two
    cyclic-extension copies inside one PN header (separated by ``delay_symbols``
    ~ 511 for PN945). Its precision is limited because the baseline is short and
    only ~16 frames are averaged. A small residual it cannot resolve (a few Hz)
    is harmless for mode1 but, accumulated across mode2's 510-frame symbol
    deinterleaver span, rotates each frame's constellation independently and
    collapses every LDPC codeword to the 0.50 plateau (see trial-log
    2026-05-31).

    This estimator instead correlates each frame's header against its detected
    cyclic PN-family reference to get that frame's absolute phase, then fits
    the phase slope against the frame index. Real PN945 broadcasts rotate PN
    phase across frames, so a fixed phase-0 reference produces a spurious
    slope. The baseline is the full frame spacing
    (``frame_symbols`` ~ 4725 for PN945), ~9x longer than the intra-header gap,
    and the slope is fit over up to ``max_frames`` frames, so the residual-CFO
    resolution improves by roughly two orders of magnitude. The result is the
    fine CFO *remaining after* the coarse correction; subtract it from the
    baseband to flatten the per-frame phase walk.

    Returns ``None`` when fewer than two usable frames are present.
    """

    if symbol_rate_sps <= 0:
        raise ValueError("symbol_rate_sps must be positive")
    if max_frames <= 1:
        raise ValueError("max_frames must be at least 2")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    header_symbols = definition.header_symbols
    frame_symbols = definition.frame_symbols

    family = pn_cyclic_family_bipolar(mode).astype(np.complex64)
    pinned_reference = (
        np.asarray(family[pn_phase % len(family)], dtype=np.complex64)
        if pn_phase is not None
        else None
    )

    frame_phases: list[float] = []
    frame_indices: list[int] = []
    weights: list[float] = []
    index = 0
    for start in range(
        phase_offset,
        signal.size - header_symbols + 1,
        frame_symbols,
    ):
        header = signal[start : start + header_symbols]
        if pinned_reference is None:
            correlations = family @ header
            reference = family[int(np.argmax(np.abs(correlations)))]
        else:
            reference = pinned_reference
        # Project the observed header onto the PN reference. The complex
        # projection's angle is the header's common phase; its magnitude
        # weights the linear fit toward cleaner headers.
        projection = complex(np.vdot(reference, header))
        if abs(projection) > 0.0:
            frame_phases.append(float(np.angle(projection)))
            frame_indices.append(index)
            weights.append(abs(projection))
        index += 1
        if index >= max_frames:
            break

    if len(frame_phases) < 2:
        return None

    idx = np.asarray(frame_indices, dtype=np.float64)
    # Unwrap the per-frame phase walk so the slope is not folded into [-pi, pi).
    phases = np.unwrap(np.asarray(frame_phases, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.max(w) if np.max(w) > 0 else np.ones_like(w)

    # Weighted least-squares slope (radians per frame).
    mean_idx = float(np.sum(w * idx) / np.sum(w))
    mean_phase = float(np.sum(w * phases) / np.sum(w))
    cov = float(np.sum(w * (idx - mean_idx) * (phases - mean_phase)))
    var = float(np.sum(w * (idx - mean_idx) ** 2))
    if var <= 0.0:
        return None
    slope_rad_per_frame = cov / var

    # Guard against fitting a spurious slope to a phase walk that is not a
    # genuine CFO ramp (e.g. random per-frame channel phase). Require the
    # weighted linear fit to explain most of the phase variance; otherwise the
    # "CFO" is not real and correcting it would do more harm than good.
    phase_var = float(np.sum(w * (phases - mean_phase) ** 2))
    if phase_var > 0.0:
        residual_var = phase_var - slope_rad_per_frame * cov
        r_squared = 1.0 - residual_var / phase_var
        if r_squared < min_fit_r_squared:
            return None

    # radians/frame -> Hz: divide by the frame period (frame_symbols / rate).
    return slope_rad_per_frame * symbol_rate_sps / (2 * np.pi * frame_symbols)


def _score_metric_by_frame_phase(
    metric: np.ndarray, *, frame_symbols: int, hit_threshold: float
) -> list[tuple[int, float, float, int, int]]:
    phase_count = min(frame_symbols, metric.size)
    scores: list[tuple[int, float, float, int, int]] = []
    for phase in range(phase_count):
        values = metric[phase::frame_symbols]
        if values.size == 0:
            continue
        hit_count = int(np.count_nonzero(values >= hit_threshold))
        hits = values[values >= hit_threshold]
        # Prefer the mean of actual threshold hits. If no hits exist, retain a
        # strongest-window score so noise-only captures still rank deterministically.
        strongest = hits if hits.size else np.sort(values)[-min(8, values.size) :]
        scores.append(
            (
                phase,
                float(np.mean(strongest)),
                float(np.max(values)),
                hit_count,
                int(values.size),
            )
        )
    return sorted(scores, key=lambda item: (item[3], item[1], item[2]), reverse=True)


def _shifted_window_similarity(
    signal: np.ndarray,
    *,
    first_offset: int,
    second_offset: int,
    length: int,
    header_symbols: int,
) -> np.ndarray:
    max_start = signal.size - header_symbols
    first = signal[first_offset : first_offset + max_start + length]
    second = signal[second_offset : second_offset + max_start + length]
    product = np.conj(first) * second
    numerator = _moving_sum(product, length)
    first_power = _moving_sum(np.abs(first) ** 2, length)
    second_power = _moving_sum(np.abs(second) ** 2, length)
    denominator = np.sqrt(np.maximum(first_power * second_power, 1e-30))
    return (np.abs(numerator) / denominator).astype(np.float32)


def _moving_sum(values: np.ndarray, length: int) -> np.ndarray:
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    cumulative = np.concatenate(
        (np.asarray([0], dtype=dtype), np.cumsum(values, dtype=dtype))
    )
    return cumulative[length:] - cumulative[:-length]


def _pn_family_metrics_at_frame_phase(
    signal: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
) -> np.ndarray:
    definition = PN_DEFINITIONS[mode]
    family = pn_cyclic_family_bipolar(mode).astype(np.complex64)
    metrics: list[float] = []
    reference_power = float(definition.header_symbols)
    for start in range(
        phase_offset,
        signal.size - definition.header_symbols + 1,
        definition.frame_symbols,
    ):
        header = signal[start : start + definition.header_symbols]
        header_power = float(np.sum(np.abs(header) ** 2))
        if header_power <= 0:
            metrics.append(0.0)
            continue
        correlations = family @ header
        metric = float(
            np.max(np.abs(correlations)) / np.sqrt(reference_power * header_power)
        )
        metrics.append(metric)
    return np.asarray(metrics, dtype=np.float32)


def _pn_family_delay_profile(
    header: np.ndarray,
    *,
    mode: PnMode,
    max_delay_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    definition = PN_DEFINITIONS[mode]
    family = pn_cyclic_family_bipolar(mode).astype(np.complex64)
    correlations = _batch_full_correlation(header, family)
    lags = np.arange(
        -min(max_delay_symbols, definition.header_symbols - 1),
        min(max_delay_symbols, definition.header_symbols - 1) + 1,
        dtype=np.int32,
    )
    indices = lags + definition.header_symbols - 1
    denominators = _lag_correlation_denominators(header, lags)
    metrics = np.abs(correlations[:, indices]) / denominators[None, :]
    phases = np.argmax(metrics, axis=0).astype(np.int32)
    best_by_lag = metrics[phases, np.arange(metrics.shape[1])]
    return lags, best_by_lag.astype(np.float32), phases


def _select_best_delay_lag(metrics: np.ndarray, lags: np.ndarray) -> int:
    """Choose the strongest lag, preferring zero/short delays on metric ties."""

    values = np.asarray(metrics, dtype=np.float64)
    lag_values = np.asarray(lags, dtype=np.int32)
    if values.ndim != 1 or lag_values.ndim != 1 or values.size != lag_values.size:
        raise ValueError("metrics and lags must be one-dimensional and aligned")
    if values.size == 0:
        raise ValueError("metrics must not be empty")
    max_metric = float(np.max(values))
    tolerance = max(1e-6, abs(max_metric) * 1e-6)
    candidates = np.flatnonzero(values >= max_metric - tolerance)
    return int(
        min(
            candidates,
            key=lambda index: (abs(int(lag_values[index])), int(lag_values[index])),
        )
    )


def _select_stable_delay_lag(metric_matrix: np.ndarray, *, hit_threshold: float) -> int:
    hit_mask = metric_matrix >= hit_threshold
    hit_counts = np.count_nonzero(hit_mask, axis=0)
    hit_sums = np.sum(np.where(hit_mask, metric_matrix, 0.0), axis=0)
    hit_means = np.divide(
        hit_sums,
        np.maximum(hit_counts, 1),
        out=np.zeros_like(hit_sums, dtype=np.float64),
        where=hit_counts > 0,
    )
    mean_metrics = np.mean(metric_matrix, axis=0)
    max_metrics = np.max(metric_matrix, axis=0)
    order = np.lexsort((max_metrics, mean_metrics, hit_means, hit_counts))
    return int(order[-1])


def _batch_full_correlation(header: np.ndarray, references: np.ndarray) -> np.ndarray:
    length = header.size + references.shape[1] - 1
    fft_length = 1 << (length - 1).bit_length()
    header_fft = np.fft.fft(header, fft_length)
    reversed_refs = np.conj(references[:, ::-1])
    ref_fft = np.fft.fft(reversed_refs, fft_length, axis=1)
    return np.fft.ifft(ref_fft * header_fft[None, :], axis=1)[:, :length]


def _lag_correlation_denominators(header: np.ndarray, lags: np.ndarray) -> np.ndarray:
    power = np.abs(header) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    denominators = []
    size = header.size
    for raw_lag in lags:
        lag = int(raw_lag)
        if lag >= 0:
            header_start = lag
            length = size - lag
        else:
            header_start = 0
            length = size + lag
        if length <= 0:
            denominators.append(np.inf)
            continue
        header_power = cumulative[header_start + length] - cumulative[header_start]
        ref_power = float(length)
        denominators.append(np.sqrt(max(header_power * ref_power, 1e-30)))
    return np.asarray(denominators, dtype=np.float64)
