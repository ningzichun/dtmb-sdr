"""Frame-body timing search diagnostics for DTMB captures."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .equalization import equalize_c3780_spectrum_with_system_info_pilots
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    delay_corrected_phase_offset,
    detect_pn_cyclic_extension_trains,
    estimate_cfo_from_pn_cyclic_extension,
    score_pn_family_delay_train,
    should_apply_pn_family_delay,
)
from .channel import (
    detect_pn_phase,
    equalize_spectrum_with_channel,
    estimate_pn_channel,
    estimate_pn_channel_compact,
    pn_circular_restore_body,
    pn_restore_circular_body_window,
    pn_tail_cancel_body,
)
from .frames import iter_signal_frames
from .frequency import (
    FRAME_BODY_SYMBOLS,
    frame_body_fft,
    frequency_deinterleave,
    split_system_info_and_data,
)
from .pn import PN_DEFINITIONS, PnMode, pn_cyclic_family_bipolar
from .qam import QamMode, qam_symbol_quality
from .system_info import classify_system_info


_WORKER_SYMBOLS: np.ndarray | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


@dataclass(frozen=True)
class FrameTimingCandidate:
    """One candidate body timing phase scored across several frames."""

    phase_offset: int
    coarse_cfo_hz: float | None
    frame_count: int
    expected_top_count: int
    expected_top3_count: int
    c3780_top_count: int
    mean_best_metric: float
    best_expected_metric: float
    median_expected_metric: float
    median_evm_rms: float | None
    min_evm_rms: float | None
    score: float
    body_window_offset_symbols: int = 0
    dominant_top_index: int | None = None
    dominant_top_frame_body_mode: str | None = None
    dominant_top_count: int = 0
    dominant_top_ratio: float = 0.0
    dominant_top_mean_metric: float | None = None
    top_index_counts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_offset": self.phase_offset,
            "coarse_cfo_hz": self.coarse_cfo_hz,
            "body_window_offset_symbols": self.body_window_offset_symbols,
            "frame_count": self.frame_count,
            "expected_top_count": self.expected_top_count,
            "expected_top3_count": self.expected_top3_count,
            "c3780_top_count": self.c3780_top_count,
            "mean_best_metric": self.mean_best_metric,
            "best_expected_metric": self.best_expected_metric,
            "median_expected_metric": self.median_expected_metric,
            "median_evm_rms": self.median_evm_rms,
            "min_evm_rms": self.min_evm_rms,
            "score": self.score,
            "dominant_top_index": self.dominant_top_index,
            "dominant_top_frame_body_mode": self.dominant_top_frame_body_mode,
            "dominant_top_count": self.dominant_top_count,
            "dominant_top_ratio": self.dominant_top_ratio,
            "dominant_top_mean_metric": self.dominant_top_mean_metric,
            "top_index_counts": list(self.top_index_counts),
        }


@dataclass(frozen=True)
class PnPayloadTimingCandidate:
    """Payload-wide quality for one PN-header/body boundary candidate."""

    phase_offset: int
    frame_count: int
    grid_evm_rms: float
    max_abs_hard_bit_bias: float
    max_abs_axis_occupancy_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_offset": self.phase_offset,
            "frame_count": self.frame_count,
            "grid_evm_rms": self.grid_evm_rms,
            "max_abs_hard_bit_bias": self.max_abs_hard_bit_bias,
            "max_abs_axis_occupancy_error": self.max_abs_axis_occupancy_error,
        }


def evaluate_pn_payload_timing(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    qam_mode: QamMode,
    max_frames: int = 24,
    pn_channel_taps: int | None = 8,
    pn_noise_variance: float | None = 0.05,
) -> PnPayloadTimingCandidate:
    """Score a boundary using PN-equalized data carriers across the full band.

    The 36 system-information carriers can remain strong when neighboring
    candidate boundaries differ by one sample because they sparsely sample the
    physical spectrum. A wrong body boundary leaves a phase ramp across the
    3744 data carriers, increasing QAM level-occupancy and bit-plane bias even
    when Gate C cannot distinguish the candidates.
    """

    if phase_offset < 0:
        raise ValueError("phase_offset must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    frames = list(
        iter_signal_frames(
            np.asarray(symbols, dtype=np.complex64),
            mode=mode,
            phase_offset=phase_offset,
            max_frames=max_frames + 1,
        )
    )
    data_frames: list[np.ndarray] = []
    for frame, next_frame in zip(frames, frames[1:], strict=False):
        channel = estimate_pn_channel_compact(
            frame.header,
            mode=mode,
            fft_size=FRAME_BODY_SYMBOLS,
            channel_taps=pn_channel_taps,
        )
        next_channel = estimate_pn_channel_compact(
            next_frame.header,
            mode=mode,
            fft_size=FRAME_BODY_SYMBOLS,
            channel_taps=pn_channel_taps,
        )
        body = pn_restore_circular_body_window(
            frame.body,
            pn_phase=channel.pn_phase,
            next_pn_phase=next_channel.pn_phase,
            mode=mode,
            taps=channel.taps,
            next_header=next_frame.header,
        )
        spectrum = equalize_spectrum_with_channel(
            frame_body_fft(body),
            channel,
            noise_variance=pn_noise_variance,
        )
        _system_info, data = split_system_info_and_data(
            frequency_deinterleave(spectrum)
        )
        data_frames.append(np.asarray(data, dtype=np.complex64))
    if not data_frames:
        raise ValueError("not enough complete frames to score PN payload timing")
    quality = qam_symbol_quality(
        np.concatenate(data_frames),
        mode=qam_mode,
        normalize=True,
    )
    occupancy = quality.get("axis_level_occupancy") or {}
    return PnPayloadTimingCandidate(
        phase_offset=int(phase_offset),
        frame_count=int(len(data_frames)),
        grid_evm_rms=float(quality["grid_evm_rms"]),
        max_abs_hard_bit_bias=float(quality["max_abs_hard_bit_bias"]),
        max_abs_axis_occupancy_error=float(
            occupancy["max_abs_fraction_error_from_uniform"]
        ),
    )


def search_pn_payload_timing(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offsets: Sequence[int],
    qam_mode: QamMode,
    max_frames: int = 24,
    pn_channel_taps: int | None = 8,
    pn_noise_variance: float | None = 0.05,
) -> dict[str, Any]:
    """Rank a bounded set of PN/body boundaries by payload-wide QAM quality."""

    phases = sorted(set(int(value) for value in phase_offsets))
    if not phases:
        raise ValueError("phase_offsets must not be empty")
    candidates = [
        evaluate_pn_payload_timing(
            symbols,
            mode=mode,
            phase_offset=phase,
            qam_mode=qam_mode,
            max_frames=max_frames,
            pn_channel_taps=pn_channel_taps,
            pn_noise_variance=pn_noise_variance,
        )
        for phase in phases
    ]
    candidates.sort(key=_pn_payload_timing_sort_key)
    return {
        "mode": mode,
        "qam_mode": qam_mode,
        "max_frames": int(max_frames),
        "pn_channel_taps": pn_channel_taps,
        "pn_noise_variance": pn_noise_variance,
        "selected": candidates[0].to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def evaluate_frame_timing(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    body_window_offset_symbols: int = 0,
    max_frames: int = 24,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    estimate_cfo: bool = True,
    coarse_cfo_hz: float | None = None,
    cfo_phase_offset: int | None = None,
) -> FrameTimingCandidate:
    """Score one frame-body timing phase from symbol-rate samples."""

    if phase_offset < 0:
        raise ValueError("phase_offset must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    signal = np.asarray(symbols, dtype=np.complex64)
    cfo_hz = coarse_cfo_hz
    if cfo_hz is None and estimate_cfo:
        cfo_hz = estimate_cfo_from_pn_cyclic_extension(
            signal,
            mode=mode,
            phase_offset=cfo_phase_offset if cfo_phase_offset is not None else phase_offset,
            symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
        )
    corrected = signal
    if cfo_hz is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=DTMB_SYMBOL_RATE_SPS,
            shift_hz=-cfo_hz,
        )

    best_metrics: list[float] = []
    expected_metrics: list[float] = []
    evm_values: list[float] = []
    expected_top_count = 0
    expected_top3_count = 0
    c3780_top_count = 0
    frame_count = 0
    top_index_counts: dict[tuple[str, int], int] = {}
    top_index_metric_sums: dict[tuple[str, int], float] = {}

    for frame in iter_signal_frames(
        corrected,
        mode=mode,
        phase_offset=phase_offset,
        max_frames=max_frames,
    ):
        body_samples = _timing_frame_body_samples(
            frame,
            corrected,
            mode=mode,
            body_window_offset_symbols=body_window_offset_symbols,
        )
        if body_samples is None:
            continue
        spectrum = frame_body_fft(body_samples)
        deinterleaved = frequency_deinterleave(spectrum)
        system_info, _data = split_system_info_and_data(deinterleaved)
        matches = classify_system_info(
            system_info,
            frame_body_modes=(expected_frame_body_mode,)
            if expected_frame_body_mode in ("C3780", "C1")
            else ("C3780", "C1"),
        )
        if not matches:
            continue
        eq = equalize_c3780_spectrum_with_system_info_pilots(
            spectrum,
            system_info_index=expected_system_info_index,
            frame_body_mode=expected_frame_body_mode,
        )
        best = matches[0]
        top_key = (str(best.frame_body_mode), int(best.index))
        top_index_counts[top_key] = top_index_counts.get(top_key, 0) + 1
        top_index_metric_sums[top_key] = (
            top_index_metric_sums.get(top_key, 0.0) + float(best.metric)
        )
        expected = [
            match
            for match in matches
            if match.index == expected_system_info_index
            and match.frame_body_mode == expected_frame_body_mode
        ]
        frame_count += 1
        best_metrics.append(best.metric)
        if best.frame_body_mode == "C3780":
            c3780_top_count += 1
        if expected:
            expected_metrics.append(expected[0].metric)
        if (
            best.index == expected_system_info_index
            and best.frame_body_mode == expected_frame_body_mode
        ):
            expected_top_count += 1
        if any(
            match.index == expected_system_info_index
            and match.frame_body_mode == expected_frame_body_mode
            for match in matches[:3]
        ):
            expected_top3_count += 1
        evm_values.append(eq.data_evm_rms)

    mean_best_metric = float(np.mean(best_metrics)) if best_metrics else 0.0
    best_expected_metric = float(np.max(expected_metrics)) if expected_metrics else 0.0
    median_expected_metric = (
        float(np.median(expected_metrics)) if expected_metrics else 0.0
    )
    median_evm = float(np.median(evm_values)) if evm_values else None
    min_evm = float(np.min(evm_values)) if evm_values else None
    top_rows = _top_index_count_rows(top_index_counts, top_index_metric_sums)
    dominant = top_rows[0] if top_rows else {}
    return FrameTimingCandidate(
        phase_offset=phase_offset,
        coarse_cfo_hz=cfo_hz,
        body_window_offset_symbols=int(body_window_offset_symbols),
        frame_count=frame_count,
        expected_top_count=expected_top_count,
        expected_top3_count=expected_top3_count,
        c3780_top_count=c3780_top_count,
        mean_best_metric=mean_best_metric,
        best_expected_metric=best_expected_metric,
        median_expected_metric=median_expected_metric,
        median_evm_rms=median_evm,
        min_evm_rms=min_evm,
        score=_timing_score(
            frame_count=frame_count,
            expected_top_count=expected_top_count,
            expected_top3_count=expected_top3_count,
            c3780_top_count=c3780_top_count,
            mean_best_metric=mean_best_metric,
            best_expected_metric=best_expected_metric,
            median_expected_metric=median_expected_metric,
        ),
        dominant_top_index=dominant.get("index"),
        dominant_top_frame_body_mode=dominant.get("frame_body_mode"),
        dominant_top_count=int(dominant.get("count") or 0),
        dominant_top_ratio=float((dominant.get("count") or 0) / frame_count)
        if frame_count
        else 0.0,
        dominant_top_mean_metric=dominant.get("mean_metric"),
        top_index_counts=tuple(top_rows),
    )


def search_frame_timing(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    center_phase_offset: int,
    search_radius_symbols: int = 512,
    step_symbols: int = 4,
    max_frames: int = 24,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    estimate_cfo: bool = True,
    per_candidate_cfo: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Search repeated frame-body timing phases around a center phase."""

    if search_radius_symbols < 0:
        raise ValueError("search_radius_symbols must be non-negative")
    if step_symbols <= 0:
        raise ValueError("step_symbols must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    definition = PN_DEFINITIONS[mode]
    phases = []
    for delta in range(-search_radius_symbols, search_radius_symbols + 1, step_symbols):
        phases.append((center_phase_offset + delta) % definition.frame_symbols)
    phases = sorted(set(phases))
    center_cfo_hz = None
    if estimate_cfo and not per_candidate_cfo:
        center_cfo_hz = estimate_cfo_from_pn_cyclic_extension(
            symbols,
            mode=mode,
            phase_offset=center_phase_offset,
            symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
        )
    worker_config = {
        "mode": mode,
        "max_frames": max_frames,
        "expected_system_info_index": expected_system_info_index,
        "expected_frame_body_mode": expected_frame_body_mode,
        "estimate_cfo": bool(estimate_cfo and per_candidate_cfo),
        "coarse_cfo_hz": center_cfo_hz,
    }
    if jobs == 1 or len(phases) <= 1:
        candidates = [
            evaluate_frame_timing(
                symbols,
                phase_offset=phase,
                **worker_config,
            )
            for phase in phases
        ]
    else:
        worker_symbols = np.asarray(symbols, dtype=np.complex64)
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_timing_worker,
            initargs=(worker_symbols, worker_config),
        ) as executor:
            candidates = list(executor.map(_evaluate_timing_phase_worker, phases))
    candidates.sort(key=_candidate_sort_key, reverse=True)
    selected = candidates[0] if candidates else None
    candidate_dicts = [
        _candidate_to_search_dict(
            candidate,
            center_phase_offset=center_phase_offset,
            frame_symbols=definition.frame_symbols,
        )
        for candidate in candidates
    ]
    return {
        "mode": mode,
        "center_phase_offset": center_phase_offset,
        "search_radius_symbols": search_radius_symbols,
        "step_symbols": step_symbols,
        "max_frames": max_frames,
        "expected_system_info_index": expected_system_info_index,
        "expected_frame_body_mode": expected_frame_body_mode,
        "per_candidate_cfo": bool(per_candidate_cfo),
        "jobs": jobs,
        "candidate_count": len(candidates),
        "selected": candidate_dicts[0] if selected else None,
        "candidates": candidate_dicts,
    }


def track_frame_timing_windows(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    center_phase_offset: int,
    start_frame_index: int = 0,
    window_frames: int = 48,
    window_step_frames: int = 48,
    max_windows: int | None = None,
    search_radius_symbols: int = 512,
    step_symbols: int = 4,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    follow_selected_phase: bool = False,
    per_candidate_cfo: bool = False,
    jobs: int = 1,
    top_candidates: int = 4,
    continuity_tracking: bool = False,
    continuity_candidate_limit: int = 8,
    continuity_bridge_candidate_limit: int | None = None,
    smoothness_penalty_per_symbol: float = 0.25,
    max_phase_step_symbols: int | None = None,
) -> dict[str, Any]:
    """Search C=3780 timing over successive windows of one symbol stream.

    Window starts are expressed in nominal DTMB frames, so the resampler phase
    stays continuous. Each window is ranked only by the independent raw system
    information classifier used by :func:`search_frame_timing`; trained EVM is
    still reported on candidates but does not affect ranking. When continuity
    tracking is enabled, a bounded dynamic-programming pass chooses a smooth
    phase trajectory from the top candidates in each independently scored
    window.
    """

    if center_phase_offset < 0:
        raise ValueError("center_phase_offset must be non-negative")
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be non-negative")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if window_step_frames <= 0:
        raise ValueError("window_step_frames must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if top_candidates <= 0:
        raise ValueError("top_candidates must be positive")
    if continuity_candidate_limit <= 0:
        raise ValueError("continuity_candidate_limit must be positive")
    if (
        continuity_bridge_candidate_limit is not None
        and continuity_bridge_candidate_limit <= 0
    ):
        raise ValueError("continuity_bridge_candidate_limit must be positive when set")
    if (
        not np.isfinite(smoothness_penalty_per_symbol)
        or smoothness_penalty_per_symbol < 0
    ):
        raise ValueError("smoothness_penalty_per_symbol must be finite non-negative")
    if max_phase_step_symbols is not None and max_phase_step_symbols < 0:
        raise ValueError("max_phase_step_symbols must be non-negative")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    frame_symbols = definition.frame_symbols
    windows: list[dict[str, Any]] = []
    trajectory_candidates: list[list[dict[str, Any]]] = []
    current_center = center_phase_offset % frame_symbols
    start = start_frame_index
    while start * frame_symbols < signal.size:
        if max_windows is not None and len(windows) >= max_windows:
            break
        analysis_start_symbol = start * frame_symbols
        analysis_stop_symbol = min(
            signal.size,
            analysis_start_symbol + (window_frames + 1) * frame_symbols,
        )
        analysis_symbols = signal[analysis_start_symbol:analysis_stop_symbol]
        search = search_frame_timing(
            analysis_symbols,
            mode=mode,
            center_phase_offset=current_center,
            search_radius_symbols=search_radius_symbols,
            step_symbols=step_symbols,
            max_frames=window_frames,
            expected_system_info_index=expected_system_info_index,
            expected_frame_body_mode=expected_frame_body_mode,
            per_candidate_cfo=per_candidate_cfo,
            jobs=jobs,
        )
        selected = search.get("selected")
        selected_phase = (
            int(selected["phase_offset"])
            if isinstance(selected, dict) and selected.get("phase_offset") is not None
            else None
        )
        if continuity_tracking:
            trajectory_candidates.append(search["candidates"])
        windows.append(
            {
                "start_frame_index": int(start),
                "analysis_start_symbol": int(analysis_start_symbol),
                "analysis_stop_symbol": int(analysis_stop_symbol),
                "center_phase_offset": int(current_center),
                "selected": selected,
                "candidate_count": int(search["candidate_count"]),
                "top_candidates": search["candidates"][:top_candidates],
            }
        )
        if follow_selected_phase and selected_phase is not None:
            current_center = selected_phase
        start += window_step_frames
    trajectory = None
    if continuity_tracking:
        trajectory = select_continuous_timing_path(
            trajectory_candidates,
            center_phase_offset=center_phase_offset % frame_symbols,
            frame_symbols=frame_symbols,
            candidate_limit=continuity_candidate_limit,
            bridge_candidate_limit=continuity_bridge_candidate_limit,
            smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
            max_phase_step_symbols=max_phase_step_symbols,
            window_metadata=windows,
        )
        if trajectory.get("available"):
            for step in trajectory["windows"]:
                window_index = int(step["window_index"])
                windows[window_index]["continuous_selected"] = step["candidate"]
                windows[window_index]["continuous_candidate_rank"] = int(
                    step["candidate_rank"]
                )
                windows[window_index]["continuous_transition_step_symbols"] = int(
                    step["transition_step_symbols"]
                )
                windows[window_index]["continuous_transition_penalty"] = float(
                    step["transition_penalty"]
                )
                windows[window_index]["continuous_path_score"] = float(
                    step["path_score"]
                )
    return {
        "mode": mode,
        "center_phase_offset": int(center_phase_offset % frame_symbols),
        "start_frame_index": int(start_frame_index),
        "window_frames": int(window_frames),
        "window_step_frames": int(window_step_frames),
        "max_windows": max_windows,
        "search_radius_symbols": int(search_radius_symbols),
        "step_symbols": int(step_symbols),
        "expected_system_info_index": int(expected_system_info_index),
        "expected_frame_body_mode": expected_frame_body_mode,
        "follow_selected_phase": bool(follow_selected_phase),
        "per_candidate_cfo": bool(per_candidate_cfo),
        "jobs": int(jobs),
        "top_candidates": int(top_candidates),
        "continuity_tracking": bool(continuity_tracking),
        "continuity_candidate_limit": int(continuity_candidate_limit),
        "continuity_bridge_candidate_limit": None
        if continuity_bridge_candidate_limit is None
        else int(continuity_bridge_candidate_limit),
        "smoothness_penalty_per_symbol": float(smoothness_penalty_per_symbol),
        "max_phase_step_symbols": max_phase_step_symbols,
        "window_count": int(len(windows)),
        "summary": _summarize_timing_windows(
            windows,
            center_phase_offset=center_phase_offset % frame_symbols,
            frame_symbols=frame_symbols,
        ),
        "trajectory": trajectory,
        "windows": windows,
    }


def build_frame_timing_trajectory(
    tracking: dict[str, Any],
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Convert windowed timing diagnostics into receiver-consumable segments.

    The trajectory is intentionally conservative: each timing window contributes
    a phase/CFO segment that is held until the next window starts. When the
    dynamic-programming continuity path is available, those candidates are used;
    otherwise the independently selected window candidates are preserved with an
    unavailable verdict so downstream gates can block promotion.
    """

    mode = str(tracking.get("mode") or "pn945")
    if mode not in PN_DEFINITIONS:
        raise ValueError(f"unsupported timing trajectory mode: {mode}")
    definition = PN_DEFINITIONS[mode]  # type: ignore[index]
    frame_symbols = definition.frame_symbols
    windows = [
        window for window in tracking.get("windows", []) if isinstance(window, dict)
    ]
    trajectory = tracking.get("trajectory") if isinstance(tracking.get("trajectory"), dict) else None
    continuous_available = bool((trajectory or {}).get("available"))
    trajectory_windows = {
        int(window.get("window_index")): window
        for window in (trajectory or {}).get("windows", [])
        if isinstance(window, dict) and window.get("window_index") is not None
    }
    source = "continuous" if continuous_available else "selected"
    segments: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        selected = None
        candidate_rank = None
        transition_step = None
        if continuous_available:
            step = trajectory_windows.get(index)
            if step is not None:
                selected = step.get("candidate") or {
                    "phase_offset": step.get("phase_offset"),
                    "coarse_cfo_hz": None,
                }
                candidate_rank = step.get("candidate_rank")
                transition_step = step.get("transition_step_symbols")
        if selected is None:
            selected = window.get("selected")
        if not isinstance(selected, dict) or selected.get("phase_offset") is None:
            continue
        start_frame = int(window.get("start_frame_index") or 0)
        if index + 1 < len(windows):
            end_frame = int(windows[index + 1].get("start_frame_index") or start_frame)
        else:
            end_frame = start_frame + int(tracking.get("window_step_frames") or 1)
        if max_frames is not None:
            end_frame = min(end_frame, int(max_frames))
        if end_frame <= start_frame:
            continue
        phase = int(selected["phase_offset"]) % frame_symbols
        segment = {
            "start_frame_index": start_frame,
            "end_frame_index": end_frame,
            "phase_offset": phase,
            "coarse_cfo_hz": selected.get("coarse_cfo_hz"),
            "source": source,
            "candidate_rank": candidate_rank,
            "transition_step_symbols": transition_step,
            "expected_top_count": selected.get("expected_top_count"),
            "expected_top3_count": selected.get("expected_top3_count"),
            "frame_count": selected.get("frame_count"),
            "best_expected_metric": selected.get("best_expected_metric"),
            "median_expected_metric": selected.get("median_expected_metric"),
            "score": selected.get("score"),
        }
        segments.append(segment)

    summary = trajectory if continuous_available else tracking.get("summary") or {}
    return {
        "version": 1,
        "mode": mode,
        "frame_symbols": frame_symbols,
        "header_symbols": definition.header_symbols,
        "policy": "window_hold",
        "source": source,
        "available": bool(continuous_available and segments),
        "reason": None if continuous_available and segments else _trajectory_unavailable_reason(tracking),
        "start_frame_index": int(tracking.get("start_frame_index") or 0),
        "window_frames": int(tracking.get("window_frames") or 0),
        "window_step_frames": int(tracking.get("window_step_frames") or 0),
        "expected_system_info_index": tracking.get("expected_system_info_index"),
        "expected_frame_body_mode": tracking.get("expected_frame_body_mode"),
        "continuity_candidate_limit": tracking.get("continuity_candidate_limit"),
        "continuity_bridge_candidate_limit": tracking.get(
            "continuity_bridge_candidate_limit"
        ),
        "continuity_candidate_counts_by_window": (trajectory or {}).get(
            "candidate_counts_by_window",
            [],
        ),
        "continuity_adjacent_candidate_steps": (trajectory or {}).get(
            "adjacent_candidate_steps",
            [],
        ),
        "phase_offsets": [int(segment["phase_offset"]) for segment in segments],
        "phase_span_symbols": int(summary.get("phase_span_symbols") or 0),
        "max_adjacent_phase_step_symbols": int(
            summary.get("max_adjacent_phase_step_symbols") or 0
        ),
        "segments": segments,
    }


def analyze_capture_timing_trajectory(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 8_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    mode: PnMode = "pn945",
    center_phase_offset: int | None = None,
    start_frame_index: int = 0,
    window_frames: int = 48,
    window_step_frames: int = 48,
    max_windows: int | None = None,
    search_radius_symbols: int = 512,
    step_symbols: int = 4,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    follow_selected_phase: bool = False,
    per_candidate_cfo: bool = False,
    jobs: int = 1,
    top_candidates: int = 4,
    continuity_candidate_limit: int = 8,
    continuity_bridge_candidate_limit: int | None = None,
    smoothness_penalty_per_symbol: float = 0.25,
    max_phase_step_symbols: int | None = 64,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Load a capture and emit a compact receiver timing trajectory artifact."""

    diagnostics = analyze_capture_timing_windows(
        capture_path,
        sample_rate_sps=sample_rate_sps,
        symbol_rate_sps=symbol_rate_sps,
        max_samples=max_samples,
        frequency_shift_hz=frequency_shift_hz,
        input_skip_samples=input_skip_samples,
        mode=mode,
        center_phase_offset=center_phase_offset,
        start_frame_index=start_frame_index,
        window_frames=window_frames,
        window_step_frames=window_step_frames,
        max_windows=max_windows,
        search_radius_symbols=search_radius_symbols,
        step_symbols=step_symbols,
        expected_system_info_index=expected_system_info_index,
        expected_frame_body_mode=expected_frame_body_mode,
        follow_selected_phase=follow_selected_phase,
        per_candidate_cfo=per_candidate_cfo,
        jobs=jobs,
        top_candidates=top_candidates,
        continuity_tracking=True,
        continuity_candidate_limit=continuity_candidate_limit,
        continuity_bridge_candidate_limit=continuity_bridge_candidate_limit,
        smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
        max_phase_step_symbols=max_phase_step_symbols,
    )
    trajectory = build_frame_timing_trajectory(
        diagnostics["tracking"],
        max_frames=max_frames,
    )
    return {
        "stage": "timing_trajectory",
        "capture_path": str(capture_path),
        "input_sample_rate_sps": int(sample_rate_sps),
        "symbol_rate_sps": int(symbol_rate_sps),
        "resample_up": diagnostics["resample_up"],
        "resample_down": diagnostics["resample_down"],
        "frequency_shift_hz": float(frequency_shift_hz),
        "input_skip_samples": int(input_skip_samples),
        "analyzed_symbols": diagnostics["analyzed_symbols"],
        "acquisition": diagnostics["acquisition"],
        "trajectory": trajectory,
        "continuity": _compact_timing_trajectory(
            diagnostics["tracking"].get("trajectory")
        ),
        "tracking_summary": diagnostics["tracking"].get("summary") or {},
    }


def _trajectory_unavailable_reason(tracking: dict[str, Any]) -> str:
    trajectory = tracking.get("trajectory")
    if isinstance(trajectory, dict) and trajectory.get("reason"):
        return str(trajectory["reason"])
    if not tracking.get("windows"):
        return "no_windows"
    return "continuity_path_unavailable"


def select_continuous_timing_path(
    window_candidates: Sequence[Sequence[dict[str, Any]]],
    *,
    center_phase_offset: int,
    frame_symbols: int,
    candidate_limit: int = 8,
    bridge_candidate_limit: int | None = None,
    smoothness_penalty_per_symbol: float = 0.25,
    max_phase_step_symbols: int | None = None,
    window_metadata: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose a smooth timing trajectory from per-window candidate lists.

    The objective is the sum of each candidate's local timing score minus a
    linear adjacent-phase smoothness penalty. The top ``candidate_limit``
    candidates from each window are always considered. When a phase-step bound
    is configured, a bounded number of lower-ranked candidates that can bridge
    to neighboring shortlist phases are also retained so a single noisy window
    cannot create a false ``no_continuous_path`` verdict. The bridge rank bound
    prevents a smooth but locally implausible phase from dominating the path.
    """

    if frame_symbols <= 0:
        raise ValueError("frame_symbols must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if bridge_candidate_limit is not None and bridge_candidate_limit <= 0:
        raise ValueError("bridge_candidate_limit must be positive when set")
    if (
        not np.isfinite(smoothness_penalty_per_symbol)
        or smoothness_penalty_per_symbol < 0
    ):
        raise ValueError("smoothness_penalty_per_symbol must be finite non-negative")
    if max_phase_step_symbols is not None and max_phase_step_symbols < 0:
        raise ValueError("max_phase_step_symbols must be non-negative")
    if window_metadata is not None and len(window_metadata) != len(window_candidates):
        raise ValueError("window_metadata must match window_candidates length")
    effective_bridge_limit = (
        max(candidate_limit, int(bridge_candidate_limit))
        if bridge_candidate_limit is not None
        else candidate_limit * 2
    )

    shortlist_phases_by_window = [
        [
            int(candidate["phase_offset"])
            for candidate in _trajectory_candidates(
                candidates,
                candidate_limit=candidate_limit,
            )
        ]
        for candidates in window_candidates
    ]
    candidates_by_window = []
    for window_index, candidates in enumerate(window_candidates):
        neighbor_shortlist_phases: list[int] = []
        if window_index > 0:
            neighbor_shortlist_phases.extend(
                shortlist_phases_by_window[window_index - 1]
            )
        if window_index + 1 < len(shortlist_phases_by_window):
            neighbor_shortlist_phases.extend(
                shortlist_phases_by_window[window_index + 1]
            )
        candidates_by_window.append(
            _trajectory_candidates(
                candidates,
                candidate_limit=candidate_limit,
                bridge_anchor_phases=neighbor_shortlist_phases,
                bridge_phase_step_symbols=max_phase_step_symbols,
                bridge_candidate_rank_limit=effective_bridge_limit,
                frame_symbols=frame_symbols,
            )
        )
    candidate_counts_by_window = [
        int(len(candidates)) for candidates in candidates_by_window
    ]
    if not candidates_by_window:
        return _empty_timing_trajectory(
            candidate_limit=candidate_limit,
            bridge_candidate_limit=effective_bridge_limit,
            smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
            max_phase_step_symbols=max_phase_step_symbols,
            candidate_counts_by_window=candidate_counts_by_window,
            reason="no_windows",
        )
    if any(not candidates for candidates in candidates_by_window):
        return _empty_timing_trajectory(
            candidate_limit=candidate_limit,
            bridge_candidate_limit=effective_bridge_limit,
            smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
            max_phase_step_symbols=max_phase_step_symbols,
            candidate_counts_by_window=candidate_counts_by_window,
            reason="empty_window_candidates",
        )

    scores: list[list[float]] = [
        [_candidate_local_score(candidate) for candidate in candidates_by_window[0]]
    ]
    parents: list[list[int | None]] = [
        [None for _candidate in candidates_by_window[0]]
    ]
    transition_steps: list[list[int]] = [[0 for _candidate in candidates_by_window[0]]]
    transition_penalties: list[list[float]] = [
        [0.0 for _candidate in candidates_by_window[0]]
    ]

    for window_index in range(1, len(candidates_by_window)):
        previous_candidates = candidates_by_window[window_index - 1]
        current_candidates = candidates_by_window[window_index]
        previous_scores = scores[-1]
        current_scores: list[float] = []
        current_parents: list[int | None] = []
        current_steps: list[int] = []
        current_penalties: list[float] = []
        for candidate in current_candidates:
            phase = int(candidate["phase_offset"])
            best_score = float("-inf")
            best_parent = None
            best_step = 0
            best_penalty = 0.0
            for previous_index, previous_candidate in enumerate(previous_candidates):
                previous_score = previous_scores[previous_index]
                if not np.isfinite(previous_score):
                    continue
                previous_phase = int(previous_candidate["phase_offset"])
                step_symbols = _phase_step_symbols(
                    previous_phase,
                    phase,
                    frame_symbols=frame_symbols,
                )
                if (
                    max_phase_step_symbols is not None
                    and step_symbols > max_phase_step_symbols
                ):
                    continue
                penalty = smoothness_penalty_per_symbol * step_symbols
                path_score = previous_score - penalty
                if path_score > best_score:
                    best_score = path_score
                    best_parent = previous_index
                    best_step = step_symbols
                    best_penalty = penalty
            if best_parent is None:
                current_scores.append(float("-inf"))
                current_parents.append(None)
                current_steps.append(0)
                current_penalties.append(0.0)
            else:
                current_scores.append(best_score + _candidate_local_score(candidate))
                current_parents.append(best_parent)
                current_steps.append(best_step)
                current_penalties.append(best_penalty)
        scores.append(current_scores)
        parents.append(current_parents)
        transition_steps.append(current_steps)
        transition_penalties.append(current_penalties)

    final_scores = scores[-1]
    if not any(np.isfinite(score) for score in final_scores):
        return _empty_timing_trajectory(
            candidate_limit=candidate_limit,
            bridge_candidate_limit=effective_bridge_limit,
            smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
            max_phase_step_symbols=max_phase_step_symbols,
            candidate_counts_by_window=candidate_counts_by_window,
            adjacent_candidate_steps=_adjacent_candidate_step_summary(
                candidates_by_window,
                frame_symbols=frame_symbols,
            ),
            reason="no_continuous_path",
        )

    selected_index = int(np.argmax(np.asarray(final_scores, dtype=np.float64)))
    path_indices = [selected_index]
    for window_index in range(len(candidates_by_window) - 1, 0, -1):
        parent = parents[window_index][path_indices[-1]]
        if parent is None:
            return _empty_timing_trajectory(
                candidate_limit=candidate_limit,
                bridge_candidate_limit=effective_bridge_limit,
                smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
                max_phase_step_symbols=max_phase_step_symbols,
                candidate_counts_by_window=candidate_counts_by_window,
                adjacent_candidate_steps=_adjacent_candidate_step_summary(
                    candidates_by_window,
                    frame_symbols=frame_symbols,
                ),
                reason="broken_backpointer",
            )
        path_indices.append(parent)
    path_indices.reverse()

    windows: list[dict[str, Any]] = []
    phase_offsets: list[int] = []
    local_scores: list[float] = []
    penalties: list[float] = []
    for window_index, candidate_index in enumerate(path_indices):
        candidate = dict(candidates_by_window[window_index][candidate_index])
        candidate_rank = int(
            candidate.pop("_trajectory_candidate_rank", candidate_index + 1)
        )
        phase_offset = int(candidate["phase_offset"])
        local_score = _candidate_local_score(candidate)
        penalty = float(transition_penalties[window_index][candidate_index])
        step_symbols = int(transition_steps[window_index][candidate_index])
        metadata = (
            window_metadata[window_index]
            if window_metadata is not None
            else {}
        )
        windows.append(
            {
                "window_index": int(window_index),
                "start_frame_index": metadata.get("start_frame_index"),
                "analysis_start_symbol": metadata.get("analysis_start_symbol"),
                "phase_offset": phase_offset,
                "relative_offset_symbols": _phase_delta(
                    phase_offset,
                    center_phase_offset=center_phase_offset,
                    frame_symbols=frame_symbols,
                ),
                "candidate_rank": candidate_rank,
                "local_score": local_score,
                "transition_step_symbols": step_symbols,
                "transition_penalty": penalty,
                "path_score": float(scores[window_index][candidate_index]),
                "candidate": candidate,
            }
        )
        phase_offsets.append(phase_offset)
        local_scores.append(local_score)
        penalties.append(penalty)

    relative = [
        _phase_delta(
            phase,
            center_phase_offset=center_phase_offset,
            frame_symbols=frame_symbols,
        )
        for phase in phase_offsets
    ]
    unwrapped = _unwrap_phase_offsets(relative, frame_symbols=frame_symbols)
    adjacent_steps = [
        abs(unwrapped[index] - unwrapped[index - 1])
        for index in range(1, len(unwrapped))
    ]
    total_local_score = float(sum(local_scores))
    total_penalty = float(sum(penalties))
    return {
        "available": True,
        "reason": None,
        "candidate_limit": int(candidate_limit),
        "bridge_candidate_limit": int(effective_bridge_limit),
        "candidate_counts_by_window": candidate_counts_by_window,
        "smoothness_penalty_per_symbol": float(smoothness_penalty_per_symbol),
        "max_phase_step_symbols": max_phase_step_symbols,
        "window_count": int(len(windows)),
        "total_local_score": total_local_score,
        "total_smoothness_penalty": total_penalty,
        "total_path_score": float(total_local_score - total_penalty),
        "phase_offsets": phase_offsets,
        "relative_offsets_symbols": relative,
        "unwrapped_relative_offsets_symbols": unwrapped,
        "phase_span_symbols": int(max(unwrapped) - min(unwrapped))
        if unwrapped
        else 0,
        "max_adjacent_phase_step_symbols": int(max(adjacent_steps))
        if adjacent_steps
        else 0,
        "windows": windows,
    }


def _trajectory_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    candidate_limit: int,
    bridge_anchor_phases: Sequence[int] = (),
    bridge_phase_step_symbols: int | None = None,
    bridge_candidate_rank_limit: int | None = None,
    frame_symbols: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_phases: set[int] = set()
    for candidate_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or candidate.get("phase_offset") is None:
            continue
        phase = int(candidate["phase_offset"])
        keep = candidate_rank <= candidate_limit
        if (
            not keep
            and bridge_phase_step_symbols is not None
            and frame_symbols is not None
            and bridge_anchor_phases
            and (
                bridge_candidate_rank_limit is None
                or candidate_rank <= int(bridge_candidate_rank_limit)
            )
        ):
            keep = _phase_is_near_any(
                phase,
                bridge_anchor_phases,
                frame_symbols=frame_symbols,
                max_step_symbols=bridge_phase_step_symbols,
            )
        if not keep or phase in seen_phases:
            continue
        item = dict(candidate)
        item["_trajectory_candidate_rank"] = candidate_rank
        selected.append(item)
        seen_phases.add(phase)
    return selected


def _phase_is_near_any(
    phase_offset: int,
    anchor_phases: Sequence[int],
    *,
    frame_symbols: int,
    max_step_symbols: int,
) -> bool:
    return any(
        _phase_step_symbols(
            int(anchor_phase),
            int(phase_offset),
            frame_symbols=frame_symbols,
        )
        <= max_step_symbols
        for anchor_phase in anchor_phases
    )


def _adjacent_candidate_step_summary(
    candidates_by_window: Sequence[Sequence[dict[str, Any]]],
    *,
    frame_symbols: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index in range(1, len(candidates_by_window)):
        previous_candidates = candidates_by_window[window_index - 1]
        current_candidates = candidates_by_window[window_index]
        min_step: int | None = None
        min_previous_phase: int | None = None
        min_current_phase: int | None = None
        for previous_candidate in previous_candidates:
            previous_phase = int(previous_candidate["phase_offset"])
            for current_candidate in current_candidates:
                current_phase = int(current_candidate["phase_offset"])
                step = _phase_step_symbols(
                    previous_phase,
                    current_phase,
                    frame_symbols=frame_symbols,
                )
                if min_step is None or step < min_step:
                    min_step = step
                    min_previous_phase = previous_phase
                    min_current_phase = current_phase
        rows.append(
            {
                "from_window_index": int(window_index - 1),
                "to_window_index": int(window_index),
                "min_step_symbols": int(min_step) if min_step is not None else None,
                "from_phase_offset": min_previous_phase,
                "to_phase_offset": min_current_phase,
            }
        )
    return rows


def _candidate_local_score(candidate: dict[str, Any]) -> float:
    raw_score = candidate.get("score", 0.0)
    score = float(raw_score) if raw_score is not None else 0.0
    return score if np.isfinite(score) else 0.0


def _phase_step_symbols(
    previous_phase_offset: int,
    phase_offset: int,
    *,
    frame_symbols: int,
) -> int:
    return abs(
        _phase_delta(
            phase_offset,
            center_phase_offset=previous_phase_offset,
            frame_symbols=frame_symbols,
        )
    )


def _empty_timing_trajectory(
    *,
    candidate_limit: int,
    bridge_candidate_limit: int | None,
    smoothness_penalty_per_symbol: float,
    max_phase_step_symbols: int | None,
    candidate_counts_by_window: Sequence[int] = (),
    adjacent_candidate_steps: Sequence[dict[str, Any]] = (),
    reason: str,
) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "candidate_limit": int(candidate_limit),
        "bridge_candidate_limit": None
        if bridge_candidate_limit is None
        else int(bridge_candidate_limit),
        "candidate_counts_by_window": [
            int(count) for count in candidate_counts_by_window
        ],
        "adjacent_candidate_steps": list(adjacent_candidate_steps),
        "smoothness_penalty_per_symbol": float(smoothness_penalty_per_symbol),
        "max_phase_step_symbols": max_phase_step_symbols,
        "window_count": int(len(candidate_counts_by_window)),
        "total_local_score": 0.0,
        "total_smoothness_penalty": 0.0,
        "total_path_score": 0.0,
        "phase_offsets": [],
        "relative_offsets_symbols": [],
        "unwrapped_relative_offsets_symbols": [],
        "phase_span_symbols": 0,
        "max_adjacent_phase_step_symbols": 0,
        "windows": [],
    }


def _init_timing_worker(symbols: np.ndarray, config: dict[str, Any]) -> None:
    global _WORKER_SYMBOLS, _WORKER_CONFIG
    _WORKER_SYMBOLS = np.asarray(symbols, dtype=np.complex64)
    _WORKER_CONFIG = dict(config)


def _evaluate_timing_phase_worker(phase_offset: int) -> FrameTimingCandidate:
    if _WORKER_SYMBOLS is None or _WORKER_CONFIG is None:
        raise RuntimeError("timing worker is not initialized")
    return evaluate_frame_timing(
        _WORKER_SYMBOLS,
        phase_offset=phase_offset,
        **_WORKER_CONFIG,
    )


def analyze_capture_timing(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 8_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    mode: PnMode = "pn945",
    center_phase_offset: int | None = None,
    search_radius_symbols: int = 512,
    step_symbols: int = 4,
    max_frames: int = 24,
    start_frame_index: int = 0,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    per_candidate_cfo: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Load a capture, acquire a center phase, and run frame timing search."""

    if input_skip_samples < 0:
        raise ValueError("input_skip_samples must be non-negative")
    samples = read_ci8(
        capture_path,
        max_samples=max_samples,
        skip_samples=input_skip_samples,
    )
    samples = remove_dc(samples)
    if frequency_shift_hz:
        samples = frequency_shift(
            samples,
            sample_rate_sps=sample_rate_sps,
            shift_hz=frequency_shift_hz,
        )
    if sample_rate_sps == symbol_rate_sps:
        symbols = samples
        up = 1
        down = 1
    else:
        symbols, up, down = resample_to_symbol_rate(
            samples,
            sample_rate_sps=sample_rate_sps,
            symbol_rate_sps=symbol_rate_sps,
        )

    acquisition = _acquire_timing_center(
        symbols,
        mode=mode,
        center_phase_offset=center_phase_offset,
        max_frames=max_frames,
    )
    definition = PN_DEFINITIONS[mode]
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be non-negative")
    analysis_start_symbol = start_frame_index * definition.frame_symbols
    analysis_symbols = symbols[analysis_start_symbol:]
    search_center = (acquisition["center_phase_offset"] - analysis_start_symbol) % definition.frame_symbols
    search = search_frame_timing(
        analysis_symbols,
        mode=mode,
        center_phase_offset=search_center,
        search_radius_symbols=search_radius_symbols,
        step_symbols=step_symbols,
        max_frames=max_frames,
        expected_system_info_index=expected_system_info_index,
        expected_frame_body_mode=expected_frame_body_mode,
        per_candidate_cfo=per_candidate_cfo,
        jobs=jobs,
    )
    return {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "resample_up": up,
        "resample_down": down,
        "frequency_shift_hz": frequency_shift_hz,
        "input_skip_samples": input_skip_samples,
        "start_frame_index": int(start_frame_index),
        "analysis_start_symbol": int(analysis_start_symbol),
        "analyzed_symbols": int(symbols.size),
        "timing_search_symbols": int(analysis_symbols.size),
        "acquisition": acquisition,
        "search": search,
    }


def analyze_capture_timing_windows(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 8_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    mode: PnMode = "pn945",
    center_phase_offset: int | None = None,
    start_frame_index: int = 0,
    window_frames: int = 48,
    window_step_frames: int = 48,
    max_windows: int | None = None,
    search_radius_symbols: int = 512,
    step_symbols: int = 4,
    expected_system_info_index: int = 23,
    expected_frame_body_mode: str = "C3780",
    follow_selected_phase: bool = False,
    per_candidate_cfo: bool = False,
    jobs: int = 1,
    top_candidates: int = 4,
    continuity_tracking: bool = False,
    continuity_candidate_limit: int = 8,
    continuity_bridge_candidate_limit: int | None = None,
    smoothness_penalty_per_symbol: float = 0.25,
    max_phase_step_symbols: int | None = None,
) -> dict[str, Any]:
    """Load a capture once and track timing quality across frame windows."""

    if input_skip_samples < 0:
        raise ValueError("input_skip_samples must be non-negative")
    samples = read_ci8(
        capture_path,
        max_samples=max_samples,
        skip_samples=input_skip_samples,
    )
    samples = remove_dc(samples)
    if frequency_shift_hz:
        samples = frequency_shift(
            samples,
            sample_rate_sps=sample_rate_sps,
            shift_hz=frequency_shift_hz,
        )
    if sample_rate_sps == symbol_rate_sps:
        symbols = samples
        up = 1
        down = 1
    else:
        symbols, up, down = resample_to_symbol_rate(
            samples,
            sample_rate_sps=sample_rate_sps,
            symbol_rate_sps=symbol_rate_sps,
        )

    acquisition = _acquire_timing_center(
        symbols,
        mode=mode,
        center_phase_offset=center_phase_offset,
        max_frames=window_frames,
    )
    tracking = track_frame_timing_windows(
        symbols,
        mode=mode,
        center_phase_offset=acquisition["center_phase_offset"],
        start_frame_index=start_frame_index,
        window_frames=window_frames,
        window_step_frames=window_step_frames,
        max_windows=max_windows,
        search_radius_symbols=search_radius_symbols,
        step_symbols=step_symbols,
        expected_system_info_index=expected_system_info_index,
        expected_frame_body_mode=expected_frame_body_mode,
        follow_selected_phase=follow_selected_phase,
        per_candidate_cfo=per_candidate_cfo,
        jobs=jobs,
        top_candidates=top_candidates,
        continuity_tracking=continuity_tracking,
        continuity_candidate_limit=continuity_candidate_limit,
        continuity_bridge_candidate_limit=continuity_bridge_candidate_limit,
        smoothness_penalty_per_symbol=smoothness_penalty_per_symbol,
        max_phase_step_symbols=max_phase_step_symbols,
    )
    return {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "resample_up": up,
        "resample_down": down,
        "frequency_shift_hz": frequency_shift_hz,
        "input_skip_samples": input_skip_samples,
        "start_frame_index": int(start_frame_index),
        "analyzed_symbols": int(symbols.size),
        "acquisition": acquisition,
        "tracking": tracking,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-timing-search",
        description="Search DTMB C=3780 frame-body timing around PN acquisition.",
    )
    parser.add_argument("capture", type=Path, help="Path to raw CI8 capture")
    parser.add_argument("--sample-rate", type=_positive_int, default=20_000_000)
    parser.add_argument("--symbol-rate", type=_positive_int, default=DTMB_SYMBOL_RATE_SPS)
    parser.add_argument("--max-samples", type=_positive_int, default=8_000_000)
    parser.add_argument("--frequency-shift", type=float, default=0.0)
    parser.add_argument(
        "--input-skip",
        type=_non_negative_int,
        default=0,
        help="Drop this many raw input samples before conditioning.",
    )
    parser.add_argument("--mode", choices=("pn420", "pn945"), default="pn945")
    parser.add_argument("--center-phase", type=_non_negative_int)
    parser.add_argument("--radius", type=_non_negative_int, default=512)
    parser.add_argument("--step", type=_positive_int, default=4)
    parser.add_argument("--frames", type=_positive_int, default=24)
    parser.add_argument(
        "--start-frame",
        type=_non_negative_int,
        default=0,
        help=(
            "Start the timing search this many nominal DTMB frames into the "
            "already-resampled symbol stream. Unlike --input-skip, this keeps "
            "the resampler phase continuous and is suitable for late-frame "
            "Gate C diagnostics."
        ),
    )
    parser.add_argument("--system-info-index", type=_positive_int, default=23)
    parser.add_argument("--frame-body-mode", choices=("C3780", "C1"), default="C3780")
    parser.add_argument("--top", type=_positive_int, default=12)
    parser.add_argument(
        "--track-windows",
        action="store_true",
        help="Run a windowed timing tracker instead of a single timing search.",
    )
    parser.add_argument(
        "--window-frames",
        type=_positive_int,
        default=48,
        help="Frames scored per window when --track-windows is enabled.",
    )
    parser.add_argument(
        "--window-step",
        type=_positive_int,
        default=48,
        help="Nominal frames between timing windows when --track-windows is enabled.",
    )
    parser.add_argument(
        "--max-windows",
        type=_positive_int,
        help="Maximum number of timing windows to score.",
    )
    parser.add_argument(
        "--follow-window-phase",
        action="store_true",
        help="Center each timing-window search on the previous selected phase.",
    )
    parser.add_argument(
        "--per-candidate-cfo",
        action="store_true",
        help=(
            "Estimate PN cyclic-extension CFO independently for every timing "
            "candidate instead of reusing the center phase estimate."
        ),
    )
    parser.add_argument(
        "--continuity-track",
        action="store_true",
        help=(
            "After scoring timing windows, choose a bounded smooth phase "
            "trajectory from the top candidates with dynamic programming."
        ),
    )
    parser.add_argument(
        "--continuity-candidates",
        type=_positive_int,
        default=8,
        help=(
            "Top candidate phases per window always considered by "
            "--continuity-track; lower-ranked bridge candidates may also be "
            "retained when --max-phase-step is set."
        ),
    )
    parser.add_argument(
        "--continuity-bridge-candidates",
        type=_positive_int,
        help=(
            "Maximum candidate rank eligible as a lower-ranked continuity "
            "bridge. Defaults to twice --continuity-candidates."
        ),
    )
    parser.add_argument(
        "--smoothness-penalty",
        type=_non_negative_float,
        default=0.25,
        help=(
            "Path-score penalty per symbol of adjacent timing motion when "
            "--continuity-track is enabled."
        ),
    )
    parser.add_argument(
        "--max-phase-step",
        type=_non_negative_int,
        help=(
            "Optional hard bound on adjacent phase motion in symbols for "
            "--continuity-track."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help="Worker processes for candidate phase scoring.",
    )
    parser.add_argument("--json", type=Path, help="Optional path to write full JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.track_windows:
        diagnostics = analyze_capture_timing_windows(
            args.capture,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
            max_samples=args.max_samples,
            frequency_shift_hz=args.frequency_shift,
            input_skip_samples=args.input_skip,
            mode=args.mode,
            center_phase_offset=args.center_phase,
            start_frame_index=args.start_frame,
            window_frames=args.window_frames,
            window_step_frames=args.window_step,
            max_windows=args.max_windows,
            search_radius_symbols=args.radius,
            step_symbols=args.step,
            expected_system_info_index=args.system_info_index,
            expected_frame_body_mode=args.frame_body_mode,
            follow_selected_phase=args.follow_window_phase,
            per_candidate_cfo=args.per_candidate_cfo,
            jobs=args.jobs,
            top_candidates=args.top,
            continuity_tracking=args.continuity_track,
            continuity_candidate_limit=args.continuity_candidates,
            continuity_bridge_candidate_limit=args.continuity_bridge_candidates,
            smoothness_penalty_per_symbol=args.smoothness_penalty,
            max_phase_step_symbols=args.max_phase_step,
        )
    else:
        diagnostics = analyze_capture_timing(
            args.capture,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
            max_samples=args.max_samples,
            frequency_shift_hz=args.frequency_shift,
            input_skip_samples=args.input_skip,
            mode=args.mode,
            center_phase_offset=args.center_phase,
            search_radius_symbols=args.radius,
            step_symbols=args.step,
            max_frames=args.frames,
            start_frame_index=args.start_frame,
            expected_system_info_index=args.system_info_index,
            expected_frame_body_mode=args.frame_body_mode,
            per_candidate_cfo=args.per_candidate_cfo,
            jobs=args.jobs,
        )
    if args.json:
        args.json.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.track_windows:
        print(_tracking_summary_json(diagnostics, top=args.top))
    else:
        print(_summary_json(diagnostics, top=args.top))
    return 0


def _acquire_timing_center(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    center_phase_offset: int | None,
    max_frames: int,
) -> dict[str, Any]:
    if center_phase_offset is not None:
        return {
            "center_phase_offset": center_phase_offset,
            "source": "manual",
            "cyclic_train": None,
            "cyclic_cfo_hz": None,
            "family_delay_train": None,
            "family_delay_applied": False,
        }
    trains = detect_pn_cyclic_extension_trains(symbols, modes=(mode,))
    if not trains:
        raise RuntimeError("no PN cyclic-extension trains found")
    train = trains[0]
    cfo_hz = estimate_cfo_from_pn_cyclic_extension(
        symbols,
        mode=mode,
        phase_offset=train.phase_offset,
        symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
    )
    corrected = symbols
    if cfo_hz is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=DTMB_SYMBOL_RATE_SPS,
            shift_hz=-cfo_hz,
        )
    family = score_pn_family_delay_train(
        corrected,
        mode=mode,
        phase_offset=train.phase_offset,
        max_frames=max_frames,
        max_delay_symbols=256,
    )
    family_delay_applied = should_apply_pn_family_delay(family)
    center = train.phase_offset
    source = "cyclic"
    if family_delay_applied:
        center = delay_corrected_phase_offset(
            train.phase_offset,
            median_delay_symbols=family.median_delay_symbols,
            frame_symbols=train.frame_symbols,
        )
        source = "pn-family-delay"
    return {
        "center_phase_offset": center,
        "source": source,
        "cyclic_train": train.to_dict(),
        "cyclic_cfo_hz": cfo_hz,
        "family_delay_train": family.to_dict(),
        "family_delay_applied": family_delay_applied,
    }


def _summary_json(diagnostics: dict[str, Any], *, top: int) -> str:
    search = diagnostics["search"]
    compact = {
        key: diagnostics[key]
        for key in (
            "capture_path",
            "input_sample_rate_sps",
            "symbol_rate_sps",
            "resample_up",
            "resample_down",
            "frequency_shift_hz",
            "input_skip_samples",
            "start_frame_index",
            "analysis_start_symbol",
            "analyzed_symbols",
            "timing_search_symbols",
        )
    }
    compact["acquisition"] = diagnostics["acquisition"]
    compact["search"] = {
        key: search[key]
        for key in (
            "mode",
            "center_phase_offset",
            "search_radius_symbols",
            "step_symbols",
            "max_frames",
            "expected_system_info_index",
            "expected_frame_body_mode",
            "per_candidate_cfo",
            "jobs",
            "candidate_count",
            "selected",
        )
    }
    compact["search"]["top_candidates"] = search["candidates"][:top]
    return json.dumps(compact, indent=2, sort_keys=True)


def _tracking_summary_json(diagnostics: dict[str, Any], *, top: int) -> str:
    tracking = diagnostics["tracking"]
    compact = {
        key: diagnostics[key]
        for key in (
            "capture_path",
            "input_sample_rate_sps",
            "symbol_rate_sps",
            "resample_up",
            "resample_down",
            "frequency_shift_hz",
            "input_skip_samples",
            "start_frame_index",
            "analyzed_symbols",
        )
    }
    compact["acquisition"] = diagnostics["acquisition"]
    compact["tracking"] = {
        key: tracking[key]
        for key in (
            "mode",
            "center_phase_offset",
            "start_frame_index",
            "window_frames",
            "window_step_frames",
            "max_windows",
            "search_radius_symbols",
            "step_symbols",
            "expected_system_info_index",
            "expected_frame_body_mode",
            "follow_selected_phase",
            "per_candidate_cfo",
            "jobs",
            "continuity_tracking",
            "continuity_candidate_limit",
            "continuity_bridge_candidate_limit",
            "smoothness_penalty_per_symbol",
            "max_phase_step_symbols",
            "window_count",
            "summary",
        )
    }
    compact["tracking"]["trajectory"] = _compact_timing_trajectory(
        tracking.get("trajectory")
    )
    compact["tracking"]["windows"] = [
        {
            key: window[key]
            for key in (
                "start_frame_index",
                "analysis_start_symbol",
                "center_phase_offset",
                "selected",
                "candidate_count",
            )
        }
        | {"top_candidates": window["top_candidates"][:top]}
        for window in tracking["windows"]
    ]
    return json.dumps(compact, indent=2, sort_keys=True)


def _compact_timing_trajectory(trajectory: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trajectory, dict):
        return None
    compact = {
        key: trajectory.get(key)
        for key in (
            "available",
            "reason",
            "candidate_limit",
            "bridge_candidate_limit",
            "candidate_counts_by_window",
            "adjacent_candidate_steps",
            "smoothness_penalty_per_symbol",
            "max_phase_step_symbols",
            "window_count",
            "total_local_score",
            "total_smoothness_penalty",
            "total_path_score",
            "phase_offsets",
            "relative_offsets_symbols",
            "unwrapped_relative_offsets_symbols",
            "phase_span_symbols",
            "max_adjacent_phase_step_symbols",
        )
    }
    compact["windows"] = [
        {
            key: window.get(key)
            for key in (
                "window_index",
                "start_frame_index",
                "analysis_start_symbol",
                "phase_offset",
                "relative_offset_symbols",
                "candidate_rank",
                "local_score",
                "transition_step_symbols",
                "transition_penalty",
                "path_score",
            )
        }
        for window in trajectory.get("windows", [])
    ]
    return compact


def _summarize_timing_windows(
    windows: Sequence[dict[str, Any]],
    *,
    center_phase_offset: int,
    frame_symbols: int,
) -> dict[str, Any]:
    selected = [
        window.get("selected")
        for window in windows
        if isinstance(window.get("selected"), dict)
    ]
    if not selected:
        return {
            "selected_window_count": 0,
            "total_scored_frames": 0,
            "expected_top_count": 0,
            "expected_top3_count": 0,
            "expected_top_ratio": 0.0,
            "expected_top3_ratio": 0.0,
            "median_expected_metric": 0.0,
            "best_expected_metric": 0.0,
            "phase_offsets": [],
            "relative_offsets_symbols": [],
            "phase_span_symbols": 0,
            "max_adjacent_phase_step_symbols": 0,
        }
    frame_counts = [int(item.get("frame_count", 0)) for item in selected]
    expected_top = [int(item.get("expected_top_count", 0)) for item in selected]
    expected_top3 = [int(item.get("expected_top3_count", 0)) for item in selected]
    expected_metrics = [float(item.get("median_expected_metric", 0.0)) for item in selected]
    best_metrics = [float(item.get("best_expected_metric", 0.0)) for item in selected]
    phases = [int(item["phase_offset"]) for item in selected]
    relative = [
        _phase_delta(
            phase,
            center_phase_offset=center_phase_offset,
            frame_symbols=frame_symbols,
        )
        for phase in phases
    ]
    unwrapped = _unwrap_phase_offsets(relative, frame_symbols=frame_symbols)
    adjacent_steps = [
        abs(unwrapped[index] - unwrapped[index - 1])
        for index in range(1, len(unwrapped))
    ]
    total_frames = int(sum(frame_counts))
    total_expected_top = int(sum(expected_top))
    total_expected_top3 = int(sum(expected_top3))
    return {
        "selected_window_count": int(len(selected)),
        "total_scored_frames": total_frames,
        "expected_top_count": total_expected_top,
        "expected_top3_count": total_expected_top3,
        "expected_top_ratio": float(total_expected_top / total_frames)
        if total_frames
        else 0.0,
        "expected_top3_ratio": float(total_expected_top3 / total_frames)
        if total_frames
        else 0.0,
        "median_expected_metric": float(np.median(expected_metrics))
        if expected_metrics
        else 0.0,
        "best_expected_metric": float(np.max(best_metrics)) if best_metrics else 0.0,
        "phase_offsets": phases,
        "relative_offsets_symbols": relative,
        "unwrapped_relative_offsets_symbols": unwrapped,
        "phase_span_symbols": int(max(unwrapped) - min(unwrapped)) if unwrapped else 0,
        "max_adjacent_phase_step_symbols": int(max(adjacent_steps))
        if adjacent_steps
        else 0,
    }


def _unwrap_phase_offsets(
    offsets: Sequence[int],
    *,
    frame_symbols: int,
) -> list[int]:
    if frame_symbols <= 0:
        raise ValueError("frame_symbols must be positive")
    if not offsets:
        return []
    unwrapped = [int(offsets[0])]
    half = frame_symbols // 2
    for offset in offsets[1:]:
        previous = unwrapped[-1]
        delta = int(offset) - (previous % frame_symbols)
        while delta > half:
            delta -= frame_symbols
        while delta < -half:
            delta += frame_symbols
        unwrapped.append(previous + delta)
    return unwrapped


def _candidate_sort_key(candidate: FrameTimingCandidate) -> tuple[float, ...]:
    return (
        candidate.expected_top_count,
        candidate.expected_top3_count,
        candidate.score,
        candidate.c3780_top_count,
        candidate.best_expected_metric,
        candidate.median_expected_metric,
        candidate.mean_best_metric,
    )


def _pn_payload_timing_sort_key(
    candidate: PnPayloadTimingCandidate,
) -> tuple[float, float, float, int]:
    return (
        candidate.grid_evm_rms,
        candidate.max_abs_axis_occupancy_error,
        candidate.max_abs_hard_bit_bias,
        candidate.phase_offset,
    )


def _candidate_to_search_dict(
    candidate: FrameTimingCandidate,
    *,
    center_phase_offset: int,
    frame_symbols: int,
) -> dict[str, Any]:
    data = candidate.to_dict()
    data["relative_offset_symbols"] = _phase_delta(
        candidate.phase_offset,
        center_phase_offset=center_phase_offset,
        frame_symbols=frame_symbols,
    )
    return data


def _timing_frame_body_samples(
    frame: Any,
    signal: np.ndarray,
    *,
    mode: PnMode,
    body_window_offset_symbols: int,
) -> np.ndarray | None:
    if int(body_window_offset_symbols) == 0:
        return np.asarray(frame.body, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    body_start = int(frame.start) + int(definition.header_symbols) + int(
        body_window_offset_symbols
    )
    body_stop = body_start + FRAME_BODY_SYMBOLS
    if body_start < 0 or body_stop > signal.size:
        return None
    return np.asarray(signal[body_start:body_stop], dtype=np.complex64)


def _top_index_count_rows(
    counts: dict[tuple[str, int], int],
    metric_sums: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    rows = [
        {
            "frame_body_mode": frame_body_mode,
            "index": int(index),
            "count": int(count),
            "mean_metric": float(metric_sums.get((frame_body_mode, index), 0.0) / count)
            if count
            else 0.0,
        }
        for (frame_body_mode, index), count in counts.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            -float(row["mean_metric"]),
            str(row["frame_body_mode"]),
            int(row["index"]),
        )
    )
    return rows


def _phase_delta(
    phase_offset: int,
    *,
    center_phase_offset: int,
    frame_symbols: int,
) -> int:
    half = frame_symbols // 2
    return int((phase_offset - center_phase_offset + half) % frame_symbols - half)


def _timing_score(
    *,
    frame_count: int,
    expected_top_count: int,
    expected_top3_count: int,
    c3780_top_count: int,
    mean_best_metric: float,
    best_expected_metric: float,
    median_expected_metric: float,
) -> float:
    if frame_count <= 0:
        return 0.0
    expected_top_ratio = expected_top_count / frame_count
    expected_top3_ratio = expected_top3_count / frame_count
    c3780_ratio = c3780_top_count / frame_count
    return (
        40.0 * expected_top_ratio
        + 15.0 * expected_top3_ratio
        + 10.0 * c3780_ratio
        + 10.0 * _normalize(best_expected_metric, low=0.18, high=0.45)
        + 5.0 * _normalize(median_expected_metric, low=0.14, high=0.35)
        + 8.0 * _normalize(mean_best_metric, low=0.18, high=0.40)
    )


def _normalize(value: float, *, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("normalization high must exceed low")
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
