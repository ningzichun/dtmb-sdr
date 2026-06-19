"""Constellation diagnostics after DTMB frame acquisition."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .carrier_axes import (
    apply_carrier_permutation_to_inserted,
    logical_position_to_coords,
    system_info_positions_for_extraction,
)
from .channel import (
    apply_body_window_offset_to_pn_channel_response,
    equalize_spectrum_with_channel,
    estimate_pn_channel,
    estimate_pn_channel_compact,
    detect_pn_phase,
    pn_restore_circular_body_window,
    shift_pn_channel_response,
)
from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .equalization import (
    correct_common_phase_error,
    equalize_c3780_spectrum_with_system_info_pilots,
    refine_c3780_spectrum_decision_directed,
)
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    delay_corrected_phase_offset,
    detect_pn_cyclic_extension_trains,
    estimate_cfo_from_pn_cyclic_extension,
    score_pn_family_delay_train,
    should_apply_pn_family_delay,
)
from .frames import SignalFrame, iter_signal_frames
from .frequency import (
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_POSITIONS,
    frame_body_fft,
    frequency_deinterleave_index_map,
    frequency_deinterleave_inserted,
    frequency_interleave_index_map,
    split_system_info_and_data,
)
from .pn import PN_DEFINITIONS, PnMode
from .qam import (
    QAM_DEFINITIONS,
    QamMode,
    normalize_qam_symbols,
    qam_nearest,
    qam_symbol_quality,
)
from .system_info import (
    SYSTEM_INFO_VECTORS,
    classify_system_info,
    system_info_symbols,
    transmission_parameters_for_index,
)
from .timing import search_frame_timing


@dataclass(frozen=True)
class ConstellationDiagnostics:
    """Compact constellation/equalization summary."""

    summary: dict[str, Any]
    raw_points: np.ndarray
    equalized_points: np.ndarray
    raw_system_info_points: np.ndarray
    aligned_system_info_points: np.ndarray
    system_info_reference_points: np.ndarray
    system_info_carrier_error_rms: np.ndarray


def analyze_constellation(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 8_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    conjugate_iq: bool = False,
    mode: PnMode = "pn945",
    phase_offset: int | None = None,
    max_frames: int = 24,
    system_info_index: int = 23,
    frame_body_mode: str = "C3780",
    qam_mode: QamMode | str = "auto",
    equalizer: str = "sparse",
    pn_isi_cancel: bool = False,
    body_window_offset_symbols: int = 0,
    fft_bin_shift: int = 0,
    frequency_deinterleaver_direction: str = "standard",
    data_carrier_order: str = "normal",
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
    system_info_source: str = "raw",
    system_info_selection: str = "low_data_evm",
    timing_policy: str = "fixed",
    timing_trajectory: dict[str, Any] | None = None,
    timing_trajectory_path: str | Path | None = None,
    correct_per_frame_cpe: bool = False,
    per_frame_cpe_max_relative_error: float = 0.7,
    point_normalization: str = "frame",
    timing_search: bool = True,
    timing_radius_symbols: int = 512,
    timing_step_symbols: int = 4,
    family_frames: int = 24,
    max_delay_symbols: int = 256,
    family_hit_threshold: float = 0.45,
    point_limit: int = 20_000,
) -> ConstellationDiagnostics:
    """Return constellation points and a JSON-serializable diagnostic summary."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if point_limit <= 0:
        raise ValueError("point_limit must be positive")
    if equalizer not in ("sparse", "dd"):
        raise ValueError("equalizer must be sparse or dd")
    requested_qam_mode = str(qam_mode)
    resolved_qam_mode = _resolve_qam_mode(
        requested_qam_mode,
        system_info_index=system_info_index,
        frame_body_mode=frame_body_mode,
    )
    frequency_deinterleaver_direction = str(frequency_deinterleaver_direction)
    data_carrier_order = str(data_carrier_order)
    carrier_permutation = str(carrier_permutation).strip().lower() or "identity"
    timing_policy = str(timing_policy)
    if frequency_deinterleaver_direction not in ("standard", "inverse"):
        raise ValueError("frequency_deinterleaver_direction must be standard or inverse")
    if data_carrier_order not in ("normal", "reverse"):
        raise ValueError("data_carrier_order must be normal or reverse")
    system_info_source = str(system_info_source)
    if system_info_source not in ("raw", "equalized", "pn", "dd-data"):
        raise ValueError("system_info_source must be raw, equalized, pn, or dd-data")
    system_info_selection = str(system_info_selection)
    if system_info_selection not in ("low_data_evm", "all"):
        raise ValueError("system_info_selection must be low_data_evm or all")
    if timing_policy not in ("fixed", "windowed", "trajectory"):
        raise ValueError("timing_policy must be fixed, windowed, or trajectory")
    if per_frame_cpe_max_relative_error <= 0.0:
        raise ValueError("per_frame_cpe_max_relative_error must be positive")
    point_normalization = str(point_normalization)
    if point_normalization not in ("frame", "global"):
        raise ValueError("point_normalization must be frame or global")
    symbols, resample_up, resample_down = _load_symbols(
        capture_path,
        sample_rate_sps=sample_rate_sps,
        symbol_rate_sps=symbol_rate_sps,
        max_samples=max_samples,
        frequency_shift_hz=frequency_shift_hz,
        input_skip_samples=input_skip_samples,
        conjugate_iq=conjugate_iq,
    )
    acquisition = _select_phase_and_cfo(
        symbols,
        mode=mode,
        phase_offset=phase_offset,
        symbol_rate_sps=symbol_rate_sps,
        max_frames=max_frames,
        timing_search=timing_search,
        timing_radius_symbols=timing_radius_symbols,
        timing_step_symbols=timing_step_symbols,
        expected_system_info_index=system_info_index,
        expected_frame_body_mode=frame_body_mode,
        family_frames=family_frames,
        max_delay_symbols=max_delay_symbols,
        family_hit_threshold=family_hit_threshold,
    )
    corrected = symbols
    if acquisition["coarse_cfo_hz"] is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=symbol_rate_sps,
            shift_hz=-float(acquisition["coarse_cfo_hz"]),
        )

    raw_frames: list[np.ndarray] = []
    equalized_frames: list[np.ndarray] = []
    system_info_frames_by_source: dict[str, list[np.ndarray]] = {
        "raw": [],
        "equalized": [],
        "pn": [],
        "dd-data": [],
    }
    system_info_gains_by_source: dict[str, list[complex]] = {
        "raw": [],
        "equalized": [],
        "pn": [],
        "dd-data": [],
    }
    system_info_errors_by_source: dict[str, list[float]] = {
        "raw": [],
        "equalized": [],
        "pn": [],
        "dd-data": [],
    }
    pilot_errors: list[float] = []
    evm_values: list[float] = []
    cpe_corrected_evm_values: list[float] = []
    per_frame_cpe_reports: list[dict[str, Any]] = []
    expected_metrics: list[float] = []
    best_metrics: list[float] = []
    raw_system_info_errors: list[float] = []
    raw_system_info_cpe: list[float] = []
    raw_system_info_gain_magnitude: list[float] = []
    pn_system_info_errors: list[float] = []
    pn_data_evm_values: list[float] = []
    pn_sparse_data_evm_values: list[float] = []
    pn_sparse_pilot_errors: list[float] = []
    pn_sparse_residual_flatness: list[float] = []
    pn_sparse_response_errors: list[float] = []
    pn_sparse_pilot_response_errors: list[float] = []
    adjacent_pn_channel_relative_rms: list[float] = []
    adjacent_pn_channel_phase: list[float] = []
    adjacent_equalizer_channel_relative_rms: list[float] = []
    adjacent_equalizer_channel_phase: list[float] = []
    adjacent_equalizer_pilot_channel_relative_rms: list[float] = []
    adjacent_equalizer_pilot_channel_phase: list[float] = []
    expected_top_count = 0
    expected_top3_count = 0
    c3780_top_count = 0
    frame_reports = []
    system_info_reference = _system_info_reference(
        system_info_index,
        frame_body_mode=frame_body_mode,
    )
    system_info_pilot_positions = np.asarray(
        system_info_positions_for_extraction(
            carrier_permutation,
            int(logical_position_shift_symbols),
        ),
        dtype=np.int32,
    )
    logical_to_spectrum_positions = _logical_to_spectrum_positions_for_direction(
        frequency_deinterleaver_direction
    )
    pilot_positions = logical_to_spectrum_positions[system_info_pilot_positions]

    # Collect up to max_frames+1 frames so each analyzed frame can see the
    # following frame's PN header (needed for PN-ISI circular restoration).
    trajectory_report = _load_timing_trajectory(
        timing_trajectory=timing_trajectory,
        timing_trajectory_path=timing_trajectory_path,
    )
    frame_entries = _select_frame_entries(
        corrected,
        mode=mode,
        max_frames=max_frames + 1,
        timing_policy=timing_policy,
        trajectory_report=trajectory_report,
        fallback_phase_offset=int(acquisition["phase_offset"]),
        fallback_cfo_hz=acquisition.get("coarse_cfo_hz"),
        symbol_rate_sps=symbol_rate_sps,
    )
    analyzed_entries = frame_entries[:max_frames]
    pn_isi_applied_count = 0
    previous_pn_response: np.ndarray | None = None
    previous_equalizer_channel: np.ndarray | None = None

    for frame_index, (frame, frame_timing) in enumerate(analyzed_entries):
        next_frame = (
            frame_entries[frame_index + 1][0]
            if frame_index + 1 < len(frame_entries)
            else None
        )
        # Choose the PN phase whose channel taps are most compact. This keeps
        # PN tail subtraction aligned with the transmitted phase while avoiding
        # the family-matcher phase/delay alias that collapses the equalized
        # constellation (see dtmb.channel.estimate_pn_channel_compact).
        pn_channel = estimate_pn_channel_compact(
            frame.header,
            mode=mode,
            fft_size=3780,
        )
        pn_phase = pn_channel.pn_phase
        pn_isi_this_frame = False
        body_samples = _frame_body_samples(
            frame,
            corrected,
            mode=mode,
            body_window_offset_symbols=int(body_window_offset_symbols),
        )
        if pn_isi_cancel and next_frame is not None:
            next_channel = estimate_pn_channel_compact(
                next_frame.header,
                mode=mode,
                fft_size=3780,
            )
            body_samples = pn_restore_circular_body_window(
                frame.body,
                body_window_offset_symbols=int(body_window_offset_symbols),
                pn_phase=pn_phase,
                next_pn_phase=next_channel.pn_phase,
                mode=mode,
                taps=pn_channel.taps,
                next_header=next_frame.header,
            )
            pn_isi_this_frame = True
            pn_isi_applied_count += 1
        spectrum = frame_body_fft(body_samples)
        if fft_bin_shift:
            spectrum = np.roll(spectrum, int(fft_bin_shift))
        deinterleaved = _frequency_deinterleave_variant(
            spectrum,
            direction=frequency_deinterleaver_direction,
            carrier_permutation=carrier_permutation,
            logical_position_shift_symbols=logical_position_shift_symbols,
        )
        system_info, _data = split_system_info_and_data(deinterleaved)
        matches = classify_system_info(
            system_info,
            frame_body_modes=(frame_body_mode,),
        )
        raw_gain = _reference_gain(system_info, system_info_reference)
        raw_error = _reference_error_rms(
            system_info,
            system_info_reference,
            gain=raw_gain,
        )
        raw_system_info_errors.append(raw_error)
        raw_system_info_cpe.append(float(np.angle(raw_gain)))
        raw_system_info_gain_magnitude.append(float(abs(raw_gain)))
        _record_system_info_source(
            "raw",
            system_info,
            system_info_reference,
            frames_by_source=system_info_frames_by_source,
            gains_by_source=system_info_gains_by_source,
            errors_by_source=system_info_errors_by_source,
            gain=raw_gain,
            error=raw_error,
        )
        eq = _equalize_constellation_spectrum(
            spectrum,
            equalizer=equalizer,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=resolved_qam_mode,
            system_info_positions=system_info_pilot_positions,
            logical_to_spectrum_positions=logical_to_spectrum_positions,
        )
        _raw_system_info, raw_data = split_system_info_and_data(deinterleaved)
        equalized_deinterleaved = _frequency_deinterleave_variant(
            eq.equalized,
            direction=frequency_deinterleaver_direction,
            carrier_permutation=carrier_permutation,
            logical_position_shift_symbols=logical_position_shift_symbols,
        )
        equalized_system_info, equalized_data = split_system_info_and_data(
            equalized_deinterleaved
        )
        equalized_gain = _reference_gain(
            equalized_system_info,
            system_info_reference,
        )
        equalized_error = _reference_error_rms(
            equalized_system_info,
            system_info_reference,
            gain=equalized_gain,
        )
        _record_system_info_source(
            "equalized",
            equalized_system_info,
            system_info_reference,
            frames_by_source=system_info_frames_by_source,
            gains_by_source=system_info_gains_by_source,
            errors_by_source=system_info_errors_by_source,
            gain=equalized_gain,
            error=equalized_error,
        )
        if data_carrier_order == "reverse":
            raw_data = raw_data[::-1].astype(np.complex64, copy=False)
            equalized_data = equalized_data[::-1].astype(np.complex64, copy=False)
        equalized_data_evm = _qam_data_evm_rms(
            equalized_data,
            mode=resolved_qam_mode,
        )
        per_frame_cpe_report = None
        if correct_per_frame_cpe:
            cpe = correct_common_phase_error(
                equalized_data,
                qam_mode=resolved_qam_mode,
                max_relative_error=per_frame_cpe_max_relative_error,
            )
            equalized_data = cpe.corrected_symbols
            per_frame_cpe_report = cpe.to_dict()
            per_frame_cpe_reports.append(per_frame_cpe_report)
            cpe_corrected_evm_values.append(
                _qam_data_evm_rms(equalized_data, mode=resolved_qam_mode)
            )
        raw_frames.append(raw_data)
        equalized_frames.append(equalized_data)
        pilot_errors.append(eq.pilot_error_rms)
        evm_values.append(equalized_data_evm)

        pn_response_channel = apply_body_window_offset_to_pn_channel_response(
            pn_channel,
            int(body_window_offset_symbols),
        )
        pn_response_channel = shift_pn_channel_response(
            pn_response_channel,
            int(fft_bin_shift),
        )
        pn_spectrum = equalize_spectrum_with_channel(spectrum, pn_response_channel)
        pn_deinterleaved = _frequency_deinterleave_variant(
            pn_spectrum,
            direction=frequency_deinterleaver_direction,
            carrier_permutation=carrier_permutation,
            logical_position_shift_symbols=logical_position_shift_symbols,
        )
        pn_system_info, _pn_data = split_system_info_and_data(pn_deinterleaved)
        pn_matches = classify_system_info(
            pn_system_info,
            frame_body_modes=(frame_body_mode,),
        )
        pn_expected = [
            match
            for match in pn_matches
            if match.index == system_info_index and match.frame_body_mode == frame_body_mode
        ]
        pn_gain = _reference_gain(pn_system_info, system_info_reference)
        pn_error = _reference_error_rms(
            pn_system_info,
            system_info_reference,
            gain=pn_gain,
        )
        _record_system_info_source(
            "pn",
            pn_system_info,
            system_info_reference,
            frames_by_source=system_info_frames_by_source,
            gains_by_source=system_info_gains_by_source,
            errors_by_source=system_info_errors_by_source,
            gain=pn_gain,
            error=pn_error,
        )
        _pn_system_info, pn_data = split_system_info_and_data(pn_deinterleaved)
        if data_carrier_order == "reverse":
            pn_data = pn_data[::-1].astype(np.complex64, copy=False)
        pn_data_evm = _qam_data_evm_rms(pn_data, mode=resolved_qam_mode)
        dd_data = refine_c3780_spectrum_decision_directed(
            spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=resolved_qam_mode,
            system_info_positions=system_info_pilot_positions,
            logical_to_spectrum_positions=logical_to_spectrum_positions,
            include_system_info_pilots=False,
        )
        dd_data_deinterleaved = _frequency_deinterleave_variant(
            dd_data.equalized,
            direction=frequency_deinterleaver_direction,
            carrier_permutation=carrier_permutation,
            logical_position_shift_symbols=logical_position_shift_symbols,
        )
        dd_data_system_info, _dd_data_symbols = split_system_info_and_data(
            dd_data_deinterleaved
        )
        dd_data_gain = _reference_gain(dd_data_system_info, system_info_reference)
        dd_data_error = _reference_error_rms(
            dd_data_system_info,
            system_info_reference,
            gain=dd_data_gain,
        )
        _record_system_info_source(
            "dd-data",
            dd_data_system_info,
            system_info_reference,
            frames_by_source=system_info_frames_by_source,
            gains_by_source=system_info_gains_by_source,
            errors_by_source=system_info_errors_by_source,
            gain=dd_data_gain,
            error=dd_data_error,
        )
        pn_sparse = equalize_c3780_spectrum_with_system_info_pilots(
            pn_spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=resolved_qam_mode,
            system_info_positions=system_info_pilot_positions,
            logical_to_spectrum_positions=logical_to_spectrum_positions,
        )
        pn_sparse_channel = _channel_flatness(pn_sparse.channel)
        pn_vs_sparse = _channel_agreement(pn_response_channel.response, eq.channel)
        pn_vs_sparse_pilots = _channel_agreement(
            pn_response_channel.response[pilot_positions],
            eq.channel[pilot_positions],
        )
        adjacent_pn_channel = None
        adjacent_equalizer_channel = None
        adjacent_equalizer_pilot_channel = None
        if previous_pn_response is not None:
            adjacent_pn_channel = _channel_agreement(
                previous_pn_response,
                pn_response_channel.response,
            )
            adjacent_pn_channel_relative_rms.append(
                adjacent_pn_channel["relative_rms"]
            )
            adjacent_pn_channel_phase.append(
                adjacent_pn_channel["gain_phase_rad"]
            )
        if previous_equalizer_channel is not None:
            adjacent_equalizer_channel = _channel_agreement(
                previous_equalizer_channel,
                eq.channel,
            )
            adjacent_equalizer_pilot_channel = _channel_agreement(
                previous_equalizer_channel[pilot_positions],
                eq.channel[pilot_positions],
            )
            adjacent_equalizer_channel_relative_rms.append(
                adjacent_equalizer_channel["relative_rms"]
            )
            adjacent_equalizer_channel_phase.append(
                adjacent_equalizer_channel["gain_phase_rad"]
            )
            adjacent_equalizer_pilot_channel_relative_rms.append(
                adjacent_equalizer_pilot_channel["relative_rms"]
            )
            adjacent_equalizer_pilot_channel_phase.append(
                adjacent_equalizer_pilot_channel["gain_phase_rad"]
            )
        previous_pn_response = pn_response_channel.response.copy()
        previous_equalizer_channel = eq.channel.copy()
        pn_system_info_errors.append(pn_error)
        pn_data_evm_values.append(pn_data_evm)
        pn_sparse_data_evm_values.append(pn_sparse.data_evm_rms)
        pn_sparse_pilot_errors.append(pn_sparse.pilot_error_rms)
        pn_sparse_residual_flatness.append(pn_sparse_channel["relative_rms"])
        pn_sparse_response_errors.append(pn_vs_sparse["relative_rms"])
        pn_sparse_pilot_response_errors.append(pn_vs_sparse_pilots["relative_rms"])
        pn_pilot_channel = _channel_flatness(pn_response_channel.response[pilot_positions])
        sparse_pilot_channel = _channel_flatness(eq.channel[pilot_positions])
        phase_sources = {
            "pn_pilot_channel_mean_phase_rad": pn_pilot_channel["gain_phase_rad"],
            "pn_pilot_channel_mean_magnitude": pn_pilot_channel["mean_magnitude"],
            "pn_pilot_channel_relative_rms": pn_pilot_channel["relative_rms"],
            "si_sparse_pilot_channel_mean_phase_rad": sparse_pilot_channel[
                "gain_phase_rad"
            ],
            "si_sparse_pilot_channel_mean_magnitude": sparse_pilot_channel[
                "mean_magnitude"
            ],
            "si_sparse_pilot_channel_relative_rms": sparse_pilot_channel[
                "relative_rms"
            ],
            "sparse_minus_pn_pilot_phase_rad": pn_vs_sparse_pilots[
                "gain_phase_rad"
            ],
            "pn_vs_sparse_pilot_relative_rms": pn_vs_sparse_pilots[
                "relative_rms"
            ],
        }

        best = matches[0] if matches else None
        expected = [
            match
            for match in matches
            if match.index == system_info_index and match.frame_body_mode == frame_body_mode
        ]
        expected_rank = next(
            (
                rank + 1
                for rank, match in enumerate(matches)
                if match.index == system_info_index
                and match.frame_body_mode == frame_body_mode
            ),
            None,
        )
        if best is not None:
            best_metrics.append(best.metric)
            if best.frame_body_mode == "C3780":
                c3780_top_count += 1
            if best.index == system_info_index and best.frame_body_mode == frame_body_mode:
                expected_top_count += 1
        if expected:
            expected_metrics.append(expected[0].metric)
        if expected_rank is not None and expected_rank <= 3:
            expected_top3_count += 1

        equalizer_report = eq.to_dict()
        equalizer_report["plot_qam_mode"] = resolved_qam_mode
        equalizer_report["plot_qam_data_evm_rms"] = equalized_data_evm
        pn_channel_report = pn_channel.to_dict()
        pn_channel_report["response_body_window_offset_symbols"] = int(
            body_window_offset_symbols
        )
        pn_channel_report["response_fft_bin_shift"] = int(fft_bin_shift)
        frame_reports.append(
            {
                "frame_index": frame_index,
                "start": frame.start,
                "timing": frame_timing,
                "pn_isi_cancelled": pn_isi_this_frame,
                "best_system_info": best.to_dict() if best else None,
                "expected_system_info_metric": expected[0].metric if expected else 0.0,
                "expected_system_info_rank": expected_rank,
                "raw_system_info": {
                    "gain_magnitude": float(abs(raw_gain)),
                    "cpe_rad": float(np.angle(raw_gain)),
                    "error_rms": raw_error,
                },
                "equalized_system_info": {
                    "gain_magnitude": float(abs(equalized_gain)),
                    "cpe_rad": float(np.angle(equalized_gain)),
                    "error_rms": equalized_error,
                    "trained_on_system_info_index": system_info_index,
                    "independent": False,
                },
                "sparse_pilot_equalization": equalizer_report,
                "per_frame_cpe_correction": per_frame_cpe_report,
                "pn_channel": pn_channel_report,
                "pn_equalization": {
                    "best_system_info": pn_matches[0].to_dict()
                    if pn_matches
                    else None,
                    "expected_system_info_metric": pn_expected[0].metric
                    if pn_expected
                    else 0.0,
                    "system_info_gain_magnitude": float(abs(pn_gain)),
                    "system_info_cpe_rad": float(np.angle(pn_gain)),
                    "system_info_error_rms": pn_error,
                    "data_evm_rms": pn_data_evm,
                    "data_power": float(np.mean(np.abs(pn_data) ** 2))
                    if pn_data.size
                    else 0.0,
                },
                "pn_sparse_equalization": pn_sparse.to_dict(),
                "dd_data_equalization": dd_data.to_dict(),
                "pn_sparse_residual_channel": pn_sparse_channel,
                "pn_vs_sparse_channel": pn_vs_sparse,
                "pn_vs_sparse_pilot_channel": pn_vs_sparse_pilots,
                "phase_sources": phase_sources,
                "adjacent_pn_channel": adjacent_pn_channel,
                "adjacent_equalizer_channel": adjacent_equalizer_channel,
                "adjacent_equalizer_pilot_channel": adjacent_equalizer_pilot_channel,
            }
        )

    raw_points = _bounded_points(
        _normalize_constellation_frames(
            raw_frames,
            mode=point_normalization,
            qam_mode=resolved_qam_mode,
        ),
        limit=point_limit,
    )
    equalized_points = _bounded_points(
        _normalize_constellation_frames(
            equalized_frames,
            mode=point_normalization,
            qam_mode=resolved_qam_mode,
        ),
        limit=point_limit,
    )
    system_info_source_summaries = _system_info_source_summaries(
        system_info_frames_by_source,
        system_info_gains_by_source,
        system_info_errors_by_source,
        system_info_reference,
        frame_reports=frame_reports,
        trained_source="equalized",
        trained_system_info_index=system_info_index,
    )
    system_info_plot = _system_info_plot_points(
        system_info_frames_by_source[system_info_source],
        system_info_gains_by_source[system_info_source],
        system_info_reference,
        frame_reports=frame_reports,
        selection_mode=system_info_selection,
    )
    system_info_carrier_error_rows = _system_info_carrier_error_rows(
        system_info_plot["carrier_error_rms"],
        system_info_positions=system_info_pilot_positions,
        logical_to_spectrum_positions=logical_to_spectrum_positions,
    )
    source_independent = system_info_source not in ("equalized", "dd-data")
    selection_independent = system_info_selection == "all"
    low_evm_system_info = _low_evm_system_info_summary(frame_reports)
    low_evm_system_info.update(
        {
            "selection_independent": False,
            "selection_trained_on_system_info_index": int(system_info_index),
            "independent_gate_c_evidence": False,
        }
    )
    frame_count = len(frame_reports)
    frame_drift_rows = _frame_drift_rows(frame_reports)
    summary = {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "resample_up": resample_up,
        "resample_down": resample_down,
        "frequency_shift_hz": frequency_shift_hz,
        "input_skip_samples": input_skip_samples,
        "conjugate_iq": bool(conjugate_iq),
        "mode": mode,
        "requested_qam_mode": requested_qam_mode,
        "qam_mode": resolved_qam_mode,
        "equalizer": equalizer,
        "pn_isi_cancel": bool(pn_isi_cancel),
        "pn_isi_cancelled_frames": int(pn_isi_applied_count),
        "body_window_offset_symbols": int(body_window_offset_symbols),
        "fft_bin_shift": int(fft_bin_shift),
        "frequency_deinterleaver_direction": frequency_deinterleaver_direction,
        "data_carrier_order": data_carrier_order,
        "carrier_permutation": carrier_permutation,
        "logical_position_shift_symbols": int(logical_position_shift_symbols),
        "system_info_source": system_info_source,
        "system_info_selection": system_info_selection,
        "system_info_source_independent": bool(source_independent),
        "system_info_selection_independent": bool(selection_independent),
        "system_info_points_independent_gate_c_evidence": bool(
            source_independent and selection_independent
        ),
        "system_info_pilot_positions": system_info_pilot_positions.tolist(),
        "logical_to_spectrum_positions": logical_to_spectrum_positions.tolist(),
        "timing_policy": timing_policy,
        "timing_trajectory_path": str(timing_trajectory_path)
        if timing_trajectory_path is not None
        else None,
        "correct_per_frame_cpe": bool(correct_per_frame_cpe),
        "per_frame_cpe_max_relative_error": float(per_frame_cpe_max_relative_error),
        "point_normalization": point_normalization,
        "max_samples": max_samples,
        "frames_requested": max_frames,
        "frames_used": frame_count,
        "phase_offset": acquisition["phase_offset"],
        "phase_offset_source": acquisition["phase_offset_source"],
        "coarse_cfo_hz": acquisition["coarse_cfo_hz"],
        "active_cfo_source": acquisition["active_cfo_source"],
        "cyclic_cfo_hz": acquisition["cyclic_cfo_hz"],
        "family_cfo_hz": acquisition["family_cfo_hz"],
        "timing_cfo_hz": acquisition["timing_cfo_hz"],
        "cyclic_phase_offset": acquisition["cyclic_phase_offset"],
        "family_delay_applied": acquisition["family_delay_applied"],
        "selected_train": acquisition["selected_train"],
        "family_delay_train": acquisition["family_delay_train"],
        "timing_search": _compact_timing_search(acquisition["timing_search"]),
        "system_info": {
            "expected_index": system_info_index,
            "expected_frame_body_mode": frame_body_mode,
            "expected_top_count": expected_top_count,
            "expected_top3_count": expected_top3_count,
            "c3780_top_count": c3780_top_count,
            "mean_best_metric": _mean_or_zero(best_metrics),
            "best_expected_metric": _max_or_zero(expected_metrics),
            "median_expected_metric": _median_or_zero(expected_metrics),
        },
        "equalization": {
            "median_pilot_error_rms": _median_or_zero(pilot_errors),
            "median_data_evm_rms": _median_or_zero(evm_values),
            "min_data_evm_rms": _min_or_zero(evm_values),
            "median_cpe_corrected_data_evm_rms": _median_or_none(
                cpe_corrected_evm_values
            ),
            "min_cpe_corrected_data_evm_rms": _min_or_none(
                cpe_corrected_evm_values
            ),
        },
        "qam_quality_candidates": _qam_quality_candidates(equalized_frames),
        "low_evm_system_info": low_evm_system_info,
        "system_info_sources": system_info_source_summaries,
        "per_frame_cpe": _per_frame_cpe_summary(
            per_frame_cpe_reports,
            enabled=correct_per_frame_cpe,
        ),
        "channel": {
            "median_raw_system_info_error_rms": _median_or_zero(
                raw_system_info_errors
            ),
            "mean_raw_system_info_cpe_rad": _mean_phase_or_zero(
                raw_system_info_cpe
            ),
            "median_raw_system_info_gain_magnitude": _median_or_zero(
                raw_system_info_gain_magnitude
            ),
            "median_pn_system_info_error_rms": _median_or_zero(
                pn_system_info_errors
            ),
            "median_pn_data_evm_rms": _median_or_zero(pn_data_evm_values),
            "min_pn_data_evm_rms": _min_or_zero(pn_data_evm_values),
            "median_pn_sparse_data_evm_rms": _median_or_zero(
                pn_sparse_data_evm_values
            ),
            "min_pn_sparse_data_evm_rms": _min_or_zero(
                pn_sparse_data_evm_values
            ),
            "median_pn_sparse_pilot_error_rms": _median_or_zero(
                pn_sparse_pilot_errors
            ),
            "median_pn_sparse_residual_channel_relative_rms": _median_or_zero(
                pn_sparse_residual_flatness
            ),
            "median_pn_vs_sparse_response_relative_rms": _median_or_zero(
                pn_sparse_response_errors
            ),
            "median_pn_vs_sparse_pilot_response_relative_rms": _median_or_zero(
                pn_sparse_pilot_response_errors
            ),
            "median_adjacent_pn_channel_relative_rms": _median_or_zero(
                adjacent_pn_channel_relative_rms
            ),
            "median_abs_adjacent_pn_channel_gain_phase_rad": _median_abs_or_zero(
                adjacent_pn_channel_phase
            ),
            "median_adjacent_equalizer_channel_relative_rms": _median_or_zero(
                adjacent_equalizer_channel_relative_rms
            ),
            "median_abs_adjacent_equalizer_channel_gain_phase_rad": _median_abs_or_zero(
                adjacent_equalizer_channel_phase
            ),
            "median_adjacent_equalizer_pilot_channel_relative_rms": _median_or_zero(
                adjacent_equalizer_pilot_channel_relative_rms
            ),
            "median_abs_adjacent_equalizer_pilot_channel_gain_phase_rad": _median_abs_or_zero(
                adjacent_equalizer_pilot_channel_phase
            ),
        },
        "points": {
            "point_limit": point_limit,
            "raw_retained": int(raw_points.size),
            "equalized_retained": int(equalized_points.size),
            "total_available": int(sum(frame.size for frame in equalized_frames)),
        },
        "system_info_points": {
            "source": system_info_source,
            "selection_mode": system_info_selection,
            "source_independent": bool(source_independent),
            "selection_independent": bool(selection_independent),
            "selection_trained_on_system_info_index": int(system_info_index)
            if not selection_independent or not source_independent
            else None,
            "independent_gate_c_evidence": bool(
                source_independent and selection_independent
            ),
            "selection": system_info_plot["selection"],
            "selected_frame_count": system_info_plot["selected_frame_count"],
            "selected_frame_indices": system_info_plot["selected_frame_indices"],
            "raw_retained": int(system_info_plot["raw"].size),
            "aligned_retained": int(system_info_plot["aligned"].size),
            "median_carrier_error_rms": _median_or_none(
                system_info_plot["carrier_error_rms"].tolist()
            ),
            "max_carrier_error_rms": _max_or_none(
                system_info_plot["carrier_error_rms"].tolist()
            ),
            "carrier_error_rows": system_info_carrier_error_rows,
            "axis_error_summary": _system_info_axis_error_summary(
                system_info_carrier_error_rows
            ),
        },
        "frame_drift": {
            "rows": frame_drift_rows,
            "summary": _frame_drift_summary(frame_drift_rows),
        },
        "frames": frame_reports,
    }
    return ConstellationDiagnostics(
        summary=summary,
        raw_points=raw_points.astype(np.complex64, copy=False),
        equalized_points=equalized_points.astype(np.complex64, copy=False),
        raw_system_info_points=system_info_plot["raw"].astype(
            np.complex64,
            copy=False,
        ),
        aligned_system_info_points=system_info_plot["aligned"].astype(
            np.complex64,
            copy=False,
        ),
        system_info_reference_points=system_info_plot["reference"].astype(
            np.complex64,
            copy=False,
        ),
        system_info_carrier_error_rms=system_info_plot["carrier_error_rms"].astype(
            np.float32,
            copy=False,
        ),
    )


def write_points_csv(
    path: str | Path,
    *,
    raw_points: np.ndarray,
    equalized_points: np.ndarray,
) -> None:
    """Write retained constellation points as a compact CSV."""

    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "real", "imag"])
        for value in raw_points:
            writer.writerow(["raw", float(value.real), float(value.imag)])
        for value in equalized_points:
            writer.writerow(["equalized", float(value.real), float(value.imag)])


def write_frame_metrics_csv(
    path: str | Path,
    *,
    frames: Sequence[dict[str, Any]],
) -> None:
    """Write one row per analyzed frame with compact equalization metrics."""

    output_path = Path(path)
    columns = [
        "frame_index",
        "start",
        "best_si_index",
        "best_si_mode",
        "expected_si_rank",
        "expected_si_metric",
        "raw_si_error_rms",
        "raw_si_cpe_rad",
        "sparse_data_evm_rms",
        "sparse_pilot_error_rms",
        "pn_metric",
        "pn_delay_symbols",
        "pn_si_error_rms",
        "pn_data_evm_rms",
        "pn_sparse_data_evm_rms",
        "pn_sparse_pilot_error_rms",
        "pn_sparse_residual_channel_relative_rms",
        "pn_vs_sparse_response_relative_rms",
        "pn_vs_sparse_pilot_response_relative_rms",
        "adjacent_pn_channel_relative_rms",
        "adjacent_pn_channel_gain_phase_rad",
        "adjacent_equalizer_channel_relative_rms",
        "adjacent_equalizer_channel_gain_phase_rad",
        "adjacent_equalizer_pilot_channel_relative_rms",
        "adjacent_equalizer_pilot_channel_gain_phase_rad",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in frames:
            best = frame.get("best_system_info") or {}
            raw = frame.get("raw_system_info") or {}
            sparse = frame.get("sparse_pilot_equalization") or {}
            pn_channel = frame.get("pn_channel") or {}
            pn = frame.get("pn_equalization") or {}
            pn_sparse = frame.get("pn_sparse_equalization") or {}
            residual = frame.get("pn_sparse_residual_channel") or {}
            pn_vs_sparse = frame.get("pn_vs_sparse_channel") or {}
            pn_vs_sparse_pilots = frame.get("pn_vs_sparse_pilot_channel") or {}
            adjacent_pn = frame.get("adjacent_pn_channel") or {}
            adjacent_eq = frame.get("adjacent_equalizer_channel") or {}
            adjacent_eq_pilots = frame.get("adjacent_equalizer_pilot_channel") or {}
            writer.writerow(
                {
                    "frame_index": frame.get("frame_index"),
                    "start": frame.get("start"),
                    "best_si_index": best.get("index"),
                    "best_si_mode": best.get("frame_body_mode"),
                    "expected_si_rank": frame.get("expected_system_info_rank"),
                    "expected_si_metric": frame.get("expected_system_info_metric"),
                    "raw_si_error_rms": raw.get("error_rms"),
                    "raw_si_cpe_rad": raw.get("cpe_rad"),
                    "sparse_data_evm_rms": sparse.get("data_evm_rms"),
                    "sparse_pilot_error_rms": sparse.get("pilot_error_rms"),
                    "pn_metric": pn_channel.get("metric"),
                    "pn_delay_symbols": pn_channel.get("delay_symbols"),
                    "pn_si_error_rms": pn.get("system_info_error_rms"),
                    "pn_data_evm_rms": pn.get("data_evm_rms"),
                    "pn_sparse_data_evm_rms": pn_sparse.get("data_evm_rms"),
                    "pn_sparse_pilot_error_rms": pn_sparse.get("pilot_error_rms"),
                    "pn_sparse_residual_channel_relative_rms": residual.get(
                        "relative_rms"
                    ),
                    "pn_vs_sparse_response_relative_rms": pn_vs_sparse.get(
                        "relative_rms"
                    ),
                    "pn_vs_sparse_pilot_response_relative_rms": pn_vs_sparse_pilots.get(
                        "relative_rms"
                    ),
                    "adjacent_pn_channel_relative_rms": adjacent_pn.get(
                        "relative_rms"
                    ),
                    "adjacent_pn_channel_gain_phase_rad": adjacent_pn.get(
                        "gain_phase_rad"
                    ),
                    "adjacent_equalizer_channel_relative_rms": adjacent_eq.get(
                        "relative_rms"
                    ),
                    "adjacent_equalizer_channel_gain_phase_rad": adjacent_eq.get(
                        "gain_phase_rad"
                    ),
                    "adjacent_equalizer_pilot_channel_relative_rms": adjacent_eq_pilots.get(
                        "relative_rms"
                    ),
                    "adjacent_equalizer_pilot_channel_gain_phase_rad": adjacent_eq_pilots.get(
                        "gain_phase_rad"
                    ),
                }
            )


def write_constellation_png(
    path: str | Path,
    *,
    raw_points: np.ndarray,
    equalized_points: np.ndarray,
    title: str,
    equalized_label: str = "Sparse-pilot equalized",
) -> None:
    """Render raw and equalized constellation scatter plots."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG constellation output requires matplotlib; "
            "run `pip install -r requirements.txt` or install `.[dsp]`"
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, points, label in (
        (axes[0], raw_points, "Raw C3780 data"),
        (axes[1], equalized_points, equalized_label),
    ):
        if points.size:
            axis.scatter(points.real, points.imag, s=2, alpha=0.25, linewidths=0)
        axis.set_title(label)
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-10, 10)
        axis.set_ylim(-10, 10)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_system_info_png(
    path: str | Path,
    *,
    raw_points: np.ndarray,
    aligned_points: np.ndarray,
    reference_points: np.ndarray,
    carrier_error_rms: np.ndarray,
    title: str,
    selection_label: str,
    source_label: str = "Raw",
) -> None:
    """Render SI-carrier diagnostics for the selected frames."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG system-information output requires matplotlib; "
            "run `pip install -r requirements.txt` or install `.[dsp]`"
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].set_title(f"{source_label} SI aligned ({selection_label})")
    if aligned_points.size:
        axes[0].scatter(
            aligned_points.real,
            aligned_points.imag,
            s=10,
            alpha=0.35,
            linewidths=0,
            label="observed",
        )
    if reference_points.size:
        axes[0].scatter(
            reference_points.real,
            reference_points.imag,
            s=80,
            marker="x",
            color="black",
            linewidths=1.4,
            label="expected",
        )
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].legend(loc="upper left")

    axes[1].set_title("Aligned SI carrier RMS error")
    errors = np.asarray(carrier_error_rms, dtype=np.float32).reshape(-1)
    if errors.size:
        axes[1].plot(np.arange(errors.size), errors, marker="o", markersize=3)
    axes[1].set_xlabel("SI carrier index")
    axes[1].set_ylabel("RMS error")
    axes[1].set_xlim(-0.5, 35.5)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_system_info_axis_error_png(
    report: dict[str, Any],
    path: str | Path,
) -> None:
    """Render SI carrier RMS error against C=3780 logical/physical axes."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG system-information axis output requires matplotlib; "
            "run `pip install -r requirements.txt` or install `.[dsp]`"
        ) from exc

    system_info_points = report.get("system_info_points", report)
    rows = list(system_info_points.get("carrier_error_rows") or [])
    if not rows:
        errors = np.empty(0, dtype=np.float32)
        logical = np.empty(0, dtype=np.int32)
        physical = np.empty(0, dtype=np.int32)
    else:
        errors = np.asarray(
            [float(row.get("error_rms") or 0.0) for row in rows],
            dtype=np.float32,
        )
        logical = np.asarray(
            [int(row.get("logical_position") or 0) for row in rows],
            dtype=np.int32,
        )
        physical = np.asarray(
            [int(row.get("physical_bin") or 0) for row in rows],
            dtype=np.int32,
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    carrier_index = np.arange(errors.size, dtype=np.int32)
    if errors.size:
        axes[0, 0].plot(carrier_index, errors, marker="o", markersize=3)
    axes[0, 0].set_title("SI carrier order")
    axes[0, 0].set_xlabel("SI carrier index")
    axes[0, 0].set_ylabel("RMS error")
    axes[0, 0].grid(True, alpha=0.25)

    if errors.size:
        axes[0, 1].scatter(physical, errors, c=carrier_index, cmap="viridis", s=28)
    axes[0, 1].set_title("Physical FFT bin")
    axes[0, 1].set_xlabel("Physical bin")
    axes[0, 1].set_ylabel("RMS error")
    axes[0, 1].grid(True, alpha=0.25)

    if errors.size:
        axes[1, 0].scatter(logical, errors, c=physical, cmap="plasma", s=28)
    axes[1, 0].set_title("Logical C=3780 position")
    axes[1, 0].set_xlabel("Logical position")
    axes[1, 0].set_ylabel("RMS error")
    axes[1, 0].grid(True, alpha=0.25)

    axis_summary = system_info_points.get("axis_error_summary") or {}
    x_labels: list[str] = []
    y_values: list[float] = []
    for axis_name in ("i", "j", "k", "l", "m", "n", "o"):
        for row in axis_summary.get(axis_name, []):
            x_labels.append(f"{axis_name}{int(row.get('value') or 0)}")
            y_values.append(float(row.get("median_error_rms") or 0.0))
    x = np.arange(len(x_labels), dtype=np.int32)
    if x_labels:
        axes[1, 1].bar(x, y_values)
    axes[1, 1].set_title("Axis-bin median RMS")
    axes[1, 1].set_xlabel("Axis bin")
    axes[1, 1].set_ylabel("Median RMS error")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    source = system_info_points.get("source") or report.get("system_info_source")
    selection = system_info_points.get("selection") or report.get(
        "system_info_selection"
    )
    fig.suptitle(f"SI carrier axis error source={source} selection={selection}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_frame_drift_png(
    report: dict[str, Any],
    path: str | Path,
) -> None:
    """Render per-frame timing, CPE, SI metric, and EVM drift diagnostics."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG frame-drift output requires matplotlib; "
            "run `pip install -r requirements.txt` or install `.[dsp]`"
        ) from exc

    drift = report.get("frame_drift", report)
    rows = list(drift.get("rows") or [])
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = _row_series(rows, "frame_index")
    phase_offset = _row_series(rows, "phase_offset")
    residual_cfo_hz = _row_series(rows, "residual_cfo_hz")
    raw_si_cpe = _row_series(rows, "raw_si_cpe_unwrapped_rad")
    equalized_si_cpe = _row_series(rows, "equalized_si_cpe_unwrapped_rad")
    per_frame_cpe = _row_series(rows, "per_frame_cpe_unwrapped_rad")
    pn_pilot_phase = _row_series(rows, "pn_pilot_channel_phase_unwrapped_rad")
    si_sparse_pilot_phase = _row_series(
        rows,
        "si_sparse_pilot_channel_phase_unwrapped_rad",
    )
    expected_metric = _row_series(rows, "expected_si_metric")
    expected_rank = _row_series(rows, "expected_si_rank")
    raw_si_error = _row_series(rows, "raw_si_error_rms")
    equalized_si_error = _row_series(rows, "equalized_si_error_rms")
    data_evm = _row_series(rows, "sparse_data_evm_rms")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].set_title("Frame timing")
    _plot_optional_series(axes[0, 0], frame_index, phase_offset, label="phase offset")
    axes[0, 0].set_xlabel("Frame index")
    axes[0, 0].set_ylabel("Phase offset (symbols)")
    axes[0, 0].grid(True, alpha=0.25)
    if _has_finite_values(residual_cfo_hz):
        twin = axes[0, 0].twinx()
        _plot_optional_series(
            twin,
            frame_index,
            residual_cfo_hz,
            label="residual CFO",
            color="tab:red",
        )
        twin.set_ylabel("Residual CFO (Hz)")

    axes[0, 1].set_title("Common phase")
    _plot_optional_series(
        axes[0, 1],
        frame_index,
        raw_si_cpe,
        label="raw SI",
        color="tab:blue",
    )
    _plot_optional_series(
        axes[0, 1],
        frame_index,
        equalized_si_cpe,
        label="equalized SI",
        color="tab:orange",
    )
    _plot_optional_series(
        axes[0, 1],
        frame_index,
        per_frame_cpe,
        label="DD CPE",
        color="tab:green",
    )
    _plot_optional_series(
        axes[0, 1],
        frame_index,
        pn_pilot_phase,
        label="PN pilot ch",
        color="tab:red",
    )
    _plot_optional_series(
        axes[0, 1],
        frame_index,
        si_sparse_pilot_phase,
        label="SI sparse ch",
        color="tab:cyan",
    )
    axes[0, 1].set_xlabel("Frame index")
    axes[0, 1].set_ylabel("Unwrapped phase (rad)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best")

    axes[1, 0].set_title("System-information score")
    _plot_optional_series(
        axes[1, 0],
        frame_index,
        expected_metric,
        label="expected metric",
        color="tab:purple",
    )
    axes[1, 0].set_xlabel("Frame index")
    axes[1, 0].set_ylabel("Expected SI metric")
    axes[1, 0].grid(True, alpha=0.25)
    if _has_finite_values(expected_rank):
        rank_axis = axes[1, 0].twinx()
        _plot_optional_series(
            rank_axis,
            frame_index,
            expected_rank,
            label="expected rank",
            color="tab:brown",
        )
        rank_axis.set_ylabel("Expected SI rank")
        rank_axis.invert_yaxis()

    axes[1, 1].set_title("SI error and data EVM")
    _plot_optional_series(
        axes[1, 1],
        frame_index,
        raw_si_error,
        label="raw SI error",
        color="tab:blue",
    )
    _plot_optional_series(
        axes[1, 1],
        frame_index,
        equalized_si_error,
        label="equalized SI error",
        color="tab:orange",
    )
    _plot_optional_series(
        axes[1, 1],
        frame_index,
        data_evm,
        label="data EVM",
        color="tab:green",
    )
    axes[1, 1].set_xlabel("Frame index")
    axes[1, 1].set_ylabel("RMS")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(loc="best")

    fig.suptitle(
        "Frame drift "
        f"policy={report.get('timing_policy')} "
        f"source={report.get('system_info_source')}"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-constellation",
        description="Generate DTMB C=3780 constellation/equalization diagnostics.",
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
    parser.add_argument(
        "--conjugate-iq",
        action="store_true",
        help="Conjugate the IQ stream before acquisition to test mirrored captures.",
    )
    parser.add_argument("--mode", choices=("pn420", "pn945"), default="pn945")
    parser.add_argument("--phase-offset", type=_non_negative_int)
    parser.add_argument("--frames", type=_positive_int, default=24)
    parser.add_argument("--system-info-index", type=_positive_int, default=23)
    parser.add_argument("--frame-body-mode", choices=("C3780", "C1"), default="C3780")
    parser.add_argument(
        "--qam",
        choices=("auto", "4qam", "16qam", "32qam", "64qam"),
        default="auto",
        help="QAM grid used for DD refinement, EVM, and constellation normalization.",
    )
    parser.add_argument("--equalizer", choices=("sparse", "dd"), default="sparse")
    parser.add_argument(
        "--pn-isi-cancel",
        action="store_true",
        help="Subtract PN-header ISI and restore body cyclicity using the next PN header.",
    )
    parser.add_argument("--body-window-offset", type=int, default=0)
    parser.add_argument("--fft-bin-shift", type=int, default=0)
    parser.add_argument(
        "--frequency-deinterleaver-direction",
        choices=("standard", "inverse"),
        default="standard",
    )
    parser.add_argument(
        "--data-carrier-order",
        choices=("normal", "reverse"),
        default="normal",
    )
    parser.add_argument("--carrier-permutation", default="identity")
    parser.add_argument("--logical-position-shift", type=int, default=0)
    parser.add_argument(
        "--timing-policy",
        choices=("fixed", "windowed", "trajectory"),
        default="fixed",
    )
    parser.add_argument("--timing-trajectory", type=Path)
    parser.add_argument(
        "--correct-per-frame-cpe",
        action="store_true",
        help="Apply decision-directed per-frame CPE correction before plotting points.",
    )
    parser.add_argument(
        "--per-frame-cpe-max-relative-error",
        type=_positive_float,
        default=0.7,
        help="Relative slicing-error threshold for --correct-per-frame-cpe.",
    )
    parser.add_argument(
        "--point-normalization",
        choices=("frame", "global"),
        default="frame",
        help="Normalize constellation points per frame for plotting, or globally.",
    )
    parser.add_argument("--no-timing-search", action="store_false", dest="timing_search")
    parser.add_argument("--timing-radius", type=_non_negative_int, default=512)
    parser.add_argument("--timing-step", type=_positive_int, default=4)
    parser.add_argument("--point-limit", type=_positive_int, default=20_000)
    parser.add_argument("--points-csv", type=Path, help="Optional retained point CSV")
    parser.add_argument(
        "--frame-metrics-csv",
        type=Path,
        help="Optional per-frame channel/equalization metrics CSV",
    )
    parser.add_argument("--png", type=Path, help="Optional constellation plot PNG")
    parser.add_argument(
        "--system-info-png",
        type=Path,
        help="Optional low-EVM system-information carrier diagnostic PNG",
    )
    parser.add_argument(
        "--system-info-axis-png",
        type=Path,
        help="Optional SI carrier error-by-axis diagnostic PNG",
    )
    parser.add_argument(
        "--frame-drift-png",
        type=Path,
        help="Optional per-frame timing/CPE/SI metric drift diagnostic PNG",
    )
    parser.add_argument(
        "--system-info-source",
        choices=("raw", "equalized", "pn", "dd-data"),
        default="raw",
        help=(
            "SI symbol source used by --system-info-png. dd-data uses the "
            "data-only final DD channel and is diagnostic, not independent "
            "Gate C evidence."
        ),
    )
    parser.add_argument(
        "--system-info-selection",
        choices=("low_data_evm", "all"),
        default="low_data_evm",
        help="Frame selection used by --system-info-png.",
    )
    parser.add_argument("--json", type=Path, help="Optional path to write summary JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diagnostics = analyze_constellation(
        args.capture,
        sample_rate_sps=args.sample_rate,
        symbol_rate_sps=args.symbol_rate,
        max_samples=args.max_samples,
        frequency_shift_hz=args.frequency_shift,
        input_skip_samples=args.input_skip,
        conjugate_iq=args.conjugate_iq,
        mode=args.mode,
        phase_offset=args.phase_offset,
        max_frames=args.frames,
        system_info_index=args.system_info_index,
        frame_body_mode=args.frame_body_mode,
        qam_mode=args.qam,
        equalizer=args.equalizer,
        pn_isi_cancel=args.pn_isi_cancel,
        body_window_offset_symbols=args.body_window_offset,
        fft_bin_shift=args.fft_bin_shift,
        frequency_deinterleaver_direction=args.frequency_deinterleaver_direction,
        data_carrier_order=args.data_carrier_order,
        carrier_permutation=args.carrier_permutation,
        logical_position_shift_symbols=args.logical_position_shift,
        system_info_source=args.system_info_source,
        system_info_selection=args.system_info_selection,
        timing_policy=args.timing_policy,
        timing_trajectory_path=args.timing_trajectory,
        correct_per_frame_cpe=args.correct_per_frame_cpe,
        per_frame_cpe_max_relative_error=args.per_frame_cpe_max_relative_error,
        point_normalization=args.point_normalization,
        timing_search=args.timing_search,
        timing_radius_symbols=args.timing_radius,
        timing_step_symbols=args.timing_step,
        point_limit=args.point_limit,
    )
    if args.points_csv:
        write_points_csv(
            args.points_csv,
            raw_points=diagnostics.raw_points,
            equalized_points=diagnostics.equalized_points,
        )
        diagnostics.summary["points"]["csv_path"] = str(args.points_csv)
    if args.frame_metrics_csv:
        write_frame_metrics_csv(
            args.frame_metrics_csv,
            frames=diagnostics.summary["frames"],
        )
        diagnostics.summary["frame_metrics_csv_path"] = str(args.frame_metrics_csv)
    if args.png:
        equalized_label = f"{args.equalizer} equalized"
        if args.correct_per_frame_cpe:
            equalized_label += " + per-frame CPE"
        equalized_label += f" ({diagnostics.summary['qam_mode']})"
        write_constellation_png(
            args.png,
            raw_points=diagnostics.raw_points,
            equalized_points=diagnostics.equalized_points,
            title=f"{args.capture.name} {args.mode}",
            equalized_label=equalized_label,
        )
        diagnostics.summary["points"]["png_path"] = str(args.png)
    if args.system_info_png:
        write_system_info_png(
            args.system_info_png,
            raw_points=diagnostics.raw_system_info_points,
            aligned_points=diagnostics.aligned_system_info_points,
            reference_points=diagnostics.system_info_reference_points,
            carrier_error_rms=diagnostics.system_info_carrier_error_rms,
            title=f"{args.capture.name} {args.mode} SI carriers",
            selection_label=diagnostics.summary["system_info_points"]["selection"],
            source_label=diagnostics.summary["system_info_points"]["source"],
        )
        diagnostics.summary["system_info_points"]["png_path"] = str(
            args.system_info_png
        )
    if args.system_info_axis_png:
        write_system_info_axis_error_png(
            diagnostics.summary,
            args.system_info_axis_png,
        )
        diagnostics.summary["system_info_points"]["axis_png_path"] = str(
            args.system_info_axis_png
        )
    if args.frame_drift_png:
        write_frame_drift_png(
            diagnostics.summary,
            args.frame_drift_png,
        )
        diagnostics.summary["frame_drift"]["png_path"] = str(args.frame_drift_png)
    encoded = json.dumps(diagnostics.summary, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    if not args.quiet:
        print(encoded)
    return 0


def _load_symbols(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int,
    max_samples: int,
    frequency_shift_hz: float,
    input_skip_samples: int = 0,
    conjugate_iq: bool = False,
) -> tuple[np.ndarray, int, int]:
    if input_skip_samples < 0:
        raise ValueError("input_skip_samples must be non-negative")
    samples = remove_dc(
        read_ci8(
            capture_path,
            max_samples=max_samples,
            skip_samples=input_skip_samples,
        )
    )
    if conjugate_iq:
        samples = np.conj(samples)
    if frequency_shift_hz:
        samples = frequency_shift(
            samples,
            sample_rate_sps=sample_rate_sps,
            shift_hz=frequency_shift_hz,
        )
    if sample_rate_sps == symbol_rate_sps:
        return samples.astype(np.complex64, copy=False), 1, 1
    symbols, up, down = resample_to_symbol_rate(
        samples,
        sample_rate_sps=sample_rate_sps,
        symbol_rate_sps=symbol_rate_sps,
    )
    return symbols, up, down


def _load_timing_trajectory(
    *,
    timing_trajectory: dict[str, Any] | None,
    timing_trajectory_path: str | Path | None,
) -> dict[str, Any] | None:
    if timing_trajectory is not None:
        return timing_trajectory
    if timing_trajectory_path is None:
        return None
    return json.loads(Path(timing_trajectory_path).read_text(encoding="utf-8"))


def _select_frame_entries(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    max_frames: int,
    timing_policy: str,
    trajectory_report: dict[str, Any] | None,
    fallback_phase_offset: int,
    fallback_cfo_hz: float | None,
    symbol_rate_sps: int,
) -> list[tuple[SignalFrame, dict[str, Any]]]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if fallback_phase_offset < 0:
        raise ValueError("fallback_phase_offset must be non-negative")

    policy = str(timing_policy)
    if policy == "fixed":
        return [
            (
                frame,
                {
                    "policy": "fixed",
                    "source": "acquisition",
                    "nominal_frame_index": index,
                    "phase_offset": int(fallback_phase_offset),
                    "coarse_cfo_hz": fallback_cfo_hz,
                    "residual_cfo_hz": 0.0,
                },
            )
            for index, frame in enumerate(
                iter_signal_frames(
                    symbols,
                    mode=mode,
                    phase_offset=fallback_phase_offset,
                    max_frames=max_frames,
                )
            )
        ]
    if policy not in ("windowed", "trajectory"):
        raise ValueError("timing_policy must be fixed, windowed, or trajectory")

    trajectory = _trajectory_payload(trajectory_report)
    if trajectory is None:
        raise ValueError(f"timing_policy={policy} requires --timing-trajectory")
    trajectory_mode = trajectory.get("mode")
    if trajectory_mode is not None and str(trajectory_mode) != str(mode):
        raise ValueError("timing trajectory mode does not match receiver mode")
    if policy == "trajectory" and not bool(trajectory.get("available")):
        reason = trajectory.get("reason") or "trajectory_unavailable"
        raise ValueError(f"timing trajectory is unavailable: {reason}")

    segments = [
        segment
        for segment in trajectory.get("segments", [])
        if isinstance(segment, dict)
    ]
    if not segments:
        raise ValueError("timing trajectory contains no usable segments")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    entries: list[tuple[SignalFrame, dict[str, Any]]] = []
    for segment in segments:
        start_frame = int(segment.get("start_frame_index") or 0)
        end_frame = int(segment.get("end_frame_index") or start_frame)
        if start_frame < 0:
            raise ValueError("timing trajectory start_frame_index must be non-negative")
        if end_frame <= start_frame or segment.get("phase_offset") is None:
            continue
        phase_offset = int(segment["phase_offset"]) % definition.frame_symbols
        segment_cfo = segment.get("coarse_cfo_hz")
        residual_cfo_hz = _residual_trajectory_cfo(segment_cfo, fallback_cfo_hz)
        for nominal_frame_index in range(start_frame, end_frame):
            if len(entries) >= max_frames:
                return entries
            frame_start = nominal_frame_index * definition.frame_symbols + phase_offset
            frame_stop = frame_start + definition.frame_symbols
            if frame_start < 0 or frame_stop > signal.size:
                continue
            frame_samples = signal[frame_start:frame_stop]
            if residual_cfo_hz is not None and abs(residual_cfo_hz) > 0.0:
                frame_samples = _frequency_shift_frame(
                    frame_samples,
                    sample_rate_sps=symbol_rate_sps,
                    shift_hz=-residual_cfo_hz,
                    absolute_start_symbol=frame_start,
                )
            header_stop = definition.header_symbols
            frame = SignalFrame(
                mode=mode,
                start=int(frame_start),
                header=frame_samples[:header_stop],
                body=frame_samples[header_stop:],
            )
            timing = _compact_trajectory_segment(segment)
            timing.update(
                {
                    "policy": policy,
                    "nominal_frame_index": int(nominal_frame_index),
                    "phase_offset": int(phase_offset),
                    "coarse_cfo_hz": segment_cfo,
                    "fallback_cfo_hz": fallback_cfo_hz,
                    "residual_cfo_hz": residual_cfo_hz,
                }
            )
            entries.append((frame, timing))
    return entries


def _trajectory_payload(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    trajectory = report.get("trajectory")
    if isinstance(trajectory, dict):
        return trajectory
    if "segments" in report or "available" in report:
        return report
    return None


def _compact_trajectory_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: segment.get(key)
        for key in (
            "start_frame_index",
            "end_frame_index",
            "source",
            "candidate_rank",
            "transition_step_symbols",
            "expected_top_count",
            "expected_top3_count",
            "frame_count",
            "best_expected_metric",
            "median_expected_metric",
            "score",
        )
    }


def _residual_trajectory_cfo(
    segment_cfo: Any,
    fallback_cfo_hz: float | None,
) -> float | None:
    if segment_cfo is None:
        return None
    residual = float(segment_cfo)
    if fallback_cfo_hz is not None:
        residual -= float(fallback_cfo_hz)
    return residual


def _frequency_shift_frame(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    shift_hz: float,
    absolute_start_symbol: int,
) -> np.ndarray:
    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    signal = np.asarray(samples, dtype=np.complex64)
    if signal.size == 0 or shift_hz == 0:
        return signal.astype(np.complex64, copy=False)
    time_index = absolute_start_symbol + np.arange(signal.size, dtype=np.float64)
    rotation = np.exp(2j * np.pi * float(shift_hz) * time_index / sample_rate_sps)
    return (signal * rotation).astype(np.complex64, copy=False)


def _select_phase_and_cfo(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int | None,
    symbol_rate_sps: int,
    max_frames: int,
    timing_search: bool,
    timing_radius_symbols: int,
    timing_step_symbols: int,
    expected_system_info_index: int,
    expected_frame_body_mode: str,
    family_frames: int,
    max_delay_symbols: int,
    family_hit_threshold: float,
) -> dict[str, Any]:
    selected_train = None
    family_delay_train = None
    family_delay_applied = False
    timing_search_report = None
    cyclic_phase_offset = phase_offset
    phase_offset_source = "manual"
    cyclic_cfo_hz = None
    family_cfo_hz = None
    timing_cfo_hz = None
    active_cfo_source = "none"

    if phase_offset is None:
        trains = detect_pn_cyclic_extension_trains(symbols, modes=(mode,))
        if not trains:
            raise RuntimeError("no PN cyclic-extension trains found")
        selected_train = trains[0]
        phase_offset = selected_train.phase_offset
        cyclic_phase_offset = phase_offset
        phase_offset_source = "cyclic"

    cfo_hz = estimate_cfo_from_pn_cyclic_extension(
        symbols,
        mode=mode,
        phase_offset=phase_offset,
        symbol_rate_sps=symbol_rate_sps,
    )
    cyclic_cfo_hz = cfo_hz
    if cfo_hz is not None:
        active_cfo_source = "cyclic"
    corrected = symbols
    if cfo_hz is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=symbol_rate_sps,
            shift_hz=-cfo_hz,
        )

    if selected_train is not None:
        family_delay_train = score_pn_family_delay_train(
            corrected,
            mode=mode,
            phase_offset=phase_offset,
            max_frames=family_frames,
            max_delay_symbols=max_delay_symbols,
            hit_threshold=family_hit_threshold,
        )
        if should_apply_pn_family_delay(
            family_delay_train,
            family_hit_threshold=family_hit_threshold,
        ):
            phase_offset = delay_corrected_phase_offset(
                phase_offset,
                median_delay_symbols=family_delay_train.median_delay_symbols,
                frame_symbols=family_delay_train.frame_symbols,
            )
            phase_offset_source = "pn-family-delay"
            family_delay_applied = True
            family_cfo_hz = estimate_cfo_from_pn_cyclic_extension(
                symbols,
                mode=mode,
                phase_offset=phase_offset,
                symbol_rate_sps=symbol_rate_sps,
            )
            if family_cfo_hz is not None:
                cfo_hz = family_cfo_hz
                active_cfo_source = "pn-family-delay"

    if timing_search:
        timing_search_report = search_frame_timing(
            symbols,
            mode=mode,
            center_phase_offset=phase_offset,
            search_radius_symbols=timing_radius_symbols,
            step_symbols=timing_step_symbols,
            max_frames=max_frames,
            expected_system_info_index=expected_system_info_index,
            expected_frame_body_mode=expected_frame_body_mode,
        )
        selected_timing = timing_search_report.get("selected")
        if selected_timing is not None:
            phase_offset = int(selected_timing["phase_offset"])
            phase_offset_source = "timing-search"
            timing_cfo_hz = selected_timing.get("coarse_cfo_hz")
            if timing_cfo_hz is not None:
                cfo_hz = timing_cfo_hz
                active_cfo_source = "timing-search"

    return {
        "phase_offset": phase_offset,
        "phase_offset_source": phase_offset_source,
        "coarse_cfo_hz": cfo_hz,
        "active_cfo_source": active_cfo_source,
        "cyclic_cfo_hz": cyclic_cfo_hz,
        "family_cfo_hz": family_cfo_hz,
        "timing_cfo_hz": timing_cfo_hz,
        "cyclic_phase_offset": cyclic_phase_offset,
        "selected_train": selected_train.to_dict() if selected_train else None,
        "family_delay_train": family_delay_train.to_dict()
        if family_delay_train
        else None,
        "family_delay_applied": family_delay_applied,
        "timing_search": timing_search_report,
    }


def _frame_body_samples(
    frame: SignalFrame,
    symbols: np.ndarray,
    *,
    mode: PnMode,
    body_window_offset_symbols: int,
) -> np.ndarray:
    if int(body_window_offset_symbols) == 0:
        return frame.body.astype(np.complex64, copy=False)
    definition = PN_DEFINITIONS[mode]
    signal = np.asarray(symbols, dtype=np.complex64)
    body_start = (
        int(frame.start)
        + int(definition.header_symbols)
        + int(body_window_offset_symbols)
    )
    body_stop = body_start + FRAME_BODY_SYMBOLS
    if body_start < 0 or body_stop > signal.size:
        raise ValueError("body_window_offset_symbols moves the C3780 body outside the capture")
    return signal[body_start:body_stop].astype(np.complex64, copy=False)


def _frequency_deinterleave_variant(
    symbols: np.ndarray,
    *,
    direction: str,
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
) -> np.ndarray:
    return _logical_from_inserted_c3780(
        _frequency_deinterleave_inserted_variant(symbols, direction=direction),
        carrier_permutation=carrier_permutation,
        logical_position_shift_symbols=logical_position_shift_symbols,
    )


def _frequency_deinterleave_inserted_variant(
    symbols: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frequency deinterleaver requires 3780 symbols")
    if direction == "standard":
        return frequency_deinterleave_inserted(values)
    if direction == "inverse":
        return values[frequency_deinterleave_index_map()]
    raise ValueError("frequency-deinterleaver direction must be standard or inverse")


def _logical_to_spectrum_positions_for_direction(direction: str) -> np.ndarray:
    if direction == "standard":
        return frequency_interleave_index_map()
    if direction == "inverse":
        return frequency_deinterleave_index_map()
    raise ValueError("frequency-deinterleaver direction must be standard or inverse")


def _logical_from_inserted_c3780(
    symbols: np.ndarray,
    *,
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body requires 3780 symbols")
    values = apply_carrier_permutation_to_inserted(
        values,
        permutation_name=carrier_permutation,
    )
    if int(logical_position_shift_symbols) != 0:
        values = np.roll(values, -int(logical_position_shift_symbols))
    return np.concatenate(
        (
            values[SYSTEM_INFO_POSITIONS],
            np.delete(values, SYSTEM_INFO_POSITIONS),
        )
    )


def _resolve_qam_mode(
    requested_qam_mode: str,
    *,
    system_info_index: int,
    frame_body_mode: str,
) -> QamMode:
    if requested_qam_mode == "auto":
        parameters = transmission_parameters_for_index(
            system_info_index,
            frame_body_mode=frame_body_mode,
        )
        if parameters.qam_mode is not None:
            return parameters.qam_mode
        return "64qam"
    if requested_qam_mode not in QAM_DEFINITIONS:
        raise ValueError("qam_mode must be auto, 4qam, 16qam, 32qam, or 64qam")
    return requested_qam_mode  # type: ignore[return-value]


def _normalize_to_qam_grid(symbols: np.ndarray, *, mode: QamMode) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.complex64)
    if values.size == 0:
        return values
    return normalize_qam_symbols(values, mode=mode)


def _equalize_constellation_spectrum(
    spectrum: np.ndarray,
    *,
    equalizer: str,
    system_info_index: int,
    frame_body_mode: str,
    qam_mode: QamMode,
    system_info_positions: Sequence[int] | np.ndarray,
    logical_to_spectrum_positions: Sequence[int] | np.ndarray,
):
    if equalizer == "sparse":
        return equalize_c3780_spectrum_with_system_info_pilots(
            spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=qam_mode,
            system_info_positions=system_info_positions,
            logical_to_spectrum_positions=logical_to_spectrum_positions,
        )
    if equalizer == "dd":
        return refine_c3780_spectrum_decision_directed(
            spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=qam_mode,
            system_info_positions=system_info_positions,
            logical_to_spectrum_positions=logical_to_spectrum_positions,
        )
    raise ValueError("equalizer must be sparse or dd")


def _system_info_reference(system_info_index: int, *, frame_body_mode: str) -> np.ndarray:
    try:
        bits = SYSTEM_INFO_VECTORS[system_info_index]
    except KeyError as exc:
        raise ValueError(f"unknown system-info index: {system_info_index}") from exc
    return system_info_symbols(bits, frame_body_mode=frame_body_mode)


def _reference_gain(observed: np.ndarray, reference: np.ndarray) -> complex:
    observed_values = np.asarray(observed, dtype=np.complex64)
    reference_values = np.asarray(reference, dtype=np.complex64)
    denominator = float(np.vdot(reference_values, reference_values).real)
    if denominator <= 1e-12:
        return 0j
    return complex(np.vdot(reference_values, observed_values) / denominator)


def _reference_error_rms(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    gain: complex | None = None,
) -> float:
    observed_values = np.asarray(observed, dtype=np.complex64)
    reference_values = np.asarray(reference, dtype=np.complex64)
    if observed_values.size != reference_values.size:
        raise ValueError("observed and reference system-information sizes differ")
    if observed_values.size == 0:
        return 0.0
    if gain is None:
        gain = _reference_gain(observed_values, reference_values)
    corrected = observed_values / (gain if abs(gain) > 1e-12 else 1.0)
    return float(np.sqrt(np.mean(np.abs(corrected - reference_values) ** 2)))


def _record_system_info_source(
    source: str,
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    frames_by_source: dict[str, list[np.ndarray]],
    gains_by_source: dict[str, list[complex]],
    errors_by_source: dict[str, list[float]],
    gain: complex | None = None,
    error: float | None = None,
) -> None:
    values = np.asarray(observed, dtype=np.complex64).reshape(-1)
    if gain is None:
        gain = _reference_gain(values, reference)
    if error is None:
        error = _reference_error_rms(values, reference, gain=gain)
    frames_by_source[source].append(values.astype(np.complex64, copy=False))
    gains_by_source[source].append(gain)
    errors_by_source[source].append(float(error))


def _qam_data_evm_rms(symbols: np.ndarray, *, mode: QamMode) -> float:
    data = np.asarray(symbols, dtype=np.complex64)
    if data.size == 0:
        return 0.0
    if float(np.mean(np.abs(data) ** 2)) <= 0.0:
        return 0.0
    reference_power = float(QAM_DEFINITIONS[mode].average_power)
    observed_power = float(np.mean(np.abs(data) ** 2))
    scale = np.sqrt(observed_power / reference_power)
    corrected = data / max(scale, 1e-12)
    nearest = qam_nearest(corrected, mode=mode)
    gain = np.vdot(nearest, data) / max(float(np.vdot(nearest, nearest).real), 1e-12)
    corrected = data / (gain if abs(gain) > 1e-12 else 1.0)
    nearest = qam_nearest(corrected, mode=mode)
    nearest_power = float(np.mean(np.abs(nearest) ** 2))
    if nearest_power <= 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(corrected - nearest) ** 2) / nearest_power))


def _channel_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_values = np.asarray(reference, dtype=np.complex64).reshape(-1)
    candidate_values = np.asarray(candidate, dtype=np.complex64).reshape(-1)
    if reference_values.size != candidate_values.size:
        raise ValueError("channel responses must have the same size")
    if reference_values.size == 0:
        return {"relative_rms": 0.0, "gain_magnitude": 0.0, "gain_phase_rad": 0.0}
    denominator = float(np.vdot(reference_values, reference_values).real)
    if denominator <= 1e-12:
        return {"relative_rms": 0.0, "gain_magnitude": 0.0, "gain_phase_rad": 0.0}
    gain = complex(np.vdot(reference_values, candidate_values) / denominator)
    aligned = reference_values * gain
    candidate_power = float(np.mean(np.abs(candidate_values) ** 2))
    if candidate_power <= 1e-12:
        relative_rms = 0.0
    else:
        relative_rms = float(
            np.sqrt(np.mean(np.abs(aligned - candidate_values) ** 2) / candidate_power)
        )
    return {
        "relative_rms": relative_rms,
        "gain_magnitude": float(abs(gain)),
        "gain_phase_rad": float(np.angle(gain)),
    }


def _channel_flatness(channel: np.ndarray) -> dict[str, float]:
    values = np.asarray(channel, dtype=np.complex64).reshape(-1)
    if values.size == 0:
        return {"relative_rms": 0.0, "mean_magnitude": 0.0, "gain_phase_rad": 0.0}
    mean = complex(np.mean(values))
    power = float(np.mean(np.abs(values) ** 2))
    if power <= 1e-12:
        relative_rms = 0.0
    else:
        relative_rms = float(np.sqrt(np.mean(np.abs(values - mean) ** 2) / power))
    return {
        "relative_rms": relative_rms,
        "mean_magnitude": float(abs(mean)),
        "gain_phase_rad": float(np.angle(mean)),
    }


def _compact_timing_search(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        key: report[key]
        for key in (
            "mode",
            "center_phase_offset",
            "search_radius_symbols",
            "step_symbols",
            "max_frames",
            "expected_system_info_index",
            "expected_frame_body_mode",
            "candidate_count",
            "selected",
        )
    } | {"top_candidates": report["candidates"][:8]}


def _bounded_points(points: np.ndarray, *, limit: int) -> np.ndarray:
    if points.size <= limit:
        return points
    indices = np.linspace(0, points.size - 1, num=limit, dtype=np.int64)
    return points[indices]


def _concatenate_or_empty(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.empty(0, dtype=np.complex64)
    return np.concatenate(frames).astype(np.complex64, copy=False)


def _qam_quality_candidates(frames: list[np.ndarray]) -> list[dict[str, Any]]:
    frame_values = [
        np.asarray(frame, dtype=np.complex64).reshape(-1)
        for frame in frames
        if np.asarray(frame).size
    ]
    symbol_count = int(sum(frame.size for frame in frame_values))
    candidates = []
    for mode in ("4qam", "16qam", "32qam", "64qam"):
        qualities = [
            qam_symbol_quality(frame, mode=mode, normalize=True)
            for frame in frame_values
        ]
        grid_evm = _finite_quality_values(qualities, "grid_evm_rms")
        hard_bit_bias = _finite_quality_values(qualities, "max_abs_hard_bit_bias")
        axis_mean = _finite_axis_values(
            qualities,
            "mean_abs_fraction_error_from_uniform",
        )
        axis_max = _finite_axis_values(
            qualities,
            "max_abs_fraction_error_from_uniform",
        )
        median_grid_evm = _median_or_none(grid_evm)
        candidates.append(
            {
                "qam_mode": mode,
                "grid_evm_rms": median_grid_evm,
                "median_grid_evm_rms": median_grid_evm,
                "min_grid_evm_rms": _min_or_none(grid_evm),
                "max_grid_evm_rms": _max_or_none(grid_evm),
                "max_abs_hard_bit_bias": _median_or_none(hard_bit_bias),
                "median_max_abs_hard_bit_bias": _median_or_none(hard_bit_bias),
                "axis_mean_abs_fraction_error_from_uniform": _median_or_none(
                    axis_mean
                ),
                "median_axis_mean_abs_fraction_error_from_uniform": _median_or_none(
                    axis_mean
                ),
                "axis_max_abs_fraction_error_from_uniform": _median_or_none(
                    axis_max
                ),
                "median_axis_max_abs_fraction_error_from_uniform": _median_or_none(
                    axis_max
                ),
                "symbol_count": symbol_count,
                "frame_count": len(qualities),
            }
        )
    return candidates


def _system_info_source_summaries(
    frames_by_source: dict[str, list[np.ndarray]],
    gains_by_source: dict[str, list[complex]],
    errors_by_source: dict[str, list[float]],
    reference: np.ndarray,
    *,
    frame_reports: Sequence[dict[str, Any]],
    trained_source: str,
    trained_system_info_index: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for source in ("raw", "pn", "dd-data", "equalized"):
        gains = gains_by_source.get(source, [])
        errors = errors_by_source.get(source, [])
        low_evm_plot = _system_info_plot_points(
            frames_by_source.get(source, []),
            gains,
            reference,
            frame_reports=frame_reports,
            selection_mode="low_data_evm",
        )
        all_plot = _system_info_plot_points(
            frames_by_source.get(source, []),
            gains,
            reference,
            frame_reports=frame_reports,
            selection_mode="all",
        )
        carrier_errors = low_evm_plot["carrier_error_rms"].tolist()
        all_carrier_errors = all_plot["carrier_error_rms"].tolist()
        trained = source in (trained_source, "dd-data")
        summaries[source] = {
            "frame_count": len(frames_by_source.get(source, [])),
            "independent": not trained,
            "trained_on_system_info_index": int(trained_system_info_index)
            if trained
            else None,
            "low_evm_selection_independent": False,
            "low_evm_selection_trained_on_system_info_index": int(
                trained_system_info_index
            ),
            "low_evm_independent_gate_c_evidence": False,
            "all_selection_independent": True,
            "all_independent_gate_c_evidence": not trained,
            "median_error_rms": _median_or_none(errors),
            "min_error_rms": _min_or_none(errors),
            "median_gain_magnitude": _median_or_none(
                [float(abs(gain)) for gain in gains]
            ),
            "mean_cpe_rad": _mean_phase_or_none([float(np.angle(gain)) for gain in gains]),
            "low_evm_selection": low_evm_plot["selection"],
            "low_evm_selected_frame_count": low_evm_plot["selected_frame_count"],
            "low_evm_selected_frame_indices": low_evm_plot[
                "selected_frame_indices"
            ],
            "low_evm_median_carrier_error_rms": _median_or_none(carrier_errors),
            "low_evm_max_carrier_error_rms": _max_or_none(carrier_errors),
            "all_selected_frame_count": all_plot["selected_frame_count"],
            "all_median_carrier_error_rms": _median_or_none(all_carrier_errors),
            "all_max_carrier_error_rms": _max_or_none(all_carrier_errors),
        }
    return summaries


def _frame_drift_rows(frame_reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fallback_index, frame in enumerate(frame_reports):
        timing = frame.get("timing") or {}
        best = frame.get("best_system_info") or {}
        raw = frame.get("raw_system_info") or {}
        equalized = frame.get("equalized_system_info") or {}
        sparse = frame.get("sparse_pilot_equalization") or {}
        cpe = frame.get("per_frame_cpe_correction") or {}
        phase_sources = frame.get("phase_sources") or {}
        raw_si_cpe = _float_or_none(raw.get("cpe_rad"))
        equalized_si_cpe = _float_or_none(equalized.get("cpe_rad"))
        per_frame_cpe = _float_or_none(cpe.get("cpe_rad"))
        pn_pilot_phase = _float_or_none(
            phase_sources.get("pn_pilot_channel_mean_phase_rad")
        )
        si_sparse_pilot_phase = _float_or_none(
            phase_sources.get("si_sparse_pilot_channel_mean_phase_rad")
        )
        row = {
            "frame_index": int(frame.get("frame_index", fallback_index)),
            "start": _int_or_none(frame.get("start")),
            "nominal_frame_index": _int_or_none(timing.get("nominal_frame_index")),
            "phase_offset": _int_or_none(timing.get("phase_offset")),
            "phase_offset_step_symbols": None,
            "residual_cfo_hz": _float_or_none(timing.get("residual_cfo_hz")),
            "coarse_cfo_hz": _float_or_none(timing.get("coarse_cfo_hz")),
            "expected_si_rank": _int_or_none(frame.get("expected_system_info_rank")),
            "expected_si_metric": _float_or_none(
                frame.get("expected_system_info_metric")
            ),
            "best_si_index": _int_or_none(best.get("index")),
            "best_si_mode": best.get("frame_body_mode"),
            "best_si_metric": _float_or_none(best.get("metric")),
            "raw_si_cpe_rad": raw_si_cpe,
            "raw_si_cpe_unwrapped_rad": None,
            "raw_si_cpe_step_rad": None,
            "raw_si_error_rms": _float_or_none(raw.get("error_rms")),
            "raw_si_gain_magnitude": _float_or_none(raw.get("gain_magnitude")),
            "equalized_si_cpe_rad": equalized_si_cpe,
            "equalized_si_cpe_unwrapped_rad": None,
            "equalized_si_cpe_step_rad": None,
            "equalized_si_error_rms": _float_or_none(equalized.get("error_rms")),
            "equalized_si_gain_magnitude": _float_or_none(
                equalized.get("gain_magnitude")
            ),
            "sparse_data_evm_rms": _frame_plot_data_evm(frame),
            "sparse_pilot_error_rms": _float_or_none(sparse.get("pilot_error_rms")),
            "per_frame_cpe_rad": per_frame_cpe,
            "per_frame_cpe_unwrapped_rad": None,
            "per_frame_cpe_step_rad": None,
            "per_frame_cpe_reliable_symbol_count": _int_or_none(
                cpe.get("reliable_symbol_count")
            ),
            "pn_pilot_channel_phase_rad": pn_pilot_phase,
            "pn_pilot_channel_phase_unwrapped_rad": None,
            "pn_pilot_channel_phase_step_rad": None,
            "pn_pilot_channel_mean_magnitude": _float_or_none(
                phase_sources.get("pn_pilot_channel_mean_magnitude")
            ),
            "pn_pilot_channel_relative_rms": _float_or_none(
                phase_sources.get("pn_pilot_channel_relative_rms")
            ),
            "si_sparse_pilot_channel_phase_rad": si_sparse_pilot_phase,
            "si_sparse_pilot_channel_phase_unwrapped_rad": None,
            "si_sparse_pilot_channel_phase_step_rad": None,
            "si_sparse_pilot_channel_mean_magnitude": _float_or_none(
                phase_sources.get("si_sparse_pilot_channel_mean_magnitude")
            ),
            "si_sparse_pilot_channel_relative_rms": _float_or_none(
                phase_sources.get("si_sparse_pilot_channel_relative_rms")
            ),
            "sparse_minus_pn_pilot_phase_rad": _float_or_none(
                phase_sources.get("sparse_minus_pn_pilot_phase_rad")
            ),
            "sparse_minus_pn_pilot_phase_unwrapped_rad": None,
            "sparse_minus_pn_pilot_phase_step_rad": None,
            "pn_vs_sparse_pilot_relative_rms": _float_or_none(
                phase_sources.get("pn_vs_sparse_pilot_relative_rms")
            ),
            "raw_si_minus_pn_pilot_phase_rad": _phase_difference_or_none(
                raw_si_cpe,
                pn_pilot_phase,
            ),
            "dd_cpe_minus_raw_si_phase_rad": _phase_difference_or_none(
                per_frame_cpe,
                raw_si_cpe,
            ),
        }
        rows.append(row)

    _annotate_optional_phase(rows, "raw_si_cpe_rad", "raw_si_cpe_unwrapped_rad")
    _annotate_optional_phase(
        rows,
        "equalized_si_cpe_rad",
        "equalized_si_cpe_unwrapped_rad",
    )
    _annotate_optional_phase(
        rows,
        "per_frame_cpe_rad",
        "per_frame_cpe_unwrapped_rad",
    )
    _annotate_optional_phase(
        rows,
        "pn_pilot_channel_phase_rad",
        "pn_pilot_channel_phase_unwrapped_rad",
    )
    _annotate_optional_phase(
        rows,
        "si_sparse_pilot_channel_phase_rad",
        "si_sparse_pilot_channel_phase_unwrapped_rad",
    )
    _annotate_optional_phase(
        rows,
        "sparse_minus_pn_pilot_phase_rad",
        "sparse_minus_pn_pilot_phase_unwrapped_rad",
    )
    _annotate_optional_steps(rows, "phase_offset", "phase_offset_step_symbols")
    _annotate_optional_steps(
        rows,
        "raw_si_cpe_unwrapped_rad",
        "raw_si_cpe_step_rad",
    )
    _annotate_optional_steps(
        rows,
        "equalized_si_cpe_unwrapped_rad",
        "equalized_si_cpe_step_rad",
    )
    _annotate_optional_steps(
        rows,
        "per_frame_cpe_unwrapped_rad",
        "per_frame_cpe_step_rad",
    )
    _annotate_optional_steps(
        rows,
        "pn_pilot_channel_phase_unwrapped_rad",
        "pn_pilot_channel_phase_step_rad",
    )
    _annotate_optional_steps(
        rows,
        "si_sparse_pilot_channel_phase_unwrapped_rad",
        "si_sparse_pilot_channel_phase_step_rad",
    )
    _annotate_optional_steps(
        rows,
        "sparse_minus_pn_pilot_phase_unwrapped_rad",
        "sparse_minus_pn_pilot_phase_step_rad",
    )
    return rows


def _frame_drift_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phase_offsets = _finite_row_values(rows, "phase_offset")
    phase_steps = _finite_row_values(rows, "phase_offset_step_symbols")
    residual_cfo = _finite_row_values(rows, "residual_cfo_hz")
    expected_ranks = [
        int(value)
        for value in _finite_row_values(rows, "expected_si_rank")
    ]
    expected_metrics = _finite_row_values(rows, "expected_si_metric")
    return {
        "frame_count": int(len(rows)),
        "phase_offset_min": _min_or_none(phase_offsets),
        "phase_offset_max": _max_or_none(phase_offsets),
        "phase_offset_span_symbols": _span_or_none(phase_offsets),
        "max_abs_phase_offset_step_symbols": _max_abs_or_none(phase_steps),
        "median_abs_residual_cfo_hz": _median_abs_or_none(residual_cfo),
        "max_abs_residual_cfo_hz": _max_abs_or_none(residual_cfo),
        "raw_si_cpe_span_rad": _span_or_none(
            _finite_row_values(rows, "raw_si_cpe_unwrapped_rad")
        ),
        "max_abs_raw_si_cpe_step_rad": _max_abs_or_none(
            _finite_row_values(rows, "raw_si_cpe_step_rad")
        ),
        "equalized_si_cpe_span_rad": _span_or_none(
            _finite_row_values(rows, "equalized_si_cpe_unwrapped_rad")
        ),
        "max_abs_equalized_si_cpe_step_rad": _max_abs_or_none(
            _finite_row_values(rows, "equalized_si_cpe_step_rad")
        ),
        "per_frame_cpe_span_rad": _span_or_none(
            _finite_row_values(rows, "per_frame_cpe_unwrapped_rad")
        ),
        "max_abs_per_frame_cpe_step_rad": _max_abs_or_none(
            _finite_row_values(rows, "per_frame_cpe_step_rad")
        ),
        "pn_pilot_channel_phase_span_rad": _span_or_none(
            _finite_row_values(rows, "pn_pilot_channel_phase_unwrapped_rad")
        ),
        "max_abs_pn_pilot_channel_phase_step_rad": _max_abs_or_none(
            _finite_row_values(rows, "pn_pilot_channel_phase_step_rad")
        ),
        "si_sparse_pilot_channel_phase_span_rad": _span_or_none(
            _finite_row_values(rows, "si_sparse_pilot_channel_phase_unwrapped_rad")
        ),
        "max_abs_si_sparse_pilot_channel_phase_step_rad": _max_abs_or_none(
            _finite_row_values(rows, "si_sparse_pilot_channel_phase_step_rad")
        ),
        "median_abs_sparse_minus_pn_pilot_phase_rad": _median_abs_or_none(
            _finite_row_values(rows, "sparse_minus_pn_pilot_phase_rad")
        ),
        "max_abs_sparse_minus_pn_pilot_phase_rad": _max_abs_or_none(
            _finite_row_values(rows, "sparse_minus_pn_pilot_phase_rad")
        ),
        "median_pn_vs_sparse_pilot_relative_rms": _median_or_none(
            _finite_row_values(rows, "pn_vs_sparse_pilot_relative_rms")
        ),
        "median_abs_raw_si_minus_pn_pilot_phase_rad": _median_abs_or_none(
            _finite_row_values(rows, "raw_si_minus_pn_pilot_phase_rad")
        ),
        "median_abs_dd_cpe_minus_raw_si_phase_rad": _median_abs_or_none(
            _finite_row_values(rows, "dd_cpe_minus_raw_si_phase_rad")
        ),
        "expected_top_count": sum(1 for rank in expected_ranks if rank == 1),
        "expected_top3_count": sum(1 for rank in expected_ranks if rank <= 3),
        "median_expected_rank": _median_or_none(expected_ranks),
        "median_expected_metric": _median_or_none(expected_metrics),
        "median_raw_si_error_rms": _median_or_none(
            _finite_row_values(rows, "raw_si_error_rms")
        ),
        "median_equalized_si_error_rms": _median_or_none(
            _finite_row_values(rows, "equalized_si_error_rms")
        ),
        "median_sparse_data_evm_rms": _median_or_none(
            _finite_row_values(rows, "sparse_data_evm_rms")
        ),
        "median_per_frame_cpe_reliable_symbol_count": _median_or_none(
            _finite_row_values(rows, "per_frame_cpe_reliable_symbol_count")
        ),
    }


def _annotate_optional_phase(
    rows: list[dict[str, Any]],
    source_key: str,
    target_key: str,
) -> None:
    values = [_float_or_none(row.get(source_key)) for row in rows]
    finite_indices = [
        index
        for index, value in enumerate(values)
        if value is not None and np.isfinite(float(value))
    ]
    if not finite_indices:
        return
    unwrapped = np.unwrap(
        np.asarray([float(values[index]) for index in finite_indices], dtype=np.float64)
    )
    for index, value in zip(finite_indices, unwrapped):
        rows[index][target_key] = float(value)


def _annotate_optional_steps(
    rows: list[dict[str, Any]],
    source_key: str,
    target_key: str,
) -> None:
    previous: float | None = None
    for row in rows:
        value = _float_or_none(row.get(source_key))
        if value is None:
            continue
        if previous is not None:
            row[target_key] = float(value - previous)
        previous = value


def _system_info_carrier_error_rows(
    carrier_error_rms: np.ndarray,
    *,
    system_info_positions: Sequence[int] | np.ndarray,
    logical_to_spectrum_positions: Sequence[int] | np.ndarray,
) -> list[dict[str, Any]]:
    errors = np.asarray(carrier_error_rms, dtype=np.float32).reshape(-1)
    logical_positions = np.asarray(system_info_positions, dtype=np.int32).reshape(-1)
    logical_to_spectrum = np.asarray(
        logical_to_spectrum_positions,
        dtype=np.int32,
    ).reshape(-1)
    rows: list[dict[str, Any]] = []
    if errors.size == 0:
        return rows
    count = min(errors.size, logical_positions.size)
    for index in range(count):
        logical = int(logical_positions[index])
        physical = int(logical_to_spectrum[logical])
        coords = logical_position_to_coords(logical)
        rows.append(
            {
                "system_info_carrier_index": int(index),
                "logical_position": logical,
                "physical_bin": physical,
                "i": int(coords[0]),
                "j": int(coords[1]),
                "k": int(coords[2]),
                "l": int(coords[3]),
                "m": int(coords[4]),
                "n": int(coords[5]),
                "o": int(coords[6]),
                "error_rms": float(errors[index]),
            }
        )
    return rows


def _system_info_axis_error_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for axis_name in ("i", "j", "k", "l", "m", "n", "o"):
        values = sorted({int(row[axis_name]) for row in rows})
        axis_rows: list[dict[str, Any]] = []
        for value in values:
            errors = [
                float(row["error_rms"])
                for row in rows
                if int(row[axis_name]) == int(value)
            ]
            axis_rows.append(
                {
                    "axis": axis_name,
                    "value": int(value),
                    "carrier_count": int(len(errors)),
                    "median_error_rms": _median_or_none(errors),
                    "mean_error_rms": _mean_or_none(errors),
                    "max_error_rms": _max_or_none(errors),
                }
            )
        summary[axis_name] = axis_rows
    return summary


def _system_info_plot_points(
    raw_system_info_frames: Sequence[np.ndarray],
    reference_gains: Sequence[complex],
    reference: np.ndarray,
    *,
    frame_reports: Sequence[dict[str, Any]],
    selection_count: int = 16,
    selection_mode: str = "low_data_evm",
) -> dict[str, Any]:
    if selection_mode == "low_data_evm":
        indices = _low_evm_frame_indices(frame_reports, count=selection_count)
    elif selection_mode == "all":
        indices = list(range(len(frame_reports)))
    else:
        raise ValueError("system-info plot selection must be low_data_evm or all")
    selected_raw: list[np.ndarray] = []
    selected_aligned: list[np.ndarray] = []
    selected_frame_indices: list[int] = []
    reference_values = np.asarray(reference, dtype=np.complex64).reshape(-1)
    for index in indices:
        if index >= len(raw_system_info_frames):
            continue
        raw = np.asarray(raw_system_info_frames[index], dtype=np.complex64).reshape(-1)
        if raw.size != reference_values.size:
            continue
        gain = reference_gains[index] if index < len(reference_gains) else 0j
        aligned = raw / (gain if abs(gain) > 1e-12 else 1.0)
        selected_raw.append(raw)
        selected_aligned.append(aligned.astype(np.complex64, copy=False))
        selected_frame_indices.append(
            int(frame_reports[index].get("frame_index", index))
        )
    if selected_aligned:
        aligned_matrix = np.stack(selected_aligned)
        reference_matrix = np.tile(reference_values, (aligned_matrix.shape[0], 1))
        carrier_error = np.sqrt(
            np.mean(np.abs(aligned_matrix - reference_matrix) ** 2, axis=0)
        ).astype(np.float32)
        raw_points = np.concatenate(selected_raw).astype(np.complex64, copy=False)
        aligned_points = aligned_matrix.reshape(-1).astype(np.complex64, copy=False)
        reference_points = reference_matrix.reshape(-1).astype(
            np.complex64,
            copy=False,
        )
    else:
        carrier_error = np.empty(0, dtype=np.float32)
        raw_points = np.empty(0, dtype=np.complex64)
        aligned_points = np.empty(0, dtype=np.complex64)
        reference_points = np.empty(0, dtype=np.complex64)
    return {
        "selection": _system_info_selection_label(
            selection_mode,
            len(selected_frame_indices),
        ),
        "selected_frame_count": len(selected_frame_indices),
        "selected_frame_indices": selected_frame_indices,
        "raw": raw_points,
        "aligned": aligned_points,
        "reference": reference_points,
        "carrier_error_rms": carrier_error,
    }


def _system_info_selection_label(selection_mode: str, frame_count: int) -> str:
    if selection_mode == "low_data_evm":
        return f"lowest_{frame_count}_data_evm_frames"
    if selection_mode == "all":
        return f"all_{frame_count}_frames"
    raise ValueError("system-info plot selection must be low_data_evm or all")


def _low_evm_frame_indices(
    frames: Sequence[dict[str, Any]],
    *,
    count: int,
) -> list[int]:
    rows = []
    for index, frame in enumerate(frames):
        evm = _frame_plot_data_evm(frame)
        if evm is not None:
            rows.append((index, evm))
    rows.sort(key=lambda row: (row[1], row[0]))
    return [index for index, _evm in rows[: max(0, int(count))]]


def _low_evm_system_info_summary(
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for fallback_index, frame in enumerate(frames):
        evm = _frame_plot_data_evm(frame)
        if evm is None:
            continue
        best = frame.get("best_system_info") or {}
        expected_rank = frame.get("expected_system_info_rank")
        expected_metric = frame.get("expected_system_info_metric")
        rows.append(
            {
                "frame_index": int(frame.get("frame_index", fallback_index)),
                "start": frame.get("start"),
                "data_evm_rms": evm,
                "best_index": best.get("index"),
                "best_frame_body_mode": best.get("frame_body_mode"),
                "best_metric": _float_or_none(best.get("metric")),
                "expected_rank": int(expected_rank)
                if expected_rank is not None
                else None,
                "expected_metric": _float_or_none(expected_metric),
            }
        )
    rows.sort(key=lambda row: (row["data_evm_rms"], row["frame_index"]))
    return {
        "available_frames": len(rows),
        "by_best_frame_count": [
            _summarize_system_info_rows(
                rows[:count],
                label=f"best_{count}_evm_frames",
            )
            for count in _low_evm_selection_counts(len(rows))
        ],
        "by_evm_threshold": [
            _summarize_system_info_rows(
                [
                    row
                    for row in rows
                    if row["data_evm_rms"] <= threshold
                ],
                label=f"evm_le_{threshold:.2f}",
            )
            for threshold in (0.20, 0.22, 0.25, 0.30)
        ],
    }


def _low_evm_selection_counts(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    counts = {min(count, frame_count) for count in (4, 8, 16, 32)}
    return sorted(counts)


def _summarize_system_info_rows(
    rows: Sequence[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    evm_values = [float(row["data_evm_rms"]) for row in rows]
    expected_metrics = [
        float(row["expected_metric"])
        for row in rows
        if row.get("expected_metric") is not None
    ]
    expected_ranks = [
        int(row["expected_rank"])
        for row in rows
        if row.get("expected_rank") is not None
    ]
    return {
        "label": label,
        "frame_count": len(rows),
        "frame_indices": [row["frame_index"] for row in rows],
        "best_index_counts": _value_counts(
            row.get("best_index") for row in rows
        ),
        "expected_top_count": sum(1 for rank in expected_ranks if rank == 1),
        "expected_top3_count": sum(1 for rank in expected_ranks if rank <= 3),
        "median_expected_rank": _median_or_none(expected_ranks),
        "median_expected_metric": _median_or_none(expected_metrics),
        "max_expected_metric": _max_or_none(expected_metrics),
        "median_data_evm_rms": _median_or_none(evm_values),
        "max_data_evm_rms": _max_or_none(evm_values),
    }


def _value_counts(values: Sequence[Any]) -> list[dict[str, Any]]:
    counts: dict[Any, int] = {}
    for value in values:
        key = value if value is not None else "none"
        counts[key] = counts.get(key, 0) + 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    ]


def _frame_plot_data_evm(frame: dict[str, Any]) -> float | None:
    sparse = frame.get("sparse_pilot_equalization") or {}
    value = sparse.get("plot_qam_data_evm_rms")
    if value is None:
        value = sparse.get("data_evm_rms")
    return _float_or_none(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _phase_difference_or_none(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    if not np.isfinite(float(left)) or not np.isfinite(float(right)):
        return None
    return float(np.angle(np.exp(1j * (float(left) - float(right)))))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if np.isfinite(float(parsed)) else None


def _finite_row_values(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _row_series(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> np.ndarray:
    return np.asarray(
        [
            np.nan if _float_or_none(row.get(key)) is None else float(row.get(key))
            for row in rows
        ],
        dtype=np.float64,
    )


def _has_finite_values(values: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(np.asarray(values, dtype=np.float64))))


def _plot_optional_series(
    axis: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    label: str,
    color: str | None = None,
) -> None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.any(mask):
        axis.plot(
            x[mask],
            y[mask],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=label,
            color=color,
        )


def _finite_quality_values(
    qualities: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for quality in qualities:
        value = quality.get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _finite_axis_values(
    qualities: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for quality in qualities:
        value = (quality.get("axis_level_occupancy") or {}).get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _normalize_constellation_frames(
    frames: list[np.ndarray],
    *,
    mode: str,
    qam_mode: QamMode,
) -> np.ndarray:
    if mode == "global":
        return _normalize_to_qam_grid(
            _concatenate_or_empty(frames),
            mode=qam_mode,
        )
    if mode == "frame":
        return _concatenate_or_empty(
            [_normalize_to_qam_grid(frame, mode=qam_mode) for frame in frames]
        )
    raise ValueError("point normalization mode must be frame or global")


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _mean_phase_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    phases = np.asarray(values, dtype=np.float64)
    return float(np.angle(np.mean(np.exp(1j * phases))))


def _mean_phase_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    phases = np.asarray(values, dtype=np.float64)
    return float(np.angle(np.mean(np.exp(1j * phases))))


def _median_or_zero(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _median_or_none(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _min_or_none(values: Sequence[float]) -> float | None:
    return float(np.min(values)) if values else None


def _max_or_none(values: Sequence[float]) -> float | None:
    return float(np.max(values)) if values else None


def _span_or_none(values: Sequence[float]) -> float | None:
    return float(np.max(values) - np.min(values)) if values else None


def _median_abs_or_none(values: Sequence[float]) -> float | None:
    return float(np.median(np.abs(values))) if values else None


def _max_abs_or_none(values: Sequence[float]) -> float | None:
    return float(np.max(np.abs(values))) if values else None


def _median_abs_or_zero(values: Sequence[float]) -> float:
    return float(np.median(np.abs(values))) if values else 0.0


def _max_or_zero(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _min_or_zero(values: Sequence[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _per_frame_cpe_summary(
    reports: Sequence[dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not reports:
        return {
            "enabled": bool(enabled),
            "source": "decision-directed",
            "frames": 0,
            "median_abs_cpe_rad": None,
            "max_abs_adjacent_cpe_step_rad": None,
            "median_reliable_symbol_count": None,
        }
    cpe = np.asarray([float(report["cpe_rad"]) for report in reports], dtype=np.float64)
    unwrapped = np.unwrap(cpe)
    adjacent = np.diff(unwrapped)
    reliable = [
        float(report.get("reliable_symbol_count", 0.0))
        for report in reports
    ]
    return {
        "enabled": bool(enabled),
        "source": "decision-directed",
        "frames": int(len(reports)),
        "median_abs_cpe_rad": float(np.median(np.abs(cpe))),
        "max_abs_adjacent_cpe_step_rad": float(np.max(np.abs(adjacent)))
        if adjacent.size
        else 0.0,
        "median_reliable_symbol_count": float(np.median(reliable)) if reliable else None,
    }


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
