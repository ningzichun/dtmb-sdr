"""Early DTMB equalization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .frequency import (
    DATA_SYMBOLS_PER_FRAME,
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_POSITIONS,
    frequency_interleave_index_map,
)
from .qam import (
    QAM_DEFINITIONS,
    QamMode,
    normalize_qam_symbols,
    qam_nearest,
    qam_symbol_quality,
)
from .system_info import system_info_symbols


QAM64_LEVELS = np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)


@dataclass(frozen=True)
class SparsePilotEqualization:
    """Result of sparse system-information pilot equalization."""

    equalized: np.ndarray
    channel: np.ndarray
    pilot_error_rms: float
    pilot_fit_error_rms: float
    data_evm_rms: float
    data_power: float
    decision_directed_carriers: int = 0
    decision_directed_error_rms: float | None = None
    data_qam_quality: dict[str, Any] | None = None
    decision_directed_bootstrap_qam_quality: dict[str, Any] | None = None
    decision_directed_reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        report = {
            "pilot_error_rms": self.pilot_error_rms,
            "pilot_fit_error_rms": self.pilot_fit_error_rms,
            "data_evm_rms": self.data_evm_rms,
            "data_power": self.data_power,
        }
        if self.decision_directed_carriers:
            report["decision_directed_carriers"] = self.decision_directed_carriers
        if self.decision_directed_error_rms is not None:
            report["decision_directed_error_rms"] = self.decision_directed_error_rms
        if self.data_qam_quality is not None:
            report["data_qam_quality"] = self.data_qam_quality
        if self.decision_directed_bootstrap_qam_quality is not None:
            report["decision_directed_bootstrap_qam_quality"] = (
                self.decision_directed_bootstrap_qam_quality
            )
        if self.decision_directed_reject_reason is not None:
            report["decision_directed_reject_reason"] = (
                self.decision_directed_reject_reason
            )
        return report


def _resolve_system_info_positions(
    positions: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if positions is None:
        return SYSTEM_INFO_POSITIONS
    values = np.asarray(positions, dtype=np.int64).reshape(-1)
    if values.size != SYSTEM_INFO_POSITIONS.size:
        raise ValueError("system_info_positions must contain 36 positions")
    if np.any(values < 0) or np.any(values >= FRAME_BODY_SYMBOLS):
        raise ValueError("system_info_positions contains out-of-range positions")
    if np.unique(values).size != values.size:
        raise ValueError("system_info_positions must be unique")
    return values.astype(np.int32)


def _resolve_logical_to_spectrum_positions(
    positions: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if positions is None:
        return frequency_interleave_index_map()
    values = np.asarray(positions, dtype=np.int64).reshape(-1)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("logical_to_spectrum_positions must contain 3780 positions")
    if np.any(values < 0) or np.any(values >= FRAME_BODY_SYMBOLS):
        raise ValueError("logical_to_spectrum_positions contains out-of-range positions")
    if np.unique(values).size != values.size:
        raise ValueError("logical_to_spectrum_positions must be unique")
    return values.astype(np.int32)


def equalize_with_system_info_pilots(
    inserted_spectrum: np.ndarray,
    *,
    system_info_index: int = 23,
    frame_body_mode: str = "C3780",
    qam_mode: QamMode = "64qam",
) -> SparsePilotEqualization:
    """Equalize one C=3780 frame with interpolated system-information pilots."""

    spectrum = np.asarray(inserted_spectrum, dtype=np.complex64)
    if spectrum.size != FRAME_BODY_SYMBOLS:
        raise ValueError("inserted_spectrum requires 3780 symbols")
    reference = system_info_symbols(
        _system_info_bits(system_info_index),
        frame_body_mode=frame_body_mode,
    )
    observed = spectrum[SYSTEM_INFO_POSITIONS]
    pilot_channel = observed / np.maximum(np.abs(reference), 1e-30) * np.exp(
        -1j * np.angle(reference)
    )
    channel = _interpolate_complex_channel(
        SYSTEM_INFO_POSITIONS.astype(np.float64),
        pilot_channel,
        FRAME_BODY_SYMBOLS,
    )
    equalized = spectrum / np.where(np.abs(channel) > 1e-12, channel, 1.0)
    pilot_error = equalized[SYSTEM_INFO_POSITIONS] - reference
    holdout_pilot_error = _leave_one_out_pilot_error_rms(
        SYSTEM_INFO_POSITIONS.astype(np.float64),
        pilot_channel,
        observed,
        reference,
        FRAME_BODY_SYMBOLS,
    )
    data = np.delete(equalized, SYSTEM_INFO_POSITIONS)
    data_evm = _qam_evm(data, qam_mode=qam_mode)
    return SparsePilotEqualization(
        equalized=equalized.astype(np.complex64, copy=False),
        channel=channel.astype(np.complex64, copy=False),
        pilot_error_rms=holdout_pilot_error,
        pilot_fit_error_rms=float(np.sqrt(np.mean(np.abs(pilot_error) ** 2))),
        data_evm_rms=data_evm,
        data_power=float(np.mean(np.abs(data) ** 2)) if data.size else 0.0,
        data_qam_quality=_data_qam_quality(data, qam_mode=qam_mode),
    )


def equalize_c3780_spectrum_with_system_info_pilots(
    spectrum: np.ndarray,
    *,
    system_info_index: int = 23,
    frame_body_mode: str = "C3780",
    qam_mode: QamMode = "64qam",
    system_info_positions: Sequence[int] | np.ndarray | None = None,
    logical_to_spectrum_positions: Sequence[int] | np.ndarray | None = None,
) -> SparsePilotEqualization:
    """Equalize one physical C=3780 FFT spectrum with system-information pilots."""

    physical = np.asarray(spectrum, dtype=np.complex64)
    if physical.size != FRAME_BODY_SYMBOLS:
        raise ValueError("spectrum requires 3780 physical FFT bins")
    reference = system_info_symbols(
        _system_info_bits(system_info_index),
        frame_body_mode=frame_body_mode,
    )
    pilot_logical_positions = _resolve_system_info_positions(system_info_positions)
    logical_to_spectrum = _resolve_logical_to_spectrum_positions(
        logical_to_spectrum_positions
    )
    pilot_positions = logical_to_spectrum[pilot_logical_positions]
    observed = physical[pilot_positions]
    pilot_channel = observed / reference
    channel = _interpolate_complex_channel(
        pilot_positions.astype(np.float64),
        pilot_channel,
        FRAME_BODY_SYMBOLS,
    )
    equalized_spectrum = physical / np.where(np.abs(channel) > 1e-12, channel, 1.0)
    equalized_inserted = equalized_spectrum[logical_to_spectrum]
    pilot_error = equalized_inserted[pilot_logical_positions] - reference
    holdout_pilot_error = _leave_one_out_pilot_error_rms(
        pilot_positions.astype(np.float64),
        pilot_channel,
        observed,
        reference,
        FRAME_BODY_SYMBOLS,
    )
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[pilot_logical_positions] = False
    data = equalized_inserted[mask]
    data_evm = _qam_evm(data, qam_mode=qam_mode)
    return SparsePilotEqualization(
        equalized=equalized_spectrum.astype(np.complex64, copy=False),
        channel=channel.astype(np.complex64, copy=False),
        pilot_error_rms=holdout_pilot_error,
        pilot_fit_error_rms=float(np.sqrt(np.mean(np.abs(pilot_error) ** 2))),
        data_evm_rms=data_evm,
        data_power=float(np.mean(np.abs(data) ** 2)) if data.size else 0.0,
        data_qam_quality=_data_qam_quality(data, qam_mode=qam_mode),
    )


def refine_c3780_spectrum_decision_directed(
    spectrum: np.ndarray,
    *,
    system_info_index: int = 23,
    frame_body_mode: str = "C3780",
    qam_mode: QamMode = "64qam",
    max_relative_error: float = 0.55,
    min_reliable_carriers: int = DATA_SYMBOLS_PER_FRAME // 2,
    min_data_power_ratio: float = 0.0,
    max_axis_inner_fraction: float = 0.60,
    max_hard_bit_bias: float = 0.0,
    bootstrap: str = "sparse",
    include_system_info_pilots: bool = True,
    system_info_positions: Sequence[int] | np.ndarray | None = None,
    logical_to_spectrum_positions: Sequence[int] | np.ndarray | None = None,
) -> SparsePilotEqualization:
    """Refine C=3780 equalization using reliable sliced data carriers."""

    if max_relative_error <= 0.0:
        raise ValueError("max_relative_error must be positive")
    if min_reliable_carriers < 0:
        raise ValueError("min_reliable_carriers must be non-negative")
    if min_data_power_ratio < 0.0:
        raise ValueError("min_data_power_ratio must be non-negative")
    if max_axis_inner_fraction < 0.0:
        raise ValueError("max_axis_inner_fraction must be non-negative")
    if max_hard_bit_bias < 0.0:
        raise ValueError("max_hard_bit_bias must be non-negative")
    pilot_logical_positions = _resolve_system_info_positions(system_info_positions)
    logical_to_spectrum = _resolve_logical_to_spectrum_positions(
        logical_to_spectrum_positions
    )
    bootstrap = str(bootstrap)
    if bootstrap == "sparse":
        coarse = equalize_c3780_spectrum_with_system_info_pilots(
            spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=qam_mode,
            system_info_positions=pilot_logical_positions,
            logical_to_spectrum_positions=logical_to_spectrum,
        )
    elif bootstrap == "raw":
        coarse = _raw_c3780_seed_equalization(
            spectrum,
            system_info_index=system_info_index,
            frame_body_mode=frame_body_mode,
            qam_mode=qam_mode,
            system_info_positions=pilot_logical_positions,
            logical_to_spectrum_positions=logical_to_spectrum,
        )
    else:
        raise ValueError("bootstrap must be 'sparse' or 'raw'")
    equalized_inserted = coarse.equalized[logical_to_spectrum]
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[pilot_logical_positions] = False
    data = equalized_inserted[mask]
    coarse = replace(
        coarse,
        data_qam_quality=_data_qam_quality(data, qam_mode=qam_mode),
    )
    bootstrap_qam_quality = coarse.data_qam_quality
    # Real DTMB captures can carry QAM data at a different absolute scale from
    # system-information pilots. The QAM slicer normalizes that scale, so this
    # guard is opt-in only and should not be used as the default DD gate.
    expected_data_power = float(QAM_DEFINITIONS[qam_mode].average_power)
    if (
        min_data_power_ratio > 0.0
        and expected_data_power > 0.0
        and coarse.data_power < expected_data_power * float(min_data_power_ratio)
    ):
        return coarse
    normalized = normalize_qam_symbols(data, mode=qam_mode)
    nearest = qam_nearest(normalized, mode=qam_mode)
    nearest_power = float(np.mean(np.abs(nearest) ** 2)) if nearest.size else 0.0
    if nearest_power <= 0.0:
        return coarse
    inner_fraction = _rectangular_axis_inner_fraction(nearest, qam_mode=qam_mode)
    if (
        inner_fraction is not None
        and max_axis_inner_fraction > 0.0
        and inner_fraction > float(max_axis_inner_fraction)
    ):
        return coarse
    hard_bit_bias = _max_abs_hard_bit_bias(coarse.data_qam_quality)
    if (
        hard_bit_bias is not None
        and max_hard_bit_bias > 0.0
        and hard_bit_bias > float(max_hard_bit_bias)
    ):
        return replace(
            coarse,
            decision_directed_reject_reason="hard_bit_balance",
        )
    relative_error = np.abs(normalized - nearest) / np.sqrt(nearest_power)
    reliable_data = relative_error <= max_relative_error
    reliable_count = int(np.count_nonzero(reliable_data))
    if reliable_count < int(min_reliable_carriers):
        return coarse

    logical_data_positions = np.flatnonzero(mask)
    data_positions = logical_to_spectrum[logical_data_positions[reliable_data]]
    data_reference = nearest[reliable_data]
    data_observed = np.asarray(spectrum, dtype=np.complex64)[data_positions]
    reference = system_info_symbols(
        _system_info_bits(system_info_index),
        frame_body_mode=frame_body_mode,
    )
    pilot_positions = logical_to_spectrum[pilot_logical_positions]
    if include_system_info_pilots:
        positions = np.concatenate((pilot_positions, data_positions)).astype(np.float64)
        channels = np.concatenate(
            (
                np.asarray(spectrum, dtype=np.complex64)[pilot_positions] / reference,
                data_observed / data_reference,
            )
        )
    else:
        positions = data_positions.astype(np.float64, copy=False)
        channels = data_observed / data_reference
    channel = _interpolate_complex_channel(positions, channels, FRAME_BODY_SYMBOLS)
    physical = np.asarray(spectrum, dtype=np.complex64)
    equalized_spectrum = physical / np.where(np.abs(channel) > 1e-12, channel, 1.0)
    refined_inserted = equalized_spectrum[logical_to_spectrum]
    pilot_error = refined_inserted[pilot_logical_positions] - reference
    refined_data = refined_inserted[mask]
    return SparsePilotEqualization(
        equalized=equalized_spectrum.astype(np.complex64, copy=False),
        channel=channel.astype(np.complex64, copy=False),
        pilot_error_rms=coarse.pilot_error_rms,
        pilot_fit_error_rms=float(np.sqrt(np.mean(np.abs(pilot_error) ** 2))),
        data_evm_rms=_qam_evm(refined_data, qam_mode=qam_mode),
        data_power=float(np.mean(np.abs(refined_data) ** 2)) if refined_data.size else 0.0,
        decision_directed_carriers=reliable_count,
        decision_directed_error_rms=float(np.sqrt(np.mean(relative_error[reliable_data] ** 2))),
        data_qam_quality=_data_qam_quality(refined_data, qam_mode=qam_mode),
        decision_directed_bootstrap_qam_quality=bootstrap_qam_quality,
    )


def qam64_nearest(symbols: np.ndarray) -> np.ndarray:
    """Slice symbols to the nearest rectangular 64QAM point."""

    values = np.asarray(symbols, dtype=np.complex64)
    real = QAM64_LEVELS[np.argmin(np.abs(values.real[:, None] - QAM64_LEVELS[None, :]), axis=1)]
    imag = QAM64_LEVELS[np.argmin(np.abs(values.imag[:, None] - QAM64_LEVELS[None, :]), axis=1)]
    return (real + 1j * imag).astype(np.complex64)


def _raw_c3780_seed_equalization(
    spectrum: np.ndarray,
    *,
    system_info_index: int = 23,
    frame_body_mode: str = "C3780",
    qam_mode: QamMode = "64qam",
    system_info_positions: Sequence[int] | np.ndarray | None = None,
    logical_to_spectrum_positions: Sequence[int] | np.ndarray | None = None,
) -> SparsePilotEqualization:
    physical = np.asarray(spectrum, dtype=np.complex64)
    if physical.size != FRAME_BODY_SYMBOLS:
        raise ValueError("spectrum requires 3780 physical FFT bins")
    reference = system_info_symbols(
        _system_info_bits(system_info_index),
        frame_body_mode=frame_body_mode,
    )
    pilot_logical_positions = _resolve_system_info_positions(system_info_positions)
    logical_to_spectrum = _resolve_logical_to_spectrum_positions(
        logical_to_spectrum_positions
    )
    inserted = physical[logical_to_spectrum]
    observed_pilots = inserted[pilot_logical_positions]
    denominator = float(np.vdot(reference, reference).real)
    gain = np.vdot(reference, observed_pilots) / max(denominator, 1e-12)
    pilot_equalized = observed_pilots / (gain if abs(gain) > 1e-12 else 1.0)
    pilot_error = pilot_equalized - reference
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[pilot_logical_positions] = False
    data = inserted[mask]
    return SparsePilotEqualization(
        equalized=physical.astype(np.complex64, copy=False),
        channel=np.ones(FRAME_BODY_SYMBOLS, dtype=np.complex64),
        pilot_error_rms=float(np.sqrt(np.mean(np.abs(pilot_error) ** 2))),
        pilot_fit_error_rms=float(np.sqrt(np.mean(np.abs(pilot_error) ** 2))),
        data_evm_rms=_qam_evm(data, qam_mode=qam_mode),
        data_power=float(np.mean(np.abs(data) ** 2)) if data.size else 0.0,
        data_qam_quality=_data_qam_quality(data, qam_mode=qam_mode),
    )


def _qam64_evm(symbols: np.ndarray) -> float:
    return _qam_evm(symbols, qam_mode="64qam")


def _qam_evm(symbols: np.ndarray, *, qam_mode: QamMode) -> float:
    data = np.asarray(symbols, dtype=np.complex64)
    if data.size != DATA_SYMBOLS_PER_FRAME:
        raise ValueError("QAM EVM expects 3744 data symbols")
    if np.mean(np.abs(data) ** 2) <= 0:
        return 0.0
    reference_power = float(QAM_DEFINITIONS[qam_mode].average_power)
    observed_power = float(np.mean(np.abs(data) ** 2))
    scale = np.sqrt(observed_power / reference_power)
    corrected = data / max(scale, 1e-12)
    nearest = qam_nearest(corrected, mode=qam_mode)
    # Refine the scale and residual common phase after the first slicing pass.
    gain = np.vdot(nearest, data) / max(float(np.vdot(nearest, nearest).real), 1e-12)
    corrected = data / (gain if abs(gain) > 1e-12 else 1.0)
    nearest = qam_nearest(corrected, mode=qam_mode)
    nearest_power = float(np.mean(np.abs(nearest) ** 2))
    if nearest_power <= 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(corrected - nearest) ** 2) / nearest_power))


def _data_qam_quality(symbols: np.ndarray, *, qam_mode: QamMode) -> dict[str, Any]:
    return qam_symbol_quality(
        symbols,
        mode=qam_mode,
        normalize=True,
    )


def _max_abs_hard_bit_bias(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    value = report.get("max_abs_hard_bit_bias")
    return None if value is None else float(value)


def _rectangular_axis_inner_fraction(
    symbols: np.ndarray,
    *,
    qam_mode: QamMode,
) -> float | None:
    definition = QAM_DEFINITIONS[qam_mode]
    if definition.levels is None or definition.levels.size < 4:
        return None
    levels = definition.levels.astype(np.float32, copy=False)
    inner = float(np.min(np.abs(levels)))
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if values.size == 0:
        return None
    i_inner = float(np.mean(np.abs(values.real) == inner))
    q_inner = float(np.mean(np.abs(values.imag) == inner))
    return max(i_inner, q_inner)


def _interpolate_complex_channel(
    positions: np.ndarray,
    values: np.ndarray,
    size: int,
) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    order = np.argsort(positions)
    positions = positions[order]
    values = values[order]
    extended_positions = np.concatenate(
        (
            [positions[-1] - size],
            positions,
            [positions[0] + size],
        )
    )
    extended_values = np.concatenate(([values[-1]], values, [values[0]]))
    real = np.interp(x, extended_positions, extended_values.real)
    imag = np.interp(x, extended_positions, extended_values.imag)
    return real + 1j * imag


def _leave_one_out_pilot_error_rms(
    positions: np.ndarray,
    pilot_channel: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
    size: int,
) -> float:
    """Score pilot consistency without fitting the pilot being scored."""

    if positions.size != pilot_channel.size:
        raise ValueError("pilot positions and channel values differ in size")
    if positions.size < 3:
        return 0.0
    errors = []
    for index in range(positions.size):
        train_positions = np.delete(positions, index)
        train_channel = np.delete(pilot_channel, index)
        predicted = _interpolate_complex_channel(
            train_positions,
            train_channel,
            size,
        )[int(positions[index])]
        equalized = observed[index] / (predicted if abs(predicted) > 1e-12 else 1.0)
        errors.append(equalized - reference[index])
    error = np.asarray(errors, dtype=np.complex64)
    return float(np.sqrt(np.mean(np.abs(error) ** 2)))


def _system_info_bits(index: int) -> str:
    from .system_info import SYSTEM_INFO_VECTORS

    try:
        return SYSTEM_INFO_VECTORS[index]
    except KeyError as exc:
        raise ValueError(f"unknown system-info index: {index}") from exc



@dataclass(frozen=True)
class PerFrameCpeCorrection:
    """Per-frame common-phase and gain correction from data-decision feedback.

    ``gain`` is the complex scalar that was applied as ``symbols / gain`` to the
    input. ``cpe_rad`` is ``angle(gain)``, the estimated common phase. A nominal
    result with zero residual CPE has ``cpe_rad == 0.0``.
    """

    gain: complex
    cpe_rad: float
    amplitude: float
    corrected_symbols: np.ndarray
    reliable_symbol_count: int
    pre_correction_power: float
    post_correction_power: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain_real": float(self.gain.real),
            "gain_imag": float(self.gain.imag),
            "cpe_rad": float(self.cpe_rad),
            "amplitude": float(self.amplitude),
            "reliable_symbol_count": int(self.reliable_symbol_count),
            "pre_correction_power": float(self.pre_correction_power),
            "post_correction_power": float(self.post_correction_power),
        }


def correct_common_phase_error(
    symbols: np.ndarray,
    *,
    qam_mode: QamMode,
    max_relative_error: float = 0.7,
    min_reliable_symbols: int = 32,
    max_iterations: int = 4,
    pilot_symbols: np.ndarray | None = None,
    pilot_reference: np.ndarray | None = None,
) -> PerFrameCpeCorrection:
    """Estimate and remove one common-phase-error and gain offset per frame.

    When ``pilot_symbols`` and ``pilot_reference`` are provided (for example
    from the C=3780 system-information pilot positions) the gain is computed
    as ``g = <pilot_reference, pilot_symbols> / <pilot_reference, pilot_reference>``.
    Pilots resolve the rectangular-QAM constellation's pi/2 rotational
    ambiguity that a blind decision-directed estimator cannot handle, and
    they are already extracted during frame-body processing, so using them
    is cheaper than another blind estimate.

    Without pilots the estimator starts from a blind fourth-power phase
    estimate (which works on any rectangular QAM because rotating by theta
    pushes ``E[x^4]`` by ``exp(j*4*theta)``) and then iteratively slices to
    the nearest QAM point and refits the gain. The fourth-power estimate is
    only unique modulo pi/2, so the decision-directed step may lock to the
    wrong rotational branch when pilots are unavailable and the input has
    been rotated by more than pi/4.

    The DTMB per-frame CPE jitter observed on ``wall_522`` real captures
    was north of one radian between adjacent frames even after coarse CFO
    and family-delay correction, which is enough to flip LDPC soft-bit
    signs on every other frame. Removing the CPE before the convolutional
    deinterleaver stitches frames together is what lets the deinterleaver
    output carry consistent LLR signs.
    """

    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    pre_power = float(np.mean(np.abs(values) ** 2)) if values.size else 0.0
    if values.size == 0 or pre_power <= 1e-24:
        return PerFrameCpeCorrection(
            gain=complex(1.0, 0.0),
            cpe_rad=0.0,
            amplitude=1.0,
            corrected_symbols=values.astype(np.complex64, copy=False),
            reliable_symbol_count=0,
            pre_correction_power=pre_power,
            post_correction_power=pre_power,
        )
    reference_power = float(QAM_DEFINITIONS[qam_mode].average_power)
    if reference_power <= 0.0:
        return PerFrameCpeCorrection(
            gain=complex(1.0, 0.0),
            cpe_rad=0.0,
            amplitude=1.0,
            corrected_symbols=values.astype(np.complex64, copy=False),
            reliable_symbol_count=0,
            pre_correction_power=pre_power,
            post_correction_power=pre_power,
        )

    if pilot_symbols is not None and pilot_reference is not None:
        observed_pilots = np.asarray(pilot_symbols, dtype=np.complex64).reshape(-1)
        reference_pilots = np.asarray(pilot_reference, dtype=np.complex64).reshape(-1)
        if observed_pilots.size != reference_pilots.size or observed_pilots.size == 0:
            raise ValueError(
                "pilot_symbols and pilot_reference must share the same non-empty size"
            )
        denominator = float(np.vdot(reference_pilots, reference_pilots).real)
        if denominator > 1e-12:
            base_gain = complex(
                np.vdot(reference_pilots, observed_pilots) / denominator
            )
            # DTMB system-information rows are stored in complement pairs
            # (for example 23/24 are 64QAM rate-3 mode1/mode2). If the
            # top-ranked classifier lands on the complementary vector, the
            # pilot fit flips sign. Try both polarities and keep the one that
            # leaves the frame's data symbols closer to the rectangular grid.
            candidate_gains = []
            if abs(base_gain) > 1e-12:
                candidate_gains.append(base_gain)
            if abs(-base_gain) > 1e-12:
                candidate_gains.append(-base_gain)
            if candidate_gains:
                best_gain = candidate_gains[0]
                best_score = -1.0
                reference_qam_power = float(
                    QAM_DEFINITIONS[qam_mode].average_power
                )
                for candidate in candidate_gains:
                    corrected_try = values / candidate
                    if reference_qam_power > 0.0:
                        nearest = qam_nearest(corrected_try, mode=qam_mode)
                        nearest_power = float(np.mean(np.abs(nearest) ** 2))
                        if nearest_power > 1e-24:
                            relative = (
                                np.abs(corrected_try - nearest)
                                / np.sqrt(nearest_power)
                            )
                            score = float(
                                np.mean(relative <= max_relative_error)
                            )
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                    if score > best_score:
                        best_score = score
                        best_gain = candidate
                gain = best_gain
                corrected = values / gain
                post_power = float(np.mean(np.abs(corrected) ** 2))
                return PerFrameCpeCorrection(
                    gain=gain,
                    cpe_rad=float(np.angle(gain)),
                    amplitude=float(abs(gain)),
                    corrected_symbols=corrected.astype(np.complex64, copy=False),
                    reliable_symbol_count=int(observed_pilots.size),
                    pre_correction_power=pre_power,
                    post_correction_power=post_power,
                )

    # Fourth-power blind phase estimate. For DTMB's rectangular QAM sets
    # (4QAM, 16QAM, 64QAM) the reference constellation's fourth moment is
    # a negative real number; rotating the constellation by ``theta`` pushes
    # the fourth moment to ``-|ref| * exp(j*4*theta)``. Cancelling the
    # built-in pi phase and dividing by four recovers theta modulo pi/2.
    # We wrap the result into ``[-pi/4, pi/4]`` so the decision-directed
    # refinement always starts on the right side of the rectangular
    # constellation's pi/2 ambiguity.
    quartic = np.mean((values.astype(np.complex128) / np.sqrt(pre_power)) ** 4)
    if np.abs(quartic) > 1e-6:
        raw_phase = float(np.angle(-quartic) / 4.0)
    else:
        raw_phase = 0.0
    blind_phase = raw_phase
    while blind_phase > np.pi / 4:
        blind_phase -= np.pi / 2
    while blind_phase < -np.pi / 4:
        blind_phase += np.pi / 2
    amplitude = np.sqrt(pre_power / reference_power)
    gain = complex(amplitude * np.cos(blind_phase), amplitude * np.sin(blind_phase))

    reliable_count = 0
    for _ in range(max(1, int(max_iterations))):
        corrected = values / gain if abs(gain) > 1e-12 else values
        nearest = qam_nearest(corrected, mode=qam_mode)
        nearest_power = float(np.mean(np.abs(nearest) ** 2))
        if nearest_power <= 1e-24:
            break
        relative_error = np.abs(corrected - nearest) / np.sqrt(nearest_power)
        reliable = relative_error <= max_relative_error
        reliable_count = int(np.count_nonzero(reliable))
        if reliable_count < max(min_reliable_symbols, int(values.size * 0.1)):
            reliable = np.ones_like(reliable)
            reliable_count = int(values.size)
        reliable_ref = nearest[reliable]
        reliable_obs = values[reliable]
        denominator = float(np.vdot(reliable_ref, reliable_ref).real)
        if denominator <= 1e-12:
            break
        new_gain = complex(np.vdot(reliable_ref, reliable_obs) / denominator)
        if abs(new_gain) <= 1e-12:
            break
        if abs(new_gain - gain) / max(abs(gain), 1e-12) < 1e-4:
            gain = new_gain
            break
        gain = new_gain
    corrected = (
        values / gain if abs(gain) > 1e-12 else values.astype(np.complex64, copy=False)
    )
    # Guard against the blind/decision-directed branch LOCKING TO THE WRONG
    # pi/2 rotation and *worsening* a frame. On real 64QAM captures the
    # fourth-power estimate can mis-lock on weak frames, inflating their grid
    # EVM far above the uncorrected frame (observed: worst-frame grid EVM rose
    # from ~0.45 to ~1.4 with blind DD CPE). Because mode2's 510-frame symbol
    # deinterleaver mixes every frame into every LDPC codeword, a single
    # mis-locked frame poisons many codewords. Compare the blind correction's
    # grid-relative error against an amplitude-only (phase-0) correction and
    # keep whichever is closer to the rectangular grid; the amplitude-only
    # fallback can never rotate a frame onto the wrong constellation branch.
    amplitude_only_gain = complex(np.sqrt(pre_power / reference_power), 0.0)
    if abs(amplitude_only_gain) > 1e-12:
        fallback = values / amplitude_only_gain
        nearest_dd = qam_nearest(corrected, mode=qam_mode)
        nearest_fb = qam_nearest(fallback, mode=qam_mode)
        dd_err = float(np.mean(np.abs(corrected - nearest_dd) ** 2))
        fb_err = float(np.mean(np.abs(fallback - nearest_fb) ** 2))
        if fb_err < dd_err:
            gain = amplitude_only_gain
            corrected = fallback
            reliable_count = 0
    post_power = float(np.mean(np.abs(corrected) ** 2))
    return PerFrameCpeCorrection(
        gain=gain,
        cpe_rad=float(np.angle(gain)),
        amplitude=float(abs(gain)),
        corrected_symbols=corrected.astype(np.complex64, copy=False),
        reliable_symbol_count=reliable_count,
        pre_correction_power=pre_power,
        post_correction_power=post_power,
    )
