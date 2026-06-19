import numpy as np
import pytest

from dtmb.carrier_axes import system_info_positions_for_extraction
from dtmb.equalization import (
    _qam64_evm,
    equalize_c3780_spectrum_with_system_info_pilots,
    equalize_with_system_info_pilots,
    refine_c3780_spectrum_decision_directed,
)
from dtmb.frequency import (
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_POSITIONS,
    frequency_deinterleave_index_map,
    frequency_deinterleave_inserted,
    frequency_interleave,
    frequency_interleave_index_map,
)
from dtmb.qam import qam_modulate
from dtmb.system_info import SYSTEM_INFO_VECTORS, system_info_symbols


def test_equalize_with_system_info_pilots_restores_known_pilots():
    spectrum = np.zeros(3780, dtype=np.complex64)
    reference = system_info_symbols(
        "00010001010111100100011101000011",
        frame_body_mode="C3780",
    )
    channel = 2 - 0.5j
    spectrum[SYSTEM_INFO_POSITIONS] = reference * channel

    result = equalize_with_system_info_pilots(spectrum, system_info_index=23)

    np.testing.assert_allclose(result.equalized[SYSTEM_INFO_POSITIONS], reference, atol=1e-5)
    assert result.pilot_error_rms < 1e-5


def test_equalize_with_system_info_pilots_accepts_full_system_info_table():
    spectrum = np.zeros(3780, dtype=np.complex64)
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[24],
        frame_body_mode="C3780",
    )
    spectrum[SYSTEM_INFO_POSITIONS] = reference * (0.8 + 1.3j)

    result = equalize_with_system_info_pilots(spectrum, system_info_index=24)

    np.testing.assert_allclose(result.equalized[SYSTEM_INFO_POSITIONS], reference, atol=1e-5)
    assert result.pilot_error_rms < 1e-5
    assert result.pilot_fit_error_rms < 1e-5


def test_system_info_pilot_error_is_not_in_sample_only():
    spectrum = np.zeros(3780, dtype=np.complex64)
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    )
    x = np.arange(3780, dtype=np.float32)
    channel = 1.0 + 0.1 * np.sin(2 * np.pi * x / 3780) + 0.2j
    spectrum[SYSTEM_INFO_POSITIONS] = reference * channel[SYSTEM_INFO_POSITIONS]

    correct = equalize_with_system_info_pilots(spectrum, system_info_index=23)
    wrong = equalize_with_system_info_pilots(spectrum, system_info_index=24)

    assert correct.pilot_error_rms < 0.01
    assert wrong.pilot_error_rms > correct.pilot_error_rms + 0.5


def test_c3780_sparse_equalizer_interpolates_in_physical_fft_order():
    rng = np.random.default_rng(20260504)
    logical = np.empty(3780, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="64qam")
    spectrum = frequency_interleave(logical)
    x = np.arange(3780, dtype=np.float32)
    channel = 1.0 + 0.18 * np.sin(2 * np.pi * x / 3780) + 0.12j * np.cos(
        4 * np.pi * x / 3780
    )
    observed = spectrum * channel.astype(np.complex64)

    result = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=23,
    )
    restored = frequency_deinterleave_inserted(result.equalized)

    np.testing.assert_allclose(restored[SYSTEM_INFO_POSITIONS], logical[:36], atol=1e-5)
    assert result.data_evm_rms < 0.15


def test_c3780_sparse_equalizer_honors_permuted_system_info_positions():
    rng = np.random.default_rng(20260528)
    si_positions = np.asarray(
        system_info_positions_for_extraction("rev_i+rev_j"),
        dtype=np.int32,
    )
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    )
    inserted = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    inserted[si_positions] = reference
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[si_positions] = False
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    inserted[mask] = qam_modulate(bits, mode="64qam")
    spectrum = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    spectrum[frequency_interleave_index_map()] = inserted
    channel = np.complex64(0.6 - 1.1j)
    observed = spectrum * channel

    result = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=23,
        system_info_positions=si_positions,
    )
    restored = frequency_deinterleave_inserted(result.equalized)

    np.testing.assert_allclose(restored[si_positions], reference, atol=1e-5)
    assert result.data_evm_rms < 1e-6


def test_c3780_sparse_equalizer_honors_inverse_direction_shifted_positions():
    rng = np.random.default_rng(20260528)
    si_positions = np.asarray(
        system_info_positions_for_extraction("rev_i+rev_j", 7),
        dtype=np.int32,
    )
    logical_to_spectrum = frequency_deinterleave_index_map()
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    )
    inserted = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    inserted[si_positions] = reference
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[si_positions] = False
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    inserted[mask] = qam_modulate(bits, mode="64qam")
    spectrum = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    spectrum[logical_to_spectrum] = inserted
    channel = np.complex64(-0.4 + 0.7j)
    observed = spectrum * channel

    result = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=23,
        system_info_positions=si_positions,
        logical_to_spectrum_positions=logical_to_spectrum,
    )
    restored = result.equalized[logical_to_spectrum]

    np.testing.assert_allclose(restored[si_positions], reference, atol=1e-5)
    assert result.data_evm_rms < 1e-6


def test_c3780_sparse_equalizer_reports_requested_qam_mode_quality():
    rng = np.random.default_rng(20260528)
    logical = np.empty(3780, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[14],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 4, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="16qam")
    observed = frequency_interleave(logical) * np.complex64(0.75 - 1.2j)

    result = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=14,
        qam_mode="16qam",
    )

    assert result.data_evm_rms < 1e-6
    assert result.data_qam_quality["grid_evm_rms"] < 1e-6
    assert len(result.data_qam_quality["axis_level_occupancy"]["levels"]) == 4


def test_decision_directed_equalizer_adds_full_band_channel_samples():
    rng = np.random.default_rng(20260507)
    logical = np.empty(3780, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[24],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="64qam")
    spectrum = frequency_interleave(logical)
    x = np.arange(3780, dtype=np.float32)
    channel = 1.0 + 0.25 * np.sin(6 * np.pi * x / 3780 + 0.2) + 0.25j * np.cos(
        4 * np.pi * x / 3780 - 0.5
    )
    observed = spectrum * channel.astype(np.complex64)

    sparse = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=24,
    )
    refined = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
    )

    assert refined.decision_directed_carriers > 3000
    assert refined.data_evm_rms < sparse.data_evm_rms * 0.5
    assert refined.to_dict()["decision_directed_carriers"] == refined.decision_directed_carriers


def test_decision_directed_equalizer_sparse_bootstrap_is_default():
    rng = np.random.default_rng(20260523)
    logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[24],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="64qam")
    spectrum = frequency_interleave(logical)
    x = np.arange(FRAME_BODY_SYMBOLS, dtype=np.float32)
    channel = 1.0 + 0.12 * np.sin(2 * np.pi * x / FRAME_BODY_SYMBOLS)
    observed = spectrum * channel.astype(np.complex64)

    default = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
    )
    explicit = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
        bootstrap="sparse",
        include_system_info_pilots=True,
    )

    np.testing.assert_allclose(default.equalized, explicit.equalized, atol=0.0)
    np.testing.assert_allclose(default.channel, explicit.channel, atol=0.0)
    assert default.to_dict() == explicit.to_dict()


def test_raw_decision_directed_bootstrap_ignores_misleading_edge_pilots():
    rng = np.random.default_rng(20260524)
    logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[22],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="64qam")
    spectrum = frequency_interleave(logical)
    observed = spectrum.copy()
    pilot_positions = frequency_interleave_index_map()[SYSTEM_INFO_POSITIONS]
    observed[pilot_positions[:18]] *= np.complex64(0.25 + 0.65j)
    observed[pilot_positions[18:]] *= np.complex64(3.1 - 0.75j)

    sparse = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=22,
    )
    raw = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=22,
        qam_mode="64qam",
        bootstrap="raw",
        include_system_info_pilots=False,
    )
    restored = frequency_deinterleave_inserted(raw.equalized)
    restored_data = np.delete(restored, SYSTEM_INFO_POSITIONS)

    assert sparse.data_evm_rms > 0.25
    assert raw.decision_directed_carriers > 3700
    assert raw.data_evm_rms < 1e-6
    np.testing.assert_allclose(restored_data, logical[36:], atol=1e-6)


def test_decision_directed_equalizer_reports_biased_qam_shape_despite_low_evm():
    rng = np.random.default_rng(20260525)
    logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[22],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = qam_modulate(bits, mode="64qam")
    observed = _distort_physical_o_edge_data_carriers(frequency_interleave(logical))

    refined = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=22,
        qam_mode="64qam",
    )
    report = refined.to_dict()
    quality = report["data_qam_quality"]
    bootstrap_quality = report["decision_directed_bootstrap_qam_quality"]

    assert refined.decision_directed_carriers > 3700
    assert refined.data_evm_rms < 1e-5
    assert bootstrap_quality["grid_evm_rms"] > quality["grid_evm_rms"]
    assert bootstrap_quality["max_abs_hard_bit_bias"] > 0.06
    assert quality["grid_evm_rms"] < 1e-5
    assert quality["max_abs_hard_bit_bias"] > 0.06
    assert (
        quality["axis_level_occupancy"]["max_abs_fraction_error_from_uniform"]
        > 0.035
    )

    guarded = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=22,
        qam_mode="64qam",
        max_hard_bit_bias=0.06,
    )

    assert guarded.decision_directed_carriers == 0
    assert guarded.decision_directed_reject_reason == "hard_bit_balance"
    assert guarded.data_evm_rms > refined.data_evm_rms


def test_decision_directed_equalizer_rejects_invalid_bootstrap():
    spectrum = np.zeros(FRAME_BODY_SYMBOLS, dtype=np.complex64)

    with pytest.raises(ValueError, match="bootstrap"):
        refine_c3780_spectrum_decision_directed(
            spectrum,
            system_info_index=24,
            qam_mode="64qam",
            bootstrap="pilots",
        )


def test_decision_directed_equalizer_accepts_low_power_data_relative_to_pilots():
    rng = np.random.default_rng(20260521)
    logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[24],
        frame_body_mode="C3780",
    )
    bits = rng.integers(0, 2, 3744 * 6, dtype=np.uint8)
    logical[36:] = 0.14 * qam_modulate(bits, mode="64qam")
    spectrum = frequency_interleave(logical)
    x = np.arange(FRAME_BODY_SYMBOLS, dtype=np.float32)
    channel = 1.0 + 0.25 * np.sin(6 * np.pi * x / 3780 + 0.2) + 0.25j * np.cos(
        4 * np.pi * x / 3780 - 0.5
    )
    observed = spectrum * channel.astype(np.complex64)

    sparse = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=24,
    )
    refined = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
    )

    assert sparse.data_power < 1.0
    assert refined.decision_directed_carriers > 3000
    assert refined.data_evm_rms < sparse.data_evm_rms * 0.5


def test_decision_directed_equalizer_rejects_zero_fill_dominated_frames():
    rng = np.random.default_rng(20260522)
    logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    logical[:36] = system_info_symbols(
        SYSTEM_INFO_VECTORS[24],
        frame_body_mode="C3780",
    )
    data = np.zeros(3744, dtype=np.complex64)
    bits = rng.integers(0, 2, 1000 * 6, dtype=np.uint8)
    data[-1000:] = qam_modulate(bits, mode="64qam")
    logical[36:] = data
    x = np.arange(FRAME_BODY_SYMBOLS, dtype=np.float32)
    channel = 1.0 + 0.1 * np.sin(2 * np.pi * x / 3780)
    observed = frequency_interleave(logical) * channel.astype(np.complex64)

    sparse = equalize_c3780_spectrum_with_system_info_pilots(
        observed,
        system_info_index=24,
    )
    refined = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
    )
    unchecked = refine_c3780_spectrum_decision_directed(
        observed,
        system_info_index=24,
        qam_mode="64qam",
        min_reliable_carriers=0,
        max_axis_inner_fraction=0.0,
    )

    assert refined.decision_directed_carriers == 0
    assert refined.data_evm_rms == sparse.data_evm_rms
    assert unchecked.decision_directed_carriers > 3000


def test_qam64_evm_is_low_for_scaled_ideal_constellation():
    levels = np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    repeated = np.resize(
        (levels[:, None] + 1j * levels[None, :]).reshape(-1),
        3744,
    ).astype(np.complex64)

    assert _qam64_evm((1.7 - 0.2j) * repeated) < 1e-6


def _distort_physical_o_edge_data_carriers(spectrum: np.ndarray) -> np.ndarray:
    distorted = np.asarray(spectrum, dtype=np.complex64).copy()
    mask = np.ones(FRAME_BODY_SYMBOLS, dtype=bool)
    mask[SYSTEM_INFO_POSITIONS] = False
    data_inserted_positions = np.flatnonzero(mask)
    physical_positions = frequency_interleave_index_map()[data_inserted_positions]
    physical_o = (physical_positions // 540) % 7
    selected = (physical_o == 0) | (physical_o == 6)
    selected_positions = physical_positions[selected]
    values = distorted[selected_positions]
    distorted[selected_positions] = (
        _nudge_outer_axis_values_inward(values.real)
        + 1j * _nudge_outer_axis_values_inward(values.imag)
    ).astype(np.complex64)
    return distorted


def _nudge_outer_axis_values_inward(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float32).copy()
    replacements = {
        3.0: 1.5,
        5.0: 3.5,
        7.0: 5.5,
    }
    for source, target in replacements.items():
        selected = np.isclose(np.abs(values), source)
        values[selected] = np.sign(values[selected]) * np.float32(target)
    return values
