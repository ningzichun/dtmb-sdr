import numpy as np

from dtmb.qam import (
    QAM_DEFINITIONS,
    constellation_points,
    qam_hard_demodulate,
    qam_modulate,
    qam_soft_demodulate,
    qam_symbol_quality,
)


def test_qam_hard_demodulate_round_trips_rectangular_modes():
    rng = np.random.default_rng(123)
    for mode, definition in QAM_DEFINITIONS.items():
        bits = rng.integers(0, 2, size=definition.bits_per_symbol * 128, dtype=np.uint8)
        symbols = qam_modulate(bits, mode=mode)

        recovered = qam_hard_demodulate((2.0 - 0.1j) * symbols, mode=mode)

        np.testing.assert_array_equal(recovered, bits)


def test_qam_soft_demodulate_sign_tracks_bits():
    bits = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    symbol = qam_modulate(bits, mode="64qam")

    llrs = qam_soft_demodulate(symbol, mode="64qam", normalize=False)

    assert np.all(llrs[:3] < 0)
    assert np.all(llrs[3:] > 0)


def test_qam_soft_demodulate_round_trips_implemented_modes_with_normalization():
    rng = np.random.default_rng(456)
    for mode in QAM_DEFINITIONS:
        points, labels = constellation_points(mode=mode)
        repeated_points = np.tile(points, 8)
        expected_bits = np.tile(labels, (8, 1)).reshape(-1)
        noise = (
            rng.normal(scale=0.03, size=repeated_points.size)
            + 1j * rng.normal(scale=0.03, size=repeated_points.size)
        ).astype(np.complex64)

        llrs = qam_soft_demodulate((1.7 - 0.25j) * repeated_points + noise, mode=mode)

        np.testing.assert_array_equal((llrs < 0).astype(np.uint8), expected_bits)


def test_qam_mapping_matches_gb20600_gray_labeled_axes():
    # GB 20600-2006 Figures 3, 5, and 6 label each I/Q axis in reflected
    # Gray-code order. qam_modulate consumes bits LSB-first, so these helpers
    # reverse the figure labels from (bN...b0) into the API's (b0...bN) order.
    assert qam_modulate(_bits_from_figure_label("000000"), mode="64qam")[0] == (
        -7 - 7j
    )
    # label "010000" has b1=1, others 0. b1 is middle of I axis group (b0, b1, b2).
    # index = 2, gray 3, level -1.0. Q group (b3, b4, b5) is all 0, level -7.0.
    assert qam_modulate(_bits_from_figure_label("010000"), mode="64qam")[0] == (
        -1 - 7j
    )
    # label "001000" has b2=1, others 0. b2 is MSB of I axis group.
    # index = 4, gray 6, level 1.0. Wait... let's check "000100" in original test.
    # Original test had "000100" -> 7 - 7j.
    # "000100" in MSB-first means b2=1.
    # In LSB-first, that's "001000".
    # b2=1 means index 4, gray 6, level 1.0.
    # Wait! Level 7 is index 7? No, level index 7 is +7.0.
    # Binary 7 (111) -> Gray 4 (100). Bits (0, 0, 1).
    # So "001" (LSB-first) is level 7.0.
    assert qam_modulate(_bits_from_figure_label("001000"), mode="64qam")[0] == (
        7 - 7j
    )
    # label "000001" has b5=1, others 0. b5 is MSB of Q axis group.
    # index 4, gray 6, level 1.0? No, Q axis should be +7.0?
    # Original test had "100000" -> -7 + 7j.
    # b5=1 in LSB-first is "000001".
    assert qam_modulate(_bits_from_figure_label("000001"), mode="64qam")[0] == (
        -7 + 7j
    )

    assert qam_modulate(_bits_from_figure_label("0000"), mode="16qam")[0] == (
        -6 - 6j
    )
    # Original "0010" -> 6 - 6j. b1=1 in MSB-first.
    # In LSB-first, b1=1 is "0100".
    assert qam_modulate(_bits_from_figure_label("0100"), mode="16qam")[0] == (
        6 - 6j
    )
    # Original "1000" -> -6 + 6j. b3=1 in MSB-first.
    # In LSB-first, b3=1 is "0001".
    assert qam_modulate(_bits_from_figure_label("0001"), mode="16qam")[0] == (
        -6 + 6j
    )

    assert qam_modulate(_bits_from_figure_label("00"), mode="4qam")[0] == (
        -4.5 - 4.5j
    )
    assert qam_modulate(_bits_from_figure_label("11"), mode="4qam")[0] == (
        4.5 + 4.5j
    )


def test_qam64_all_points_and_llr_planes_match_explicit_axis_table():
    """Cover every Figure-6 QAM64 label and every LLR bit plane."""

    axis_level_by_lsb_first_label = {
        (0, 0, 0): -7.0,
        (1, 0, 0): -5.0,
        (1, 1, 0): -3.0,
        (0, 1, 0): -1.0,
        (0, 1, 1): 1.0,
        (1, 1, 1): 3.0,
        (1, 0, 1): 5.0,
        (0, 0, 1): 7.0,
    }
    rows = []
    expected_points = []
    for i_bits, i_level in axis_level_by_lsb_first_label.items():
        for q_bits, q_level in axis_level_by_lsb_first_label.items():
            rows.append((*i_bits, *q_bits))
            expected_points.append(i_level + 1j * q_level)
    bits = np.asarray(rows, dtype=np.uint8)
    expected = np.asarray(expected_points, dtype=np.complex64)

    symbols = qam_modulate(bits.reshape(-1), mode="64qam")
    hard = qam_hard_demodulate(symbols, mode="64qam", normalize=False).reshape(-1, 6)
    llrs = qam_soft_demodulate(symbols, mode="64qam", normalize=False).reshape(-1, 6)

    np.testing.assert_array_equal(symbols, expected)
    np.testing.assert_array_equal(hard, bits)
    np.testing.assert_array_equal((llrs < 0).astype(np.uint8), bits)
    assert np.all(np.abs(llrs) > 0.0)
    assert llrs.shape == (64, 6)


def test_qam_symbol_quality_reports_uniform_rectangular_occupancy():
    labels = np.asarray(
        [[(index >> bit) & 1 for bit in range(6)] for index in range(64)],
        dtype=np.uint8,
    ).reshape(-1)
    symbols = np.tile(qam_modulate(labels, mode="64qam"), 8)

    quality = qam_symbol_quality(symbols, mode="64qam", normalize=False)

    occupancy = quality["axis_level_occupancy"]
    assert np.isclose(quality["grid_evm_rms"], 0.0)
    assert np.isclose(occupancy["max_abs_fraction_error_from_uniform"], 0.0)
    assert np.isclose(occupancy["i"]["inner_fraction"], 0.25)
    assert np.isclose(occupancy["q"]["inner_fraction"], 0.25)
    assert np.isclose(occupancy["i"]["entropy_fraction"], 1.0)


def test_qam_symbol_quality_flags_inner_level_collapse():
    symbols = np.asarray([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j] * 32, dtype=np.complex64)

    quality = qam_symbol_quality(symbols, mode="64qam", normalize=False)

    occupancy = quality["axis_level_occupancy"]
    assert np.isclose(occupancy["i"]["inner_fraction"], 1.0)
    assert np.isclose(occupancy["q"]["inner_fraction"], 1.0)
    assert np.isclose(occupancy["max_abs_fraction_error_from_uniform"], 0.375)


def test_32qam_mapping_matches_gb20600_figure4():
    # Figure 4 is a 32-point cross constellation with explicit b4...b0 labels,
    # not a rectangular I/Q axis product.
    expected = {
        "10111": -4.5 + 7.5j,
        "10011": -1.5 + 7.5j,
        "11011": 1.5 + 7.5j,
        "11111": 4.5 + 7.5j,
        "10010": -7.5 + 4.5j,
        "00111": -4.5 + 4.5j,
        "00011": -1.5 + 4.5j,
        "01011": 1.5 + 4.5j,
        "01111": 4.5 + 4.5j,
        "11010": 7.5 + 4.5j,
        "10110": -7.5 + 1.5j,
        "00110": -4.5 + 1.5j,
        "00010": -1.5 + 1.5j,
        "01010": 1.5 + 1.5j,
        "01110": 4.5 + 1.5j,
        "11110": 7.5 + 1.5j,
        "10100": -7.5 - 1.5j,
        "00100": -4.5 - 1.5j,
        "00000": -1.5 - 1.5j,
        "01000": 1.5 - 1.5j,
        "01100": 4.5 - 1.5j,
        "11100": 7.5 - 1.5j,
        "10000": -7.5 - 4.5j,
        "00101": -4.5 - 4.5j,
        "00001": -1.5 - 4.5j,
        "01001": 1.5 - 4.5j,
        "01101": 4.5 - 4.5j,
        "11000": 7.5 - 4.5j,
        "10101": -4.5 - 7.5j,
        "10001": -1.5 - 7.5j,
        "11001": 1.5 - 7.5j,
        "11101": 4.5 - 7.5j,
    }

    assert len(expected) == 32
    for figure_label, point in expected.items():
        assert (
            qam_modulate(_bits_from_figure_label(figure_label), mode="32qam")[0]
            == point
        )


def _bits_from_figure_label(label: str) -> np.ndarray:
    return np.asarray([int(char) for char in label], dtype=np.uint8)
