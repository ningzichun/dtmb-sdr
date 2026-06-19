import numpy as np

from dtmb.system_info import (
    SYSTEM_INFO_VECTORS,
    classify_system_info,
    system_info_symbols,
    transmission_parameters_for_index,
)


def test_system_info_vectors_match_gb20600_walsh_g31_generation():
    generated = _generate_gb20600_system_info_vectors()

    assert generated == SYSTEM_INFO_VECTORS


def test_system_info_rows_21_to_24_keep_64qam_rate_interleaver_semantics():
    expected = {
        21: ("64qam", 2, "mode1"),
        22: ("64qam", 2, "mode2"),
        23: ("64qam", 3, "mode1"),
        24: ("64qam", 3, "mode2"),
    }

    for index, (qam_mode, fec_rate_index, interleaver_mode) in expected.items():
        parameters = transmission_parameters_for_index(index, frame_body_mode="C3780")
        assert parameters.qam_mode == qam_mode
        assert parameters.fec_rate_index == fec_rate_index
        assert parameters.interleaver_mode == interleaver_mode


def test_system_info_classifier_identifies_64qam_rate3_c3780():
    symbols = system_info_symbols(
        "00010001010111100100011101000011",
        frame_body_mode="C3780",
    )

    matches = classify_system_info(symbols)

    assert matches[0].index == 23
    assert matches[0].frame_body_mode == "C3780"
    assert matches[0].metric > 0.99
    assert matches[0].to_dict()["parameters"]["qam_mode"] == "64qam"


def test_system_info_vectors_cover_appendix_g_all_64_rows():
    assert len(SYSTEM_INFO_VECTORS) == 64
    assert SYSTEM_INFO_VECTORS[25] == "00101101011000100111101101111111"
    assert SYSTEM_INFO_VECTORS[33] == "01111000110010001101000100101010"
    assert SYSTEM_INFO_VECTORS[64] == "11101110101000010100011101000011"


def test_system_info_symbols_prepend_frame_body_mode_bits():
    c1 = system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C1")
    c3780 = system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")

    np.testing.assert_array_equal(
        c1[:4],
        np.full(4, 1 + 1j, dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        c3780[:4],
        np.full(4, -1 - 1j, dtype=np.complex64),
    )


def test_system_info_parameters_describe_supported_c3780_mode():
    parameters = transmission_parameters_for_index(9, frame_body_mode="C3780")

    assert parameters.qam_mode == "4qam"
    assert parameters.fec_rate_index == 3
    assert parameters.interleaver_mode == "mode1"
    assert parameters.supported_by_receiver is True


def test_system_info_classifier_identifies_complement_as_mode2():
    symbols = system_info_symbols(SYSTEM_INFO_VECTORS[24], frame_body_mode="C3780")

    matches = classify_system_info(symbols)

    assert matches[0].index == 24
    assert matches[0].to_dict()["parameters"]["interleaver_mode"] == "mode2"
    assert matches[0].polarity == 1


def test_system_info_classifier_can_report_inverted_polarity_for_known_mode():
    symbols = -system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")

    default_matches = classify_system_info(symbols, frame_body_modes=("C3780",))
    polarity_matches = classify_system_info(
        symbols,
        allow_polarity_inversion=True,
        frame_body_modes=("C3780",),
    )

    assert default_matches[0].index == 24
    assert polarity_matches[0].index == 23
    assert polarity_matches[0].polarity == -1


def test_system_info_parameters_mark_unsupported_modes():
    assert transmission_parameters_for_index(3).supported_by_receiver is False
    assert transmission_parameters_for_index(17).qam_mode == "32qam"
    assert transmission_parameters_for_index(17).fec_rate_index == 3
    assert transmission_parameters_for_index(17).supported_by_receiver is False
    assert transmission_parameters_for_index(25).qam_label == "reserved"
    assert transmission_parameters_for_index(25).supported_by_receiver is False


def test_system_info_parameters_describe_supported_c1_mode():
    parameters = transmission_parameters_for_index(23, frame_body_mode="C1")

    assert parameters.qam_mode == "64qam"
    assert parameters.fec_rate_index == 3
    assert parameters.interleaver_mode == "mode1"
    assert parameters.supported_by_receiver is True


def test_system_info_classifier_handles_bounded_common_phase_and_awgn():
    rng = np.random.default_rng(20260429)
    rotation = np.complex64(np.exp(0.42j))
    for index, bits in SYSTEM_INFO_VECTORS.items():
        for frame_body_mode in ("C3780", "C1"):
            symbols = system_info_symbols(bits, frame_body_mode=frame_body_mode)
            noise = (
                rng.normal(scale=0.05, size=symbols.size)
                + 1j * rng.normal(scale=0.05, size=symbols.size)
            ).astype(np.complex64)

            matches = classify_system_info(rotation * symbols + noise)

            assert matches[0].index == index
            assert matches[0].frame_body_mode == frame_body_mode
            assert matches[0].metric > 0.95


def test_system_info_classifier_rejects_unbounded_common_phase():
    symbols = system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")
    rotated = np.complex64(np.exp(0.8j)) * symbols

    bounded = classify_system_info(rotated)
    explicit_wide = classify_system_info(rotated, max_common_phase_radians=0.9)

    assert bounded[0].metric < 0.8
    assert explicit_wide[0].index == 23
    assert explicit_wide[0].frame_body_mode == "C3780"
    assert explicit_wide[0].metric > 0.99


def _generate_gb20600_system_info_vectors() -> dict[int, str]:
    """Regenerate Appendix G vectors from the GB 20600-2006 construction.

    Section 4.6.3 constructs each 32-bit vector by taking one row from the
    W32 Walsh block, mapping ``+1 -> 1`` and ``-1 -> 0``, then XORing the
    32-bit G31 randomizer. Adjacent Appendix G rows are the mode-1/mode-2
    pair, so the even row uses the inverted Walsh row.
    """

    walsh = _walsh32_binary_rows()
    randomizer = _g31_randomizer_bits()
    vectors: dict[int, str] = {}
    for pair_index, walsh_row in enumerate(_GB20600_APPENDIX_G_WALSH_ROW_ORDER):
        odd_index = 2 * pair_index + 1
        base = walsh[walsh_row]
        odd_bits = base ^ randomizer
        even_bits = (1 - base) ^ randomizer
        vectors[odd_index] = _bits_to_string(odd_bits)
        vectors[odd_index + 1] = _bits_to_string(even_bits)
    return vectors


_GB20600_APPENDIX_G_WALSH_ROW_ORDER = (
    # Walsh row per Appendix G row pair, in 1-based row order:
    # (1,2), (3,4), ..., (63,64). This table is deliberately separate from
    # SYSTEM_INFO_VECTORS so the test regenerates the implementation table
    # instead of reformatting it.
    3,
    0,
    4,
    5,
    2,
    7,
    8,
    1,
    12,
    13,
    14,
    15,
    9,
    6,
    10,
    11,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
)


def _walsh32_binary_rows() -> np.ndarray:
    block = np.asarray([[1]], dtype=np.int8)
    for _ in range(5):
        block = np.block([[block, block], [block, -block]])
    return np.where(block == 1, 1, 0).astype(np.uint8)


def _g31_randomizer_bits() -> np.ndarray:
    state = [int(bit) for bit in "00001"]
    output: list[int] = []
    for _ in range(31):
        output.append(state[-1])
        feedback = state[0] ^ state[2] ^ state[3] ^ state[4]
        state = [feedback] + state[:-1]
    output.append(0)
    return np.asarray(output, dtype=np.uint8)


def _bits_to_string(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)
