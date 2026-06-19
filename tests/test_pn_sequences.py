import numpy as np
import pytest

from dtmb.pn import (
    bits_from_hex,
    lfsr_bits,
    pn_bipolar,
    pn_bits,
    pn_cyclic_family_bipolar,
    pn_header_symbols,
    pn_header_symbols_for_body_power,
)


PN420_PHASE0_HEX = (
    "B0A5E9FEA1CF0D9A3DC7407C4A22D5C8C938109BCCEFCB2B69063"
    "58AA60BAFB7614BD3FD439E1B347B8E80F89445AB91927021379"
)

PN945_PHASE0_HEX = (
    "FB946DFF3259AD7E9A7C9CDC081A82531A292661E8F1233A432F2D8055B"
    "ABDE0EBA16959060BE1BD4B9EDAA88E44D953B15C5DA03FB3F1884F38EEF"
    "AC28708B1F728DBFE64B35AFD34F939B8103504A634524CC3D1E24674865"
    "E5B00AB757BC1D742D2B20C17C37A973DB5511C89B2A762B8BB407F678"
)


def test_pn420_matches_gb20600_appendix_phase0_vector():
    expected = bits_from_hex(PN420_PHASE0_HEX, bit_count=420)

    np.testing.assert_array_equal(pn_bits("pn420"), expected)


def test_pn945_matches_gb20600_appendix_phase0_vector_without_padding_bits():
    expected = bits_from_hex(PN945_PHASE0_HEX, bit_count=945)

    np.testing.assert_array_equal(pn_bits("pn945"), expected)


def test_pn595_uses_fixed_standard_seed_and_length():
    bits = pn_bits("pn595")

    assert bits.size == 595
    np.testing.assert_array_equal(bits[:10], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8))


def test_pn_bipolar_maps_zero_to_plus_one_and_one_to_minus_one():
    chips = pn_bipolar("pn595")

    assert chips[0] == 1.0
    assert chips[9] == -1.0
    assert set(np.unique(chips)) == {-1.0, 1.0}


def test_pn_header_symbols_use_equal_i_q_4qam_mapping():
    chips = pn_bipolar("pn945")
    symbols = pn_header_symbols("pn945")

    np.testing.assert_array_equal(symbols.real, chips)
    np.testing.assert_array_equal(symbols.imag, chips)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 2.0)


def test_pn_header_symbols_match_normative_body_power_ratios():
    body_power = 0.125

    for mode, expected_ratio in (("pn420", 2.0), ("pn595", 1.0), ("pn945", 2.0)):
        symbols = pn_header_symbols_for_body_power(mode, body_power=body_power)
        actual_ratio = float(np.mean(np.abs(symbols) ** 2) / body_power)
        assert actual_ratio == pytest.approx(expected_ratio)


def test_pn945_cyclic_family_contains_phase0():
    family = pn_cyclic_family_bipolar("pn945")

    assert family.shape == (511, 945)
    np.testing.assert_allclose(family[0], pn_bipolar("pn945"))


def test_lfsr_rejects_all_zero_seed():
    with pytest.raises(ValueError, match="all-zero"):
        lfsr_bits(
            seed_d_high_to_low="00000000",
            degree=8,
            recurrence_taps=(0, 2, 3, 7),
            length=420,
        )
