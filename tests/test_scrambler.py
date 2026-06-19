import numpy as np

from dtmb.scrambler import scramble_bits, scrambling_sequence


def test_scrambler_sequence_matches_gb_20600_figure_2_prefix():
    sequence = scrambling_sequence(16)

    assert "".join(str(int(bit)) for bit in sequence) == "0000001111110110"


def test_scrambler_is_self_inverse():
    bits = np.arange(64, dtype=np.uint8) % 2

    scrambled = scramble_bits(bits)

    np.testing.assert_array_equal(scramble_bits(scrambled), bits)
