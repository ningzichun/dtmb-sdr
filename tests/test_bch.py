import numpy as np

from dtmb.bch import BCH_CODE_BITS, BCH_MESSAGE_BITS, bch_decode, bch_encode, bch_syndrome


def test_bch_encode_produces_zero_syndrome():
    rng = np.random.default_rng(123)
    message = rng.integers(0, 2, size=BCH_MESSAGE_BITS, dtype=np.uint8)

    codeword = bch_encode(message)

    assert codeword.size == BCH_CODE_BITS
    assert not np.any(bch_syndrome(codeword))


def test_bch_decode_corrects_single_bit_error():
    rng = np.random.default_rng(456)
    message = rng.integers(0, 2, size=BCH_MESSAGE_BITS, dtype=np.uint8)
    codeword = bch_encode(message)
    codeword[311] ^= 1

    decoded, corrected, clean = bch_decode(codeword)

    assert clean
    assert corrected == 1
    np.testing.assert_array_equal(decoded, message)


def test_bch_encode_matches_shortened_1023_1013_construction():
    rng = np.random.default_rng(20600)
    messages = [
        np.zeros(BCH_MESSAGE_BITS, dtype=np.uint8),
        np.ones(BCH_MESSAGE_BITS, dtype=np.uint8),
        rng.integers(0, 2, size=BCH_MESSAGE_BITS, dtype=np.uint8),
    ]

    for message in messages:
        shortened = _encode_shortened_parent_bch(message)

        np.testing.assert_array_equal(bch_encode(message), shortened)


def _encode_shortened_parent_bch(message: np.ndarray) -> np.ndarray:
    """Encode by shortening the GB parent BCH(1023,1013) code by 261 bits."""

    values = np.asarray(message, dtype=np.uint8).reshape(-1)
    assert values.size == BCH_MESSAGE_BITS
    parent_message = np.concatenate((np.zeros(261, dtype=np.uint8), values))
    parent_codeword = _systematic_parent_bch_encode(parent_message)
    return parent_codeword[261:].astype(np.uint8)


def _systematic_parent_bch_encode(message: np.ndarray) -> np.ndarray:
    values = np.asarray(message, dtype=np.uint8).reshape(-1)
    assert values.size == 1013
    work = np.concatenate((values, np.zeros(10, dtype=np.uint8)))
    generator = np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
    for index in range(values.size):
        if work[index]:
            work[index : index + generator.size] ^= generator
    parity = work[-10:]
    return np.concatenate((values, parity)).astype(np.uint8)
