import numpy as np

from dtmb.fec import (
    decode_frame_bch_descramble_from_ldpc_codewords,
    decode_frame_bch_descramble_from_ldpc_llr,
    decode_frame_bch_descramble_from_ldpc_messages,
    encode_frame_bch_ldpc_codewords,
    encode_frame_bch_ldpc_message_bits,
    extract_ldpc_message_bits_from_codewords,
)
from dtmb.ldpc import (
    collapse_dtmb_full_codeword_bits,
    dtmb_ldpc_profile,
    expand_dtmb_transmitted_llr,
    extract_dtmb_message_bits_from_full_codewords,
    llr_from_hard_bits,
)


def test_bch_descrambler_round_trips_one_rate3_ldpc_message():
    transport = bytes((index * 7) & 0xFF for index in range(752))

    message_bits = encode_frame_bch_ldpc_message_bits(transport, fec_rate_index=3)
    result = decode_frame_bch_descramble_from_ldpc_messages(
        message_bits,
        fec_rate_index=3,
    )

    assert result.transport_bytes == transport
    assert result.bch_blocks == 8
    assert result.bch_unclean_blocks == 0


def test_extract_ldpc_message_bits_assumes_parity_bits_first():
    profile = dtmb_ldpc_profile(3)
    message = np.arange(profile.message_bits, dtype=np.uint8) & 1
    parity = np.ones(profile.parity_bits, dtype=np.uint8)
    codeword = np.concatenate((parity, message))

    extracted = extract_ldpc_message_bits_from_codewords(
        codeword,
        fec_rate_index=3,
        parity_position="front",
    )

    np.testing.assert_array_equal(extracted, message)


def test_bch_descrambler_from_ldpc_codewords_round_trips():
    transport = bytes((0x47 + index) & 0xFF for index in range(752))
    codeword = encode_frame_bch_ldpc_codewords(transport, fec_rate_index=3)

    result = decode_frame_bch_descramble_from_ldpc_codewords(
        codeword,
        fec_rate_index=3,
        parity_position="front",
    )

    assert result.transport_bytes == transport


def test_bch_descrambler_round_trips_multiple_rate3_ldpc_codewords_as_one_frame():
    transport = bytes((0x80 + 17 * index) & 0xFF for index in range(3 * 752))
    codewords = encode_frame_bch_ldpc_codewords(transport, fec_rate_index=3)

    result = decode_frame_bch_descramble_from_ldpc_codewords(
        codewords,
        fec_rate_index=3,
        parity_position="front",
    )

    assert codewords.size == 3 * dtmb_ldpc_profile(3).codeword_bits
    assert result.bch_blocks == 24
    assert result.bch_unclean_blocks == 0
    assert result.transport_bytes == transport


def test_ldpc_bch_descrambler_from_llr_round_trips():
    transport = bytes((0x47 + index) & 0xFF for index in range(752))
    codeword = encode_frame_bch_ldpc_codewords(transport, fec_rate_index=3)
    llr = llr_from_hard_bits(codeword, magnitude=12.0)

    result = decode_frame_bch_descramble_from_ldpc_llr(
        llr,
        fec_rate_index=3,
        max_ldpc_iterations=20,
    )

    assert result.ldpc_codewords == 1
    assert result.ldpc_converged_codewords == 1
    assert result.bch_unclean_blocks == 0
    assert result.transport_bytes == transport


def test_ldpc_bch_descrambler_from_noisy_llr_round_trips_all_rates():
    sigma = 0.45

    for rate_index in (1, 2, 3):
        profile = dtmb_ldpc_profile(rate_index)
        transport_bytes = profile.bch_blocks_per_codeword * 752 // 8
        transport = bytes(
            (0x47 + rate_index * index) & 0xFF
            for index in range(transport_bytes)
        )
        codeword = encode_frame_bch_ldpc_codewords(transport, fec_rate_index=rate_index)
        rng = np.random.default_rng(20260429 + rate_index)
        symbols = np.where(codeword == 0, 1.0, -1.0).astype(np.float32)
        noisy = symbols + rng.normal(0.0, sigma, codeword.size).astype(np.float32)
        llr = (2.0 * noisy / (sigma * sigma)).astype(np.float32)

        hard_errors = int(np.count_nonzero((llr < 0).astype(np.uint8) != codeword))
        result = decode_frame_bch_descramble_from_ldpc_llr(
            llr,
            fec_rate_index=rate_index,
            max_ldpc_iterations=30,
        )

        assert hard_errors > 0
        assert result.ldpc_codewords == 1
        assert result.ldpc_converged_codewords == 1
        assert result.ldpc_syndrome_weights == (0,)
        assert result.bch_unclean_blocks == 0
        assert result.transport_bytes == transport


def test_ldpc_codeword_encoder_outputs_transmitted_codeword_size():
    profile = dtmb_ldpc_profile(3)
    transport = bytes((index * 3) & 0xFF for index in range(752))

    codeword = encode_frame_bch_ldpc_codewords(transport, fec_rate_index=3)

    assert codeword.size == profile.codeword_bits
    assert np.count_nonzero(codeword[: profile.parity_bits]) > 0


def test_independent_rate2_transmitted_layout_recovers_transport_bytes():
    profile = dtmb_ldpc_profile(2)
    codewords_per_frame = 3
    transport = b"".join(
        bytes([0x47, packet & 0xFF, (packet * 3) & 0xFF, (packet * 5) & 0xFF])
        + bytes(((packet + byte) & 0xFF) for byte in range(184))
        for packet in range(9)
    )
    assert len(transport) == (
        codewords_per_frame * profile.bch_blocks_per_codeword * 752 // 8
    )

    payload_bits = np.unpackbits(
        np.frombuffer(transport, dtype=np.uint8),
        bitorder="big",
    )
    scrambled = _independent_scramble_bits(payload_bits)
    message_bits = np.concatenate(
        [
            _independent_shortened_parent_bch_encode(scrambled[start : start + 752])
            for start in range(0, scrambled.size, 752)
        ]
    ).astype(np.uint8)
    assert message_bits.size == codewords_per_frame * profile.message_bits

    rows = []
    full_rows = []
    for codeword_index in range(codewords_per_frame):
        message = message_bits[
            codeword_index * profile.message_bits :
            (codeword_index + 1) * profile.message_bits
        ]
        erased_parity = (
            (np.arange(profile.erased_parity_bits, dtype=np.uint8) + codeword_index)
            & 1
        )
        transmitted_parity = (
            (np.arange(profile.parity_bits, dtype=np.uint8) + 1 + codeword_index)
            & 1
        )
        rows.append(np.concatenate((transmitted_parity, message)))
        full_rows.append(np.concatenate((erased_parity, transmitted_parity, message)))
    transmitted = np.concatenate(rows).astype(np.uint8)
    full = np.concatenate(full_rows).astype(np.uint8)

    np.testing.assert_array_equal(
        collapse_dtmb_full_codeword_bits(full, fec_rate_index=2),
        transmitted,
    )
    np.testing.assert_array_equal(
        extract_dtmb_message_bits_from_full_codewords(full, fec_rate_index=2),
        message_bits,
    )
    expanded_llr = expand_dtmb_transmitted_llr(
        llr_from_hard_bits(transmitted, magnitude=3.0),
        fec_rate_index=2,
        erased_llr=0.0,
    ).reshape(codewords_per_frame, profile.full_codeword_bits)
    assert np.all(expanded_llr[:, : profile.erased_parity_bits] == 0.0)
    np.testing.assert_array_equal(
        (expanded_llr[:, profile.full_parity_bits :] < 0).astype(np.uint8).reshape(-1),
        message_bits,
    )

    decoded = decode_frame_bch_descramble_from_ldpc_codewords(
        transmitted,
        fec_rate_index=2,
        parity_position="front",
    )
    wrong_position = decode_frame_bch_descramble_from_ldpc_codewords(
        transmitted,
        fec_rate_index=2,
        parity_position="back",
        correct_bch=False,
    )

    assert decoded.bch_blocks == codewords_per_frame * profile.bch_blocks_per_codeword
    assert decoded.bch_unclean_blocks == 0
    assert decoded.transport_bytes == transport
    assert wrong_position.transport_bytes != transport


def _independent_scramble_bits(bits: np.ndarray) -> np.ndarray:
    state = [int(bit) for bit in "100101010000000"]
    sequence = np.empty(bits.size, dtype=np.uint8)
    for index in range(bits.size):
        next_bit = state[13] ^ state[14]
        sequence[index] = next_bit
        state = [next_bit] + state[:-1]
    return (np.asarray(bits, dtype=np.uint8).reshape(-1) ^ sequence).astype(np.uint8)


def _independent_shortened_parent_bch_encode(message: np.ndarray) -> np.ndarray:
    values = np.asarray(message, dtype=np.uint8).reshape(-1)
    assert values.size == 752
    parent_message = np.concatenate((np.zeros(261, dtype=np.uint8), values))
    parent_codeword = _independent_systematic_parent_bch_encode(parent_message)
    return parent_codeword[261:].astype(np.uint8)


def _independent_systematic_parent_bch_encode(message: np.ndarray) -> np.ndarray:
    values = np.asarray(message, dtype=np.uint8).reshape(-1)
    assert values.size == 1013
    work = np.concatenate((values, np.zeros(10, dtype=np.uint8)))
    generator = np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
    for index in range(values.size):
        if work[index]:
            work[index : index + generator.size] ^= generator
    parity = work[-10:]
    return np.concatenate((values, parity)).astype(np.uint8)
