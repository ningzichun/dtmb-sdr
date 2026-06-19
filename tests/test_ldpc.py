import pytest
from pathlib import Path
import numpy as np

from dtmb.ldpc import (
    collapse_dtmb_full_codeword_bits,
    dtmb_ldpc_decode_transmitted_llr,
    dtmb_ldpc_profile,
    dtmb_ldpc_encode_message_bits,
    dtmb_ldpc_generator_hex,
    dtmb_ldpc_generator_polynomials,
    dtmb_ldpc_parity_mismatch_counts,
    dtmb_ldpc_parity_check_shifts,
    dtmb_ldpc_syndrome_weight,
    expand_dtmb_transmitted_llr,
    extract_dtmb_message_bits_from_full_codewords,
    ldpc_decode_min_sum,
    llr_from_hard_bits,
)


def test_dtmb_ldpc_profiles_record_message_and_codeword_sizes():
    assert dtmb_ldpc_profile(1).message_bits == 3048
    assert dtmb_ldpc_profile(2).message_bits == 4572
    profile = dtmb_ldpc_profile(3)
    assert profile.message_bits == 6096
    assert profile.codeword_bits == 7488
    assert profile.full_codeword_bits == 7493
    assert profile.full_parity_bits == 1397
    assert profile.parity_bits == 1392
    assert profile.erased_parity_bits == 5
    assert profile.bch_blocks_per_codeword == 8


def test_ldpc_min_sum_decodes_single_parity_check():
    parity_check = np.asarray([[1, 1, 1]], dtype=np.uint8)
    transmitted = np.asarray([0, 1, 1], dtype=np.uint8)
    llr = llr_from_hard_bits(transmitted)
    llr[0] = 0.1

    result = ldpc_decode_min_sum(llr, parity_check, max_iterations=8)

    assert result.converged
    np.testing.assert_array_equal(result.bits, transmitted)


def test_dtmb_ldpc_puncturing_helpers_restore_full_layout():
    profile = dtmb_ldpc_profile(3)
    transmitted = np.arange(profile.codeword_bits, dtype=np.float32)

    expanded = expand_dtmb_transmitted_llr(transmitted, fec_rate_index=3)

    assert expanded.size == profile.full_codeword_bits
    np.testing.assert_array_equal(
        expanded[: profile.erased_parity_bits],
        np.zeros(profile.erased_parity_bits, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        expanded[profile.erased_parity_bits : profile.full_parity_bits],
        transmitted[: profile.parity_bits],
    )
    np.testing.assert_array_equal(
        expanded[profile.full_parity_bits :],
        transmitted[profile.parity_bits :],
    )


def test_dtmb_ldpc_bit_layout_helpers_extract_transmitted_and_message_bits():
    profile = dtmb_ldpc_profile(3)
    full = (np.arange(profile.full_codeword_bits) & 1).astype(np.uint8)

    collapsed = collapse_dtmb_full_codeword_bits(full, fec_rate_index=3)
    message = extract_dtmb_message_bits_from_full_codewords(full, fec_rate_index=3)

    assert collapsed.size == profile.codeword_bits
    np.testing.assert_array_equal(
        collapsed[: profile.parity_bits],
        full[profile.erased_parity_bits : profile.full_parity_bits],
    )
    np.testing.assert_array_equal(message, full[profile.full_parity_bits :])


def test_dtmb_ldpc_generator_matrix_dimensions_match_profiles():
    expected = {
        1: (24, 35, 127),
        2: (36, 23, 127),
        3: (48, 11, 127),
    }

    for rate_index, shape in expected.items():
        hex_matrix = dtmb_ldpc_generator_hex(rate_index)
        polynomials = dtmb_ldpc_generator_polynomials(rate_index)

        assert len(hex_matrix) == shape[0]
        assert len(hex_matrix[0]) == shape[1]
        assert polynomials.shape == shape


def test_dtmb_ldpc_parity_check_shift_dimensions_match_profiles():
    expected = {
        1: (35, 59),
        2: (23, 59),
        3: (11, 59),
    }

    for rate_index, (row_count, column_count) in expected.items():
        rows = dtmb_ldpc_parity_check_shifts(rate_index)

        assert len(rows) == row_count
        assert all(0 <= column < column_count for row in rows for column, _shift in row)
        assert all(0 <= shift < 127 for row in rows for _column, shift in row)


def test_dtmb_ldpc_encoder_uses_standard_parity_first_layout():
    profile = dtmb_ldpc_profile(3)
    message = np.zeros(profile.message_bits, dtype=np.uint8)
    message[0] = 1
    polynomials = dtmb_ldpc_generator_polynomials(3)
    expected_full_parity = polynomials[0].reshape(-1)

    full = dtmb_ldpc_encode_message_bits(
        message,
        fec_rate_index=3,
        transmitted=False,
    )
    transmitted = dtmb_ldpc_encode_message_bits(message, fec_rate_index=3)

    np.testing.assert_array_equal(
        full[: profile.full_parity_bits],
        expected_full_parity,
    )
    np.testing.assert_array_equal(full[profile.full_parity_bits :], message)
    np.testing.assert_array_equal(
        transmitted[: profile.parity_bits],
        expected_full_parity[profile.erased_parity_bits :],
    )
    np.testing.assert_array_equal(transmitted[profile.parity_bits :], message)


def test_dtmb_ldpc_encoder_satisfies_standard_parity_check():
    rng = np.random.default_rng(123)
    for rate_index in (1, 2, 3):
        profile = dtmb_ldpc_profile(rate_index)
        message = rng.integers(0, 2, profile.message_bits, dtype=np.uint8)
        full = dtmb_ldpc_encode_message_bits(
            message,
            fec_rate_index=rate_index,
            transmitted=False,
        )
        damaged = full.copy()
        damaged[17] ^= 1

        assert dtmb_ldpc_syndrome_weight(full, fec_rate_index=rate_index) == 0
        assert dtmb_ldpc_syndrome_weight(damaged, fec_rate_index=rate_index) > 0


def test_dtmb_ldpc_parity_mismatch_counts_validates_codewords():
    profile = dtmb_ldpc_profile(3)
    message = (np.arange(profile.message_bits) & 1).astype(np.uint8)
    codeword = dtmb_ldpc_encode_message_bits(message, fec_rate_index=3)

    clean = dtmb_ldpc_parity_mismatch_counts(codeword, fec_rate_index=3)
    damaged = codeword.copy()
    damaged[profile.parity_bits + 17] ^= 1
    dirty = dtmb_ldpc_parity_mismatch_counts(damaged, fec_rate_index=3)

    np.testing.assert_array_equal(clean, np.asarray([0], dtype=np.int32))
    assert dirty[0] > 0


def test_dtmb_ldpc_decode_transmitted_llr_recovers_erased_parity_bits():
    profile = dtmb_ldpc_profile(3)
    message = np.zeros(profile.message_bits, dtype=np.uint8)
    message[::257] = 1
    full = dtmb_ldpc_encode_message_bits(
        message,
        fec_rate_index=3,
        transmitted=False,
    )
    transmitted = collapse_dtmb_full_codeword_bits(full, fec_rate_index=3)
    llr = llr_from_hard_bits(transmitted, magnitude=12.0)

    result = dtmb_ldpc_decode_transmitted_llr(
        llr,
        fec_rate_index=3,
        max_iterations=20,
    )

    assert result.converged
    assert result.syndrome_weight == 0
    np.testing.assert_array_equal(result.bits, full)


def test_dtmb_ldpc_appendix_b_syndrome_counts_on_valid_and_random_codewords():
    from dtmb.ldpc import dtmb_ldpc_appendix_b_syndrome_counts

    rng = np.random.default_rng(20260510)
    for rate_index in (1, 2, 3):
        profile = dtmb_ldpc_profile(rate_index)
        message = rng.integers(0, 2, size=profile.message_bits, dtype=np.uint8)
        valid = dtmb_ldpc_encode_message_bits(
            message,
            fec_rate_index=rate_index,
            transmitted=True,
        )
        weights, clean_rows = dtmb_ldpc_appendix_b_syndrome_counts(
            valid, fec_rate_index=rate_index
        )
        assert clean_rows > 0
        assert weights.shape == (1,)
        assert int(weights[0]) == 0, (
            f"valid rate-{rate_index} codeword must have zero Appendix-B syndrome"
        )

        scrambled = valid.copy()
        flip = rng.integers(0, 2, size=scrambled.size, dtype=np.uint8)
        scrambled ^= flip
        weights_random, _ = dtmb_ldpc_appendix_b_syndrome_counts(
            scrambled, fec_rate_index=rate_index
        )
        ratio = float(weights_random[0]) / clean_rows
        assert 0.3 < ratio < 0.7, (
            f"random bits should give syndrome ratio near 0.5 (got {ratio:.3f})"
        )


def _parse_standard_ldpc_appendices() -> tuple[dict, dict]:
    """Parse Appendix A (G hex) and Appendix B (A shifts) from the GB text.

    Returns ``(generator, parity)`` where each is ``{rate: {(r, c): value}}``.
    The standard interleaves both appendices under ``LDPC(7493, NNNN)`` headers
    that select the FEC rate by information-bit count.
    """

    import re
    from pathlib import Path

    refs = Path(__file__).resolve().parents[1] / "references"
    std_path = next(refs.glob("GB_20600*2006.txt"))
    text = std_path.read_text(encoding="utf-8", errors="replace")

    msg_to_rate = {3048: 1, 4572: 2, 6096: 3}
    hdr_re = re.compile(r"LDPC\(\s*7493\s*,\s*(\d+)\s*\)")
    g_re = re.compile(r"G\[\s*(\d+)\]\[\s*(\d+)\]\s*:\s*([0-9A-Fa-f]{32})")
    a_re = re.compile(r"A\[\s*(\d+)\]\[\s*(\d+)\]\s*=\s*(\d+)")

    headers = [(m.start(), int(m.group(1))) for m in hdr_re.finditer(text)]
    gen: dict = {1: {}, 2: {}, 3: {}}
    par: dict = {1: {}, 2: {}, 3: {}}
    for idx, (pos, msg) in enumerate(headers):
        if msg not in msg_to_rate:
            continue
        rate = msg_to_rate[msg]
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
        for m in g_re.finditer(text, pos, end):
            gen[rate][(int(m.group(1)), int(m.group(2)))] = m.group(3).upper()
        for m in a_re.finditer(text, pos, end):
            par[rate][(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    return gen, par



@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "references").exists(),
    reason="GB 20600-2006 reference text not available in open-source tree",
)
def test_stored_ldpc_matrices_match_gb20600_normative_text():
    """Stored G/H matrices must match GB 20600-2006 Appendix A/B byte-exact.

    A self-consistent synthetic loopback proves the generator and parity-check
    matrices are mutually consistent but NOT that they match the normative
    standard. If both were transcribed from a wrong source, the synthetic
    round-trip still passes byte-exact while every real broadcast codeword
    fails LDPC parity at the 0.50 random plateau. This regression guards the
    one fault the loopback can never catch. See trial-log 2026-05-31.
    """

    gen_shape = {1: (24, 35), 2: (36, 23), 3: (48, 11)}
    gen_std, par_std = _parse_standard_ldpc_appendices()

    for rate in (1, 2, 3):
        # --- Generator (Appendix A) ---
        hex_matrix = dtmb_ldpc_generator_hex(rate)
        k, c = gen_shape[rate]
        assert len(gen_std[rate]) == k * c, (
            f"rate{rate} generator: expected {k*c} circulants in standard, "
            f"parsed {len(gen_std[rate])}"
        )
        for i in range(k):
            for j in range(c):
                assert hex_matrix[i][j].upper() == gen_std[rate][(i, j)], (
                    f"rate{rate} G[{i}][{j}] differs from GB 20600-2006"
                )

        # --- Parity check (Appendix B) ---
        rows = dtmb_ldpc_parity_check_shifts(rate)
        stored_map = {
            (r, int(col)): int(shift)
            for r, row in enumerate(rows)
            for col, shift in row
        }
        assert stored_map == par_std[rate], (
            f"rate{rate} parity-check shifts differ from GB 20600-2006 "
            f"(stored {len(stored_map)} entries, standard {len(par_std[rate])})"
        )
