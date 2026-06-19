import numpy as np
import pytest

from dtmb.frequency import DATA_SYMBOLS_PER_FRAME
from dtmb.ldpc import dtmb_ldpc_profile
from dtmb.symbol_interleaver import (
    SYMBOL_INTERLEAVERS,
    convolutional_deinterleave,
    convolutional_deinterleave_source_indices,
    convolutional_deinterleave_source_indices_custom,
    convolutional_deinterleave_custom,
    convolutional_interleave,
    convolutional_interleave_custom,
    source_symbol_for_output_symbol,
)


def test_custom_convolutional_interleaver_round_trips_after_latency():
    values = (np.arange(64) + 1j * np.arange(64)[::-1]).astype(np.complex64)
    branch_count = 4
    delay_step = 2
    latency = (branch_count - 1) * delay_step * branch_count

    interleaved = convolutional_interleave_custom(
        values,
        branch_count=branch_count,
        delay_step=delay_step,
        flush=True,
    )
    restored = convolutional_deinterleave_custom(
        interleaved,
        branch_count=branch_count,
        delay_step=delay_step,
        flush=True,
    )

    np.testing.assert_allclose(restored[latency : latency + values.size], values)


def test_custom_convolutional_interleaver_round_trips_with_commutator_phase():
    values = (np.arange(80) - 1j * np.arange(80)[::-1]).astype(np.complex64)
    branch_count = 5
    delay_step = 3
    latency = (branch_count - 1) * delay_step * branch_count

    interleaved = convolutional_interleave_custom(
        values,
        branch_count=branch_count,
        delay_step=delay_step,
        flush=True,
        phase=2,
    )
    restored = convolutional_deinterleave_custom(
        interleaved,
        branch_count=branch_count,
        delay_step=delay_step,
        flush=True,
        phase=2,
    )

    np.testing.assert_allclose(restored[latency : latency + values.size], values)


def test_dtmb_interleaver_specs_match_standard_modes():
    assert SYMBOL_INTERLEAVERS["mode1"].branch_count == 52
    assert SYMBOL_INTERLEAVERS["mode1"].delay_step == 240
    assert SYMBOL_INTERLEAVERS["mode2"].delay_step == 720


def test_dtmb_mode_functions_accept_empty_input():
    assert convolutional_interleave(np.empty(0), mode="mode1").size == 0
    assert convolutional_deinterleave(np.empty(0), mode="mode2").size == 0


@pytest.mark.parametrize(
    ("mode", "phase", "extra_symbols"),
    [
        ("mode1", 0, 137),
        ("mode1", 17, 211),
        ("mode2", 0, 113),
    ],
)
def test_dtmb_interleaver_no_flush_restores_numbered_stream_after_latency(
    mode,
    phase,
    extra_symbols,
):
    spec = SYMBOL_INTERLEAVERS[mode]
    payload_symbols = 2 * 3744 + extra_symbols
    total_symbols = spec.full_stream_latency_symbols + payload_symbols
    symbol_ids = np.arange(total_symbols, dtype=np.float32)
    values = (symbol_ids + 1j * (symbol_ids % 997)).astype(np.complex64)

    interleaved = convolutional_interleave(
        values,
        mode=mode,
        flush=False,
        phase=phase,
    )
    restored = convolutional_deinterleave(
        interleaved,
        mode=mode,
        flush=False,
        phase=phase,
    )

    recovered = restored[
        spec.full_stream_latency_symbols : spec.full_stream_latency_symbols
        + payload_symbols
    ]
    np.testing.assert_array_equal(recovered, values[:payload_symbols])


def test_deinterleave_source_indices_match_integer_label_stream():
    mode = "mode2"
    phase = 9
    output_start = 37
    count = 4096
    spec = SYMBOL_INTERLEAVERS[mode]
    source_indices = convolutional_deinterleave_source_indices(
        count,
        mode=mode,
        phase=phase,
        output_start=output_start,
    )
    input_count = max(
        int(source_indices.max()) + 1,
        spec.full_stream_latency_symbols + output_start + count,
    )
    values = np.arange(input_count, dtype=np.float32).astype(np.complex64)

    recovered = convolutional_deinterleave(
        values,
        mode=mode,
        phase=phase,
        flush=False,
        fill=-1 + 0j,
    )[
        spec.full_stream_latency_symbols
        + output_start : spec.full_stream_latency_symbols
        + output_start
        + count
    ]

    np.testing.assert_array_equal(
        np.asarray(recovered.real, dtype=np.int64),
        source_indices,
    )


def test_mode2_tagged_chronology_fixture_covers_all_phases():
    """Trace post-deinterleaver symbols to FEC/QAM/source tags for every phase."""

    mode = "mode2"
    spec = SYMBOL_INTERLEAVERS[mode]
    bits_per_symbol = 6
    profile = dtmb_ldpc_profile(2)
    symbols_per_codeword = profile.codeword_bits // bits_per_symbol
    assert profile.codeword_bits % bits_per_symbol == 0
    assert DATA_SYMBOLS_PER_FRAME == 3 * symbols_per_codeword

    output_symbol_count = 2 * DATA_SYMBOLS_PER_FRAME
    output_indices = np.arange(output_symbol_count, dtype=np.int64)
    expected_fec_frames = output_indices // DATA_SYMBOLS_PER_FRAME
    expected_codeword_slots = (
        output_indices % DATA_SYMBOLS_PER_FRAME
    ) // symbols_per_codeword
    expected_qam_symbols = output_indices % symbols_per_codeword
    expected_output_carriers = output_indices % DATA_SYMBOLS_PER_FRAME

    for phase in range(spec.branch_count):
        source_indices = convolutional_deinterleave_source_indices(
            output_symbol_count,
            mode=mode,
            phase=phase,
        )
        input_symbols = max(
            int(source_indices.max()) + 1,
            spec.full_stream_latency_symbols + output_symbol_count,
        )
        tagged_input = np.arange(input_symbols, dtype=np.float32).astype(np.complex64)
        recovered = convolutional_deinterleave(
            tagged_input,
            mode=mode,
            phase=phase,
            fill=-1 + 0j,
        )[
            spec.full_stream_latency_symbols : spec.full_stream_latency_symbols
            + output_symbol_count
        ]

        np.testing.assert_array_equal(
            np.asarray(recovered.real, dtype=np.int64),
            source_indices,
        )
        assert source_symbol_for_output_symbol(0, mode=mode, phase=phase) == int(
            source_indices[0]
        )
        assert source_symbol_for_output_symbol(
            DATA_SYMBOLS_PER_FRAME,
            mode=mode,
            phase=phase,
        ) == int(source_indices[DATA_SYMBOLS_PER_FRAME])

        source_frames = source_indices // DATA_SYMBOLS_PER_FRAME
        source_carriers = source_indices % DATA_SYMBOLS_PER_FRAME
        source_branches = (
            (source_indices % spec.branch_count) + phase
        ) % spec.branch_count

        assert np.all(expected_fec_frames[:DATA_SYMBOLS_PER_FRAME] == 0)
        assert np.all(expected_fec_frames[DATA_SYMBOLS_PER_FRAME:] == 1)
        np.testing.assert_array_equal(
            expected_output_carriers[:DATA_SYMBOLS_PER_FRAME],
            np.arange(DATA_SYMBOLS_PER_FRAME, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            expected_codeword_slots[:DATA_SYMBOLS_PER_FRAME],
            np.repeat(np.arange(3, dtype=np.int64), symbols_per_codeword),
        )
        np.testing.assert_array_equal(
            expected_qam_symbols[:symbols_per_codeword],
            np.arange(symbols_per_codeword, dtype=np.int64),
        )
        assert int(source_frames.min()) >= 0
        assert int(source_frames.max()) == 511
        assert set(np.unique(expected_codeword_slots).tolist()) == {0, 1, 2}
        assert np.unique(source_branches).size == spec.branch_count
        branch_counts = np.bincount(source_branches, minlength=spec.branch_count)
        assert int(branch_counts.min()) == output_symbol_count // spec.branch_count
        assert int(branch_counts.max()) == output_symbol_count // spec.branch_count
        assert np.unique(source_carriers).size == DATA_SYMBOLS_PER_FRAME


def test_mode2_branch_zero_standard_trace_spans_510_received_frames():
    # With phase 0, the first symbol of a 3744-symbol C=3780 data block is
    # branch 0. The first post-latency output cycle therefore pulls branch b
    # from received input frame 10*b for mode2 (M=720).
    indices = convolutional_deinterleave_source_indices(52, mode="mode2", phase=0)
    expected = np.arange(52, dtype=np.int64) * (52 * 720 + 1)

    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(indices // 3744, np.arange(52, dtype=np.int64) * 10)
    np.testing.assert_array_equal(indices % 3744, np.arange(52, dtype=np.int64))


def test_custom_deinterleave_source_indices_report_raw_leading_fill():
    indices = convolutional_deinterleave_source_indices_custom(
        8,
        branch_count=4,
        delay_step=2,
        phase=0,
        after_latency_discard=False,
    )

    np.testing.assert_array_equal(
        indices,
        np.asarray([-1, -1, -1, 3, -1, -1, -1, 7], dtype=np.int64),
    )


def test_source_symbol_for_output_symbol_reports_raw_leading_fill():
    assert (
        source_symbol_for_output_symbol(
            0,
            mode="mode2",
            phase=0,
            after_latency_discard=False,
        )
        == -1
    )
