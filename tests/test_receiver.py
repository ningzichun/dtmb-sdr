import json

import numpy as np

import dtmb.receiver as receiver
from dtmb.bits import unpack_bytes_msb
from dtmb.fec import encode_frame_bch_ldpc_codewords
from dtmb.frame_sync import PnFrameTrain
from dtmb.ldpc import dtmb_ldpc_encode_message_bits, dtmb_ldpc_profile
from dtmb.carrier_axes import system_info_positions_for_permutation
from dtmb.frequency import (
    DATA_SYMBOLS_PER_FRAME,
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_POSITIONS,
    frequency_interleave,
)
from dtmb.pn import pn_header_symbols
from dtmb.qam import qam_modulate
from dtmb.receiver import _summarize_system_info, receive_capture
from dtmb.symbol_interleaver import convolutional_interleave
from dtmb.system_info import (
    SYSTEM_INFO_VECTORS,
    classify_system_info,
    system_info_symbols,
)
from dtmb.ts import TS_SYNC_BYTE


def test_receiver_parser_accepts_raw_decision_directed_equalizers():
    parser = receiver.build_parser()

    args = parser.parse_args(["capture.ci8", "--equalizer", "dd-raw"])
    pn_args = parser.parse_args(["capture.ci8", "--equalizer", "pn-dd-raw"])
    data_args = parser.parse_args(["capture.ci8", "--equalizer", "dd-data"])
    pn_data_args = parser.parse_args(["capture.ci8", "--equalizer", "pn-dd-data"])
    guarded = parser.parse_args(
        ["capture.ci8", "--equalizer", "dd", "--dd-max-hard-bit-bias", "0.06"]
    )

    assert args.equalizer == "dd-raw"
    assert pn_args.equalizer == "pn-dd-raw"
    assert data_args.equalizer == "dd-data"
    assert pn_data_args.equalizer == "pn-dd-data"
    assert guarded.dd_max_hard_bit_bias == 0.06


def test_receiver_parser_accepts_wideband_pn_controls():
    args = receiver.build_parser().parse_args(
        [
            "capture.ci8",
            "--pn-estimator",
            "wideband",
            "--pn-wideband-block-frames",
            "32",
        ]
    )

    assert args.pn_estimator == "wideband"
    assert args.pn_wideband_block_frames == 32


def test_receiver_parser_accepts_llr_conditioning_controls():
    parser = receiver.build_parser()

    args = parser.parse_args(
        [
            "capture.ci8",
            "--llr-scale",
            "0.5",
            "--llr-clip",
            "2.0",
            "--llr-erase-fraction",
            "0.01",
            "--llr-plane-scales",
            "1,0.1,1,1,0.1,1",
        ]
    )

    assert args.llr_scale == 0.5
    assert args.llr_clip == 2.0
    assert args.llr_erase_fraction == 0.01
    assert args.llr_plane_scales == "1,0.1,1,1,0.1,1"


def test_receiver_parser_accepts_symbols_only_pre_ldpc_dump():
    args = receiver.build_parser().parse_args(
        [
            "capture.ci8",
            "--dump-pre-ldpc",
            "capture.pre_ldpc.npz",
            "--dump-pre-ldpc-symbols-only",
        ]
    )

    assert args.dump_pre_ldpc.name == "capture.pre_ldpc.npz"
    assert args.dump_pre_ldpc_symbols_only is True


def test_receiver_parser_accepts_csi_demap():
    parser = receiver.build_parser()

    args = parser.parse_args(["capture.ci8", "--equalizer", "pn", "--csi-demap"])

    assert args.csi_demap is True


def test_receive_capture_rejects_csi_demap_without_pn_equalizer(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "requires a PN equalizer"):
        receive_capture(
            tmp_path / "missing.ci8",
            sample_rate_sps=7_560_000,
            equalizer="none",
            csi_demap=True,
        )


def test_receiver_parser_accepts_pn_payload_timing_refine_control():
    parser = receiver.build_parser()

    default = parser.parse_args(["capture.ci8"])
    disabled = parser.parse_args(["capture.ci8", "--no-pn-payload-timing-refine"])

    assert default.pn_payload_timing_refine is True
    assert disabled.pn_payload_timing_refine is False


def test_receiver_parser_accepts_carrier_erase_controls():
    parser = receiver.build_parser()

    args = parser.parse_args(
        [
            "capture.ci8",
            "--carrier-erase-metric",
            "evm",
            "--carrier-erase-fraction",
            "0.01",
            "--carrier-erase-reliability-threshold",
            "0.6",
        ]
    )

    assert args.carrier_erase_metric == "evm"
    assert args.carrier_erase_fraction == 0.01
    assert args.carrier_erase_reliability_threshold == 0.6


def test_receiver_parser_accepts_carrier_permutation_controls():
    parser = receiver.build_parser()

    args = parser.parse_args(
        [
            "capture.ci8",
            "--carrier-permutation",
            "rev_i+rev_j",
            "--logical-position-shift",
            "10",
        ]
    )

    assert args.carrier_permutation == "rev_i+rev_j"
    assert args.logical_position_shift == 10


def test_receiver_main_passes_carrier_permutation(monkeypatch, tmp_path):
    seen = {}

    def fake_receive_capture(*args, **kwargs):
        seen["kwargs"] = kwargs
        return {"ts": {"lock": None}}

    monkeypatch.setattr(receiver, "receive_capture", fake_receive_capture)

    rc = receiver.main(
        [
            "capture.ci8",
            "--carrier-permutation",
            "rev_i+rev_j",
            "--logical-position-shift",
            "10",
            "--json",
            str(tmp_path / "out.json"),
            "--quiet",
        ]
    )

    assert rc == 2
    assert seen["kwargs"]["carrier_permutation"] == "rev_i+rev_j"
    assert seen["kwargs"]["logical_position_shift_symbols"] == 10


def test_logical_from_inserted_c3780_applies_position_shift():
    inserted = np.zeros(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    ).astype(np.complex64)
    shifted_positions = (SYSTEM_INFO_POSITIONS + 2) % FRAME_BODY_SYMBOLS
    inserted[shifted_positions] = reference

    logical = receiver._logical_from_inserted_c3780(
        inserted,
        logical_position_shift_symbols=2,
    )

    np.testing.assert_allclose(logical[:36], reference)


def test_logical_from_inserted_c3780_applies_carrier_permutation():
    inserted = np.zeros(FRAME_BODY_SYMBOLS, dtype=np.complex64)
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[23],
        frame_body_mode="C3780",
    ).astype(np.complex64)
    positions = np.asarray(
        system_info_positions_for_permutation("swap_lm"),
        dtype=np.int32,
    )
    inserted[positions] = reference

    logical = receiver._logical_from_inserted_c3780(
        inserted,
        carrier_permutation="swap_lm",
    )

    np.testing.assert_allclose(logical[:36], reference)


def test_receiver_parser_accepts_branch_gain_controls():
    parser = receiver.build_parser()

    args = parser.parse_args(
        [
            "capture.ci8",
            "--branch-gain-branches",
            "34,41",
            "--branch-gain-reliability-threshold",
            "0.6",
            "--branch-gain-min-symbols",
            "48",
        ]
    )

    assert args.branch_gain_branches == "34,41"
    assert args.branch_gain_reliability_threshold == 0.6
    assert args.branch_gain_min_symbols == 48


def test_receive_capture_data_only_dd_excludes_system_info_pilots(
    tmp_path,
    monkeypatch,
):
    capture_path = tmp_path / "synthetic_dd_data.ci8"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(4), frames=1)
    original_refine = receiver.refine_c3780_spectrum_decision_directed
    seen = []

    def recording_refine(*args, **kwargs):
        seen.append(kwargs.get("include_system_info_pilots"))
        return original_refine(*args, **kwargs)

    monkeypatch.setattr(
        receiver,
        "refine_c3780_spectrum_decision_directed",
        recording_refine,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="dd-data",
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=1,
    )

    assert seen == [False]
    assert diagnostics["equalizer"] == "dd-data"


def test_receive_capture_passes_dd_hard_bit_bias_guard(tmp_path, monkeypatch):
    capture_path = tmp_path / "synthetic_dd_guard.ci8"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(4), frames=1)
    original_refine = receiver.refine_c3780_spectrum_decision_directed
    seen = []

    def recording_refine(*args, **kwargs):
        seen.append(kwargs.get("max_hard_bit_bias"))
        return original_refine(*args, **kwargs)

    monkeypatch.setattr(
        receiver,
        "refine_c3780_spectrum_decision_directed",
        recording_refine,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="dd",
        dd_max_hard_bit_bias=0.06,
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=1,
    )

    assert seen == [0.06]
    assert diagnostics["dd_max_hard_bit_bias"] == 0.06


def test_receive_capture_recovers_uncoded_synthetic_ts(tmp_path):
    capture_path = tmp_path / "synthetic_ts.ci8"
    output_path = tmp_path / "out.ts"
    llr_path = tmp_path / "out.llr.f32"
    packet = _ts_packet(pid=0x0100, continuity_counter=0, unit_start=True)
    ts_bytes = _ts_stream(30)
    _write_uncoded_c3780_capture(capture_path, ts_bytes=ts_bytes, frames=2)

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=5,
        output_path=output_path,
        llr_output_path=llr_path,
    )

    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0
    assert diagnostics["ts"]["packet_count"] >= 29
    assert diagnostics["ts"]["stream"]["valid_packet_count"] >= 29
    assert diagnostics["ts"]["stream"]["continuity_error_count"] == 0
    assert diagnostics["demapped_llrs"] == 3744 * 6 * 2
    assert llr_path.stat().st_size == diagnostics["demapped_llrs"] * 4
    assert output_path.read_bytes().startswith(packet)


def test_receive_capture_applies_llr_conditioning_to_output_and_dump(tmp_path):
    capture_path = tmp_path / "synthetic_llr_conditioning.ci8"
    raw_llr_path = tmp_path / "raw.llr.f32"
    conditioned_llr_path = tmp_path / "conditioned.llr.f32"
    conditioned_dump_path = tmp_path / "conditioned.pre_ldpc.npz"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(8), frames=2)

    receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=1,
        llr_output_path=raw_llr_path,
    )
    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=1,
        llr_output_path=conditioned_llr_path,
        pre_ldpc_dump_path=conditioned_dump_path,
        pre_ldpc_dump_symbols_only=True,
        llr_scale=0.5,
        llr_clip=1.0,
        llr_erase_fraction=0.25,
        llr_plane_scales=(1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    )

    raw_llr = np.fromfile(raw_llr_path, dtype="<f4")
    conditioned_llr = np.fromfile(conditioned_llr_path, dtype="<f4")
    expected_llr = receiver._condition_demapped_llr(
        raw_llr,
        scale=0.5,
        clip=1.0,
        erase_fraction=0.25,
        bits_per_symbol=6,
        plane_scales=(1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(conditioned_llr, expected_llr)
    assert diagnostics["llr_conditioning"]["scale"] == 0.5
    assert diagnostics["llr_conditioning"]["clip"] == 1.0
    assert diagnostics["llr_conditioning"]["erase_fraction"] == 0.25
    assert diagnostics["llr_conditioning"]["plane_scales"] == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert diagnostics["llr_conditioning"]["zero_count"] > 0

    dump_sidecar = json.loads(conditioned_dump_path.with_suffix(".json").read_text())
    assert dump_sidecar["dump_mode"] == "symbols_only"
    assert dump_sidecar["llr_conditioning"]["scale"] == 0.5
    assert dump_sidecar["llr_conditioning"]["clip"] == 1.0
    assert dump_sidecar["llr_conditioning"]["erase_fraction"] == 0.25
    assert dump_sidecar["llr_conditioning"]["plane_scales"] == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    with np.load(conditioned_dump_path, allow_pickle=False) as dump:
        assert "data_symbols_before_symbol_deinterleave" in dump.files
        assert "data_symbols_after_symbol_deinterleave" in dump.files
        assert "data_symbol_frame_indices_before_deinterleave" in dump.files
        assert "frame_start_symbols" in dump.files
        assert "metadata_json" in dump.files
        assert "hard_bits" not in dump.files
        assert "llr" not in dump.files
        assert "frame_indices" not in dump.files
        assert "codeword_indices" not in dump.files
        assert "bit_indices_in_codeword" not in dump.files


def test_condition_demapped_llr_applies_plane_scales_before_clip_and_erase():
    llr = np.asarray(
        [
            4.0,
            3.0,
            2.0,
            1.0,
            -1.0,
            -2.0,
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            1.0,
            2.0,
        ],
        dtype=np.float32,
    )

    conditioned = receiver._condition_demapped_llr(
        llr,
        scale=0.5,
        clip=1.0,
        erase_fraction=0.0,
        bits_per_symbol=6,
        plane_scales=(1.0, 0.0, 0.5, 1.0, 0.0, 0.5),
    )

    np.testing.assert_allclose(
        conditioned,
        np.asarray(
            [1.0, 0.0, 0.5, 0.5, -0.0, -0.5, -1.0, -0.0, -0.5, -0.5, 0.0, 0.5],
            dtype=np.float32,
        ),
    )


def test_carrier_erase_mask_selects_outlier_carrier():
    profile = dtmb_ldpc_profile(3)
    message = np.zeros(profile.message_bits, dtype=np.uint8)
    bits = dtmb_ldpc_encode_message_bits(message, fec_rate_index=3)
    symbols = qam_modulate(bits, mode="64qam")
    frame = np.resize(symbols, DATA_SYMBOLS_PER_FRAME).astype(np.complex64)
    bad_carrier = 17
    frame[bad_carrier] = np.complex64(20 + 20j)

    mask, summary = receiver._carrier_erase_mask(
        frame,
        qam_mode="64qam",
        metric="evm",
        fraction=1.0 / DATA_SYMBOLS_PER_FRAME,
        reliability_threshold=0.55,
    )

    assert int(np.count_nonzero(mask)) == 1
    assert bool(mask[bad_carrier])
    assert summary["evm_rms_max"] > 1.0


def test_branch_gain_correction_recovers_branch_distorted_symbols():
    rng = np.random.default_rng(20260523)
    symbol_count = 52 * 40
    bits = rng.integers(0, 2, size=symbol_count * 6, dtype=np.uint8)
    symbols = qam_modulate(bits, mode="64qam")
    branches = receiver._source_branches_for_post_deinterleaver_symbols(
        symbol_count,
        mode="mode2",
        phase=0,
    )
    distorted = symbols.astype(np.complex64, copy=True)
    distorted[branches == 34] *= np.complex64(0.55 + 0.28j)
    distorted[branches == 41] *= np.complex64(1.18 - 0.31j)

    baseline_bits = receiver.qam_hard_demodulate(
        distorted,
        mode="64qam",
        normalize=True,
    )
    corrected, report = receiver._apply_post_deinterleave_branch_gain_correction(
        distorted,
        qam_mode="64qam",
        interleaver_mode="mode2",
        interleaver_phase=0,
        selected_branches=(34, 41),
        reliability_threshold=0.55,
        min_symbols=16,
    )
    corrected_bits = receiver.qam_hard_demodulate(
        corrected,
        mode="64qam",
        normalize=True,
    )

    baseline_errors = int(np.count_nonzero(baseline_bits != bits))
    corrected_errors = int(np.count_nonzero(corrected_bits != bits))

    assert baseline_errors > 0
    assert corrected_errors < baseline_errors
    assert report["enabled"] is True
    assert report["selected_branches"] == [34, 41]
    assert all(item["applied"] for item in report["corrected_branches"])


def test_receive_capture_can_force_decision_directed_cpe_source(tmp_path, monkeypatch):
    capture_path = tmp_path / "synthetic_cpe_source.ci8"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(4), frames=1)
    calls = []

    class FakeCpe:
        def __init__(self, symbols):
            self.corrected_symbols = np.asarray(symbols, dtype=np.complex64)

        def to_dict(self):
            return {
                "gain_real": 1.0,
                "gain_imag": 0.0,
                "cpe_rad": 0.0,
                "amplitude": 1.0,
                "reliable_symbol_count": 3744,
                "pre_correction_power": 1.0,
                "post_correction_power": 1.0,
            }

    def fake_correct_common_phase_error(symbols, **kwargs):
        calls.append(
            (
                kwargs.get("pilot_symbols") is None,
                kwargs.get("pilot_reference") is None,
            )
        )
        return FakeCpe(symbols)

    monkeypatch.setattr(
        receiver,
        "correct_common_phase_error",
        fake_correct_common_phase_error,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        correct_per_frame_cpe=True,
        per_frame_cpe_source="decision-directed",
        min_ts_packets=1,
    )

    assert calls == [(True, True)]
    assert diagnostics["per_frame_cpe"]["requested_source"] == "decision-directed"
    assert diagnostics["per_frame_cpe"]["source_counts"] == {"decision-directed": 1}


def test_receive_capture_raw_dd_auto_cpe_uses_decision_directed(tmp_path, monkeypatch):
    capture_path = tmp_path / "synthetic_dd_raw_auto_cpe_source.ci8"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(4), frames=1)
    calls = []

    class FakeCpe:
        def __init__(self, symbols):
            self.corrected_symbols = np.asarray(symbols, dtype=np.complex64)

        def to_dict(self):
            return {
                "gain_real": 1.0,
                "gain_imag": 0.0,
                "cpe_rad": 0.0,
                "amplitude": 1.0,
                "reliable_symbol_count": 3744,
                "pre_correction_power": 1.0,
                "post_correction_power": 1.0,
            }

    def fake_correct_common_phase_error(symbols, **kwargs):
        calls.append(
            (
                kwargs.get("pilot_symbols") is None,
                kwargs.get("pilot_reference") is None,
            )
        )
        return FakeCpe(symbols)

    monkeypatch.setattr(
        receiver,
        "correct_common_phase_error",
        fake_correct_common_phase_error,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="dd-raw",
        timing_search=False,
        uncoded_payload=True,
        correct_per_frame_cpe=True,
        min_ts_packets=1,
    )

    assert calls == [(True, True)]
    assert diagnostics["per_frame_cpe"]["requested_source"] == "auto"
    assert diagnostics["per_frame_cpe"]["source_counts"] == {"decision-directed": 1}


def test_receive_capture_can_use_raw_system_info_cpe_source(tmp_path, monkeypatch):
    capture_path = tmp_path / "synthetic_raw_sysinfo_cpe_source.ci8"
    _write_uncoded_c3780_capture(capture_path, ts_bytes=_ts_stream(4), frames=1)
    calls = []

    class FakeCpe:
        def __init__(self, symbols):
            self.corrected_symbols = np.asarray(symbols, dtype=np.complex64)

        def to_dict(self):
            return {
                "gain_real": 1.0,
                "gain_imag": 0.0,
                "cpe_rad": 0.0,
                "amplitude": 1.0,
                "reliable_symbol_count": 36,
                "pre_correction_power": 1.0,
                "post_correction_power": 1.0,
            }

    def fake_correct_common_phase_error(symbols, **kwargs):
        calls.append(
            (
                kwargs.get("pilot_symbols") is not None,
                kwargs.get("pilot_reference") is not None,
            )
        )
        return FakeCpe(symbols)

    monkeypatch.setattr(
        receiver,
        "correct_common_phase_error",
        fake_correct_common_phase_error,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        correct_per_frame_cpe=True,
        per_frame_cpe_source="raw-system-info",
        system_info_index=23,
        min_ts_packets=1,
    )

    assert calls == [(True, True)]
    assert diagnostics["per_frame_cpe"]["requested_source"] == "raw-system-info"
    assert diagnostics["per_frame_cpe"]["source_counts"] == {"raw-system-info": 1}


def test_receive_capture_can_use_raw_polarity_system_info_source(tmp_path):
    capture_path = tmp_path / "synthetic_raw_polarity_source.ci8"
    _write_uncoded_c3780_capture(
        capture_path,
        ts_bytes=_ts_stream(4),
        frames=1,
        gain=-1 + 0j,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        system_info_index=23,
        min_ts_packets=1,
    )

    frame = diagnostics["frames"][0]
    assert frame["system_info_source"] == "raw-polarity"
    assert frame["best_system_info"]["index"] == 23
    assert frame["best_system_info"]["polarity"] == -1
    evidence = {entry["source"]: entry for entry in frame["system_info_evidence"]}
    assert evidence["raw"]["best"]["index"] == 24
    assert evidence["raw-polarity"]["best"]["index"] == 23


def test_phase_only_system_info_gain_discards_amplitude():
    reference = system_info_symbols(SYSTEM_INFO_VECTORS[22], frame_body_mode="C3780")
    phase = np.exp(0.73j)
    observed = (0.17 * phase * reference).astype(np.complex64)

    gain = receiver._phase_only_system_info_gain(reference, observed)

    assert gain is not None
    assert abs(abs(gain) - 1.0) < 1e-6
    assert abs(np.angle(gain) - 0.73) < 1e-6


def test_receive_capture_truncates_output_when_no_ts_lock(tmp_path):
    capture_path = tmp_path / "synthetic_no_ts.ci8"
    output_path = tmp_path / "out.ts"
    output_path.write_bytes(b"stale")
    _write_uncoded_c3780_capture(capture_path, ts_bytes=b"\x00" * 188, frames=1)

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=10_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="64qam",
        equalizer="none",
        timing_search=False,
        uncoded_payload=True,
        min_ts_packets=5,
        output_path=output_path,
    )

    assert diagnostics["ts"]["lock"] is None
    assert diagnostics["ts"]["output_path"] == str(output_path)
    assert output_path.read_bytes() == b""


def test_receive_capture_recovers_bch_descrambled_synthetic_ts(tmp_path):
    capture_path = tmp_path / "synthetic_fec.ts.ci8"
    ts_bytes = _ts_stream(8)
    _write_bch_assume_ldpc_c3780_capture(capture_path, ts_bytes=ts_bytes, frames=2)

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="auto",
        equalizer="none",
        timing_search=False,
        fec_mode="bch-assume-ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["receive_config"]["qam_mode"] == "4qam"
    assert diagnostics["receive_config"]["fec_rate_index"] == 3
    assert diagnostics["system_info"]["selected"]["index"] == 9
    assert diagnostics["fec"]["ldpc_parity_check"]["zero_mismatch_codewords"] == 2
    assert diagnostics["fec"]["ldpc_parity_check"]["mean_mismatch_ratio"] == 0.0
    assert diagnostics["fec"]["frame_reports"][0]["bch_unclean_blocks"] == 0
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0
    assert diagnostics["ts"]["stream"]["valid_packet_count"] == 8
    assert diagnostics["ts"]["stream"]["continuity_error_count"] == 0


def test_receive_capture_auto_acquisition_recovers_c3780_synthetic_ts(tmp_path):
    capture_path = tmp_path / "synthetic_auto_fec.ts.ci8"
    ts_bytes = _ts_stream(8)
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=2,
        leading_zeros=23,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        max_frames=2,
        qam_mode="auto",
        equalizer="none",
        system_info_index=9,
        fec_mode="bch-assume-ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["acquisition"]["selected_train"]["mode"] == "pn945"
    assert diagnostics["acquisition"]["cyclic_phase_offset"] == 23
    assert diagnostics["acquisition"]["phase_offset_source"] == "timing-search"
    assert diagnostics["receive_config"]["qam_mode"] == "4qam"
    assert diagnostics["system_info"]["selected"]["index"] == 9
    assert diagnostics["system_info"]["expected"]["index"] == 9
    assert diagnostics["system_info"]["expected"]["top_count"] == 2
    assert diagnostics["system_info"]["expected"]["top3_count"] == 2
    assert diagnostics["frames"][0]["expected_system_info"]["rank"] == 1
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_uses_timing_trajectory_for_frame_slicing(tmp_path):
    capture_path = tmp_path / "synthetic_trajectory_fec.ts.ci8"
    ts_bytes = _ts_stream(8)
    leading_zeros = 23
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=2,
        leading_zeros=leading_zeros,
    )
    timing_report = {
        "stage": "timing_trajectory",
        "trajectory": {
            "version": 1,
            "mode": "pn945",
            "frame_symbols": 4725,
            "header_symbols": 945,
            "policy": "window_hold",
            "source": "continuous",
            "available": True,
            "reason": None,
            "phase_offsets": [leading_zeros],
            "segments": [
                {
                    "start_frame_index": 0,
                    "end_frame_index": 3,
                    "phase_offset": leading_zeros,
                    "coarse_cfo_hz": None,
                    "source": "continuous",
                    "candidate_rank": 1,
                }
            ],
        },
    }

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=leading_zeros,
        max_frames=2,
        qam_mode="auto",
        equalizer="none",
        system_info_index=9,
        timing_search=False,
        timing_policy="trajectory",
        timing_trajectory=timing_report,
        fec_mode="bch-assume-ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["timing_policy"] == "trajectory"
    assert diagnostics["timing_trajectory"]["available"] is True
    assert diagnostics["frames"][0]["start"] == leading_zeros
    assert diagnostics["frames"][0]["timing"]["policy"] == "trajectory"
    assert diagnostics["frames"][0]["timing"]["source"] == "continuous"
    assert diagnostics["receive_config"]["qam_mode"] == "4qam"
    assert diagnostics["system_info"]["selected"]["index"] == 9
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_pn_equalizer_succeeds_with_multipath_tail_cancel(tmp_path):
    capture_path = tmp_path / "synthetic_multipath_fec.ts.ci8"
    ts_bytes = _ts_stream(12)
    channel = np.zeros(33, dtype=np.complex64)
    channel[0] = 0.25 + 0.1j
    channel[7] = 0.5 - 0.4j
    channel[31] = 0.12 + 0.18j
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=3,
        channel=channel,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="auto",
        equalizer="pn",
        system_info_index=9,
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    # With proper PN tail cancellation and circular restore, the PN equalizer
    # correctly reconstructs the system-info vector and recovers the payload.
    assert diagnostics["frames"][0]["system_info_source"] == "pn-channel"
    assert diagnostics["frames"][0]["pn_channel"]["tap_count"] >= channel.size
    assert diagnostics["frames"][0]["pn_channel"]["response_fft_bin_shift"] == 0
    assert diagnostics["frames"][0]["system_info_iq_source"] == "post-equalizer"
    assert "pn-channel" in diagnostics["frames"][0]["system_info_iq_by_source"]
    assert diagnostics["receive_config"]["qam_mode"] == "4qam"
    assert diagnostics["system_info"]["selected"]["index"] == 9
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 1
    assert diagnostics["fec"]["frame_reports"][1]["ldpc_converged_codewords"] == 1
    assert diagnostics["fec"]["ldpc_parity_check"]["zero_mismatch_codewords"] == 2
    assert diagnostics["fec"]["ldpc_parity_check"]["mean_mismatch_ratio"] == 0.0
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0
    assert diagnostics["ts"]["stream"]["valid_packet_count"] == 8


def test_receive_capture_wideband_pn_equalizer_recovers_far_echo(tmp_path):
    capture_path = tmp_path / "synthetic_wideband_multipath_fec.ts.ci8"
    ts_bytes = _ts_stream(12)
    channel = np.zeros(96, dtype=np.complex64)
    channel[0] = 1.0
    channel[7] = np.complex64(0.5 * np.exp(0.7j))
    channel[80] = np.complex64(0.7 * np.exp(-0.2j))
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=3,
        channel=channel,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="auto",
        equalizer="pn",
        pn_estimator="wideband",
        system_info_index=9,
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["wideband_channel"]["span_symbols"] > 80
    assert diagnostics["frames"][0]["pn_channel"]["estimator"] == "wideband"
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 1
    assert diagnostics["fec"]["frame_reports"][1]["ldpc_converged_codewords"] == 1
    assert diagnostics["fec"]["frame_reports"][0]["bch_unclean_blocks"] == 0
    assert diagnostics["fec"]["frame_reports"][1]["bch_unclean_blocks"] == 0
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_pn_equalizer_supports_body_window_offset(tmp_path):
    capture_path = tmp_path / "synthetic_multipath_offset_fec.ts.ci8"
    ts_bytes = _ts_stream(12)
    channel = np.zeros(33, dtype=np.complex64)
    channel[0] = 0.25 + 0.1j
    channel[7] = 0.5 - 0.4j
    channel[31] = 0.12 + 0.18j
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=3,
        channel=channel,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="auto",
        equalizer="pn",
        system_info_index=9,
        body_window_offset_symbols=3,
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["frames"][0]["system_info_source"] == "pn-channel"
    assert (
        diagnostics["frames"][0]["pn_channel"][
            "response_body_window_offset_symbols"
        ]
        == 3
    )
    assert diagnostics["fec"]["ldpc_parity_check"]["mean_mismatch_ratio"] == 0.0
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_pn_equalizer_reports_zero_delay_pn_phase(tmp_path):
    capture_path = tmp_path / "synthetic_pn_phase_source.ts.ci8"
    _write_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=_ts_stream(8),
        frames=2,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=1,
        qam_mode="4qam",
        equalizer="pn",
        system_info_index=9,
        timing_search=False,
    )

    pn_report = diagnostics["frames"][0]["pn_channel"]
    assert pn_report["metric"] > 0.99
    assert pn_report["zero_delay_pn_phase"] == 0
    assert pn_report["pn_phase_delta_from_zero_delay"] == 0


def test_receive_capture_keeps_raw_system_info_when_pn_equalization_worsens(tmp_path):
    capture_path = tmp_path / "synthetic_bad_header_channel.ts.ci8"
    _write_header_mismatch_c3780_capture(capture_path, frames=2)

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="64qam",
        equalizer="pn",
        system_info_index=9,
        timing_search=False,
    )

    assert diagnostics["frames"][0]["system_info_source"] == "raw"
    assert diagnostics["frames"][0]["best_system_info"]["index"] == 9
    assert diagnostics["frames"][0]["expected_system_info"]["rank"] == 1


def test_system_info_summary_does_not_auto_select_reserved_vectors():
    reserved = classify_system_info(
        system_info_symbols(SYSTEM_INFO_VECTORS[25], frame_body_mode="C3780")
    )[0].to_dict()
    frame_reports = [{"best_system_info": reserved} for _ in range(3)]

    summary = _summarize_system_info(
        frame_reports,
        min_metric=0.75,
        min_agreement=0.5,
    )

    assert summary["qualified_frame_count"] == 3
    assert summary["auto_eligible_frame_count"] == 0
    assert summary["selected"] is None
    assert summary["top_counts"] == []
    assert summary["top_counts_all"] == [
        {"frame_body_mode": "C3780", "index": 25, "count": 3}
    ]


def test_system_info_summary_reports_expected_low_metric_consensus():
    expected = classify_system_info(
        system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")
    )[0].to_dict()
    expected["metric"] = 0.5
    frame_reports = [
        {
            "best_system_info": expected,
            "expected_system_info": {
                "index": 23,
                "frame_body_mode": "C3780",
                "rank": 1,
                "top": True,
                "top3": True,
                "metric": 0.5,
                "match": expected,
            },
        }
        for _ in range(4)
    ]

    summary = _summarize_system_info(
        frame_reports,
        min_metric=0.75,
        min_agreement=0.5,
        expected_index=23,
        expected_frame_body_mode="C3780",
    )

    assert summary["selected"] is None
    assert summary["qualified_frame_count"] == 0
    assert summary["expected"]["top_count"] == 4
    assert summary["expected"]["top3_count"] == 4
    assert summary["expected"]["qualified_frame_count"] == 0
    assert summary["expected"]["median_metric"] == 0.5


def test_system_info_summary_separates_trained_from_independent_evidence():
    expected = classify_system_info(
        system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")
    )[0].to_dict()
    frame_reports = [
        {
            "best_system_info": expected,
            "system_info_source": "post-equalizer",
            "system_info_independent": False,
            "expected_system_info": {
                "index": 23,
                "frame_body_mode": "C3780",
                "rank": 1,
                "top": True,
                "top3": True,
                "metric": expected["metric"],
                "match": expected,
            },
        }
        for _ in range(3)
    ]

    summary = _summarize_system_info(
        frame_reports,
        min_metric=0.75,
        min_agreement=0.5,
        expected_index=23,
        expected_frame_body_mode="C3780",
    )

    assert summary["selected"]["index"] == 23
    assert summary["trained_or_circular_frame_count"] == 3
    assert summary["independent_frame_count"] == 0
    assert summary["independent"]["selected"] is None
    assert summary["independent"]["qualified_frame_count"] == 0


def test_system_info_summary_includes_supported_raw_oracle():
    expected = classify_system_info(
        system_info_symbols(SYSTEM_INFO_VECTORS[23], frame_body_mode="C3780")
    )[0].to_dict()
    frame_reports = [
        {
            "best_system_info": expected,
            "system_info_source": "raw",
            "system_info_independent": True,
            "expected_system_info": {
                "index": 23,
                "frame_body_mode": "C3780",
                "rank": 1,
                "top": True,
                "top3": True,
                "metric": expected["metric"],
                "match": expected,
            },
            "supported_system_info_oracle": {
                "best": {
                    "index": 21,
                    "frame_body_mode": "C3780",
                    "bit_errors": 10,
                    "evm_rms": 2.5,
                    "gain_phase_rad": 0.1,
                    "parameters": {
                        "index": 21,
                        "frame_body_mode": "C3780",
                        "qam_mode": "64qam",
                        "qam_label": "64QAM",
                        "fec_rate_index": 2,
                        "interleaver_mode": "mode1",
                        "nr_mapping": False,
                        "supported_by_receiver": True,
                    },
                },
                "top3": [
                    {
                        "index": 21,
                        "frame_body_mode": "C3780",
                        "bit_errors": 10,
                        "evm_rms": 2.5,
                        "gain_phase_rad": 0.1,
                        "parameters": {
                            "index": 21,
                            "frame_body_mode": "C3780",
                            "qam_mode": "64qam",
                            "qam_label": "64QAM",
                            "fec_rate_index": 2,
                            "interleaver_mode": "mode1",
                            "nr_mapping": False,
                            "supported_by_receiver": True,
                        },
                    },
                    {
                        "index": 23,
                        "frame_body_mode": "C3780",
                        "bit_errors": 12,
                        "evm_rms": 3.0,
                        "gain_phase_rad": 0.2,
                        "parameters": {
                            "index": 23,
                            "frame_body_mode": "C3780",
                            "qam_mode": "64qam",
                            "qam_label": "64QAM",
                            "fec_rate_index": 3,
                            "interleaver_mode": "mode1",
                            "nr_mapping": False,
                            "supported_by_receiver": True,
                        },
                    },
                ],
            },
        }
        for _ in range(3)
    ]

    summary = _summarize_system_info(
        frame_reports,
        min_metric=0.75,
        min_agreement=0.5,
        expected_index=23,
        expected_frame_body_mode="C3780",
    )

    supported = summary["supported_raw_oracle"]
    assert supported["best_supported_index"] == 21
    assert supported["best_choice_counts"] == [{"index": 21, "count": 3}]
    assert supported["candidate_summary"][0]["index"] == 21


def test_select_phase_keeps_prior_cfo_when_timing_candidate_has_none(monkeypatch):
    calls = {}

    def fake_estimate_cfo(symbols, *, mode, phase_offset, symbol_rate_sps):
        del symbols, mode, symbol_rate_sps
        return 123.0 if phase_offset == 10 else None

    def fake_search_frame_timing(symbols, **kwargs):
        del symbols
        calls.update(kwargs)
        return {
            "center_phase_offset": kwargs["center_phase_offset"],
            "per_candidate_cfo": kwargs["per_candidate_cfo"],
            "candidate_count": 1,
            "selected": {
                "phase_offset": 11,
                "coarse_cfo_hz": None,
                "score": 1.0,
            },
            "candidates": [],
        }

    monkeypatch.setattr(
        receiver,
        "estimate_cfo_from_pn_cyclic_extension",
        fake_estimate_cfo,
    )
    monkeypatch.setattr(receiver, "search_frame_timing", fake_search_frame_timing)

    acquisition = receiver._select_phase_and_cfo(
        np.zeros(128, dtype=np.complex64),
        mode="pn945",
        phase_offset=10,
        symbol_rate_sps=7_560_000,
        max_frames=1,
        use_family_delay=True,
        family_frames=1,
        max_delay_symbols=16,
        family_hit_threshold=0.45,
        timing_search=True,
        timing_radius_symbols=4,
        timing_step_symbols=1,
        timing_jobs=1,
        timing_per_candidate_cfo=True,
        expected_system_info_index=23,
        expected_frame_body_mode="C3780",
    )

    assert calls["per_candidate_cfo"] is True
    assert acquisition["phase_offset"] == 11
    assert acquisition["phase_offset_source"] == "timing-search"
    assert acquisition["coarse_cfo_hz"] == 123.0
    assert acquisition["cyclic_cfo_hz"] == 123.0
    assert acquisition["timing_cfo_hz"] is None
    assert acquisition["active_cfo_source"] == "cyclic"
    assert acquisition["timing_search"]["per_candidate_cfo"] is True


def test_select_phase_refines_automatic_neighbor_with_pn_payload(monkeypatch):
    train = PnFrameTrain(
        mode="pn945",
        phase_offset=10,
        frame_symbols=4725,
        header_symbols=945,
        mean_metric=0.99,
        max_metric=0.99,
        hit_count=3,
        observed_frames=3,
    )
    calls = {}

    monkeypatch.setattr(
        receiver,
        "detect_pn_cyclic_extension_trains",
        lambda symbols, modes: [train],
    )
    monkeypatch.setattr(
        receiver,
        "estimate_cfo_from_pn_cyclic_extension",
        lambda *args, **kwargs: None,
    )

    def fake_payload_timing(symbols, **kwargs):
        del symbols
        calls.update(kwargs)
        return {
            "selected": {"phase_offset": 11, "grid_evm_rms": 0.1},
            "candidates": [],
        }

    monkeypatch.setattr(receiver, "search_pn_payload_timing", fake_payload_timing)

    acquisition = receiver._select_phase_and_cfo(
        np.zeros(20_000, dtype=np.complex64),
        mode="pn945",
        phase_offset=None,
        symbol_rate_sps=7_560_000,
        max_frames=24,
        use_family_delay=False,
        family_frames=1,
        max_delay_symbols=16,
        family_hit_threshold=0.45,
        timing_search=False,
        timing_radius_symbols=4,
        timing_step_symbols=1,
        timing_jobs=1,
        timing_per_candidate_cfo=False,
        expected_system_info_index=22,
        expected_frame_body_mode="C3780",
        pn_payload_timing_refine=True,
        pn_payload_qam_mode="64qam",
    )

    assert calls["phase_offsets"] == (8, 9, 10, 11, 12)
    assert acquisition["phase_offset"] == 11
    assert acquisition["phase_offset_source"] == "pn-payload-timing"
    assert acquisition["pn_payload_timing"]["selected"]["phase_offset"] == 11


def test_per_frame_cpe_summary_unwraps_phase_steps():
    frame_reports = [
        {
            "data_symbols": 100,
            "per_frame_cpe_correction": {
                "cpe_rad": 3.13,
                "amplitude": 1.0,
                "reliable_symbol_count": 80,
                "source": "system-info-pilot",
            },
        },
        {
            "data_symbols": 100,
            "per_frame_cpe_correction": {
                "cpe_rad": -3.12,
                "amplitude": 1.1,
                "reliable_symbol_count": 90,
                "source": "system-info-pilot",
            },
        },
        {
            "data_symbols": 100,
            "per_frame_cpe_correction": {
                "cpe_rad": -3.08,
                "amplitude": 0.9,
                "reliable_symbol_count": 70,
                "source": "decision-directed",
            },
        },
    ]

    summary = receiver._summarize_per_frame_cpe(
        frame_reports,
        enabled=True,
        max_relative_error=0.7,
        requested_source="auto",
    )

    assert summary["frames"] == 3
    assert summary["requested_source"] == "auto"
    assert summary["max_abs_adjacent_cpe_step_rad"] < 0.05
    assert summary["median_abs_adjacent_cpe_step_rad"] < 0.05
    assert summary["linear_cpe_residual_rms_rad"] < 0.01
    assert summary["source_counts"] == {
        "system-info-pilot": 2,
        "decision-directed": 1,
    }


def test_receive_capture_recovers_c1_synthetic_ts(tmp_path):
    capture_path = tmp_path / "synthetic_c1_fec.ts.ci8"
    ts_bytes = _ts_stream(8)
    _write_bch_assume_ldpc_c1_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=2,
        gain=0.35 + 0.8j,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=20_000,
        mode="pn945",
        phase_offset=0,
        max_frames=2,
        qam_mode="auto",
        equalizer="sparse",
        frame_body_mode="C1",
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["frame_body_mode"] == "C1"
    assert diagnostics["receive_config"]["qam_mode"] == "4qam"
    assert diagnostics["receive_config"]["fec_rate_index"] == 3
    assert diagnostics["system_info"]["selected"]["frame_body_mode"] == "C1"
    assert diagnostics["system_info"]["independent"]["selected"] is None
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 1
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_discards_symbol_deinterleaver_latency(tmp_path):
    capture_path = tmp_path / "synthetic_interleaved_fec.ts.ci8"
    frames = 171
    ts_bytes = _ts_stream(frames * 4)
    _write_interleaved_bch_assume_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=frames,
        symbol_interleave="mode1",
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=900_000,
        mode="pn945",
        phase_offset=0,
        max_frames=frames,
        qam_mode="auto",
        equalizer="none",
        symbol_deinterleave="mode1",
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=3,
    )

    assert diagnostics["fec"]["ldpc_decode"] == "enabled"
    assert diagnostics["symbol_deinterleave_latency_symbols"] == 3744 * 170
    assert diagnostics["symbol_deinterleave_discarded_symbols"] == 3744 * 170
    assert diagnostics["data_symbols_after_symbol_deinterleave"] == 3744
    assert diagnostics["demapped_bits"] == 3744 * 2
    assert diagnostics["fec"]["ldpc_parity_check"]["zero_mismatch_codewords"] == 1
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 1
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0


def test_receive_capture_recovers_64qam_interleaved_ldpc_synthetic_ts(tmp_path):
    capture_path = tmp_path / "synthetic_64qam_interleaved_fec.ts.ci8"
    frames = 171
    ts_bytes = _ts_stream(12)
    _write_interleaved_64qam_bch_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=frames,
        symbol_interleave="mode1",
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=900_000,
        mode="pn945",
        phase_offset=0,
        max_frames=frames,
        qam_mode="auto",
        equalizer="none",
        symbol_deinterleave="mode1",
        timing_search=False,
        scan_symbol_deinterleave_phases=True,
        scan_symbol_deinterleave_codewords=1,
        fec_mode="ldpc",
        fec_rate_index="auto",
        min_ts_packets=8,
    )

    assert diagnostics["receive_config"]["qam_mode"] == "64qam"
    assert diagnostics["receive_config"]["fec_rate_index"] == 3
    assert diagnostics["symbol_deinterleave_latency_symbols"] == 3744 * 170
    assert diagnostics["symbol_deinterleave_discarded_symbols"] == 3744 * 170
    assert diagnostics["symbol_deinterleave_phase_scan"]["best_phase"] == 0
    assert (
        diagnostics["symbol_deinterleave_phase_scan"]["best"]["zero_mismatch_codewords"]
        == 1
    )
    assert (
        diagnostics["symbol_deinterleave_phase_scan"]["best"]["mean_mismatch_ratio"]
        == 0.0
    )
    assert diagnostics["data_symbols_after_symbol_deinterleave"] == 3744
    assert (
        diagnostics["qam_symbol_quality"]["after_symbol_deinterleave"]["symbol_count"]
        == 3744
    )
    assert "axis_level_occupancy" in diagnostics["qam_symbol_quality"][
        "after_symbol_deinterleave"
    ]
    assert diagnostics["demapped_bits"] == 3744 * 6
    assert diagnostics["fec"]["ldpc_decode"] == "enabled"
    assert diagnostics["fec"]["ldpc_parity_check"]["zero_mismatch_codewords"] == 3
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_codewords"] == 3
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 3
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_syndrome_weights"] == [0, 0, 0]
    assert diagnostics["fec"]["frame_reports"][0]["bch_unclean_blocks"] == 0
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0
    assert diagnostics["ts"]["stream"]["valid_packet_count"] == 12
    assert diagnostics["ts"]["stream"]["continuity_error_count"] == 0


def test_receive_capture_pn_equalizer_recovers_64qam_ts_on_multipath(tmp_path):
    """End-to-end Gate B->F: PN equalization recovers 64QAM TS on a
    frequency-selective channel where pilot/DD equalizers fail.

    The channel below has a strong delay-3 echo (inside the 36 scattered
    system-information pilot spacing) plus echoes at delay 9 and 30, which
    biases pilot/DD channel estimates ~50% (see AGENTS.md 2026-05-29 and
    tests/test_equalizer_bit_path_multipath.py). The PN-header equalizer with
    the compact phase selector recovers the exact transport stream. This is
    the fixture that guards the real-capture PN-equalizer win.
    """

    capture_path = tmp_path / "synthetic_64qam_multipath_fec.ts.ci8"
    frames = 171
    ts_bytes = _ts_stream(12)
    channel = np.zeros(64, dtype=np.complex64)
    channel[0] = 1.0
    channel[3] = 0.6
    channel[9] = 0.45 - 0.15j
    channel[30] = 0.25 + 0.20j
    _write_interleaved_64qam_bch_ldpc_c3780_capture(
        capture_path,
        ts_bytes=ts_bytes,
        frames=frames,
        symbol_interleave="mode1",
        channel=channel,
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=900_000,
        mode="pn945",
        phase_offset=0,
        max_frames=frames,
        qam_mode="64qam",
        equalizer="pn",
        symbol_deinterleave="mode1",
        timing_search=False,
        scan_symbol_deinterleave_phases=True,
        scan_symbol_deinterleave_codewords=1,
        fec_mode="ldpc",
        fec_rate_index=3,
        system_info_index=23,
        correct_per_frame_cpe=True,
        min_ts_packets=8,
    )

    # The PN equalizer must reconstruct the system-information vector and the
    # exact transport stream through the full FEC chain on this multipath
    # channel.
    assert diagnostics["frames"][0]["system_info_source"] == "pn-channel"
    assert diagnostics["fec"]["frame_reports"][0]["ldpc_converged_codewords"] == 3
    assert diagnostics["fec"]["frame_reports"][0]["bch_unclean_blocks"] == 0
    assert diagnostics["ts"]["lock"]["packet_size"] == 188
    assert diagnostics["ts"]["lock"]["sync_ratio"] == 1.0
    assert diagnostics["ts"]["stream"]["valid_packet_count"] == 12
    assert diagnostics["ts"]["stream"]["continuity_error_count"] == 0


def test_receive_capture_pn_equalizer_skips_ambiguous_per_frame_cpe(
    tmp_path, monkeypatch
):
    """PN equalization already locks absolute phase via the PN-header channel
    estimate, so the pi/2-ambiguous blind/decision-directed CPE must be
    suppressed for PN equalizers. Applying it re-aliases each frame's phase by
    a 90deg quadrant and, across mode2's 510-frame deinterleaver span, produces
    the 0.50 LDPC parity plateau (see scripts/probe_mode2_phase_strategy.py:
    mode2 multipath+5Hz-CFO decodes byte-exact without CPE, plateaus with it)."""

    capture_path = tmp_path / "synthetic_pn_skip_cpe.ci8"
    _write_interleaved_64qam_bch_ldpc_c3780_capture(
        capture_path,
        ts_bytes=_ts_stream(12),
        frames=3,
        symbol_interleave="mode1",
    )

    calls = []

    def fake_correct_common_phase_error(symbols, **kwargs):  # pragma: no cover
        calls.append(kwargs)
        return receiver.correct_common_phase_error  # never reached in assertion

    monkeypatch.setattr(
        receiver, "correct_common_phase_error", fake_correct_common_phase_error
    )

    diagnostics = receive_capture(
        capture_path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=200_000,
        mode="pn945",
        phase_offset=0,
        max_frames=3,
        qam_mode="64qam",
        equalizer="pn",
        symbol_deinterleave="mode1",
        timing_search=False,
        fec_mode="ldpc",
        fec_rate_index=3,
        system_info_index=23,
        correct_per_frame_cpe=True,
        min_ts_packets=1,
    )

    assert calls == []
    assert diagnostics["per_frame_cpe"]["enabled"] is True
    assert diagnostics["per_frame_cpe"]["source_counts"] == {
        "skipped-pn-absolute-phase-lock": 3
    }


def test_phase_scan_sort_key_prefers_appendix_b_before_mismatch():
    weaker_mismatch_better_syndrome = {
        "phase": 17,
        "mean_mismatch_ratio": 0.49,
        "min_mismatch_ratio": 0.47,
        "appendix_b_mean_syndrome_ratio": 0.30,
        "appendix_b_min_syndrome_ratio": 0.28,
    }
    stronger_mismatch_worse_syndrome = {
        "phase": 3,
        "mean_mismatch_ratio": 0.40,
        "min_mismatch_ratio": 0.39,
        "appendix_b_mean_syndrome_ratio": 0.45,
        "appendix_b_min_syndrome_ratio": 0.44,
    }

    assert receiver._phase_scan_sort_key(weaker_mismatch_better_syndrome) < (
        receiver._phase_scan_sort_key(stronger_mismatch_worse_syndrome)
    )


def _ts_stream(packet_count: int, *, pid: int = 0x0100) -> bytes:
    return b"".join(
        _ts_packet(
            pid=pid,
            continuity_counter=index & 0x0F,
            unit_start=index == 0,
        )
        for index in range(packet_count)
    )


def _ts_packet(
    *,
    pid: int,
    continuity_counter: int,
    unit_start: bool = False,
) -> bytes:
    header = bytes(
        (
            TS_SYNC_BYTE,
            (0x40 if unit_start else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0x10 | (continuity_counter & 0x0F),
        )
    )
    payload = bytes((pid + continuity_counter + index) & 0xFF for index in range(184))
    return header + payload


def _write_uncoded_c3780_capture(
    path,
    *,
    ts_bytes: bytes,
    frames: int,
    gain: complex = 1 + 0j,
) -> None:
    bits = unpack_bytes_msb(ts_bytes)
    bits_per_frame = 3744 * 6
    needed = bits_per_frame * frames
    if bits.size < needed:
        repeats = int(np.ceil(needed / bits.size))
        bits = np.tile(bits, repeats)
    bits = bits[:needed]
    frame_samples = []
    for frame_index in range(frames):
        chunk = bits[frame_index * bits_per_frame : (frame_index + 1) * bits_per_frame]
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(
            "00010001010111100100011101000011",
            frame_body_mode="C3780",
        )
        logical[36:] = qam_modulate(chunk, mode="64qam")
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        frame = np.concatenate((0.25 * pn_header_symbols("pn945"), body))
        frame = np.asarray(gain, dtype=np.complex64) * frame
        frame_samples.append(frame)
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_interleaved_bch_assume_ldpc_c3780_capture(
    path,
    *,
    ts_bytes: bytes,
    frames: int,
    symbol_interleave: str,
) -> None:
    if len(ts_bytes) != frames * 752:
        raise ValueError("rate-3 4QAM synthetic frames need 4 TS packets per frame")
    data_chunks = []
    for frame_index in range(frames):
        chunk = ts_bytes[frame_index * 752 : (frame_index + 1) * 752]
        codeword = encode_frame_bch_ldpc_codewords(chunk, fec_rate_index=3)
        data_chunks.append(qam_modulate(codeword, mode="4qam"))
    data_stream = convolutional_interleave(
        np.concatenate(data_chunks),
        mode=symbol_interleave,
    )
    frame_samples = []
    for frame_index in range(frames):
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(
            "01001011111110110001110111100110",
            frame_body_mode="C3780",
        )
        logical[36:] = data_stream[frame_index * 3744 : (frame_index + 1) * 3744]
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        frame = np.concatenate((0.25 * pn_header_symbols("pn945"), body))
        frame_samples.append(frame)
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_interleaved_64qam_bch_ldpc_c3780_capture(
    path,
    *,
    ts_bytes: bytes,
    frames: int,
    symbol_interleave: str,
    channel: np.ndarray | None = None,
) -> None:
    bytes_per_codeword = 752
    codewords_per_frame = 3
    frame_payload_bytes = bytes_per_codeword * codewords_per_frame
    if len(ts_bytes) != frame_payload_bytes:
        raise ValueError("rate-3 64QAM synthetic fixture needs one 12-packet frame")
    data_chunks = [
        qam_modulate(
            encode_frame_bch_ldpc_codewords(ts_bytes, fec_rate_index=3),
            mode="64qam",
        )
    ]
    data_chunks.extend(
        np.zeros(3744, dtype=np.complex64)
        for _frame_index in range(frames - 1)
    )
    data_stream = convolutional_interleave(
        np.concatenate(data_chunks),
        mode=symbol_interleave,
    )
    frame_samples = []
    for frame_index in range(frames):
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(
            SYSTEM_INFO_VECTORS[23],
            frame_body_mode="C3780",
        )
        logical[36:] = data_stream[frame_index * 3744 : (frame_index + 1) * 3744]
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        frame = np.concatenate((0.25 * pn_header_symbols("pn945"), body))
        frame_samples.append(frame)
    capture = np.concatenate(frame_samples)
    if channel is not None:
        capture = np.convolve(capture, channel)[: capture.size].astype(np.complex64)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_bch_assume_ldpc_c1_capture(
    path,
    *,
    ts_bytes: bytes,
    frames: int,
    gain: complex = 1 + 0j,
) -> None:
    if len(ts_bytes) != frames * 752:
        raise ValueError("rate-3 4QAM synthetic frames need 4 TS packets per frame")
    frame_samples = []
    for frame_index in range(frames):
        chunk = ts_bytes[frame_index * 752 : (frame_index + 1) * 752]
        codeword = encode_frame_bch_ldpc_codewords(chunk, fec_rate_index=3)
        body = np.empty(3780, dtype=np.complex64)
        body[:36] = system_info_symbols(
            "01001011111110110001110111100110",
            frame_body_mode="C1",
        )
        body[36:] = qam_modulate(codeword, mode="4qam")
        frame = np.concatenate((0.25 * pn_header_symbols("pn945"), body))
        frame = np.asarray(gain, dtype=np.complex64) * frame
        frame_samples.append(frame)
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_header_mismatch_c3780_capture(path, *, frames: int) -> None:
    frame_samples = []
    header_channel = np.zeros(64, dtype=np.complex64)
    header_channel[0] = 1.0 + 0j
    header_channel[11] = 0.7 - 0.4j
    header_channel[39] = -0.3 + 0.2j
    rng = np.random.default_rng(1234)
    for _frame_index in range(frames):
        data_bits = rng.integers(0, 2, 3744 * 2, dtype=np.uint8)
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(
            "01001011111110110001110111100110",
            frame_body_mode="C3780",
        )
        logical[36:] = qam_modulate(data_bits, mode="4qam")
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        header = 0.05 * pn_header_symbols("pn945")
        header = np.convolve(header, header_channel)[: header.size].astype(np.complex64)
        frame_samples.append(np.concatenate((header, body)))
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_bch_assume_ldpc_c3780_capture(
    path,
    *,
    ts_bytes: bytes,
    frames: int,
    leading_zeros: int = 0,
    channel: np.ndarray | None = None,
) -> None:
    if len(ts_bytes) != frames * 752:
        raise ValueError("rate-3 4QAM synthetic frames need 4 TS packets per frame")
    frame_samples = []
    if leading_zeros:
        frame_samples.append(np.zeros(leading_zeros, dtype=np.complex64))
    for frame_index in range(frames):
        chunk = ts_bytes[frame_index * 752 : (frame_index + 1) * 752]
        codeword = encode_frame_bch_ldpc_codewords(chunk, fec_rate_index=3)
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(
            "01001011111110110001110111100110",
            frame_body_mode="C3780",
        )
        logical[36:] = qam_modulate(codeword, mode="4qam")
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        frame = np.concatenate((0.25 * pn_header_symbols("pn945"), body))
        frame_samples.append(frame)
    capture = np.concatenate(frame_samples)
    if channel is not None:
        capture = np.convolve(capture, channel)[:capture.size].astype(np.complex64)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())
