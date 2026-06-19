"""Offline DTMB receive-path diagnostics through transport-stream alignment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .bits import pack_bits_msb
from .carrier_axes import (
    apply_carrier_permutation_to_inserted,
    system_info_positions_for_extraction,
)
from .channel import (
    apply_body_window_offset_to_pn_channel_response,
    build_wideband_pn_channel_model,
    canonicalize_pn_channel_estimate,
    detect_pn_phase,
    equalize_spectrum_with_channel,
    estimate_pn_channel,
    estimate_pn_channel_compact,
    estimate_frequency_domain_noise_variance,
    pn_restore_circular_body_window,
    shift_pn_channel_response,
    wideband_pn_channel_estimate_for_header,
)
from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .equalization import (
    equalize_c3780_spectrum_with_system_info_pilots,
    equalize_with_system_info_pilots,
    refine_c3780_spectrum_decision_directed,
    correct_common_phase_error,
)
from .fec import (
    decode_frame_bch_descramble_from_ldpc_codewords,
    decode_frame_bch_descramble_from_ldpc_llr,
)
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    delay_corrected_phase_offset,
    detect_pn_cyclic_extension_trains,
    estimate_cfo_from_pn_cyclic_extension,
    estimate_residual_cfo_from_frame_headers,
    score_pn_family_delay_train,
    should_apply_pn_family_delay,
)
from .frames import SignalFrame, iter_signal_frames
from .frequency import (
    DATA_SYMBOLS_PER_FRAME,
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_POSITIONS,
    frame_body_fft,
    frequency_deinterleave_inserted,
    frequency_deinterleave_index_map,
    frequency_interleave_index_map,
    split_system_info_and_data,
)
from .ldpc import (
    dtmb_ldpc_appendix_b_syndrome_counts,
    dtmb_ldpc_parity_mismatch_counts,
    dtmb_ldpc_profile,
)
from .pn import PN_DEFINITIONS, PnMode
from .qam import (
    QamMode,
    normalize_qam_symbols,
    qam_hard_demodulate,
    qam_nearest,
    qam_soft_demodulate,
    qam_symbol_quality,
)
from .system_info import (
    SYSTEM_INFO_VECTORS,
    classify_system_info,
    system_info_symbols,
    transmission_parameters_for_index,
)
from .symbol_interleaver import (
    SYMBOL_INTERLEAVERS,
    InterleaverMode,
    convolutional_deinterleave,
    convolutional_deinterleave_source_indices,
)
from .timing import search_frame_timing, search_pn_payload_timing
from .ts import analyze_ts_lock_candidates, extract_ts_packets


EQUALIZER_CHOICES = (
    "none",
    "sparse",
    "pn",
    "pn-sparse",
    "dd",
    "pn-dd",
    "dd-data",
    "pn-dd-data",
    "dd-raw",
    "pn-dd-raw",
)
PN_EQUALIZERS = ("pn", "pn-sparse", "pn-dd", "pn-dd-data", "pn-dd-raw")
SPARSE_PILOT_EQUALIZERS = ("sparse", "pn-sparse")
DECISION_DIRECTED_EQUALIZERS = (
    "dd",
    "pn-dd",
    "dd-data",
    "pn-dd-data",
    "dd-raw",
    "pn-dd-raw",
)
TRAINED_SYSTEM_INFO_EQUALIZERS = SPARSE_PILOT_EQUALIZERS + DECISION_DIRECTED_EQUALIZERS
RAW_POLARITY_WIDE_MAX_COMMON_PHASE_RADIANS = 1.2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-receive",
        description=(
            "Run offline DTMB frame extraction through QAM demapping and TS alignment. "
            "The default uncoded mode is a diagnostic path; real DTMB still requires full LDPC decode."
        ),
    )
    parser.add_argument(
        "capture",
        type=str,
        help="Path to raw CI8 capture, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--streaming-mode",
        action="store_true",
        help=(
            "Enable streaming mode for real-time processing. Processes input in chunks "
            "with bounded memory, suitable for live HackRF streaming via stdin."
        ),
    )
    parser.add_argument(
        "--streaming-buffer-mb",
        type=_positive_int,
        default=200,
        help="Maximum input buffer size in MB for streaming mode (default: 200).",
    )
    parser.add_argument(
        "--streaming-batch-frames",
        type=_positive_int,
        default=16,
        help="Process this many frames per batch in streaming mode (default: 16).",
    )
    parser.add_argument("--sample-rate", type=_positive_int, default=20_000_000)
    parser.add_argument("--symbol-rate", type=_positive_int, default=DTMB_SYMBOL_RATE_SPS)
    parser.add_argument("--max-samples", type=_positive_int, default=8_000_000)
    parser.add_argument("--frequency-shift", type=float, default=0.0)
    parser.add_argument(
        "--input-skip",
        type=_non_negative_int,
        default=0,
        help="Drop this many raw input samples before conditioning and resampling.",
    )
    parser.add_argument(
        "--conjugate-iq",
        action="store_true",
        help=(
            "Conjugate the IQ stream before acquisition. Reverses spectrum "
            "around DC and flips the Q axis; used to test whether a capture "
            "carries mirrored IQ from the front end."
        ),
    )
    parser.add_argument("--mode", choices=("pn420", "pn945"), default="pn945")
    parser.add_argument("--phase-offset", type=_non_negative_int)
    parser.add_argument("--frames", type=_positive_int, default=24)
    parser.add_argument(
        "--qam",
        choices=("auto", "4qam", "16qam", "32qam", "64qam"),
        default="64qam",
        help="QAM mode to demap, or auto to use stable system information.",
    )
    parser.add_argument(
        "--equalizer",
        choices=EQUALIZER_CHOICES,
        default="sparse",
    )
    parser.add_argument(
        "--dd-max-hard-bit-bias",
        type=_ratio,
        default=0.0,
        help=(
            "Opt-in decision-directed equalizer guard. When >0, reject DD "
            "refinement for a frame if any sliced QAM bit plane differs from "
            "0.5 by more than this fraction. Default 0 keeps current behavior."
        ),
    )
    parser.add_argument(
        "--symbol-deinterleave",
        choices=("none", "auto", "mode1", "mode2"),
        default="none",
        help="Apply DTMB convolutional symbol deinterleaving before QAM demapping.",
    )
    parser.add_argument(
        "--symbol-deinterleave-phase",
        type=_non_negative_int,
        default=0,
        help="52-branch commutator phase for DTMB symbol deinterleaving.",
    )
    parser.add_argument(
        "--scan-symbol-deinterleave-phases",
        action="store_true",
        help=(
            "Report LDPC parity mismatch for all 52 symbol-deinterleaver phases "
            "using the selected QAM/FEC mode."
        ),
    )
    parser.add_argument(
        "--scan-symbol-deinterleave-codewords",
        type=_positive_int,
        default=24,
        help="Maximum LDPC codewords to score per deinterleaver phase scan.",
    )
    parser.add_argument(
        "--fec-mode",
        choices=("none", "bch-assume-ldpc", "ldpc"),
        default="none",
        help=(
            "FEC post-processing mode. ldpc runs QC-LDPC min-sum before BCH; "
            "bch-assume-ldpc treats each 7488-bit LDPC codeword as if its "
            "message bits are already reliable."
        ),
    )
    parser.add_argument(
        "--fec-rate",
        type=_fec_rate_or_auto,
        default=3,
        help="DTMB FEC rate index, or auto to use stable system information.",
    )
    parser.add_argument(
        "--ldpc-parity-position",
        choices=("front", "back"),
        default="front",
        help="Position of LDPC parity bits when using bch-assume-ldpc.",
    )
    parser.add_argument("--system-info-index", type=_positive_int, default=23)
    parser.add_argument(
        "--fft-bin-shift",
        type=int,
        default=0,
        help="Circularly shift each C3780 frame-body FFT before frequency deinterleaving.",
    )
    parser.add_argument(
        "--body-window-offset",
        type=int,
        default=0,
        help=(
            "Diagnostic C=3780 body-window offset in symbols before the FFT. "
            "Positive values start the FFT body later than the selected frame timing."
        ),
    )
    parser.add_argument(
        "--frequency-deinterleaver-direction",
        choices=("standard", "inverse"),
        default="standard",
        help="Diagnostic sweep for the C=3780 frequency-deinterleaver index direction.",
    )
    parser.add_argument(
        "--data-carrier-order",
        choices=("normal", "reverse"),
        default="normal",
        help="Diagnostic sweep for per-frame C=3780 data-carrier extraction order.",
    )
    parser.add_argument(
        "--carrier-permutation",
        default="identity",
        help=(
            "Diagnostic C=3780 logical carrier-axis permutation applied to the "
            "inserted logical frame after frequency deinterleaving."
        ),
    )
    parser.add_argument(
        "--logical-position-shift",
        type=int,
        default=0,
        help=(
            "Diagnostic C=3780 logical inserted-domain shift before extracting "
            "system-information and data carriers. Positive values move the "
            "expected system-information holes later in logical carrier order."
        ),
    )
    parser.add_argument(
        "--system-info-min-metric",
        type=_ratio,
        default=0.75,
        help="Minimum per-frame system-information match metric for auto config.",
    )
    parser.add_argument(
        "--system-info-min-agreement",
        type=_ratio,
        default=0.5,
        help="Minimum frame agreement ratio for auto config.",
    )
    parser.add_argument("--frame-body-mode", choices=("C3780", "C1"), default="C3780")
    parser.add_argument("--family-frames", type=_positive_int, default=24)
    parser.add_argument("--max-delay", type=_non_negative_int, default=256)
    parser.add_argument("--family-hit-threshold", type=_positive_float, default=0.45)
    parser.set_defaults(use_family_delay=True)
    parser.add_argument(
        "--use-family-delay",
        action="store_true",
        dest="use_family_delay",
        help="Use stable PN-family delay as the C=3780 timing-search center.",
    )
    parser.add_argument(
        "--no-family-delay",
        action="store_false",
        dest="use_family_delay",
        help="Disable PN-family delay correction before timing search.",
    )
    parser.add_argument("--no-timing-search", action="store_false", dest="timing_search")
    parser.add_argument(
        "--timing-policy",
        choices=("fixed", "windowed", "trajectory"),
        default="fixed",
        help=(
            "Frame slicing policy. fixed uses one global phase; trajectory/windowed "
            "consume --timing-trajectory if provided."
        ),
    )
    parser.add_argument(
        "--timing-trajectory",
        type=Path,
        help="Receiver timing trajectory JSON produced by scripts/pipeline/timing_trajectory.py.",
    )
    parser.add_argument("--timing-radius", type=_non_negative_int, default=512)
    parser.add_argument("--timing-step", type=_positive_int, default=4)
    parser.add_argument(
        "--timing-jobs",
        type=_positive_int,
        default=1,
        help="Worker processes for C=3780 timing candidate scoring.",
    )
    parser.add_argument(
        "--timing-per-candidate-cfo",
        action="store_true",
        help=(
            "Estimate PN cyclic-extension CFO independently for every timing "
            "candidate instead of reusing the center phase estimate."
        ),
    )
    parser.set_defaults(pn_payload_timing_refine=True)
    parser.add_argument(
        "--pn-payload-timing-refine",
        action="store_true",
        dest="pn_payload_timing_refine",
        help=(
            "For an automatically acquired PN-equalized C=3780 stream, break "
            "neighboring phase ties using full-band PN-equalized QAM EVM."
        ),
    )
    parser.add_argument(
        "--no-pn-payload-timing-refine",
        action="store_false",
        dest="pn_payload_timing_refine",
        help="Disable the bounded PN payload timing tie-break.",
    )
    parser.add_argument(
        "--no-track-residual-cfo",
        dest="track_residual_cfo",
        action="store_false",
        help=(
            "Disable fine residual-CFO tracking from the PN-header phase trend. "
            "Residual CFO that mode1 tolerates collapses mode2 LDPC to the 0.50 "
            "plateau, so tracking is on by default."
        ),
    )
    parser.set_defaults(track_residual_cfo=True)
    parser.add_argument(
        "--uncoded",
        action="store_true",
        help=(
            "Treat demapped frame-body bits as payload bytes before FEC. "
            "This is only valid for synthetic fixtures or external pre-FEC experiments."
        ),
    )
    parser.add_argument(
        "--correct-per-frame-cpe",
        action="store_true",
        help=(
            "After equalization, slice each frame's data symbols against the "
            "selected QAM constellation and divide by the fitted complex gain. "
            "Removes residual common phase error that otherwise flips soft-bit "
            "signs between frames once the deinterleaver stitches them."
        ),
    )
    parser.add_argument(
        "--per-frame-cpe-max-relative-error",
        type=_positive_float,
        default=0.7,
        help="Max relative slicing error for a symbol to join the CPE fit.",
    )
    parser.add_argument(
        "--per-frame-cpe-source",
        choices=("auto", "pilot", "decision-directed", "raw-system-info"),
        default="auto",
        help=(
            "Reference source for --correct-per-frame-cpe. auto preserves the "
            "current behavior: use system-information pilots when available, "
            "otherwise fall back to decision-directed QAM slicing. "
            "raw-system-info is a diagnostic mode that derives a phase-only "
            "gain from the independent raw system-information carriers."
        ),
    )
    parser.add_argument("--min-ts-packets", type=_positive_int, default=5)
    parser.add_argument("--min-ts-sync-ratio", type=_ratio, default=0.8)
    parser.add_argument("--min-ts-valid-ratio", type=_ratio, default=0.8)
    parser.add_argument(
        "--output",
        type=str,
        help="Optional .ts output path; '-' writes recovered TS packets to stdout.",
    )
    parser.add_argument(
        "--llr-output",
        type=Path,
        help="Optional float32 little-endian soft-bit LLR output path.",
    )
    parser.add_argument(
        "--dump-frame-fft",
        type=Path,
        help="Optional .npz path for per-frame FFT diagnostics.",
    )

    parser.add_argument(
        "--dump-pre-ldpc",
        type=Path,
        help=(
            "Optional .npz path for post-demapper/pre-LDPC diagnostics. "
            "Writes hard bits, LLRs, QAM symbol streams, and a JSON sidecar."
        ),
    )
    parser.add_argument(
        "--dump-pre-ldpc-symbols-only",
        action="store_true",
        help=(
            "With --dump-pre-ldpc, omit hard bits, LLRs, and per-bit index "
            "arrays. Keeps QAM symbol streams, source-frame indices, frame "
            "starts, and metadata for disk-bounded symbol convention probes."
        ),
    )
    parser.add_argument(
        "--llr-scale",
        type=_positive_float,
        default=1.0,
        help="Multiply demapped soft bits by this factor before hard slicing and LDPC.",
    )
    parser.add_argument(
        "--llr-clip",
        type=_positive_float,
        help="Optionally clip demapped soft bits to +/- this magnitude before LDPC.",
    )
    parser.add_argument(
        "--llr-erase-fraction",
        type=_ratio,
        default=0.0,
        help="Zero the lowest-confidence fraction of demapped LLRs before LDPC.",
    )
    parser.add_argument(
        "--llr-plane-scales",
        default=None,
        help=(
            "Optional comma-separated per-bit-plane LLR multipliers applied after "
            "--llr-scale, e.g. 1,0.1,1,1,0.1,1 for 64QAM."
        ),
    )
    parser.add_argument(
        "--csi-demap",
        action="store_true",
        help=(
            "Per-carrier unbiased demapping for PN equalizers: undo the MMSE "
            "per-carrier shrinkage (which under-occupies outer QAM levels on "
            "rippled channels) and weight each symbol's LLRs by its carrier's "
            "|H|^2 so the LDPC decoder trusts spectral notches less."
        ),
    )
    parser.add_argument(
        "--carrier-erase-metric",
        choices=("none", "evm", "bit_bias", "unreliable"),
        default="none",
        help="Optionally zero LLRs from the worst source carriers before LDPC.",
    )
    parser.add_argument(
        "--carrier-erase-fraction",
        type=_ratio,
        default=0.0,
        help="Fraction of source carriers to erase when --carrier-erase-metric is enabled.",
    )
    parser.add_argument(
        "--carrier-erase-reliability-threshold",
        type=_ratio,
        default=0.55,
        help="Relative-error threshold used by carrier reliability scoring.",
    )
    parser.add_argument(
        "--branch-gain-branches",
        default=None,
        help=(
            "Optional comma-separated interleaver branch indexes to correct with "
            "branch-local DD complex gain after symbol deinterleave; use 'all' "
            "for every branch."
        ),
    )
    parser.add_argument(
        "--branch-gain-reliability-threshold",
        type=_ratio,
        default=0.55,
        help="Relative-error threshold for branch-local DD gain estimation.",
    )
    parser.add_argument(
        "--branch-gain-min-symbols",
        type=_positive_int,
        default=32,
        help="Minimum reliable symbols required before correcting one branch.",
    )
    parser.add_argument(
        "--pn-mmse",
        default="off",
        help=(
            "PN equalizer MMSE regularization: 'off' (pure zero-forcing), "
            "'auto' (estimate noise variance from PN tap noise), or a float "
            "noise-variance value. MMSE limits noise amplification at channel "
            "nulls and is the field-SNR robustness path for long-span (mode2) "
            "deinterleaving on frequency-selective channels."
        ),
    )
    parser.add_argument(
        "--pn-channel-taps",
        type=int,
        default=None,
        help=(
            "Limit the PN channel estimate to the strongest N taps. The "
            "delay-tolerant matcher otherwise lets noise spread across the "
            "full guard (observed: 511-tap estimate with a spurious ~-500 "
            "delay alias on real 522 MHz RF) which scrambles the equalized "
            "per-carrier phase. A compact limit (e.g. 8-32) keeps the "
            "estimate physical. Default None preserves the legacy behaviour."
        ),
    )
    parser.add_argument(
        "--pn-estimator",
        choices=("compact", "wideband"),
        default="compact",
        help=(
            "PN channel estimator. 'compact' (default) estimates per frame "
            "and keeps the strongest leading taps; correct for single-cluster "
            "channels. 'wideband' averages the full cyclic-core deconvolution "
            "across all frame headers and thresholds taps against the "
            "averaged noise floor, supporting multi-cluster channels with "
            "echoes anywhere in the PN guard (e.g. SFN reception); it ignores "
            "--pn-channel-taps."
        ),
    )
    parser.add_argument(
        "--pn-wideband-block-frames",
        type=_positive_int,
        default=16,
        help="Headers per local wideband PN channel model (default: 16).",
    )
    parser.add_argument("--json", type=Path, help="Optional path to write diagnostics JSON")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress the diagnostics JSON on stdout. Implied when --output=- "
            "is used so stdout only carries the recovered TS packets."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.streaming_mode:
        return receive_streaming(
            args.capture,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
            buffer_mb=args.streaming_buffer_mb,
            batch_frames=args.streaming_batch_frames,
            frequency_shift_hz=args.frequency_shift,
            input_skip_samples=args.input_skip,
            conjugate_iq=args.conjugate_iq,
            mode=args.mode,
            phase_offset=args.phase_offset,
            qam_mode=args.qam,
            equalizer=args.equalizer,
            dd_max_hard_bit_bias=args.dd_max_hard_bit_bias,
            symbol_deinterleave=args.symbol_deinterleave,
            symbol_deinterleave_phase=args.symbol_deinterleave_phase,
            fec_mode=args.fec_mode,
            fec_rate_index=args.fec_rate,
            ldpc_parity_position=args.ldpc_parity_position,
            system_info_index=args.system_info_index,
            fft_bin_shift=args.fft_bin_shift,
            body_window_offset_symbols=args.body_window_offset,
            frequency_deinterleaver_direction=args.frequency_deinterleaver_direction,
            data_carrier_order=args.data_carrier_order,
            carrier_permutation=args.carrier_permutation,
            logical_position_shift_symbols=args.logical_position_shift,
            frame_body_mode=args.frame_body_mode,
            timing_trajectory_path=args.timing_trajectory,
            track_residual_cfo=args.track_residual_cfo,
            uncoded_payload=args.uncoded,
            correct_per_frame_cpe=args.correct_per_frame_cpe,
            per_frame_cpe_max_relative_error=args.per_frame_cpe_max_relative_error,
            per_frame_cpe_source=args.per_frame_cpe_source,
            llr_output_path=args.llr_output,
            llr_scale=args.llr_scale,
            llr_clip=args.llr_clip,
            llr_erase_fraction=args.llr_erase_fraction,
            llr_plane_scales=args.llr_plane_scales,
            carrier_erase_metric=args.carrier_erase_metric,
            carrier_erase_fraction=args.carrier_erase_fraction,
            carrier_erase_reliability_threshold=args.carrier_erase_reliability_threshold,
            csi_demap=args.csi_demap,
            branch_gain_branches=args.branch_gain_branches,
            branch_gain_reliability_threshold=args.branch_gain_reliability_threshold,
            branch_gain_min_symbols=args.branch_gain_min_symbols,
            pn_equalizer_noise_variance=_parse_pn_mmse(args.pn_mmse),
            pn_channel_taps=args.pn_channel_taps,
            pn_estimator=args.pn_estimator,
            pn_wideband_block_frames=args.pn_wideband_block_frames,
        )

    diagnostics = receive_capture(
        args.capture,
        sample_rate_sps=args.sample_rate,
        symbol_rate_sps=args.symbol_rate,
        max_samples=args.max_samples,
        frequency_shift_hz=args.frequency_shift,
        input_skip_samples=args.input_skip,
        conjugate_iq=args.conjugate_iq,
        mode=args.mode,
        phase_offset=args.phase_offset,
        max_frames=args.frames,
        qam_mode=args.qam,
        equalizer=args.equalizer,
        dd_max_hard_bit_bias=args.dd_max_hard_bit_bias,
        symbol_deinterleave=args.symbol_deinterleave,
        symbol_deinterleave_phase=args.symbol_deinterleave_phase,
        scan_symbol_deinterleave_phases=args.scan_symbol_deinterleave_phases,
        scan_symbol_deinterleave_codewords=args.scan_symbol_deinterleave_codewords,
        fec_mode=args.fec_mode,
        fec_rate_index=args.fec_rate,
        ldpc_parity_position=args.ldpc_parity_position,
        system_info_index=args.system_info_index,
        fft_bin_shift=args.fft_bin_shift,
        body_window_offset_symbols=args.body_window_offset,
        frequency_deinterleaver_direction=args.frequency_deinterleaver_direction,
        data_carrier_order=args.data_carrier_order,
        carrier_permutation=args.carrier_permutation,
        logical_position_shift_symbols=args.logical_position_shift,
        system_info_min_metric=args.system_info_min_metric,
        system_info_min_agreement=args.system_info_min_agreement,
        frame_body_mode=args.frame_body_mode,
        use_family_delay=args.use_family_delay,
        family_frames=args.family_frames,
        max_delay_symbols=args.max_delay,
        family_hit_threshold=args.family_hit_threshold,
        timing_search=args.timing_search,
        timing_policy=args.timing_policy,
        timing_trajectory_path=args.timing_trajectory,
        timing_radius_symbols=args.timing_radius,
        timing_step_symbols=args.timing_step,
        timing_jobs=args.timing_jobs,
        timing_per_candidate_cfo=args.timing_per_candidate_cfo,
        pn_payload_timing_refine=args.pn_payload_timing_refine,
        track_residual_cfo=args.track_residual_cfo,
        uncoded_payload=args.uncoded,
        correct_per_frame_cpe=args.correct_per_frame_cpe,
        per_frame_cpe_max_relative_error=args.per_frame_cpe_max_relative_error,
        per_frame_cpe_source=args.per_frame_cpe_source,
        min_ts_packets=args.min_ts_packets,
        min_ts_sync_ratio=args.min_ts_sync_ratio,
        min_ts_valid_ratio=args.min_ts_valid_ratio,
        output_path=args.output,
        llr_output_path=args.llr_output,
        pre_ldpc_dump_path=args.dump_pre_ldpc,
        pre_ldpc_dump_symbols_only=args.dump_pre_ldpc_symbols_only,
        dump_frame_fft_path=args.dump_frame_fft,
        llr_scale=args.llr_scale,
        llr_clip=args.llr_clip,
        llr_erase_fraction=args.llr_erase_fraction,
        llr_plane_scales=args.llr_plane_scales,
        carrier_erase_metric=args.carrier_erase_metric,
        carrier_erase_fraction=args.carrier_erase_fraction,
        carrier_erase_reliability_threshold=args.carrier_erase_reliability_threshold,
        csi_demap=args.csi_demap,
        branch_gain_branches=args.branch_gain_branches,
        branch_gain_reliability_threshold=args.branch_gain_reliability_threshold,
        branch_gain_min_symbols=args.branch_gain_min_symbols,
        pn_equalizer_noise_variance=_parse_pn_mmse(args.pn_mmse),
        pn_channel_taps=args.pn_channel_taps,
        pn_estimator=args.pn_estimator,
        pn_wideband_block_frames=args.pn_wideband_block_frames,
    )
    encoded = json.dumps(diagnostics, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    if args.output != "-" and not args.quiet:
        print(encoded)
    return 0 if diagnostics["ts"]["lock"] is not None else 2


def receive_capture(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 8_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    conjugate_iq: bool = False,
    mode: PnMode = "pn945",
    phase_offset: int | None = None,
    max_frames: int = 24,
    qam_mode: QamMode | Literal["auto"] = "64qam",
    equalizer: str = "sparse",
    dd_max_hard_bit_bias: float = 0.0,
    symbol_deinterleave: InterleaverMode | Literal["none", "auto"] | str = "none",
    symbol_deinterleave_phase: int = 0,
    scan_symbol_deinterleave_phases: bool = False,
    scan_symbol_deinterleave_codewords: int = 24,
    fec_mode: str = "none",
    fec_rate_index: int | Literal["auto"] = 3,
    ldpc_parity_position: str = "front",
    system_info_index: int = 23,
    fft_bin_shift: int = 0,
    body_window_offset_symbols: int = 0,
    frequency_deinterleaver_direction: Literal["standard", "inverse"] | str = "standard",
    data_carrier_order: Literal["normal", "reverse"] | str = "normal",
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
    system_info_min_metric: float = 0.75,
    system_info_min_agreement: float = 0.5,
    frame_body_mode: str = "C3780",
    use_family_delay: bool = True,
    family_frames: int = 24,
    max_delay_symbols: int = 256,
    family_hit_threshold: float = 0.45,
    timing_search: bool = True,
    timing_policy: Literal["fixed", "windowed", "trajectory"] | str = "fixed",
    timing_trajectory_path: str | Path | None = None,
    timing_trajectory: dict[str, Any] | None = None,
    timing_radius_symbols: int = 512,
    timing_step_symbols: int = 4,
    timing_jobs: int = 1,
    timing_per_candidate_cfo: bool = False,
    pn_payload_timing_refine: bool = True,
    track_residual_cfo: bool = True,
    uncoded_payload: bool = False,
    correct_per_frame_cpe: bool = False,
    per_frame_cpe_max_relative_error: float = 0.7,
    per_frame_cpe_source: Literal[
        "auto",
        "pilot",
        "decision-directed",
        "raw-system-info",
    ] | str = "auto",
    min_ts_packets: int = 5,
    min_ts_sync_ratio: float = 0.8,
    min_ts_valid_ratio: float = 0.8,
    output_path: str | Path | None = None,
    llr_output_path: str | Path | None = None,
    pre_ldpc_dump_path: str | Path | None = None,
    pre_ldpc_dump_symbols_only: bool = False,
    dump_frame_fft_path: str | Path | None = None,
    llr_scale: float = 1.0,
    llr_clip: float | None = None,
    llr_erase_fraction: float = 0.0,
    llr_plane_scales: Sequence[float] | str | None = None,
    carrier_erase_metric: str = "none",
    carrier_erase_fraction: float = 0.0,
    carrier_erase_reliability_threshold: float = 0.55,
    csi_demap: bool = False,
    branch_gain_branches: Sequence[int] | str | None = None,
    branch_gain_reliability_threshold: float = 0.55,
    branch_gain_min_symbols: int = 32,
    pn_equalizer_noise_variance: float | Literal["auto"] | None = None,
    pn_channel_taps: int | None = None,
    pn_estimator: str = "compact",
    pn_wideband_block_frames: int = 16,
) -> dict[str, Any]:
    """Receive a capture through demapping and optional uncoded TS alignment."""

    pn_estimator = str(pn_estimator)
    if pn_estimator not in ("compact", "wideband"):
        raise ValueError("pn_estimator must be 'compact' or 'wideband'")
    if pn_wideband_block_frames <= 0:
        raise ValueError("pn_wideband_block_frames must be positive")
    equalizer = str(equalizer)
    if equalizer not in EQUALIZER_CHOICES:
        raise ValueError(f"equalizer must be one of: {', '.join(EQUALIZER_CHOICES)}")
    if csi_demap and equalizer not in PN_EQUALIZERS:
        raise ValueError("csi_demap requires a PN equalizer")
    frequency_deinterleaver_direction = str(frequency_deinterleaver_direction)
    data_carrier_order = str(data_carrier_order)
    carrier_permutation = str(carrier_permutation).strip().lower() or "identity"
    if frequency_deinterleaver_direction not in ("standard", "inverse"):
        raise ValueError("frequency_deinterleaver_direction must be standard or inverse")
    if data_carrier_order not in ("normal", "reverse"):
        raise ValueError("data_carrier_order must be normal or reverse")
    system_info_pilot_positions = np.asarray(
        system_info_positions_for_extraction(
            carrier_permutation,
            int(logical_position_shift_symbols),
        ),
        dtype=np.int32,
    )
    logical_to_spectrum_positions = _logical_to_spectrum_positions_for_direction(
        frequency_deinterleaver_direction
    )
    if dd_max_hard_bit_bias < 0.0 or dd_max_hard_bit_bias > 1.0:
        raise ValueError("dd_max_hard_bit_bias must be between 0 and 1")
    per_frame_cpe_source = str(per_frame_cpe_source)
    if per_frame_cpe_source not in (
        "auto",
        "pilot",
        "decision-directed",
        "raw-system-info",
    ):
        raise ValueError(
            "per_frame_cpe_source must be auto, pilot, decision-directed, "
            "or raw-system-info"
        )
    if llr_scale <= 0.0:
        raise ValueError("llr_scale must be positive")
    if llr_clip is not None and llr_clip <= 0.0:
        raise ValueError("llr_clip must be positive when provided")
    if llr_erase_fraction < 0.0 or llr_erase_fraction > 1.0:
        raise ValueError("llr_erase_fraction must be between 0 and 1")
    if carrier_erase_metric not in ("none", "evm", "bit_bias", "unreliable"):
        raise ValueError("carrier_erase_metric must be none, evm, bit_bias, or unreliable")
    if carrier_erase_fraction < 0.0 or carrier_erase_fraction > 1.0:
        raise ValueError("carrier_erase_fraction must be between 0 and 1")
    if carrier_erase_reliability_threshold < 0.0 or carrier_erase_reliability_threshold > 1.0:
        raise ValueError("carrier_erase_reliability_threshold must be between 0 and 1")
    if branch_gain_reliability_threshold < 0.0 or branch_gain_reliability_threshold > 1.0:
        raise ValueError("branch_gain_reliability_threshold must be between 0 and 1")
    if branch_gain_min_symbols <= 0:
        raise ValueError("branch_gain_min_symbols must be positive")
    symbols, up, down = _load_symbols(
        capture_path,
        sample_rate_sps=sample_rate_sps,
        symbol_rate_sps=symbol_rate_sps,
        max_samples=max_samples,
        frequency_shift_hz=frequency_shift_hz,
        input_skip_samples=input_skip_samples,
        conjugate_iq=conjugate_iq,
    )
    pn_payload_qam_mode = (
        str(qam_mode)
        if qam_mode in ("4qam", "16qam", "64qam")
        else transmission_parameters_for_index(
            int(system_info_index),
            frame_body_mode=frame_body_mode,
        ).qam_mode
    )
    acquisition = _select_phase_and_cfo(
        symbols,
        mode=mode,
        phase_offset=phase_offset,
        symbol_rate_sps=symbol_rate_sps,
        max_frames=max_frames,
        use_family_delay=use_family_delay,
        family_frames=family_frames,
        max_delay_symbols=max_delay_symbols,
        family_hit_threshold=family_hit_threshold,
        timing_search=timing_search,
        timing_radius_symbols=timing_radius_symbols,
        timing_step_symbols=timing_step_symbols,
        timing_jobs=timing_jobs,
        timing_per_candidate_cfo=timing_per_candidate_cfo,
        pn_payload_timing_refine=bool(
            pn_payload_timing_refine
            and equalizer in PN_EQUALIZERS
            and pn_payload_qam_mode in ("4qam", "16qam", "64qam")
        ),
        pn_payload_qam_mode=pn_payload_qam_mode
        if pn_payload_qam_mode in ("4qam", "16qam", "64qam")
        else None,
        expected_system_info_index=system_info_index,
        expected_frame_body_mode=frame_body_mode,
    )
    corrected = symbols
    if acquisition["coarse_cfo_hz"] is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=symbol_rate_sps,
            shift_hz=-float(acquisition["coarse_cfo_hz"]),
        )

    # Fine residual-CFO tracking. The coarse PN cyclic-extension estimate leaves
    # a few-Hz residual that mode1's 170-frame symbol-deinterleaver span
    # tolerates but mode2's 510-frame span does not: the residual rotates each
    # C=3780 frame's constellation independently and collapses every LDPC
    # codeword to the 0.50 plateau (trial-log 2026-05-31). Estimating the slope
    # of the PN-header phase across all frames resolves it to well under 1 Hz
    # and flattens the per-frame phase walk before body processing.
    residual_cfo_hz = None
    if track_residual_cfo:
        residual_cfo_hz = estimate_residual_cfo_from_frame_headers(
            corrected,
            mode=mode,
            phase_offset=int(acquisition["phase_offset"]),
            symbol_rate_sps=symbol_rate_sps,
            max_frames=max_frames + 1,
        )
        if residual_cfo_hz is not None and abs(residual_cfo_hz) > 0.0:
            corrected = frequency_shift(
                corrected,
                sample_rate_sps=symbol_rate_sps,
                shift_hz=-float(residual_cfo_hz),
            )

    frame_reports: list[dict[str, Any]] = []
    data_symbol_chunks: list[np.ndarray] = []
    csi_weight_chunks: list[np.ndarray] = []
    data_symbols_per_frame: list[int] = []
    frame_fft_dumps: list[dict[str, Any]] = []
    dd_qam_mode = _decision_directed_qam_mode(
        qam_mode,
        system_info_index=system_info_index,
        frame_body_mode=frame_body_mode,
    )
    trajectory_report = _load_timing_trajectory(
        timing_trajectory=timing_trajectory,
        timing_trajectory_path=timing_trajectory_path,
    )
    frame_entries = _select_frame_entries(
        corrected,
        mode=mode,
        max_frames=max_frames + 1,
        timing_policy=timing_policy,
        trajectory_report=trajectory_report,
        fallback_phase_offset=int(acquisition["phase_offset"]),
        fallback_cfo_hz=acquisition.get("coarse_cfo_hz"),
        symbol_rate_sps=symbol_rate_sps,
    )
    analyzed_entries = frame_entries[:max_frames]

    # The wideband estimator needs every header up front: it averages the full
    # cyclic-core deconvolution across frames to drop the per-tap estimation
    # noise far enough below multi-cluster echo shelves to threshold them
    # reliably (trial-log 2026-06-11: per-frame compact estimates cannot be
    # widened on such channels without making the estimate worse).
    wideband_channel_models = []
    wideband_channel_report: dict[str, Any] | None = None
    if (
        pn_estimator == "wideband"
        and equalizer in PN_EQUALIZERS
        and frame_body_mode == "C3780"
        and frame_entries
    ):
        block_reports = []
        for start in range(0, len(frame_entries), pn_wideband_block_frames):
            block_entries = frame_entries[
                start : min(
                    start + pn_wideband_block_frames + 1,
                    len(frame_entries),
                )
            ]
            model = build_wideband_pn_channel_model(
                [entry[0].header for entry in block_entries],
                mode=mode,
                fft_size=FRAME_BODY_SYMBOLS,
            )
            wideband_channel_models.append(model)
            block_report = model.to_dict()
            block_report["start_frame"] = int(start)
            block_reports.append(block_report)
        wideband_channel_report = {
            **block_reports[0],
            "block_frames": int(pn_wideband_block_frames),
            "model_count": int(len(block_reports)),
            "models": block_reports,
        }

    for frame_index, (frame, frame_timing) in enumerate(analyzed_entries):
        pn_zero_delay_phase: int | None = None
        frame_csi_h2: np.ndarray | None = None
        next_frame = (
            frame_entries[frame_index + 1][0]
            if frame_index + 1 < len(frame_entries)
            else None
        )
        wideband_channel_model = (
            wideband_channel_models[
                min(
                    frame_index // pn_wideband_block_frames,
                    len(wideband_channel_models) - 1,
                )
            ]
            if wideband_channel_models
            else None
        )
        if frame_body_mode == "C3780":
            body_samples = _frame_body_samples(
                frame,
                corrected,
                mode=mode,
                body_window_offset_symbols=body_window_offset_symbols,
            )
            _raw_body_samples = body_samples.copy()
            pn_isi_cancelled = False
            # Only apply PN tail cancellation and circular restore if we have
            # the next frame's header to seal the cyclic boundary and we are
            # using PN channel estimation (which needs the circular property).
            if (
                next_frame is not None
                and equalizer in PN_EQUALIZERS
            ):
                pn_zero_delay_phase = detect_pn_phase(frame.header, mode=mode)
                if wideband_channel_model is not None:
                    pn_channel = wideband_pn_channel_estimate_for_header(
                        frame.header, wideband_channel_model,
                    )
                    next_channel = wideband_pn_channel_estimate_for_header(
                        next_frame.header, wideband_channel_model,
                    )
                else:
                    # Choose the PN phase whose channel taps are most compact
                    # so a family-matcher phase/delay alias cannot corrupt the
                    # taps used for PN-ISI restoration or equalization. See
                    # dtmb.channel.estimate_pn_channel_compact and AGENTS.md
                    # (2026-05-29).
                    pn_channel = estimate_pn_channel_compact(
                        frame.header, mode=mode, fft_size=3780,
                        channel_taps=pn_channel_taps,
                    )
                    next_channel = estimate_pn_channel_compact(
                        next_frame.header, mode=mode, fft_size=3780,
                        channel_taps=pn_channel_taps,
                    )
                body_samples = pn_restore_circular_body_window(
                    frame.body,
                    body_window_offset_symbols=int(body_window_offset_symbols),
                    pn_phase=pn_channel.pn_phase,
                    next_pn_phase=next_channel.pn_phase,
                    mode=mode,
                    taps=pn_channel.taps,
                    next_header=next_frame.header,
                )
                pn_isi_cancelled = True

            spectrum = frame_body_fft(body_samples)
            _fft_raw = spectrum.copy()
            if fft_bin_shift:
                spectrum = np.roll(spectrum, fft_bin_shift)
            _fft_shifted = spectrum.copy()
            raw_deinterleaved = _frequency_deinterleave_variant(
                spectrum,
                direction=frequency_deinterleaver_direction,
                carrier_permutation=carrier_permutation,
                logical_position_shift_symbols=logical_position_shift_symbols,
            )
            raw_system_info, _raw_data = split_system_info_and_data(raw_deinterleaved)
            system_info_iq_by_source = {
                "raw": _complex_symbol_pairs(raw_system_info),
            }
            supported_system_info_oracle = _supported_system_info_oracle(
                raw_system_info,
                frame_body_mode="C3780",
            )
            raw_matches = classify_system_info(
                raw_system_info,
                frame_body_modes=("C3780",),
            )
            matches = raw_matches
            system_info_source = "raw"
            system_info_independent = True
            channel_report = None
            system_info_evidence = [
                _system_info_evidence_report(
                    "raw",
                    raw_matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            ]
            raw_polarity_matches = classify_system_info(
                raw_system_info,
                allow_polarity_inversion=True,
                frame_body_modes=("C3780",),
            )
            system_info_evidence.append(
                _system_info_evidence_report(
                    "raw-polarity",
                    raw_polarity_matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            )
            if _best_system_info_metric(raw_polarity_matches) > _best_system_info_metric(
                matches
            ):
                matches = raw_polarity_matches
                system_info_source = "raw-polarity"
            raw_polarity_wide_matches = classify_system_info(
                raw_system_info,
                allow_polarity_inversion=True,
                max_common_phase_radians=RAW_POLARITY_WIDE_MAX_COMMON_PHASE_RADIANS,
                frame_body_modes=("C3780",),
            )
            system_info_evidence.append(
                _system_info_evidence_report(
                    "raw-polarity-wide",
                    raw_polarity_wide_matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            )
            if _best_system_info_metric(raw_polarity_wide_matches) > _best_system_info_metric(
                matches
            ):
                matches = raw_polarity_wide_matches
                system_info_source = "raw-polarity-wide"
            if equalizer in PN_EQUALIZERS:
                if not pn_isi_cancelled:
                    if wideband_channel_model is not None:
                        pn_channel = wideband_pn_channel_estimate_for_header(
                            frame.header, wideband_channel_model,
                        )
                    else:
                        pn_channel = estimate_pn_channel_compact(
                            frame.header,
                            mode=mode,
                            fft_size=spectrum.size,
                            channel_taps=pn_channel_taps,
                        )
                pn_response_channel = apply_body_window_offset_to_pn_channel_response(
                    pn_channel,
                    int(body_window_offset_symbols),
                )
                pn_response_channel = shift_pn_channel_response(
                    pn_response_channel,
                    fft_bin_shift,
                )
                if pn_equalizer_noise_variance is None:
                    pn_noise_variance = None
                elif pn_equalizer_noise_variance == "auto":
                    if wideband_channel_model is not None:
                        # The wideband template zeroes its noise taps, so the
                        # trailing-tap proxy below would report ~0; use the
                        # cross-frame measured single-header tap noise instead.
                        pn_noise_variance = wideband_channel_model.noise_variance
                    else:
                        pn_noise_variance = estimate_frequency_domain_noise_variance(
                            pn_channel
                        )
                else:
                    pn_noise_variance = float(pn_equalizer_noise_variance)
                spectrum = equalize_spectrum_with_channel(
                    spectrum,
                    pn_response_channel,
                    noise_variance=pn_noise_variance,
                )
                if csi_demap:
                    h2 = (
                        np.abs(pn_response_channel.response).astype(np.float64) ** 2
                    )
                    if pn_noise_variance:
                        # Undo the per-carrier MMSE shrinkage g = |H|^2/(|H|^2+N)
                        # (clipped so deep notches do not blow up the global
                        # demap normalization); the notch noise this re-inflates
                        # is handled by the per-carrier LLR weights instead.
                        gain = h2 / (h2 + float(pn_noise_variance))
                        spectrum = (
                            spectrum / np.maximum(gain, 0.25)
                        ).astype(np.complex64)
                    frame_csi_h2 = h2.astype(np.float32)
                channel_report = pn_channel.to_dict()
                channel_report["estimator"] = pn_estimator
                if pn_noise_variance is not None:
                    channel_report["mmse_noise_variance"] = float(pn_noise_variance)
                channel_report["response_body_window_offset_symbols"] = int(
                    body_window_offset_symbols
                )
                channel_report["response_fft_bin_shift"] = int(fft_bin_shift)
                if pn_zero_delay_phase is None:
                    pn_zero_delay_phase = detect_pn_phase(frame.header, mode=mode)
                channel_report.update(
                    _pn_phase_alias_report(
                        channel_phase=pn_channel.pn_phase,
                        zero_delay_phase=pn_zero_delay_phase,
                        mode=mode,
                    )
                )
                pn_deinterleaved = _frequency_deinterleave_variant(
                    spectrum,
                    direction=frequency_deinterleaver_direction,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                )
                pn_system_info, _pn_data = split_system_info_and_data(pn_deinterleaved)
                system_info_iq_by_source["pn-channel"] = _complex_symbol_pairs(
                    pn_system_info
                )
                pn_matches = classify_system_info(
                    pn_system_info,
                    frame_body_modes=("C3780",),
                )
                system_info_evidence.append(
                    _system_info_evidence_report(
                        "pn-channel",
                        pn_matches,
                        independent=True,
                        expected_index=system_info_index,
                        expected_frame_body_mode=frame_body_mode,
                    )
                )
                if pn_matches and _best_system_info_metric(pn_matches) > _best_system_info_metric(matches):
                    matches = pn_matches
                    system_info_source = "pn-channel"
            if equalizer in SPARSE_PILOT_EQUALIZERS:
                sparse = equalize_c3780_spectrum_with_system_info_pilots(
                    spectrum,
                    system_info_index=system_info_index,
                    frame_body_mode="C3780",
                    system_info_positions=system_info_pilot_positions,
                    logical_to_spectrum_positions=logical_to_spectrum_positions,
                )
                spectrum = sparse.equalized
                sparse_report = sparse.to_dict()
            elif equalizer in DECISION_DIRECTED_EQUALIZERS:
                sparse = refine_c3780_spectrum_decision_directed(
                    spectrum,
                    system_info_index=system_info_index,
                    frame_body_mode="C3780",
                    qam_mode=dd_qam_mode,
                    max_hard_bit_bias=dd_max_hard_bit_bias,
                    system_info_positions=system_info_pilot_positions,
                    logical_to_spectrum_positions=logical_to_spectrum_positions,
                    bootstrap="raw" if equalizer.endswith("-raw") else "sparse",
                    include_system_info_pilots=not equalizer.endswith(
                        ("-raw", "-data")
                    ),
                )
                spectrum = sparse.equalized
                sparse_report = sparse.to_dict()
            else:
                sparse_report = None
            inserted = _frequency_deinterleave_inserted_variant(
                spectrum,
                direction=frequency_deinterleaver_direction,
            )
            if equalizer in SPARSE_PILOT_EQUALIZERS:
                deinterleaved = _logical_from_inserted_c3780(
                    inserted,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                )
            elif equalizer in DECISION_DIRECTED_EQUALIZERS:
                deinterleaved = _logical_from_inserted_c3780(
                    inserted,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                )
            else:
                deinterleaved = _frequency_deinterleave_variant(
                    spectrum,
                    direction=frequency_deinterleaver_direction,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                )
            frame_system_info, data_symbols = split_system_info_and_data(deinterleaved)
            if data_carrier_order == "reverse":
                data_symbols = data_symbols[::-1].astype(np.complex64, copy=False)
            post_matches = classify_system_info(
                frame_system_info,
                frame_body_modes=("C3780",),
            )
            system_info_iq_by_source["post-equalizer"] = _complex_symbol_pairs(
                frame_system_info
            )
            post_trained = (
                equalizer in TRAINED_SYSTEM_INFO_EQUALIZERS
                and not equalizer.endswith("-raw")
            )
            if per_frame_cpe_source == "auto" and equalizer.endswith("-raw"):
                cpe_pilot_matches = None
            elif per_frame_cpe_source == "pilot" and equalizer.endswith("-raw"):
                cpe_pilot_matches = post_matches
            else:
                cpe_pilot_matches = post_matches if post_trained else matches
            system_info_evidence.append(
                _system_info_evidence_report(
                    "post-equalizer",
                    post_matches,
                    independent=not post_trained,
                    trained_on_system_info_index=system_info_index
                    if post_trained
                    else None,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            )
        elif frame_body_mode == "C1":
            raw_system_info = frame.body[:36]
            frame_system_info = raw_system_info
            system_info_iq_by_source = {
                "raw": _complex_symbol_pairs(raw_system_info),
            }
            supported_system_info_oracle = _supported_system_info_oracle(
                raw_system_info,
                frame_body_mode="C1",
            )
            matches = classify_system_info(
                raw_system_info,
                frame_body_modes=("C1",),
            )
            system_info_source = "raw"
            system_info_independent = True
            system_info_evidence = [
                _system_info_evidence_report(
                    "raw",
                    matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            ]
            raw_polarity_matches = classify_system_info(
                raw_system_info,
                allow_polarity_inversion=True,
                frame_body_modes=("C1",),
            )
            system_info_evidence.append(
                _system_info_evidence_report(
                    "raw-polarity",
                    raw_polarity_matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            )
            if _best_system_info_metric(raw_polarity_matches) > _best_system_info_metric(
                matches
            ):
                matches = raw_polarity_matches
                system_info_source = "raw-polarity"
            raw_polarity_wide_matches = classify_system_info(
                raw_system_info,
                allow_polarity_inversion=True,
                max_common_phase_radians=RAW_POLARITY_WIDE_MAX_COMMON_PHASE_RADIANS,
                frame_body_modes=("C1",),
            )
            system_info_evidence.append(
                _system_info_evidence_report(
                    "raw-polarity-wide",
                    raw_polarity_wide_matches,
                    independent=True,
                    expected_index=system_info_index,
                    expected_frame_body_mode=frame_body_mode,
                )
            )
            if _best_system_info_metric(raw_polarity_wide_matches) > _best_system_info_metric(
                matches
            ):
                matches = raw_polarity_wide_matches
                system_info_source = "raw-polarity-wide"
            channel_report = None
            if equalizer == "none":
                sparse_report = None
                data_symbols = frame.body[36:]
            elif equalizer == "sparse":
                reference = system_info_symbols(
                    SYSTEM_INFO_VECTORS[system_info_index],
                    frame_body_mode="C1",
                )
                gain = np.vdot(reference, raw_system_info) / max(
                    float(np.vdot(reference, reference).real),
                    1e-12,
                )
                equalized_body = frame.body / (gain if abs(gain) > 1e-12 else 1.0)
                equalized_system_info = equalized_body[:36]
                frame_system_info = equalized_system_info
                matches = classify_system_info(
                    equalized_system_info,
                    frame_body_modes=("C1",),
                )
                system_info_source = "sparse-pilot"
                system_info_independent = False
                system_info_evidence.append(
                    _system_info_evidence_report(
                        "sparse-pilot",
                        matches,
                        independent=False,
                        trained_on_system_info_index=system_info_index,
                        expected_index=system_info_index,
                        expected_frame_body_mode=frame_body_mode,
                    )
                )
                pilot_error = equalized_system_info - reference
                data_symbols = equalized_body[36:]
                system_info_iq_by_source["post-equalizer"] = _complex_symbol_pairs(
                    frame_system_info
                )
                sparse_report = {
                    "pilot_error_rms": float(
                        np.sqrt(np.mean(np.abs(pilot_error) ** 2))
                    ),
                    "data_power": float(np.mean(np.abs(data_symbols) ** 2))
                    if data_symbols.size
                    else 0.0,
                    "gain_real": float(np.real(gain)),
                    "gain_imag": float(np.imag(gain)),
                }
            else:
                raise ValueError("C1 frame-body mode supports equalizer=none or sparse")
            system_info_iq_by_source.setdefault(
                "post-equalizer",
                _complex_symbol_pairs(frame_system_info),
            )
            cpe_pilot_matches = matches
        else:
            raise ValueError("frame_body_mode must be C3780 or C1")
        expected_system_info = _expected_system_info_report(
            matches,
            expected_index=system_info_index,
            expected_frame_body_mode=frame_body_mode,
        )
        per_frame_cpe_report = None
        # PN-equalized frames are already absolute-phase-locked by the per-frame
        # PN-header channel estimate: X/H deconvolves against the known PN
        # sequence, so the equalized body carries the TRUE absolute orientation,
        # not the pi/2-ambiguous one a blind estimator leaves. Applying the
        # blind/decision-directed CPE on top RE-ALIASES each frame's phase by a
        # k*90deg quadrant (on weak frames the 4th-power estimate locks to the
        # wrong branch). Across mode2's 510-frame symbol-deinterleaver span those
        # independent per-frame rotations randomize every LDPC codeword -- the
        # exact 0.50 parity plateau. Proven on the synthetic mode2
        # multipath+5Hz-CFO fixture (scripts/probe_mode2_phase_strategy.py):
        # pre-LDPC BER 0.047 and byte-exact decode (6/6 codewords) WITHOUT CPE
        # vs BER 0.27 and zero converged codewords WITH CPE. So suppress the
        # per-frame CPE whenever a PN equalizer already provides the absolute
        # phase reference; keep it for the sparse/DD/none paths that do not.
        if correct_per_frame_cpe and equalizer in PN_EQUALIZERS:
            power = (
                float(np.mean(np.abs(data_symbols) ** 2)) if data_symbols.size else 0.0
            )
            per_frame_cpe_report = {
                "source": "skipped-pn-absolute-phase-lock",
                "requested_source": per_frame_cpe_source,
                "gain_real": 1.0,
                "gain_imag": 0.0,
                "cpe_rad": 0.0,
                "amplitude": 1.0,
                "reliable_symbol_count": 0,
                "pre_correction_power": power,
                "post_correction_power": power,
            }
        elif correct_per_frame_cpe:
            # Per-frame CPE correction is only well-defined once we have a QAM
            # mode to slice against. If we're in auto mode we defer to the
            # decision-directed QAM mode used by the equalizer; that matches
            # what the downstream soft demapper assumes.
            cpe_qam_mode = dd_qam_mode
            pilot_reference = None
            pilot_observed = None
            if per_frame_cpe_source == "raw-system-info":
                raw_bits = SYSTEM_INFO_VECTORS.get(int(system_info_index))
                if raw_bits is not None:
                    raw_reference = system_info_symbols(
                        raw_bits,
                        frame_body_mode=frame_body_mode,
                    )
                    raw_phase_gain = _phase_only_system_info_gain(
                        raw_reference,
                        raw_system_info,
                    )
                    if raw_phase_gain is not None:
                        pilot_reference = raw_reference
                        pilot_observed = (
                            raw_reference * raw_phase_gain
                        ).astype(np.complex64, copy=False)
            elif per_frame_cpe_source != "decision-directed" and cpe_pilot_matches:
                # Pick the system-info reference that matches what we actually
                # observed; this resolves the rectangular-QAM 4-fold rotational
                # ambiguity that a blind CPE estimator cannot recover.
                best_bits = SYSTEM_INFO_VECTORS.get(int(cpe_pilot_matches[0].index))
                if best_bits is not None:
                    try:
                        pilot_reference = system_info_symbols(
                            best_bits,
                            frame_body_mode=frame_body_mode,
                        ) * cpe_pilot_matches[0].polarity
                    except Exception:  # pragma: no cover - defensive fallback
                        pilot_reference = None
                    if pilot_reference is not None:
                        pilot_observed = frame_system_info
            cpe = correct_common_phase_error(
                data_symbols,
                qam_mode=cpe_qam_mode,
                max_relative_error=per_frame_cpe_max_relative_error,
                pilot_symbols=pilot_observed,
                pilot_reference=pilot_reference,
            )
            data_symbols = cpe.corrected_symbols
            per_frame_cpe_report = cpe.to_dict()
            cpe_source = (
                "raw-system-info"
                if per_frame_cpe_source == "raw-system-info"
                and pilot_observed is not None
                and pilot_reference is not None
                else (
                    "system-info-pilot"
                    if pilot_observed is not None and pilot_reference is not None
                    else "decision-directed"
                )
            )
            per_frame_cpe_report["source"] = (
                cpe_source
            )
            per_frame_cpe_report["requested_source"] = per_frame_cpe_source
        if csi_demap:
            if frame_csi_h2 is not None and frame_csi_h2.size:
                csi_logical = _frequency_deinterleave_variant(
                    frame_csi_h2.astype(np.complex64),
                    direction=frequency_deinterleaver_direction,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                )
                _csi_si, csi_data = split_system_info_and_data(csi_logical)
                csi_data = csi_data.real.astype(np.float32)
                if data_carrier_order == "reverse":
                    csi_data = csi_data[::-1].astype(np.float32, copy=False)
                if csi_data.size != data_symbols.size:
                    csi_data = np.ones(int(data_symbols.size), dtype=np.float32)
            else:
                csi_data = np.ones(int(data_symbols.size), dtype=np.float32)
            csi_weight_chunks.append(csi_data)
        data_symbol_chunks.append(data_symbols.astype(np.complex64, copy=False))
        data_symbols_per_frame.append(int(data_symbols.size))
        if dump_frame_fft_path is not None:
            is_c3780 = frame_body_mode == "C3780"
            frame_fft_dumps.append({
                "frame_start_symbols": frame.start,
                "body_samples": _raw_body_samples if is_c3780 else frame.body,
                "body_samples_after_pn_restore": body_samples if is_c3780 else None,
                "fft_raw": _fft_raw if is_c3780 else None,
                "fft_shifted": _fft_shifted if is_c3780 else None,
                "fft_equalized": spectrum if is_c3780 else None,
                "logical_carriers_raw": raw_deinterleaved if is_c3780 else None,
                "logical_carriers_equalized": deinterleaved if is_c3780 else None,
                "system_info_symbols_raw": raw_system_info,
                "system_info_symbols_post_eq": frame_system_info,
                "data_symbols_per_frame": data_symbols,
            })
        frame_reports.append(
            {
                "start": frame.start,
                "timing": frame_timing,
                "best_system_info": matches[0].to_dict() if matches else None,
                "system_info_top3": [match.to_dict() for match in matches[:3]],
                "expected_system_info": expected_system_info,
                "system_info_source": system_info_source,
                "system_info_independent": system_info_independent,
                "system_info_evidence": system_info_evidence,
                "supported_system_info_oracle": supported_system_info_oracle,
                "data_symbols": int(data_symbols.size),
                "body_window_offset_symbols": int(body_window_offset_symbols),
                "frequency_deinterleaver_direction": frequency_deinterleaver_direction,
                "data_carrier_order": data_carrier_order,
                "carrier_permutation": carrier_permutation,
                "logical_position_shift_symbols": int(logical_position_shift_symbols),
                "system_info_pilot_positions": system_info_pilot_positions.tolist(),
                "logical_to_spectrum_positions": logical_to_spectrum_positions.tolist(),
                "pn_channel": channel_report,
                "sparse_pilot_equalization": sparse_report,
                "per_frame_cpe_correction": per_frame_cpe_report,
                "system_info_iq": _complex_symbol_pairs(frame_system_info),
                "system_info_iq_source": "post-equalizer",
                "system_info_iq_note": (
                    "legacy post-equalizer SI IQ; prefer system_info_iq_by_source "
                    "for source-specific Gate C evidence"
                ),
                "system_info_iq_by_source": system_info_iq_by_source,
            }
        )

    data_symbol_stream = _concat_symbols(data_symbol_chunks)
    data_symbol_stream_before_deinterleave = data_symbol_stream
    system_info_summary = _summarize_system_info(
        frame_reports,
        min_metric=system_info_min_metric,
        min_agreement=system_info_min_agreement,
        expected_index=system_info_index,
        expected_frame_body_mode=frame_body_mode,
    )
    effective_config = _resolve_receive_config(
        qam_mode=qam_mode,
        fec_rate_index=fec_rate_index,
        symbol_deinterleave=symbol_deinterleave,
        system_info_summary=system_info_summary,
    )
    effective_symbol_deinterleave = effective_config["symbol_deinterleave"]
    effective_qam_mode = effective_config["qam_mode"]
    effective_fec_rate_index = effective_config["fec_rate_index"]
    symbol_deinterleave_latency = 0
    symbol_deinterleave_discarded = 0
    data_symbols_before_symbol_deinterleave = int(data_symbol_stream.size)
    symbol_deinterleave_phase_scan = None
    if scan_symbol_deinterleave_phases:
        symbol_deinterleave_phase_scan = _symbol_deinterleave_phase_scan(
            data_symbol_stream,
            mode=effective_symbol_deinterleave,
            qam_mode=effective_qam_mode,
            fec_rate_index=effective_fec_rate_index,
            max_codewords=scan_symbol_deinterleave_codewords,
        )
    if effective_symbol_deinterleave != "none":
        spec = SYMBOL_INTERLEAVERS[effective_symbol_deinterleave]  # type: ignore[index]
        if symbol_deinterleave_phase >= spec.branch_count:
            raise ValueError("symbol_deinterleave_phase must be 0..51")
        symbol_deinterleave_latency = spec.full_stream_latency_symbols
        data_symbol_stream = convolutional_deinterleave(
            data_symbol_stream,
            mode=effective_symbol_deinterleave,  # type: ignore[arg-type]
            phase=symbol_deinterleave_phase,
        )
        symbol_deinterleave_discarded = min(
            symbol_deinterleave_latency,
            int(data_symbol_stream.size),
        )
        data_symbol_stream = data_symbol_stream[symbol_deinterleave_discarded:]
    data_symbol_stream_after_deinterleave = data_symbol_stream
    branch_gain_report = _branch_gain_report(
        selected_branches=None,
        reliability_threshold=branch_gain_reliability_threshold,
        min_symbols=branch_gain_min_symbols,
    )
    if branch_gain_branches is not None and data_symbol_stream_after_deinterleave.size:
        data_symbol_stream_after_deinterleave, branch_gain_report = (
            _apply_post_deinterleave_branch_gain_correction(
                data_symbol_stream_after_deinterleave,
                qam_mode=effective_qam_mode,
                interleaver_mode=effective_symbol_deinterleave,
                interleaver_phase=symbol_deinterleave_phase,
                selected_branches=branch_gain_branches,
                reliability_threshold=branch_gain_reliability_threshold,
                min_symbols=branch_gain_min_symbols,
            )
        )
    data_symbol_stream = data_symbol_stream_after_deinterleave
    csi_demap_report: dict[str, Any] | None = None
    post_csi_weights: np.ndarray | None = None
    if csi_demap:
        csi_weight_stream = (
            np.concatenate(csi_weight_chunks)
            if csi_weight_chunks
            else np.empty(0, dtype=np.float32)
        )
        post_count = int(data_symbol_stream.size)
        if csi_weight_stream.size and post_count:
            if effective_symbol_deinterleave != "none":
                weight_sources = convolutional_deinterleave_source_indices(
                    post_count,
                    mode=effective_symbol_deinterleave,  # type: ignore[arg-type]
                    phase=symbol_deinterleave_phase,
                )
                post_csi_weights = csi_weight_stream[weight_sources]
            else:
                post_csi_weights = csi_weight_stream[:post_count]
            positive = post_csi_weights > 0.0
            scale_ref = (
                float(np.median(post_csi_weights[positive]))
                if bool(np.any(positive))
                else 1.0
            )
            post_csi_weights = np.clip(
                post_csi_weights / max(scale_ref, 1e-12), 0.0, 4.0
            ).astype(np.float32)
        csi_demap_report = {
            "enabled": True,
            "weighted_symbols": int(
                post_csi_weights.size if post_csi_weights is not None else 0
            ),
            "weight_p10": (
                float(np.percentile(post_csi_weights, 10))
                if post_csi_weights is not None and post_csi_weights.size
                else None
            ),
            "weight_median": 1.0 if post_csi_weights is not None else None,
            "weight_p90": (
                float(np.percentile(post_csi_weights, 90))
                if post_csi_weights is not None and post_csi_weights.size
                else None
            ),
            "mmse_unbias_gain_floor": 0.25,
        }
    raw_demapped_llr = qam_soft_demodulate(data_symbol_stream, mode=effective_qam_mode)
    if post_csi_weights is not None and raw_demapped_llr.size:
        bit_weights = np.repeat(
            post_csi_weights, _bits_per_symbol(effective_qam_mode)
        )[: raw_demapped_llr.size]
        raw_demapped_llr = (raw_demapped_llr * bit_weights).astype(np.float32)
    carrier_erase_report = _carrier_erase_report(
        metric=carrier_erase_metric,
        fraction=carrier_erase_fraction,
        reliability_threshold=carrier_erase_reliability_threshold,
    )
    if carrier_erase_metric != "none" and carrier_erase_fraction > 0.0 and raw_demapped_llr.size:
        carrier_mask, quality_summary = _carrier_erase_mask(
            data_symbol_stream_before_deinterleave,
            qam_mode=effective_qam_mode,
            metric=carrier_erase_metric,
            fraction=carrier_erase_fraction,
            reliability_threshold=carrier_erase_reliability_threshold,
        )
        symbol_mask = carrier_mask[
            _source_carriers_for_post_deinterleaver_symbols(
                int(data_symbol_stream_after_deinterleave.size),
                mode=effective_symbol_deinterleave,
                phase=symbol_deinterleave_phase,
            )
        ]
        bit_mask = np.repeat(symbol_mask, _bits_per_symbol(effective_qam_mode))[
            : raw_demapped_llr.size
        ]
        raw_demapped_llr = np.array(raw_demapped_llr, dtype=np.float32, copy=True)
        raw_demapped_llr[bit_mask] = np.float32(0.0)
        carrier_erase_report = _carrier_erase_report(
            metric=carrier_erase_metric,
            fraction=carrier_erase_fraction,
            reliability_threshold=carrier_erase_reliability_threshold,
            quality_summary=quality_summary,
            erased_carriers=int(np.count_nonzero(carrier_mask)),
            erased_symbols=int(np.count_nonzero(symbol_mask)),
            erased_llrs=int(np.count_nonzero(bit_mask)),
            llr_count=int(bit_mask.size),
        )
    demapped_llr = _condition_demapped_llr(
        raw_demapped_llr,
        scale=llr_scale,
        clip=llr_clip,
        erase_fraction=llr_erase_fraction,
        bits_per_symbol=_bits_per_symbol(effective_qam_mode),
        plane_scales=llr_plane_scales,
    )
    llr_conditioning_report = _llr_conditioning_report(
        raw_demapped_llr,
        demapped_llr,
        scale=llr_scale,
        clip=llr_clip,
        erase_fraction=llr_erase_fraction,
        bits_per_symbol=_bits_per_symbol(effective_qam_mode),
        plane_scales=llr_plane_scales,
    )
    qam_quality_report = _qam_symbol_quality_report(
        data_symbol_stream_before_deinterleave,
        data_symbol_stream_after_deinterleave,
        qam_mode=effective_qam_mode,
    )
    if llr_output_path is not None:
        llr_path = Path(llr_output_path)
        llr_path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(demapped_llr, dtype="<f4").tofile(str(llr_path))
    demapped_bits = (demapped_llr < 0).astype(np.uint8)
    bits_per_frame = (
        data_symbols_per_frame[0]
        * _bits_per_symbol(effective_qam_mode)
        if data_symbols_per_frame
        else 0
    )
    fec_outcome = decode_llr_stream_to_transport_bytes(
        demapped_llr,
        demapped_bits=demapped_bits,
        bits_per_frame=bits_per_frame,
        fec_mode=fec_mode,
        fec_rate_index=effective_fec_rate_index,
        ldpc_parity_position=ldpc_parity_position,
        uncoded_payload=uncoded_payload,
    )
    ldpc_parity_report = fec_outcome["ldpc_parity_check"]
    fec_reports = fec_outcome["frame_reports"]
    payload_bytes = fec_outcome["payload_bytes"]
    pre_ldpc_dump_sidecar_path = None
    if dump_frame_fft_path is not None:
        _write_frame_fft_dump(dump_frame_fft_path, frame_fft_dumps)

    if pre_ldpc_dump_path is not None:
        pre_ldpc_dump_sidecar_path = _write_pre_ldpc_dump(
            pre_ldpc_dump_path,
            capture_path=capture_path,
            llr=demapped_llr,
            hard_bits=demapped_bits,
            symbols_before_deinterleave=data_symbol_stream_before_deinterleave,
            symbols_after_deinterleave=data_symbol_stream_after_deinterleave,
            data_symbols_per_frame=data_symbols_per_frame,
            frame_reports=frame_reports,
            effective_qam_mode=effective_qam_mode,
            requested_qam_mode=qam_mode,
            effective_fec_rate_index=effective_fec_rate_index,
            requested_fec_rate_index=fec_rate_index,
            effective_symbol_deinterleave=effective_symbol_deinterleave,
            requested_symbol_deinterleave=symbol_deinterleave,
            symbol_deinterleave_phase=symbol_deinterleave_phase,
            symbol_deinterleave_latency=symbol_deinterleave_latency,
            symbol_deinterleave_discarded=symbol_deinterleave_discarded,
            bits_per_frame=bits_per_frame,
            equalizer=equalizer,
            correct_per_frame_cpe=correct_per_frame_cpe,
            frame_body_mode=frame_body_mode,
            mode=mode,
            fft_bin_shift=fft_bin_shift,
            body_window_offset_symbols=body_window_offset_symbols,
            frequency_deinterleaver_direction=frequency_deinterleaver_direction,
            data_carrier_order=data_carrier_order,
            carrier_permutation=carrier_permutation,
            logical_position_shift_symbols=logical_position_shift_symbols,
            system_info_index=system_info_index,
            system_info_summary=system_info_summary,
            acquisition=acquisition,
            qam_quality_report=qam_quality_report,
            dd_max_hard_bit_bias=dd_max_hard_bit_bias,
            llr_conditioning=llr_conditioning_report,
            carrier_erase=carrier_erase_report,
            branch_gain_correction=branch_gain_report,
            symbols_only=pre_ldpc_dump_symbols_only,
        )
    if llr_output_path is not None:
        llr_path = Path(llr_output_path)
        sidecar_path = llr_path.with_suffix(llr_path.suffix + ".json")
        sidecar = {
            "capture_path": str(capture_path),
            "llr_file": str(llr_path),
            "llr_dtype": "<f4",
            "llr_count": int(demapped_llr.size),
            "qam_mode": effective_qam_mode,
            "bits_per_symbol": int(_bits_per_symbol(effective_qam_mode)),
            "bits_per_frame": int(bits_per_frame),
            "symbol_deinterleave": effective_symbol_deinterleave,
            "symbol_deinterleave_phase": int(symbol_deinterleave_phase),
            "symbol_deinterleave_latency_symbols": int(symbol_deinterleave_latency),
            "symbol_deinterleave_discarded_symbols": int(symbol_deinterleave_discarded),
            "fec_rate_index": int(effective_fec_rate_index),
            "ldpc_parity_position": ldpc_parity_position,
            "system_info_index": int(system_info_index),
            "frame_body_mode": frame_body_mode,
            "frames_used": int(len(frame_reports)),
            "llr_conditioning": llr_conditioning_report,
            "carrier_erase": carrier_erase_report,
            "branch_gain_correction": branch_gain_report,
        }
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ts_candidates = (
        analyze_ts_lock_candidates(payload_bytes, min_packets=min_ts_packets)
        if payload_bytes
        else []
    )
    selected_candidate = next(
        (
            candidate
            for candidate in ts_candidates
            if candidate.lock.sync_ratio >= min_ts_sync_ratio
            and candidate.valid_packet_ratio >= min_ts_valid_ratio
        ),
        None,
    )
    selected_lock = selected_candidate.lock if selected_candidate else None
    packets = extract_ts_packets(payload_bytes, selected_lock) if selected_lock else []
    stream_summary = selected_candidate.stream if selected_candidate else None
    if output_path is not None:
        _write_ts_bytes(output_path, b"".join(packets))

    return {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "resample_up": up,
        "resample_down": down,
        "frequency_shift_hz": frequency_shift_hz,
        "input_skip_samples": input_skip_samples,
        "mode": mode,
        "qam_mode": effective_qam_mode,
        "requested_qam_mode": qam_mode,
        "equalizer": equalizer,
        "dd_max_hard_bit_bias": float(dd_max_hard_bit_bias),
        "frame_body_mode": frame_body_mode,
        "fft_bin_shift": fft_bin_shift,
        "body_window_offset_symbols": int(body_window_offset_symbols),
        "frequency_deinterleaver_direction": frequency_deinterleaver_direction,
        "data_carrier_order": data_carrier_order,
        "carrier_permutation": carrier_permutation,
        "logical_position_shift_symbols": int(logical_position_shift_symbols),
        "system_info_pilot_positions": system_info_pilot_positions.tolist(),
        "logical_to_spectrum_positions": logical_to_spectrum_positions.tolist(),
        "per_frame_cpe_source": per_frame_cpe_source,
        "use_family_delay": use_family_delay,
        "family_frames": family_frames,
        "max_delay_symbols": max_delay_symbols,
        "family_hit_threshold": family_hit_threshold,
        "timing_jobs": timing_jobs,
        "timing_per_candidate_cfo": timing_per_candidate_cfo,
        "pn_payload_timing_refine": bool(pn_payload_timing_refine),
        "timing_policy": timing_policy,
        "timing_trajectory_path": str(timing_trajectory_path)
        if timing_trajectory_path is not None
        else None,
        "timing_trajectory": _compact_receiver_trajectory(trajectory_report),
        "symbol_deinterleave": effective_symbol_deinterleave,
        "requested_symbol_deinterleave": symbol_deinterleave,
        "symbol_deinterleave_phase": symbol_deinterleave_phase,
        "symbol_deinterleave_latency_symbols": symbol_deinterleave_latency,
        "symbol_deinterleave_discarded_symbols": symbol_deinterleave_discarded,
        "symbol_deinterleave_phase_scan": symbol_deinterleave_phase_scan,
        "data_symbols_before_symbol_deinterleave": data_symbols_before_symbol_deinterleave,
        "data_symbols_after_symbol_deinterleave": int(data_symbol_stream.size),
        "qam_symbol_quality": qam_quality_report,
        "uncoded_payload": uncoded_payload,
        "system_info": system_info_summary,
        "receive_config": effective_config,
        "per_frame_cpe": _summarize_per_frame_cpe(
            frame_reports,
            enabled=correct_per_frame_cpe,
            max_relative_error=per_frame_cpe_max_relative_error,
            requested_source=per_frame_cpe_source,
        ),
        "llr_conditioning": llr_conditioning_report,
        "carrier_erase": carrier_erase_report,
        "csi_demap": csi_demap_report,
        "branch_gain_correction": branch_gain_report,
        "fec": {
            "mode": fec_mode,
            "fec_rate_index": effective_fec_rate_index,
            "requested_fec_rate_index": fec_rate_index,
            "ldpc_parity_position": ldpc_parity_position,
            "ldpc_parity_check": ldpc_parity_report,
            "frame_reports": fec_reports,
            "ldpc_decode": "enabled" if fec_mode == "ldpc" else "not_run",
            "note": (
                "LDPC mode uses the standard QC parity-check table with "
                "normalized min-sum decoding. Capture recovery still depends "
                "on correct synchronization, equalization, and QAM mapping."
            ),
        },
        "acquisition": acquisition,
        "wideband_channel": wideband_channel_report,
        "residual_cfo_hz": residual_cfo_hz,
        "frames_used": len(frame_reports),
        "demapped_bits": int(demapped_bits.size),
        "demapped_llrs": int(demapped_llr.size),
        "llr_output_path": str(llr_output_path) if llr_output_path is not None else None,
        "pre_ldpc_dump_path": str(pre_ldpc_dump_path)
        if pre_ldpc_dump_path is not None
        else None,
        "pre_ldpc_dump_json_path": str(pre_ldpc_dump_sidecar_path)
        if pre_ldpc_dump_sidecar_path is not None
        else None,
        "payload_bytes": len(payload_bytes),
        "ts": {
            "lock": selected_lock.to_dict() if selected_lock else None,
            "candidate_locks": [candidate.to_dict() for candidate in ts_candidates[:8]],
            "min_sync_ratio": min_ts_sync_ratio,
            "min_valid_ratio": min_ts_valid_ratio,
            "packet_count": len(packets),
            "stream": stream_summary.to_dict() if stream_summary else None,
            "output_path": str(output_path) if output_path is not None else None,
        },
        "frames": frame_reports,
    }


def receive_streaming(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    buffer_mb: int = 200,
    batch_frames: int = 16,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    conjugate_iq: bool = False,
    mode: PnMode = "pn945",
    phase_offset: int | None = None,
    qam_mode: QamMode | Literal["auto"] = "64qam",
    equalizer: str = "sparse",
    dd_max_hard_bit_bias: float = 0.0,
    symbol_deinterleave: InterleaverMode | Literal["none", "auto"] | str = "none",
    symbol_deinterleave_phase: int = 0,
    fec_mode: str = "none",
    fec_rate_index: int | Literal["auto"] = 3,
    ldpc_parity_position: str = "front",
    system_info_index: int = 23,
    fft_bin_shift: int = 0,
    body_window_offset_symbols: int = 0,
    frequency_deinterleaver_direction: Literal["standard", "inverse"] | str = "standard",
    data_carrier_order: Literal["normal", "reverse"] | str = "normal",
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
    frame_body_mode: str = "C3780",
    timing_trajectory_path: str | Path | None = None,
    track_residual_cfo: bool = True,
    uncoded_payload: bool = False,
    correct_per_frame_cpe: bool = False,
    per_frame_cpe_max_relative_error: float = 0.7,
    per_frame_cpe_source: Literal["auto", "pilot", "decision-directed", "raw-system-info"] | str = "auto",
    llr_output_path: str | Path | None = None,
    llr_scale: float = 1.0,
    llr_clip: float | None = None,
    llr_erase_fraction: float = 0.0,
    llr_plane_scales: Sequence[float] | str | None = None,
    carrier_erase_metric: str = "none",
    carrier_erase_fraction: float = 0.0,
    carrier_erase_reliability_threshold: float = 0.55,
    csi_demap: bool = False,
    branch_gain_branches: Sequence[int] | str | None = None,
    branch_gain_reliability_threshold: float = 0.55,
    branch_gain_min_symbols: int = 32,
    pn_equalizer_noise_variance: float | Literal["auto"] | None = None,
    pn_channel_taps: int | None = None,
    pn_estimator: str = "compact",
    pn_wideband_block_frames: int = 16,
) -> int:
    """Streaming mode for real-time DTMB processing with bounded memory.

    Processes frames in batches, emits LLRs incrementally, and maintains a
    sliding window for timing trajectory. Designed for live HackRF streaming.
    """
    import sys
    from collections import deque

    # Configuration and validation
    if str(capture_path) != "-":
        raise ValueError("streaming mode requires stdin input (use '-' as capture path)")

    if llr_output_path is None or str(llr_output_path) != "-":
        raise ValueError("streaming mode requires --llr-output - for stdout")

    if timing_trajectory_path is None:
        raise ValueError("streaming mode requires --timing-trajectory")

    max_buffer_bytes = buffer_mb * 1024 * 1024
    definition = PN_DEFINITIONS[mode]
    samples_per_frame = definition.frame_symbols

    # Load timing trajectory
    trajectory_report = _load_timing_trajectory(
        timing_trajectory=None,
        timing_trajectory_path=timing_trajectory_path,
    )

    # Initialize streaming state
    input_buffer = bytearray()
    processed_samples = 0
    total_frames_processed = 0
    timing_window: deque[dict[str, Any]] = deque(maxlen=50)  # Keep last 50 frame timings
    llr_output = sys.stdout.buffer

    # Telemetry via stderr
    def emit_telemetry(**kwargs: Any) -> None:
        telemetry = {"timestamp": total_frames_processed, **kwargs}
        sys.stderr.write(json.dumps(telemetry) + "\n")
        sys.stderr.flush()

    emit_telemetry(event="streaming_start", buffer_mb=buffer_mb, batch_frames=batch_frames)

    try:
        while True:
            # Read chunk from stdin
            chunk = sys.stdin.buffer.read(65536)
            if not chunk:
                # EOF reached
                emit_telemetry(event="eof", total_frames=total_frames_processed)
                break

            input_buffer.extend(chunk)

            # Check buffer pressure
            if len(input_buffer) > max_buffer_bytes:
                emit_telemetry(
                    event="backpressure_warning",
                    buffer_bytes=len(input_buffer),
                    max_bytes=max_buffer_bytes,
                )

            # Process batches
            while True:
                # Calculate needed bytes for next batch
                needed_samples = (batch_frames + 2) * samples_per_frame  # Extra for frame boundaries
                needed_bytes = needed_samples * 2  # CI8 = 2 bytes per sample

                if len(input_buffer) < needed_bytes:
                    break

                # Extract batch
                batch_bytes = bytes(input_buffer[:needed_bytes])
                input_buffer = input_buffer[needed_bytes:]

                # Convert CI8 to complex
                batch_samples = np.frombuffer(batch_bytes, dtype=np.int8).astype(np.float32)
                batch_iq = batch_samples[0::2] + 1j * batch_samples[1::2]
                batch_iq = batch_iq.astype(np.complex64) / 127.0

                # Conditioning
                batch_iq = remove_dc(batch_iq)
                if conjugate_iq:
                    batch_iq = np.conj(batch_iq)
                if frequency_shift_hz:
                    batch_iq = frequency_shift(
                        batch_iq,
                        sample_rate_sps=sample_rate_sps,
                        shift_hz=frequency_shift_hz,
                    )

                # Resample if needed
                if sample_rate_sps != symbol_rate_sps:
                    batch_iq, _, _ = resample_to_symbol_rate(
                        batch_iq,
                        sample_rate_sps=sample_rate_sps,
                        symbol_rate_sps=symbol_rate_sps,
                    )

                # Process batch using trajectory timing
                batch_llr = _process_streaming_batch(
                    batch_iq,
                    mode=mode,
                    phase_offset=phase_offset,
                    trajectory_report=trajectory_report,
                    frame_offset=total_frames_processed,
                    batch_frames=batch_frames,
                    qam_mode=qam_mode,
                    equalizer=equalizer,
                    dd_max_hard_bit_bias=dd_max_hard_bit_bias,
                    symbol_deinterleave=symbol_deinterleave,
                    symbol_deinterleave_phase=symbol_deinterleave_phase,
                    fec_rate_index=fec_rate_index,
                    system_info_index=system_info_index,
                    fft_bin_shift=fft_bin_shift,
                    body_window_offset_symbols=body_window_offset_symbols,
                    frequency_deinterleaver_direction=frequency_deinterleaver_direction,
                    data_carrier_order=data_carrier_order,
                    carrier_permutation=carrier_permutation,
                    logical_position_shift_symbols=logical_position_shift_symbols,
                    frame_body_mode=frame_body_mode,
                    track_residual_cfo=track_residual_cfo,
                    correct_per_frame_cpe=correct_per_frame_cpe,
                    per_frame_cpe_max_relative_error=per_frame_cpe_max_relative_error,
                    per_frame_cpe_source=per_frame_cpe_source,
                    llr_scale=llr_scale,
                    llr_clip=llr_clip,
                    llr_erase_fraction=llr_erase_fraction,
                    llr_plane_scales=llr_plane_scales,
                    carrier_erase_metric=carrier_erase_metric,
                    carrier_erase_fraction=carrier_erase_fraction,
                    carrier_erase_reliability_threshold=carrier_erase_reliability_threshold,
                    csi_demap=csi_demap,
                    branch_gain_branches=branch_gain_branches,
                    branch_gain_reliability_threshold=branch_gain_reliability_threshold,
                    branch_gain_min_symbols=branch_gain_min_symbols,
                    pn_equalizer_noise_variance=pn_equalizer_noise_variance,
                    pn_channel_taps=pn_channel_taps,
                    pn_estimator=pn_estimator,
                    symbol_rate_sps=symbol_rate_sps,
                )

                # Write LLR output immediately
                if batch_llr is not None and batch_llr.size > 0:
                    np.asarray(batch_llr, dtype="<f4").tofile(llr_output)
                    llr_output.flush()

                total_frames_processed += batch_frames
                processed_samples += needed_samples

                # Emit telemetry
                emit_telemetry(
                    event="batch_complete",
                    frames_processed=batch_frames,
                    total_frames=total_frames_processed,
                    buffer_bytes=len(input_buffer),
                    llr_count=int(batch_llr.size) if batch_llr is not None else 0,
                )

                # Clear processed data
                batch_iq = None
                batch_llr = None

    except KeyboardInterrupt:
        emit_telemetry(event="interrupted", total_frames=total_frames_processed)
        return 1
    except Exception as e:
        emit_telemetry(event="error", error=str(e), total_frames=total_frames_processed)
        raise

    emit_telemetry(event="streaming_complete", total_frames=total_frames_processed)
    return 0


def _process_streaming_batch(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int | None,
    trajectory_report: dict[str, Any] | None,
    frame_offset: int,
    batch_frames: int,
    qam_mode: QamMode | Literal["auto"],
    equalizer: str,
    dd_max_hard_bit_bias: float,
    symbol_deinterleave: InterleaverMode | Literal["none", "auto"] | str,
    symbol_deinterleave_phase: int,
    fec_rate_index: int | Literal["auto"],
    system_info_index: int,
    fft_bin_shift: int,
    body_window_offset_symbols: int,
    frequency_deinterleaver_direction: str,
    data_carrier_order: str,
    carrier_permutation: str,
    logical_position_shift_symbols: int,
    frame_body_mode: str,
    track_residual_cfo: bool,
    correct_per_frame_cpe: bool,
    per_frame_cpe_max_relative_error: float,
    per_frame_cpe_source: str,
    llr_scale: float,
    llr_clip: float | None,
    llr_erase_fraction: float,
    llr_plane_scales: Sequence[float] | str | None,
    carrier_erase_metric: str,
    carrier_erase_fraction: float,
    carrier_erase_reliability_threshold: float,
    csi_demap: bool,
    branch_gain_branches: Sequence[int] | str | None,
    branch_gain_reliability_threshold: float,
    branch_gain_min_symbols: int,
    pn_equalizer_noise_variance: float | Literal["auto"] | None,
    pn_channel_taps: int | None,
    pn_estimator: str,
    symbol_rate_sps: int,
) -> np.ndarray | None:
    """Process a single batch of frames and return LLRs."""

    # Use minimal receive_capture with trajectory timing
    # This reuses the existing frame processing logic
    effective_qam_mode = (
        str(qam_mode)
        if qam_mode in ("4qam", "16qam", "64qam")
        else transmission_parameters_for_index(
            int(system_info_index),
            frame_body_mode=frame_body_mode,
        ).qam_mode
    )

    effective_fec_rate = (
        int(fec_rate_index)
        if fec_rate_index != "auto"
        else transmission_parameters_for_index(
            int(system_info_index),
            frame_body_mode=frame_body_mode,
        ).fec_rate_index or 3
    )

    effective_symbol_deinterleave = (
        str(symbol_deinterleave)
        if symbol_deinterleave != "auto"
        else transmission_parameters_for_index(
            int(system_info_index),
            frame_body_mode=frame_body_mode,
        ).interleaver_mode or "none"
    )

    # Extract frames using trajectory (simplified - just get phase offsets)
    trajectory = _trajectory_payload(trajectory_report)
    if trajectory is None:
        return None

    # Find trajectory segment for this frame offset
    frame_entries = _select_frame_entries(
        symbols,
        mode=mode,
        max_frames=batch_frames,
        timing_policy="trajectory",
        trajectory_report=trajectory_report,
        fallback_phase_offset=phase_offset or 0,
        fallback_cfo_hz=None,
        symbol_rate_sps=symbol_rate_sps,
    )

    if not frame_entries:
        return None

    # Process frames (simplified path - no diagnostics)
    data_symbol_chunks: list[np.ndarray] = []

    for frame, _ in frame_entries[:batch_frames]:
        if frame_body_mode == "C3780":
            body_samples = _frame_body_samples(
                frame,
                symbols,
                mode=mode,
                body_window_offset_symbols=body_window_offset_symbols,
            )
            spectrum = frame_body_fft(body_samples)
            if fft_bin_shift:
                spectrum = np.roll(spectrum, fft_bin_shift)

            # Apply equalizer (simplified)
            if equalizer == "sparse":
                system_info_pilot_positions = np.asarray(
                    system_info_positions_for_extraction(
                        carrier_permutation,
                        int(logical_position_shift_symbols),
                    ),
                    dtype=np.int32,
                )
                logical_to_spectrum_positions = _logical_to_spectrum_positions_for_direction(
                    frequency_deinterleaver_direction
                )
                sparse = equalize_c3780_spectrum_with_system_info_pilots(
                    spectrum,
                    system_info_index=system_info_index,
                    frame_body_mode="C3780",
                    system_info_positions=system_info_pilot_positions,
                    logical_to_spectrum_positions=logical_to_spectrum_positions,
                )
                spectrum = sparse.equalized

            inserted = _frequency_deinterleave_inserted_variant(
                spectrum,
                direction=frequency_deinterleaver_direction,
            )
            deinterleaved = _logical_from_inserted_c3780(
                inserted,
                carrier_permutation=carrier_permutation,
                logical_position_shift_symbols=logical_position_shift_symbols,
            )
            _, data_symbols = split_system_info_and_data(deinterleaved)
            if data_carrier_order == "reverse":
                data_symbols = data_symbols[::-1].astype(np.complex64, copy=False)

            data_symbol_chunks.append(data_symbols.astype(np.complex64, copy=False))

    if not data_symbol_chunks:
        return None

    data_symbol_stream = _concat_symbols(data_symbol_chunks)

    # Symbol deinterleave
    if effective_symbol_deinterleave != "none":
        spec = SYMBOL_INTERLEAVERS[effective_symbol_deinterleave]  # type: ignore[index]
        data_symbol_stream = convolutional_deinterleave(
            data_symbol_stream,
            mode=effective_symbol_deinterleave,  # type: ignore[arg-type]
            phase=symbol_deinterleave_phase,
        )
        discarded = min(spec.full_stream_latency_symbols, int(data_symbol_stream.size))
        data_symbol_stream = data_symbol_stream[discarded:]

    # Soft demodulate
    raw_demapped_llr = qam_soft_demodulate(data_symbol_stream, mode=effective_qam_mode)  # type: ignore[arg-type]

    # Condition LLRs
    demapped_llr = _condition_demapped_llr(
        raw_demapped_llr,
        scale=llr_scale,
        clip=llr_clip,
        erase_fraction=llr_erase_fraction,
        bits_per_symbol=_bits_per_symbol(effective_qam_mode),  # type: ignore[arg-type]
        plane_scales=llr_plane_scales,
    )

    return demapped_llr


def decode_llr_stream_to_transport_bytes(
    demapped_llr: np.ndarray,
    *,
    demapped_bits: np.ndarray | None = None,
    bits_per_frame: int,
    fec_mode: str,
    fec_rate_index: int,
    ldpc_parity_position: str = "front",
    uncoded_payload: bool = False,
) -> dict[str, Any]:
    """Run the FEC + descrambling stage on a flat post-deinterleaver LLR stream.

    This is the seam that lets the pipeline take apart the receiver: any
    upstream stage that produces per-frame LLRs (the in-tree receiver, or
    an offline LLR dump loaded by a pipeline script, or a future external
    demapper) can pass its stream here and get the same (payload bytes,
    per-frame FEC reports, LDPC parity consistency check) contract the
    live receiver uses internally.

    Returns a dict with keys:
        - ``payload_bytes`` (``bytes``): transport bytes from descrambled BCH.
        - ``frame_reports`` (``list[dict]``): per-frame FEC diagnostic.
        - ``ldpc_parity_check`` (``dict``): hard-decision parity check and
          Appendix-B syndrome summary, same schema as before.
    """

    values = np.asarray(demapped_llr, dtype=np.float32).reshape(-1)
    if demapped_bits is None:
        demapped_bits = (values < 0).astype(np.uint8)
    else:
        demapped_bits = np.asarray(demapped_bits, dtype=np.uint8).reshape(-1)
        if demapped_bits.size != values.size:
            raise ValueError("demapped_bits and demapped_llr must have matching length")

    ldpc_parity_report = _ldpc_parity_report(
        demapped_bits,
        fec_rate_index=fec_rate_index,
    )
    fec_reports: list[dict[str, Any]] = []
    if uncoded_payload:
        payload_bytes = pack_bits_msb(demapped_bits, truncate=True)
    elif fec_mode == "bch-assume-ldpc":
        payload_parts: list[bytes] = []
        for frame_bits in _iter_frame_bits(demapped_bits, bits_per_frame=bits_per_frame):
            try:
                result = decode_frame_bch_descramble_from_ldpc_codewords(
                    frame_bits,
                    fec_rate_index=fec_rate_index,
                    parity_position=ldpc_parity_position,  # type: ignore[arg-type]
                )
            except ValueError as exc:
                fec_reports.append({"error": str(exc), "input_bits": int(frame_bits.size)})
                continue
            fec_reports.append(result.to_dict())
            payload_parts.append(result.transport_bytes)
        payload_bytes = b"".join(payload_parts)
    elif fec_mode == "ldpc":
        payload_parts = []
        for frame_llr in _iter_frame_llrs(values, llrs_per_frame=bits_per_frame):
            try:
                result = decode_frame_bch_descramble_from_ldpc_llr(
                    frame_llr,
                    fec_rate_index=fec_rate_index,
                )
            except ValueError as exc:
                fec_reports.append({"error": str(exc), "input_llrs": int(frame_llr.size)})
                continue
            fec_reports.append(result.to_dict())
            payload_parts.append(result.transport_bytes)
        payload_bytes = b"".join(payload_parts)
    elif fec_mode == "none":
        payload_bytes = b""
    else:
        raise ValueError("unsupported fec_mode")
    return {
        "payload_bytes": payload_bytes,
        "frame_reports": fec_reports,
        "ldpc_parity_check": ldpc_parity_report,
    }


def _frame_body_samples(
    frame: SignalFrame,
    symbols: np.ndarray,
    *,
    mode: PnMode,
    body_window_offset_symbols: int,
) -> np.ndarray:
    """Return a C=3780 body slice with an optional diagnostic window offset."""

    if body_window_offset_symbols == 0:
        return frame.body
    definition = PN_DEFINITIONS[mode]
    signal = np.asarray(symbols, dtype=np.complex64)
    body_start = (
        int(frame.start)
        + int(definition.header_symbols)
        + int(body_window_offset_symbols)
    )
    body_stop = body_start + FRAME_BODY_SYMBOLS
    if body_start < 0 or body_stop > signal.size:
        raise ValueError("body_window_offset_symbols moves the C3780 body outside the capture")
    return signal[body_start:body_stop].astype(np.complex64, copy=False)


def _frequency_deinterleave_variant(
    symbols: np.ndarray,
    *,
    direction: str,
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
) -> np.ndarray:
    return _logical_from_inserted_c3780(
        _frequency_deinterleave_inserted_variant(symbols, direction=direction),
        carrier_permutation=carrier_permutation,
        logical_position_shift_symbols=logical_position_shift_symbols,
    )


def _frequency_deinterleave_inserted_variant(
    symbols: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frequency deinterleaver requires 3780 symbols")
    if direction == "standard":
        return frequency_deinterleave_inserted(values)
    if direction == "inverse":
        return values[frequency_deinterleave_index_map()]
    raise ValueError("frequency-deinterleaver direction must be standard or inverse")


def _logical_to_spectrum_positions_for_direction(direction: str) -> np.ndarray:
    if direction == "standard":
        return frequency_interleave_index_map()
    if direction == "inverse":
        return frequency_deinterleave_index_map()
    raise ValueError("frequency-deinterleaver direction must be standard or inverse")


def _logical_from_inserted_c3780(
    symbols: np.ndarray,
    *,
    carrier_permutation: str = "identity",
    logical_position_shift_symbols: int = 0,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.size != FRAME_BODY_SYMBOLS:
        raise ValueError("frame body requires 3780 symbols")
    values = apply_carrier_permutation_to_inserted(
        values,
        permutation_name=carrier_permutation,
    )
    if int(logical_position_shift_symbols) != 0:
        values = np.roll(values, -int(logical_position_shift_symbols))
    return np.concatenate(
        (
            values[SYSTEM_INFO_POSITIONS],
            np.delete(values, SYSTEM_INFO_POSITIONS),
        )
    )


def _load_symbols(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int,
    max_samples: int,
    frequency_shift_hz: float,
    input_skip_samples: int = 0,
    conjugate_iq: bool = False,
) -> tuple[np.ndarray, int, int]:
    if input_skip_samples < 0:
        raise ValueError("input_skip_samples must be non-negative")
    samples = read_ci8(
        capture_path,
        max_samples=max_samples,
        skip_samples=input_skip_samples,
    )
    samples = remove_dc(samples)
    if conjugate_iq:
        # Reverses the baseband spectrum around DC. A HackRF front end that
        # delivers a mirrored IQ stream (common when mixing above vs below
        # the target RF) would otherwise feed every real-valued DSP stage
        # (PN cyclic-extension correlation, 64QAM constellation) a correct
        # answer while routing each data carrier onto the bin that its
        # mirror image expected, scrambling LDPC bits in exactly the way
        # the 0.5 plateau shows.
        samples = np.conj(samples)
    if frequency_shift_hz:
        samples = frequency_shift(
            samples,
            sample_rate_sps=sample_rate_sps,
            shift_hz=frequency_shift_hz,
        )
    if sample_rate_sps == symbol_rate_sps:
        return samples.astype(np.complex64, copy=False), 1, 1
    symbols, up, down = resample_to_symbol_rate(
        samples,
        sample_rate_sps=sample_rate_sps,
        symbol_rate_sps=symbol_rate_sps,
    )
    return symbols, up, down


def _load_timing_trajectory(
    *,
    timing_trajectory: dict[str, Any] | None,
    timing_trajectory_path: str | Path | None,
) -> dict[str, Any] | None:
    if timing_trajectory is not None:
        return timing_trajectory
    if timing_trajectory_path is None:
        return None
    return json.loads(Path(timing_trajectory_path).read_text(encoding="utf-8"))


def _select_frame_entries(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    max_frames: int,
    timing_policy: Literal["fixed", "windowed", "trajectory"] | str,
    trajectory_report: dict[str, Any] | None,
    fallback_phase_offset: int,
    fallback_cfo_hz: float | None,
    symbol_rate_sps: int,
) -> list[tuple[SignalFrame, dict[str, Any]]]:
    """Return frame slices paired with timing metadata for diagnostics."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if fallback_phase_offset < 0:
        raise ValueError("fallback_phase_offset must be non-negative")

    policy = str(timing_policy)
    if policy == "fixed":
        return [
            (
                frame,
                {
                    "policy": "fixed",
                    "source": "acquisition",
                    "nominal_frame_index": index,
                    "phase_offset": int(fallback_phase_offset),
                    "coarse_cfo_hz": fallback_cfo_hz,
                    "residual_cfo_hz": 0.0,
                },
            )
            for index, frame in enumerate(
                iter_signal_frames(
                    symbols,
                    mode=mode,
                    phase_offset=fallback_phase_offset,
                    max_frames=max_frames,
                )
            )
        ]
    if policy not in ("windowed", "trajectory"):
        raise ValueError("timing_policy must be fixed, windowed, or trajectory")

    trajectory = _trajectory_payload(trajectory_report)
    if trajectory is None:
        raise ValueError(f"timing_policy={policy} requires --timing-trajectory")
    trajectory_mode = trajectory.get("mode")
    if trajectory_mode is not None and str(trajectory_mode) != str(mode):
        raise ValueError("timing trajectory mode does not match receiver mode")
    if policy == "trajectory" and not bool(trajectory.get("available")):
        reason = trajectory.get("reason") or "trajectory_unavailable"
        raise ValueError(f"timing trajectory is unavailable: {reason}")

    segments = [
        segment
        for segment in trajectory.get("segments", [])
        if isinstance(segment, dict)
    ]
    if not segments:
        raise ValueError("timing trajectory contains no usable segments")

    signal = np.asarray(symbols, dtype=np.complex64)
    definition = PN_DEFINITIONS[mode]
    entries: list[tuple[SignalFrame, dict[str, Any]]] = []
    for segment in segments:
        start_frame = int(segment.get("start_frame_index") or 0)
        end_frame = int(segment.get("end_frame_index") or start_frame)
        if start_frame < 0:
            raise ValueError("timing trajectory start_frame_index must be non-negative")
        if end_frame <= start_frame:
            continue
        if segment.get("phase_offset") is None:
            continue
        phase_offset = int(segment["phase_offset"]) % definition.frame_symbols
        segment_cfo = segment.get("coarse_cfo_hz")
        residual_cfo_hz = _residual_trajectory_cfo(segment_cfo, fallback_cfo_hz)
        for nominal_frame_index in range(start_frame, end_frame):
            if len(entries) >= max_frames:
                return entries
            frame_start = nominal_frame_index * definition.frame_symbols + phase_offset
            frame_stop = frame_start + definition.frame_symbols
            if frame_start < 0 or frame_stop > signal.size:
                continue
            frame_samples = signal[frame_start:frame_stop]
            if residual_cfo_hz is not None and abs(residual_cfo_hz) > 0.0:
                frame_samples = _frequency_shift_frame(
                    frame_samples,
                    sample_rate_sps=symbol_rate_sps,
                    shift_hz=-residual_cfo_hz,
                    absolute_start_symbol=frame_start,
                )
            header_stop = definition.header_symbols
            frame = SignalFrame(
                mode=mode,
                start=int(frame_start),
                header=frame_samples[:header_stop],
                body=frame_samples[header_stop:],
            )
            timing = _compact_trajectory_segment(segment)
            timing.update(
                {
                    "policy": policy,
                    "nominal_frame_index": int(nominal_frame_index),
                    "phase_offset": int(phase_offset),
                    "coarse_cfo_hz": segment_cfo,
                    "fallback_cfo_hz": fallback_cfo_hz,
                    "residual_cfo_hz": residual_cfo_hz,
                }
            )
            entries.append((frame, timing))
    return entries


def _trajectory_payload(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    trajectory = report.get("trajectory")
    if isinstance(trajectory, dict):
        return trajectory
    if "segments" in report or "available" in report:
        return report
    return None


def _trajectory_segment_for_frame(
    trajectory: dict[str, Any],
    frame_index: int,
) -> dict[str, Any] | None:
    for segment in trajectory.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        start = int(segment.get("start_frame_index") or 0)
        end = int(segment.get("end_frame_index") or start)
        if start <= frame_index < end:
            return segment
    return None


def _residual_trajectory_cfo(
    segment_cfo: Any,
    fallback_cfo_hz: float | None,
) -> float | None:
    if segment_cfo is None:
        return None
    residual = float(segment_cfo)
    if fallback_cfo_hz is not None:
        residual -= float(fallback_cfo_hz)
    return residual


def _frequency_shift_frame(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    shift_hz: float,
    absolute_start_symbol: int,
) -> np.ndarray:
    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    signal = np.asarray(samples, dtype=np.complex64)
    if signal.size == 0 or shift_hz == 0:
        return signal.astype(np.complex64, copy=False)
    time_index = absolute_start_symbol + np.arange(signal.size, dtype=np.float64)
    rotation = np.exp(2j * np.pi * float(shift_hz) * time_index / sample_rate_sps)
    return (signal * rotation).astype(np.complex64, copy=False)


def _compact_trajectory_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: segment.get(key)
        for key in (
            "start_frame_index",
            "end_frame_index",
            "source",
            "candidate_rank",
            "transition_step_symbols",
            "expected_top_count",
            "expected_top3_count",
            "frame_count",
            "best_expected_metric",
            "median_expected_metric",
            "score",
        )
    }


def _compact_receiver_trajectory(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    trajectory = _trajectory_payload(report)
    if trajectory is None:
        return None
    compact = {
        key: trajectory.get(key)
        for key in (
            "version",
            "mode",
            "frame_symbols",
            "header_symbols",
            "policy",
            "source",
            "available",
            "reason",
            "start_frame_index",
            "window_frames",
            "window_step_frames",
            "expected_system_info_index",
            "expected_frame_body_mode",
            "phase_offsets",
            "phase_span_symbols",
            "max_adjacent_phase_step_symbols",
        )
    }
    compact["segments"] = [
        _compact_trajectory_segment(segment)
        | {
            "phase_offset": segment.get("phase_offset"),
            "coarse_cfo_hz": segment.get("coarse_cfo_hz"),
        }
        for segment in trajectory.get("segments", [])
        if isinstance(segment, dict)
    ]
    return compact


def _select_phase_and_cfo(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int | None,
    symbol_rate_sps: int,
    max_frames: int,
    use_family_delay: bool,
    family_frames: int,
    max_delay_symbols: int,
    family_hit_threshold: float,
    timing_search: bool,
    timing_radius_symbols: int,
    timing_step_symbols: int,
    timing_jobs: int,
    timing_per_candidate_cfo: bool,
    expected_system_info_index: int,
    expected_frame_body_mode: str,
    pn_payload_timing_refine: bool = False,
    pn_payload_qam_mode: QamMode | None = None,
) -> dict[str, Any]:
    phase_was_manual = phase_offset is not None
    selected_train = None
    family_delay_train = None
    family_delay_applied = False
    cyclic_phase_offset = phase_offset
    cyclic_cfo_hz = None
    family_cfo_hz = None
    timing_cfo_hz = None
    active_cfo_source = "none"
    source = "manual"
    if phase_offset is None:
        trains = detect_pn_cyclic_extension_trains(symbols, modes=(mode,))
        if not trains:
            raise RuntimeError("no PN cyclic-extension trains found")
        selected_train = trains[0]
        phase_offset = selected_train.phase_offset
        cyclic_phase_offset = phase_offset
        source = "cyclic"
    cfo_hz = estimate_cfo_from_pn_cyclic_extension(
        symbols,
        mode=mode,
        phase_offset=phase_offset,
        symbol_rate_sps=symbol_rate_sps,
    )
    cyclic_cfo_hz = cfo_hz
    if cfo_hz is not None:
        active_cfo_source = "cyclic"
    corrected = symbols
    if cfo_hz is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=symbol_rate_sps,
            shift_hz=-cfo_hz,
        )
    if selected_train is not None and use_family_delay:
        family_delay_train = score_pn_family_delay_train(
            corrected,
            mode=mode,
            phase_offset=phase_offset,
            max_frames=family_frames,
            max_delay_symbols=max_delay_symbols,
            hit_threshold=family_hit_threshold,
        )
        if should_apply_pn_family_delay(
            family_delay_train,
            family_hit_threshold=family_hit_threshold,
        ):
            phase_offset = delay_corrected_phase_offset(
                phase_offset,
                median_delay_symbols=family_delay_train.median_delay_symbols,
                frame_symbols=family_delay_train.frame_symbols,
            )
            family_delay_applied = True
            source = "pn-family-delay"
            family_cfo_hz = estimate_cfo_from_pn_cyclic_extension(
                symbols,
                mode=mode,
                phase_offset=phase_offset,
                symbol_rate_sps=symbol_rate_sps,
            )
            if family_cfo_hz is not None:
                cfo_hz = family_cfo_hz
                active_cfo_source = "pn-family-delay"
    timing_report = None
    if timing_search:
        timing_report = search_frame_timing(
            symbols,
            mode=mode,
            center_phase_offset=phase_offset,
            search_radius_symbols=timing_radius_symbols,
            step_symbols=timing_step_symbols,
            max_frames=max_frames,
            expected_system_info_index=expected_system_info_index,
            expected_frame_body_mode=expected_frame_body_mode,
            per_candidate_cfo=timing_per_candidate_cfo,
            jobs=timing_jobs,
        )
        selected = timing_report.get("selected")
        if selected is not None:
            phase_offset = int(selected["phase_offset"])
            timing_cfo_hz = selected.get("coarse_cfo_hz")
            if timing_cfo_hz is not None:
                cfo_hz = timing_cfo_hz
                active_cfo_source = "timing-search"
            source = "timing-search"
    pn_payload_timing_report = None
    if (
        pn_payload_timing_refine
        and not phase_was_manual
        and pn_payload_qam_mode is not None
        and expected_frame_body_mode == "C3780"
    ):
        payload_corrected = symbols
        if cfo_hz is not None:
            payload_corrected = frequency_shift(
                payload_corrected,
                sample_rate_sps=symbol_rate_sps,
                shift_hz=-cfo_hz,
            )
        definition = PN_DEFINITIONS[mode]
        payload_phases = tuple(
            (int(phase_offset) + delta) % int(definition.frame_symbols)
            for delta in range(-2, 3)
        )
        try:
            pn_payload_timing_report = search_pn_payload_timing(
                payload_corrected,
                mode=mode,
                phase_offsets=payload_phases,
                qam_mode=pn_payload_qam_mode,
                max_frames=min(24, max_frames),
                pn_channel_taps=8,
                pn_noise_variance=0.05,
            )
            selected_payload = pn_payload_timing_report["selected"]
            selected_payload_phase = int(selected_payload["phase_offset"])
            if selected_payload_phase != int(phase_offset):
                phase_offset = selected_payload_phase
                source = "pn-payload-timing"
        except ValueError as exc:
            pn_payload_timing_report = {
                "selected": None,
                "candidates": [],
                "reason": str(exc),
            }
    return {
        "phase_offset": phase_offset,
        "phase_offset_source": source,
        "cyclic_phase_offset": cyclic_phase_offset,
        "coarse_cfo_hz": cfo_hz,
        "active_cfo_source": active_cfo_source,
        "cyclic_cfo_hz": cyclic_cfo_hz,
        "family_cfo_hz": family_cfo_hz,
        "timing_cfo_hz": timing_cfo_hz,
        "timing_per_candidate_cfo": bool(timing_per_candidate_cfo),
        "selected_train": selected_train.to_dict() if selected_train else None,
        "family_delay_train": family_delay_train.to_dict()
        if family_delay_train
        else None,
        "family_delay_applied": family_delay_applied,
        "timing_search": _compact_timing(timing_report),
        "pn_payload_timing": pn_payload_timing_report,
    }


def _compact_timing(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "selected": report.get("selected"),
        "candidate_count": report.get("candidate_count"),
        "center_phase_offset": report.get("center_phase_offset"),
        "per_candidate_cfo": report.get("per_candidate_cfo"),
        "top_candidates": report.get("candidates", [])[:8],
    }


def _complex_symbol_pairs(values: np.ndarray) -> list[list[float]]:
    return [
        [float(value.real), float(value.imag)]
        for value in np.asarray(values, dtype=np.complex64).reshape(-1)
    ]


def _phase_only_system_info_gain(
    reference: np.ndarray,
    observed: np.ndarray,
) -> complex | None:
    """Return unit-magnitude raw system-info gain, or None if unusable."""

    ref = np.asarray(reference, dtype=np.complex64).reshape(-1)
    obs = np.asarray(observed, dtype=np.complex64).reshape(-1)
    if ref.size != obs.size or ref.size == 0:
        return None
    denominator = float(np.vdot(ref, ref).real)
    if denominator <= 1e-12:
        return None
    gain = complex(np.vdot(ref, obs) / denominator)
    if abs(gain) <= 1e-12:
        return None
    return complex(gain / abs(gain))


def _system_info_oracle_candidate(
    observed: np.ndarray,
    *,
    index: int,
    frame_body_mode: str,
) -> dict[str, Any]:
    reference = system_info_symbols(
        SYSTEM_INFO_VECTORS[int(index)],
        frame_body_mode=frame_body_mode,
    ).astype(np.complex64, copy=False)
    values = np.asarray(observed, dtype=np.complex64).reshape(-1)
    denominator = complex(np.vdot(reference, reference))
    gain = (
        complex(np.vdot(reference, values) / denominator)
        if abs(denominator) > 1e-12
        else (1.0 + 0.0j)
    )
    equalized = values / gain if abs(gain) > 1e-12 else values.copy()
    error = equalized - reference
    ref_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    evm = float(np.sqrt(np.mean(np.abs(error) ** 2)) / max(ref_rms, 1e-12))
    decisions = (
        np.real(equalized * np.conj(np.complex64(1.0 + 1.0j))) < 0.0
    ).astype(np.uint8)
    expected_bits = np.asarray(
        [int(bit) for bit in ("1111" if frame_body_mode == "C3780" else "0000") + SYSTEM_INFO_VECTORS[int(index)]],
        dtype=np.uint8,
    )
    bit_errors = int(np.count_nonzero(decisions != expected_bits))
    return {
        "index": int(index),
        "frame_body_mode": frame_body_mode,
        "bit_errors": bit_errors,
        "bit_error_rate": float(bit_errors / 36.0),
        "evm_rms": evm,
        "gain_amplitude": float(abs(gain)),
        "gain_phase_rad": float(np.angle(gain)),
        "parameters": transmission_parameters_for_index(
            int(index),
            frame_body_mode=frame_body_mode,
        ).to_dict(),
    }


def _supported_system_info_oracle(
    observed: np.ndarray,
    *,
    frame_body_mode: str,
) -> dict[str, Any]:
    candidates = [
        _system_info_oracle_candidate(
            observed,
            index=index,
            frame_body_mode=frame_body_mode,
        )
        for index in sorted(SYSTEM_INFO_VECTORS)
        if transmission_parameters_for_index(
            index,
            frame_body_mode=frame_body_mode,
        ).supported_by_receiver
    ]
    candidates.sort(
        key=lambda row: (
            int(row["bit_errors"]),
            float(row["evm_rms"]),
            abs(float(row["gain_phase_rad"])),
            int(row["index"]),
        )
    )
    return {
        "best": candidates[0] if candidates else None,
        "top3": candidates[:3],
    }


def _pn_phase_alias_report(
    *,
    channel_phase: int,
    zero_delay_phase: int,
    mode: PnMode,
) -> dict[str, Any]:
    phase_count = (1 << PN_DEFINITIONS[mode].degree) - 1
    delta = (int(channel_phase) - int(zero_delay_phase)) % phase_count
    if delta > phase_count // 2:
        delta -= phase_count
    return {
        "zero_delay_pn_phase": int(zero_delay_phase),
        "pn_phase_delta_from_zero_delay": int(delta),
    }


def _expected_system_info_report(
    matches: Sequence[Any],
    *,
    expected_index: int,
    expected_frame_body_mode: str,
) -> dict[str, Any]:
    for rank, match in enumerate(matches, start=1):
        if (
            int(match.index) == expected_index
            and str(match.frame_body_mode) == expected_frame_body_mode
        ):
            return {
                "index": expected_index,
                "frame_body_mode": expected_frame_body_mode,
                "rank": rank,
                "top": rank == 1,
                "top3": rank <= 3,
                "metric": float(match.metric),
                "match": match.to_dict(),
            }
    return {
        "index": expected_index,
        "frame_body_mode": expected_frame_body_mode,
        "rank": None,
        "top": False,
        "top3": False,
        "metric": 0.0,
        "match": None,
    }


def _system_info_evidence_report(
    source: str,
    matches: Sequence[Any],
    *,
    independent: bool,
    expected_index: int,
    expected_frame_body_mode: str,
    trained_on_system_info_index: int | None = None,
) -> dict[str, Any]:
    """Describe one system-info classifier pass and whether it is independent."""

    return {
        "source": source,
        "independent": independent,
        "trained_on_system_info_index": trained_on_system_info_index,
        "best": matches[0].to_dict() if matches else None,
        "top3": [match.to_dict() for match in matches[:3]],
        "expected": _expected_system_info_report(
            matches,
            expected_index=expected_index,
            expected_frame_body_mode=expected_frame_body_mode,
        ),
    }


def _best_system_info_metric(matches: Sequence[Any]) -> float:
    if not matches:
        return float("-inf")
    return float(matches[0].metric)


def _summarize_system_info(
    frame_reports: list[dict[str, Any]],
    *,
    min_metric: float,
    min_agreement: float,
    expected_index: int | None = None,
    expected_frame_body_mode: str | None = None,
) -> dict[str, Any]:
    if min_metric < 0.0 or min_metric > 1.0:
        raise ValueError("min_metric must be between 0 and 1")
    if min_agreement < 0.0 or min_agreement > 1.0:
        raise ValueError("min_agreement must be between 0 and 1")
    tops: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for report in frame_reports:
        top = report.get("best_system_info")
        if top is None:
            continue
        independent = bool(report.get("system_info_independent", True))
        compact = {
            "index": top["index"],
            "meaning": top["meaning"],
            "frame_body_mode": top["frame_body_mode"],
            "metric": top["metric"],
            "parameters": top["parameters"],
            "source": report.get("system_info_source"),
            "independent": independent,
        }
        tops.append(compact)
        if float(top["metric"]) >= min_metric:
            qualified.append(compact)

    all_top_counts = Counter(
        (str(match["frame_body_mode"]), int(match["index"])) for match in tops
    )
    selected, agreement_ratio, counts, auto_eligible = _select_system_info(
        qualified,
        frame_count=len(frame_reports),
        min_agreement=min_agreement,
    )
    independent_frame_reports = [
        report
        for report in frame_reports
        if bool(report.get("system_info_independent", True))
    ]
    independent_qualified = [
        match for match in qualified if bool(match.get("independent", True))
    ]
    (
        independent_selected,
        independent_agreement_ratio,
        independent_counts,
        independent_auto_eligible,
    ) = _select_system_info(
        independent_qualified,
        frame_count=len(frame_reports),
        min_agreement=min_agreement,
    )
    supported_raw_oracle = _summarize_supported_system_info_oracle(frame_reports)

    return {
        "selected": selected,
        "min_metric": min_metric,
        "min_agreement": min_agreement,
        "frame_count": len(frame_reports),
        "qualified_frame_count": len(qualified),
        "auto_eligible_frame_count": len(auto_eligible),
        "source_policy": "best_system_info_all_sources",
        "independent_frame_count": len(independent_frame_reports),
        "trained_or_circular_frame_count": len(frame_reports)
        - len(independent_frame_reports),
        "top_counts": [
            {"index": index, "count": count}
            for index, count in counts.most_common()
        ],
        "top_counts_all": [
            {"frame_body_mode": frame_body_mode, "index": index, "count": count}
            for (frame_body_mode, index), count in all_top_counts.most_common()
        ],
        "agreement_ratio": agreement_ratio,
        "expected": _summarize_expected_system_info(
            frame_reports,
            expected_index=expected_index,
            expected_frame_body_mode=expected_frame_body_mode,
            min_metric=min_metric,
        ),
        "supported_raw_oracle": supported_raw_oracle,
        "independent": {
            "selected": independent_selected,
            "frame_count": len(independent_frame_reports),
            "qualified_frame_count": len(independent_qualified),
            "auto_eligible_frame_count": len(independent_auto_eligible),
            "top_counts": [
                {"index": index, "count": count}
                for index, count in independent_counts.most_common()
            ],
            "agreement_ratio": independent_agreement_ratio,
            "expected": _summarize_expected_system_info(
                independent_frame_reports,
                expected_index=expected_index,
                expected_frame_body_mode=expected_frame_body_mode,
                min_metric=min_metric,
            ),
        },
        "frame_tops": tops,
    }


def _select_system_info(
    qualified: list[dict[str, Any]],
    *,
    frame_count: int,
    min_agreement: float,
) -> tuple[dict[str, Any] | None, float, Counter[int], list[dict[str, Any]]]:
    auto_eligible = [
        match
        for match in qualified
        if match["parameters"]["supported_by_receiver"]
    ]
    counts: Counter[int] = Counter(int(match["index"]) for match in auto_eligible)
    selected = None
    agreement_ratio = 0.0
    if counts:
        index, count = counts.most_common(1)[0]
        agreement_ratio = count / max(1, frame_count)
        if agreement_ratio >= min_agreement:
            frame_body_counts = Counter(
                str(match["frame_body_mode"])
                for match in auto_eligible
                if int(match["index"]) == index
            )
            frame_body_mode = frame_body_counts.most_common(1)[0][0]
            parameters = transmission_parameters_for_index(
                index,
                frame_body_mode=frame_body_mode,
            )
            selected = {
                "index": index,
                "frame_body_mode": frame_body_mode,
                "agreement_count": count,
                "agreement_ratio": agreement_ratio,
                "mean_metric": float(
                    np.mean(
                        [
                            match["metric"]
                            for match in auto_eligible
                            if int(match["index"]) == index
                        ]
                    )
                ),
                "parameters": parameters.to_dict(),
            }
    return selected, agreement_ratio, counts, auto_eligible


def _summarize_expected_system_info(
    frame_reports: list[dict[str, Any]],
    *,
    expected_index: int | None,
    expected_frame_body_mode: str | None,
    min_metric: float,
) -> dict[str, Any] | None:
    if expected_index is None or expected_frame_body_mode is None:
        return None
    reports = [
        report.get("expected_system_info")
        for report in frame_reports
        if isinstance(report.get("expected_system_info"), dict)
    ]
    if not reports:
        return None
    metrics = np.asarray(
        [float(report.get("metric", 0.0) or 0.0) for report in reports],
        dtype=np.float64,
    )
    ranks = [
        int(report["rank"])
        for report in reports
        if report.get("rank") is not None
    ]
    frame_count = len(frame_reports)
    top_count = sum(1 for report in reports if bool(report.get("top")))
    top3_count = sum(1 for report in reports if bool(report.get("top3")))
    qualified_count = sum(
        1 for report in reports if float(report.get("metric", 0.0) or 0.0) >= min_metric
    )
    return {
        "index": expected_index,
        "frame_body_mode": expected_frame_body_mode,
        "frame_count": frame_count,
        "ranked_frame_count": len(ranks),
        "top_count": top_count,
        "top3_count": top3_count,
        "top_ratio": top_count / max(1, frame_count),
        "top3_ratio": top3_count / max(1, frame_count),
        "qualified_frame_count": qualified_count,
        "best_metric": float(np.max(metrics)) if metrics.size else 0.0,
        "mean_metric": float(np.mean(metrics)) if metrics.size else 0.0,
        "median_metric": float(np.median(metrics)) if metrics.size else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else None,
        "worst_rank": max(ranks) if ranks else None,
    }


def _summarize_supported_system_info_oracle(
    frame_reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates_by_index: dict[int, list[dict[str, Any]]] = {}
    best_rows: list[dict[str, Any]] = []
    for report in frame_reports:
        oracle = report.get("supported_system_info_oracle")
        if not isinstance(oracle, dict):
            continue
        best = oracle.get("best")
        if isinstance(best, dict):
            best_rows.append(best)
        for candidate in oracle.get("top3") or []:
            if not isinstance(candidate, dict):
                continue
            candidates_by_index.setdefault(int(candidate["index"]), []).append(candidate)
    if not best_rows and not candidates_by_index:
        return None

    best_counts = Counter(
        int(row["index"]) for row in best_rows if row.get("index") is not None
    )
    candidate_summary = []
    for index, values in sorted(candidates_by_index.items()):
        bit_errors = np.asarray(
            [int(value.get("bit_errors") or 0) for value in values],
            dtype=np.float64,
        )
        evm = np.asarray(
            [float(value.get("evm_rms") or 0.0) for value in values],
            dtype=np.float64,
        )
        phase = np.asarray(
            [float(value.get("gain_phase_rad") or 0.0) for value in values],
            dtype=np.float64,
        )
        frame_body_mode = str(values[0].get("frame_body_mode") or "C3780")
        top_count = int(best_counts.get(index, 0))
        candidate_summary.append(
            {
                "index": int(index),
                "frame_body_mode": frame_body_mode,
                "frame_count": int(len(values)),
                "best_choice_count": top_count,
                "best_choice_ratio": float(top_count / max(1, len(best_rows))),
                "mean_bit_errors": float(np.mean(bit_errors)) if bit_errors.size else None,
                "median_bit_errors": float(np.median(bit_errors)) if bit_errors.size else None,
                "mean_evm_rms": float(np.mean(evm)) if evm.size else None,
                "median_evm_rms": float(np.median(evm)) if evm.size else None,
                "median_gain_phase_rad": float(np.median(phase)) if phase.size else None,
                "phase_inverted_frames": int(np.count_nonzero(np.abs(phase) > np.pi / 2)),
                "parameters": transmission_parameters_for_index(
                    int(index),
                    frame_body_mode=frame_body_mode,
                ).to_dict(),
            }
        )
    candidate_summary.sort(
        key=lambda row: (
            -int(row["best_choice_count"]),
            float(row["mean_bit_errors"])
            if row["mean_bit_errors"] is not None
            else float("inf"),
            float(row["mean_evm_rms"])
            if row["mean_evm_rms"] is not None
            else float("inf"),
            abs(
                float(row["median_gain_phase_rad"])
                if row["median_gain_phase_rad"] is not None
                else 0.0
            ),
            int(row["index"]),
        )
    )
    return {
        "frame_count": int(len(best_rows)),
        "best_choice_counts": [
            {"index": int(index), "count": int(count)}
            for index, count in best_counts.most_common()
        ],
        "candidate_summary": candidate_summary,
        "best_supported_index": candidate_summary[0]["index"] if candidate_summary else None,
    }


def _resolve_receive_config(
    *,
    qam_mode: QamMode | Literal["auto"],
    fec_rate_index: int | Literal["auto"],
    symbol_deinterleave: InterleaverMode | Literal["none", "auto"] | str,
    system_info_summary: dict[str, Any],
) -> dict[str, Any]:
    selected = system_info_summary.get("selected")
    parameters = selected.get("parameters") if selected is not None else None

    def auto_value(field: str) -> Any:
        if parameters is None:
            raise RuntimeError("auto receive config requires stable system information")
        if not parameters["supported_by_receiver"]:
            raise RuntimeError(
                "decoded system-information mode is not supported by this receiver"
            )
        value = parameters[field]
        if value is None:
            raise RuntimeError(f"system information does not provide {field}")
        return value

    effective_qam_mode = auto_value("qam_mode") if qam_mode == "auto" else qam_mode
    effective_fec_rate = (
        int(auto_value("fec_rate_index"))
        if fec_rate_index == "auto"
        else int(fec_rate_index)
    )
    effective_symbol_deinterleave = (
        auto_value("interleaver_mode")
        if symbol_deinterleave == "auto"
        else symbol_deinterleave
    )
    if effective_symbol_deinterleave not in ("none", "mode1", "mode2"):
        raise ValueError("symbol_deinterleave must be none, auto, mode1, or mode2")
    return {
        "qam_mode": effective_qam_mode,
        "fec_rate_index": effective_fec_rate,
        "symbol_deinterleave": effective_symbol_deinterleave,
        "auto_source": selected,
    }


def _decision_directed_qam_mode(
    qam_mode: QamMode | Literal["auto"],
    *,
    system_info_index: int,
    frame_body_mode: str,
) -> QamMode:
    if qam_mode != "auto":
        return qam_mode
    parameters = transmission_parameters_for_index(
        system_info_index,
        frame_body_mode=frame_body_mode,
    )
    return parameters.qam_mode or "64qam"


def _symbol_deinterleave_phase_scan(
    symbols: np.ndarray,
    *,
    mode: InterleaverMode | Literal["none"] | str,
    qam_mode: QamMode,
    fec_rate_index: int,
    max_codewords: int,
) -> dict[str, Any]:
    """Score every symbol-deinterleaver phase with LDPC parity mismatch."""

    if mode == "none":
        return {
            "enabled": False,
            "reason": "symbol deinterleaving is disabled",
            "phases": [],
            "best_phase": None,
        }
    if mode not in SYMBOL_INTERLEAVERS:
        raise ValueError("symbol_deinterleave phase scan requires mode1 or mode2")
    if max_codewords <= 0:
        raise ValueError("max_codewords must be positive")
    spec = SYMBOL_INTERLEAVERS[mode]  # type: ignore[index]
    profile = dtmb_ldpc_profile(fec_rate_index)
    bits_per_symbol = _bits_per_symbol(qam_mode)
    needed_symbols = (
        (max_codewords * profile.codeword_bits) + bits_per_symbol - 1
    ) // bits_per_symbol
    scan_symbol_count = min(
        int(np.asarray(symbols).size),
        spec.full_stream_latency_symbols + needed_symbols,
    )
    scan_symbols = np.asarray(symbols, dtype=np.complex64).reshape(-1)[
        :scan_symbol_count
    ]
    phase_reports = []
    best_report = None
    for phase in range(spec.branch_count):
        deinterleaved = convolutional_deinterleave(
            scan_symbols,
            mode=mode,  # type: ignore[arg-type]
            phase=phase,
        )
        discarded = min(spec.full_stream_latency_symbols, int(deinterleaved.size))
        deinterleaved = deinterleaved[discarded:]
        llr = qam_soft_demodulate(deinterleaved, mode=qam_mode)
        bits = (llr < 0).astype(np.uint8)
        codewords = min(bits.size // profile.codeword_bits, max_codewords)
        if codewords == 0:
            report = {
                "phase": phase,
                "codewords": 0,
                "discarded_symbols": discarded,
                "mean_mismatch_ratio": None,
                "min_mismatch_ratio": None,
                "max_mismatch_ratio": None,
                "zero_mismatch_codewords": 0,
                "appendix_b_mean_syndrome_ratio": None,
                "appendix_b_min_syndrome_ratio": None,
                "appendix_b_max_syndrome_ratio": None,
                "appendix_b_zero_syndrome_codewords": 0,
                "sample_mismatch_counts": [],
                "sample_syndrome_weights": [],
            }
        else:
            usable = codewords * profile.codeword_bits
            counts = dtmb_ldpc_parity_mismatch_counts(
                bits[:usable],
                fec_rate_index=fec_rate_index,
            )
            ratios = counts.astype(np.float64) / profile.parity_bits
            syndrome_weights, clean_rows = dtmb_ldpc_appendix_b_syndrome_counts(
                bits[:usable],
                fec_rate_index=fec_rate_index,
            )
            if clean_rows > 0:
                syndrome_ratios = syndrome_weights.astype(np.float64) / float(clean_rows)
                appendix_mean = float(np.mean(syndrome_ratios))
                appendix_min = float(np.min(syndrome_ratios))
                appendix_max = float(np.max(syndrome_ratios))
            else:
                appendix_mean = None
                appendix_min = None
                appendix_max = None
            report = {
                "phase": phase,
                "codewords": int(counts.size),
                "discarded_symbols": discarded,
                "mean_mismatch_ratio": float(np.mean(ratios)),
                "min_mismatch_ratio": float(np.min(ratios)),
                "max_mismatch_ratio": float(np.max(ratios)),
                "zero_mismatch_codewords": int(np.count_nonzero(counts == 0)),
                "appendix_b_mean_syndrome_ratio": appendix_mean,
                "appendix_b_min_syndrome_ratio": appendix_min,
                "appendix_b_max_syndrome_ratio": appendix_max,
                "appendix_b_zero_syndrome_codewords": int(
                    np.count_nonzero(syndrome_weights == 0)
                ),
                "sample_mismatch_counts": [int(value) for value in counts[:8]],
                "sample_syndrome_weights": [int(value) for value in syndrome_weights[:8]],
            }
            if best_report is None or _phase_scan_sort_key(report) < _phase_scan_sort_key(
                best_report
            ):
                best_report = report
        phase_reports.append(report)
    return {
        "enabled": True,
        "mode": mode,
        "qam_mode": qam_mode,
        "fec_rate_index": fec_rate_index,
        "max_codewords": max_codewords,
        "latency_symbols": spec.full_stream_latency_symbols,
        "input_symbols_scored": int(scan_symbols.size),
        "best_phase": best_report["phase"] if best_report is not None else None,
        "best": best_report,
        "phases": phase_reports,
    }


def _phase_scan_sort_key(report: dict[str, Any]) -> tuple[float, float, float, float, int]:
    appendix_mean = report.get("appendix_b_mean_syndrome_ratio")
    appendix_min = report.get("appendix_b_min_syndrome_ratio")
    mean = report.get("mean_mismatch_ratio")
    minimum = report.get("min_mismatch_ratio")
    return (
        float("inf") if appendix_mean is None else float(appendix_mean),
        float("inf") if appendix_min is None else float(appendix_min),
        float("inf") if mean is None else float(mean),
        float("inf") if minimum is None else float(minimum),
        int(report["phase"]),
    )


def _summarize_per_frame_cpe(
    frame_reports: list[dict[str, Any]],
    *,
    enabled: bool,
    max_relative_error: float,
    requested_source: str,
) -> dict[str, Any]:
    cpe_pairs = [
        (report, report["per_frame_cpe_correction"])
        for report in frame_reports
        if report.get("per_frame_cpe_correction") is not None
    ]
    cpe_reports = [cpe for _frame, cpe in cpe_pairs]
    if not cpe_reports:
        return {
            "enabled": enabled,
            "max_relative_error": max_relative_error,
            "requested_source": requested_source,
            "frames": 0,
            "cpe_rad": [],
            "cpe_unwrapped_rad": [],
            "amplitude": [],
            "mean_cpe_rad": None,
            "std_cpe_rad": None,
            "max_abs_cpe_rad": None,
            "max_abs_adjacent_cpe_step_rad": None,
            "median_abs_adjacent_cpe_step_rad": None,
            "linear_cpe_slope_rad_per_frame": None,
            "linear_cpe_residual_rms_rad": None,
            "mean_amplitude": None,
            "reliable_symbol_ratio_mean": None,
            "source_counts": {},
        }
    cpe_rad = np.asarray([float(report["cpe_rad"]) for report in cpe_reports])
    cpe_unwrapped = np.unwrap(cpe_rad)
    adjacent_steps = np.diff(cpe_unwrapped)
    abs_adjacent_steps = np.abs(adjacent_steps)
    if cpe_unwrapped.size > 1:
        x = np.arange(cpe_unwrapped.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, cpe_unwrapped, 1)
        residual = cpe_unwrapped - (slope * x + intercept)
        residual_rms = float(np.sqrt(np.mean(residual**2)))
    else:
        slope = 0.0
        residual_rms = 0.0
    amplitudes = np.asarray([float(report["amplitude"]) for report in cpe_reports])
    reliable = np.asarray(
        [float(report["reliable_symbol_count"]) for report in cpe_reports]
    )
    data_counts = np.asarray(
        [
            max(1.0, float(frame.get("data_symbols", 0)))
            for frame, _cpe in cpe_pairs
        ]
    )
    source_counts: Counter[str] = Counter(
        str(report.get("source", "unknown")) for report in cpe_reports
    )
    return {
        "enabled": enabled,
        "max_relative_error": max_relative_error,
        "requested_source": requested_source,
        "frames": int(cpe_rad.size),
        "cpe_rad": [float(value) for value in cpe_rad],
        "cpe_unwrapped_rad": [float(value) for value in cpe_unwrapped],
        "amplitude": [float(value) for value in amplitudes],
        "mean_cpe_rad": float(np.mean(cpe_rad)),
        "std_cpe_rad": float(np.std(cpe_rad)) if cpe_rad.size > 1 else 0.0,
        "max_abs_cpe_rad": float(np.max(np.abs(cpe_rad))),
        "max_abs_adjacent_cpe_step_rad": float(np.max(abs_adjacent_steps))
        if abs_adjacent_steps.size
        else 0.0,
        "median_abs_adjacent_cpe_step_rad": float(np.median(abs_adjacent_steps))
        if abs_adjacent_steps.size
        else 0.0,
        "linear_cpe_slope_rad_per_frame": float(slope),
        "linear_cpe_residual_rms_rad": residual_rms,
        "mean_amplitude": float(np.mean(amplitudes)),
        "reliable_symbol_ratio_mean": float(
            np.mean(reliable / data_counts)
        ),
        "source_counts": dict(source_counts),
    }


def _ldpc_parity_report(bits: np.ndarray, *, fec_rate_index: int) -> dict[str, Any]:
    profile = dtmb_ldpc_profile(fec_rate_index)
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    usable = (values.size // profile.codeword_bits) * profile.codeword_bits
    if usable == 0:
        return {
            "codewords": 0,
            "unused_bits": int(values.size),
            "parity_bits_per_codeword": profile.parity_bits,
            "mean_mismatch_ratio": None,
            "min_mismatch_ratio": None,
            "max_mismatch_ratio": None,
            "zero_mismatch_codewords": 0,
            "sample_mismatch_counts": [],
            "appendix_b_syndrome": {
                "codewords": 0,
                "clean_rows_per_codeword": 0,
                "mean_syndrome_ratio": None,
                "min_syndrome_ratio": None,
                "max_syndrome_ratio": None,
                "zero_syndrome_codewords": 0,
                "sample_syndrome_weights": [],
            },
            "interpretation": "no_codewords",
        }
    counts = dtmb_ldpc_parity_mismatch_counts(
        values[:usable],
        fec_rate_index=fec_rate_index,
    )
    ratios = counts.astype(np.float64) / profile.parity_bits
    syndrome_weights, clean_rows = dtmb_ldpc_appendix_b_syndrome_counts(
        values[:usable],
        fec_rate_index=fec_rate_index,
    )
    syndrome_ratios = (
        syndrome_weights.astype(np.float64) / clean_rows
        if clean_rows > 0
        else np.zeros(syndrome_weights.size, dtype=np.float64)
    )
    return {
        "codewords": int(counts.size),
        "unused_bits": int(values.size - usable),
        "parity_bits_per_codeword": profile.parity_bits,
        "mean_mismatch_ratio": float(np.mean(ratios)),
        "min_mismatch_ratio": float(np.min(ratios)),
        "max_mismatch_ratio": float(np.max(ratios)),
        "zero_mismatch_codewords": int(np.count_nonzero(counts == 0)),
        "sample_mismatch_counts": [int(value) for value in counts[:16]],
        "appendix_b_syndrome": {
            "codewords": int(syndrome_weights.size),
            "clean_rows_per_codeword": int(clean_rows),
            "mean_syndrome_ratio": float(np.mean(syndrome_ratios))
            if syndrome_ratios.size
            else None,
            "min_syndrome_ratio": float(np.min(syndrome_ratios))
            if syndrome_ratios.size
            else None,
            "max_syndrome_ratio": float(np.max(syndrome_ratios))
            if syndrome_ratios.size
            else None,
            "zero_syndrome_codewords": int(
                np.count_nonzero(syndrome_weights == 0)
            ),
            "sample_syndrome_weights": [
                int(value) for value in syndrome_weights[:16]
            ],
        },
        "interpretation": _ldpc_parity_interpretation(
            generator_ratio=float(np.mean(ratios)),
            syndrome_ratio=float(np.mean(syndrome_ratios))
            if syndrome_ratios.size
            else 0.0,
        ),
    }


def _ldpc_parity_interpretation(
    *,
    generator_ratio: float,
    syndrome_ratio: float,
    consistent_threshold: float = 0.05,
    random_threshold: float = 0.3,
) -> str:
    """Classify the LDPC parity readout for the fixture gate interpretation."""

    generator_ok = generator_ratio <= consistent_threshold
    syndrome_ok = syndrome_ratio <= consistent_threshold
    generator_random = generator_ratio >= random_threshold
    syndrome_random = syndrome_ratio >= random_threshold
    if generator_ok and syndrome_ok:
        return "codewords_consistent_proceed_to_bch"
    if generator_random and syndrome_random:
        return "hard_parity_uninformative_decode_soft_bits_before_diagnosing_bit_order"
    if generator_random and syndrome_ok:
        return "ldpc_generator_convention_mismatch"
    if generator_ok and syndrome_random:
        return "appendix_b_only_inconsistent_check_parity_mapping"
    return "partial_consistency_investigate_individual_codewords"


def _write_frame_fft_dump(
    path: str | Path,
    frame_fft_dumps: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract lists for np.savez
    arrays = {}
    for key in frame_fft_dumps[0].keys():
        arrays[key] = []
    
    for row in frame_fft_dumps:
        for k, v in row.items():
            arrays[k].append(v if v is not None else np.nan)
            
    # Convert to arrays, handling complex and mixed types carefully
    save_dict = {}
    for k, v in arrays.items():
        try:
            save_dict[k] = np.array(v)
        except Exception:
            save_dict[k] = np.array(v, dtype=object)
            
    np.savez_compressed(path, **save_dict)


def _write_pre_ldpc_dump(
    path: str | Path,
    *,
    capture_path: str | Path,
    llr: np.ndarray,
    hard_bits: np.ndarray,
    symbols_before_deinterleave: np.ndarray,
    symbols_after_deinterleave: np.ndarray,
    data_symbols_per_frame: Sequence[int],
    frame_reports: Sequence[dict[str, Any]],
    effective_qam_mode: str,
    requested_qam_mode: Any,
    effective_fec_rate_index: int,
    requested_fec_rate_index: Any,
    effective_symbol_deinterleave: str,
    requested_symbol_deinterleave: Any,
    symbol_deinterleave_phase: int,
    symbol_deinterleave_latency: int,
    symbol_deinterleave_discarded: int,
    bits_per_frame: int,
    equalizer: str,
    correct_per_frame_cpe: bool,
    frame_body_mode: str,
    mode: str,
    fft_bin_shift: int,
    body_window_offset_symbols: int,
    frequency_deinterleaver_direction: str,
    data_carrier_order: str,
    carrier_permutation: str,
    logical_position_shift_symbols: int,
    system_info_index: int,
    system_info_summary: dict[str, Any],
    acquisition: dict[str, Any],
    qam_quality_report: dict[str, Any],
    dd_max_hard_bit_bias: float = 0.0,
    llr_conditioning: dict[str, Any] | None = None,
    carrier_erase: dict[str, Any] | None = None,
    branch_gain_correction: dict[str, Any] | None = None,
    symbols_only: bool = False,
) -> Path:
    """Write the receiver's pre-LDPC seam as an NPZ plus JSON sidecar."""

    dump_path = Path(path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = dump_path.with_suffix(".json")
    values = np.asarray(hard_bits, dtype=np.uint8).reshape(-1)
    llr_values = np.asarray(llr, dtype=np.float32).reshape(-1)
    bits_per_symbol = _bits_per_symbol(effective_qam_mode)
    profile = dtmb_ldpc_profile(effective_fec_rate_index)
    before_symbols = np.asarray(
        symbols_before_deinterleave, dtype=np.complex64
    ).reshape(-1)
    after_symbols = np.asarray(
        symbols_after_deinterleave, dtype=np.complex64
    ).reshape(-1)
    before_symbol_frame_indices = _symbol_frame_indices(data_symbols_per_frame)
    per_frame_evm = [
        _frame_data_evm_rms(frame)
        for frame in frame_reports
    ]
    per_frame_cpe_rad = [
        _frame_cpe_rad(frame)
        for frame in frame_reports
    ]
    metadata = {
        "capture_path": str(capture_path),
        "npz_file": str(dump_path),
        "json_file": str(sidecar_path),
        "dump_mode": "symbols_only" if symbols_only else "full",
        "hard_bits_dtype": "uint8",
        "hard_bits_encoding": "one_bit_per_byte",
        "llr_dtype": "<f4",
        "llr_count": int(llr_values.size),
        "hard_bit_count": int(values.size),
        "bits_per_symbol": int(bits_per_symbol),
        "bits_per_frame": int(bits_per_frame),
        "codeword_bits": int(profile.codeword_bits),
        "codewords": int(values.size // profile.codeword_bits),
        "unused_bits": int(values.size % profile.codeword_bits),
        "chosen_qam": effective_qam_mode,
        "requested_qam": str(requested_qam_mode),
        "chosen_fec_rate": int(effective_fec_rate_index),
        "requested_fec_rate": str(requested_fec_rate_index),
        "chosen_interleaver_mode": effective_symbol_deinterleave,
        "requested_interleaver_mode": str(requested_symbol_deinterleave),
        "symbol_deinterleave_phase": int(symbol_deinterleave_phase),
        "symbol_deinterleave_latency_symbols": int(symbol_deinterleave_latency),
        "symbol_deinterleave_discarded_symbols": int(symbol_deinterleave_discarded),
        "chosen_system_info_vector": int(system_info_index),
        "system_info": system_info_summary,
        "frame_body_mode": frame_body_mode,
        "pn_mode": mode,
        "fft_window_offset": int(acquisition.get("phase_offset") or 0),
        "fft_bin_shift": int(fft_bin_shift),
        "body_window_offset_symbols": int(body_window_offset_symbols),
        "data_carrier_order": data_carrier_order,
        "carrier_permutation": carrier_permutation,
        "logical_position_shift_symbols": int(logical_position_shift_symbols),
        "carrier_order_version": (
            f"C3780_{data_carrier_order}_data_order_after_frequency_deinterleave"
            if frame_body_mode == "C3780"
            else "C1_body_order"
        ),
        "frequency_deinterleaver_version": (
            f"{frequency_deinterleaver_direction}:frequency_deinterleave_inserted"
            if frame_body_mode == "C3780"
            else "none"
        ),
        "equalizer_mode": equalizer,
        "dd_max_hard_bit_bias": float(dd_max_hard_bit_bias),
        "per_frame_cpe_enabled": bool(correct_per_frame_cpe),
        "llr_conditioning": llr_conditioning or _llr_conditioning_report(
            llr_values,
            llr_values,
            scale=1.0,
            clip=None,
            erase_fraction=0.0,
            bits_per_symbol=bits_per_symbol,
            plane_scales=None,
        ),
        "carrier_erase": carrier_erase or _carrier_erase_report(
            metric="none",
            fraction=0.0,
            reliability_threshold=0.55,
        ),
        "branch_gain_correction": branch_gain_correction or _branch_gain_report(
            selected_branches=None,
            reliability_threshold=0.55,
            min_symbols=32,
        ),
        "per_frame_data_evm_rms": per_frame_evm,
        "median_data_evm_rms": _finite_median_or_none(per_frame_evm),
        "min_data_evm_rms": _finite_min_or_none(per_frame_evm),
        "per_frame_cpe_rad": per_frame_cpe_rad,
        "llr_stats": _llr_stats(llr_values),
        "hard_bit_balance_per_plane": _hard_bit_balance_per_plane(
            values,
            bits_per_symbol=bits_per_symbol,
        ),
        "qam_symbol_quality": qam_quality_report,
        "data_symbols_before_symbol_deinterleave": int(before_symbols.size),
        "data_symbols_after_symbol_deinterleave": int(after_symbols.size),
        "data_symbols_per_frame": [int(value) for value in data_symbols_per_frame],
        "frames_used": int(len(frame_reports)),
        "arrays": {
            "data_symbols_before_symbol_deinterleave": "complex64 QAM symbols before convolutional deinterleave",
            "data_symbols_after_symbol_deinterleave": "complex64 QAM symbols after latency discard",
            "data_symbol_frame_indices_before_deinterleave": "int32 source frame index per pre-deinterleave data symbol",
            "frame_start_symbols": "int64 signal-frame start symbol offsets",
        },
    }
    if not symbols_only:
        metadata["arrays"].update(
            {
                "hard_bits": "uint8 flat pre-LDPC hard decisions",
                "llr": "float32 flat pre-LDPC LLRs; positive means bit 0",
                "frame_indices": "int32 output-frame index per hard bit",
                "codeword_indices": "int32 LDPC codeword index per hard bit, -1 for tail",
                "bit_indices_in_codeword": "int32 bit position inside LDPC codeword, -1 for tail",
            }
        )
    frame_starts = np.asarray(
        [int(frame.get("start", 0)) for frame in frame_reports],
        dtype=np.int64,
    )
    arrays = {
        "data_symbols_before_symbol_deinterleave": before_symbols,
        "data_symbols_after_symbol_deinterleave": after_symbols,
        "data_symbol_frame_indices_before_deinterleave": before_symbol_frame_indices,
        "frame_start_symbols": frame_starts,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if not symbols_only:
        frame_indices, codeword_indices, bit_indices_in_codeword = (
            _pre_ldpc_bit_indices(
                values.size,
                bits_per_frame=bits_per_frame,
                codeword_bits=profile.codeword_bits,
            )
        )
        arrays.update(
            {
                "hard_bits": values,
                "llr": llr_values,
                "frame_indices": frame_indices,
                "codeword_indices": codeword_indices,
                "bit_indices_in_codeword": bit_indices_in_codeword,
            }
        )
    np.savez(dump_path, **arrays)
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def _pre_ldpc_bit_indices(
    bit_count: int,
    *,
    bits_per_frame: int,
    codeword_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = np.full(bit_count, -1, dtype=np.int32)
    if bits_per_frame > 0:
        frame_indices = (
            np.arange(bit_count, dtype=np.int64) // int(bits_per_frame)
        ).astype(np.int32)
    codeword_indices = np.full(bit_count, -1, dtype=np.int32)
    bit_indices = np.full(bit_count, -1, dtype=np.int32)
    usable = (bit_count // codeword_bits) * codeword_bits
    if usable:
        idx = np.arange(usable, dtype=np.int64)
        codeword_indices[:usable] = (idx // codeword_bits).astype(np.int32)
        bit_indices[:usable] = (idx % codeword_bits).astype(np.int32)
    return frame_indices, codeword_indices, bit_indices


def _symbol_frame_indices(data_symbols_per_frame: Sequence[int]) -> np.ndarray:
    chunks = [
        np.full(int(count), index, dtype=np.int32)
        for index, count in enumerate(data_symbols_per_frame)
        if int(count) > 0
    ]
    if not chunks:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32, copy=False)


def _frame_data_evm_rms(frame: dict[str, Any]) -> float | None:
    report = frame.get("sparse_pilot_equalization")
    if not isinstance(report, dict):
        return None
    value = report.get("data_evm_rms")
    return float(value) if value is not None else None


def _frame_cpe_rad(frame: dict[str, Any]) -> float | None:
    report = frame.get("per_frame_cpe_correction")
    if not isinstance(report, dict):
        return None
    value = report.get("cpe_rad")
    return float(value) if value is not None else None


def _finite_median_or_none(values: Sequence[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.median(array))


def _finite_min_or_none(values: Sequence[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.min(array))


def _llr_stats(llr: np.ndarray) -> dict[str, Any]:
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "positive_fraction": None,
            "negative_fraction": None,
            "zero_fraction": None,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def _llr_conditioning_report(
    raw_llr: np.ndarray,
    conditioned_llr: np.ndarray,
    *,
    scale: float,
    clip: float | None,
    erase_fraction: float,
    bits_per_symbol: int,
    plane_scales: Sequence[float] | str | None,
) -> dict[str, Any]:
    raw_values = np.asarray(raw_llr, dtype=np.float32).reshape(-1)
    conditioned_values = np.asarray(conditioned_llr, dtype=np.float32).reshape(-1)
    resolved_plane_scales = _resolve_llr_plane_scales(
        plane_scales,
        bits_per_symbol=bits_per_symbol,
    )
    return {
        "scale": float(scale),
        "clip": None if clip is None else float(clip),
        "erase_fraction": float(erase_fraction),
        "plane_scales": list(resolved_plane_scales)
        if resolved_plane_scales is not None
        else None,
        "raw_stats": _llr_stats(raw_values),
        "conditioned_stats": _llr_stats(conditioned_values),
        "zero_count": int(np.count_nonzero(conditioned_values == 0.0)),
    }


def _carrier_erase_report(
    *,
    metric: str,
    fraction: float,
    reliability_threshold: float,
    quality_summary: dict[str, Any] | None = None,
    erased_carriers: int = 0,
    erased_symbols: int = 0,
    erased_llrs: int = 0,
    llr_count: int = 0,
) -> dict[str, Any]:
    return {
        "enabled": bool(metric != "none" and fraction > 0.0),
        "metric": str(metric),
        "fraction": float(fraction),
        "reliability_threshold": float(reliability_threshold),
        "quality_summary": quality_summary,
        "erased_carriers": int(erased_carriers),
        "erased_symbols": int(erased_symbols),
        "erased_llrs": int(erased_llrs),
        "erased_llr_fraction": (
            None if int(llr_count) <= 0 else float(erased_llrs) / float(llr_count)
        ),
    }


def _branch_gain_report(
    *,
    selected_branches: Sequence[int] | None,
    reliability_threshold: float,
    min_symbols: int,
    corrected_branches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": bool(selected_branches),
        "selected_branches": [int(value) for value in selected_branches]
        if selected_branches is not None
        else None,
        "reliability_threshold": float(reliability_threshold),
        "min_symbols": int(min_symbols),
        "corrected_branches": corrected_branches or [],
    }


def _condition_demapped_llr(
    llr: np.ndarray,
    *,
    scale: float,
    clip: float | None,
    erase_fraction: float,
    bits_per_symbol: int,
    plane_scales: Sequence[float] | str | None,
) -> np.ndarray:
    values = np.asarray(llr, dtype=np.float32).reshape(-1).copy()
    values *= np.float32(scale)
    resolved_plane_scales = _resolve_llr_plane_scales(
        plane_scales,
        bits_per_symbol=bits_per_symbol,
    )
    if resolved_plane_scales is not None and values.size:
        usable = (values.size // bits_per_symbol) * bits_per_symbol
        if usable:
            grouped = values[:usable].reshape(-1, bits_per_symbol)
            grouped *= np.asarray(resolved_plane_scales, dtype=np.float32).reshape(1, -1)
    if clip is not None:
        np.clip(values, -float(clip), float(clip), out=values)
    if erase_fraction > 0.0 and values.size:
        _erase_low_confidence_llr_in_place(values, float(erase_fraction))
    return values


def _erase_low_confidence_llr_in_place(values: np.ndarray, fraction: float) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    erase_count = int(np.floor(flat.size * fraction))
    if erase_count <= 0:
        return
    abs_values = np.abs(flat)
    indices = np.argpartition(abs_values, erase_count - 1)[:erase_count]
    flat[indices] = 0.0


def _source_carriers_for_post_deinterleaver_symbols(
    symbol_count: int,
    *,
    mode: str,
    phase: int,
) -> np.ndarray:
    count = max(0, int(symbol_count))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    selected_mode = str(mode).lower()
    if selected_mode == "none":
        return (np.arange(count, dtype=np.int64) % DATA_SYMBOLS_PER_FRAME).astype(
            np.int32,
            copy=False,
        )
    spec = SYMBOL_INTERLEAVERS[selected_mode]
    output_indices = spec.full_stream_latency_symbols + np.arange(count, dtype=np.int64)
    starts = output_indices % spec.branch_count
    branches = (starts + int(phase)) % spec.branch_count
    branch_delays = (spec.branch_count - 1 - branches) * spec.delay_step
    branch_offsets = (output_indices - starts) // spec.branch_count
    source_offsets = branch_offsets - branch_delays
    source_indices = starts + spec.branch_count * source_offsets
    if bool(np.any(source_indices < 0)):
        raise ValueError("computed invalid deinterleaver source index")
    return (source_indices % DATA_SYMBOLS_PER_FRAME).astype(np.int32, copy=False)


def _carrier_erase_mask(
    symbols_before_deinterleave: np.ndarray,
    *,
    qam_mode: QamMode,
    metric: str,
    fraction: float,
    reliability_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(symbols_before_deinterleave, dtype=np.complex64).reshape(-1)
    frame_count = int(values.size // DATA_SYMBOLS_PER_FRAME)
    if frame_count <= 0:
        return np.zeros(DATA_SYMBOLS_PER_FRAME, dtype=np.bool_), {
            "frames": 0,
            "carrier_count": DATA_SYMBOLS_PER_FRAME,
        }
    framed = values[: frame_count * DATA_SYMBOLS_PER_FRAME].reshape(
        frame_count,
        DATA_SYMBOLS_PER_FRAME,
    )
    normalized = normalize_qam_symbols(
        framed.reshape(-1),
        mode=qam_mode,
    ).reshape(frame_count, DATA_SYMBOLS_PER_FRAME)
    nearest = qam_nearest(
        normalized.reshape(-1),
        mode=qam_mode,
    ).reshape(frame_count, DATA_SYMBOLS_PER_FRAME)
    nearest_power = np.maximum(np.mean(np.abs(nearest) ** 2, axis=0), 1e-24)
    error_power = np.mean(np.abs(normalized - nearest) ** 2, axis=0)
    evm = np.sqrt(error_power / nearest_power).astype(np.float64)
    relative_error = np.abs(normalized - nearest) / np.sqrt(nearest_power)[None, :]
    reliability = np.mean(relative_error <= float(reliability_threshold), axis=0)
    hard = qam_hard_demodulate(
        normalized.reshape(-1),
        mode=qam_mode,
        normalize=False,
    ).reshape(frame_count, DATA_SYMBOLS_PER_FRAME, -1)
    bit_balance = np.mean(hard, axis=0)
    bit_bias = np.max(np.abs(bit_balance - 0.5), axis=1).astype(np.float64)
    summary = {
        "frames": int(frame_count),
        "carrier_count": int(DATA_SYMBOLS_PER_FRAME),
        "evm_rms_mean": float(np.mean(evm)),
        "evm_rms_p95": float(np.quantile(evm, 0.95)),
        "evm_rms_max": float(np.max(evm)),
        "reliability_mean": float(np.mean(reliability)),
        "reliability_min": float(np.min(reliability)),
        "bit_bias_mean": float(np.mean(bit_bias)),
        "bit_bias_p95": float(np.quantile(bit_bias, 0.95)),
        "bit_bias_max": float(np.max(bit_bias)),
    }
    if metric == "none" or fraction <= 0.0:
        return np.zeros(DATA_SYMBOLS_PER_FRAME, dtype=np.bool_), summary
    if metric == "evm":
        order = np.argsort(evm)[::-1]
    elif metric == "bit_bias":
        order = np.argsort(bit_bias)[::-1]
    elif metric == "unreliable":
        order = np.argsort(reliability)
    else:
        raise ValueError("carrier_erase_metric must be none, evm, bit_bias, or unreliable")
    erase_count = min(DATA_SYMBOLS_PER_FRAME, max(1, int(np.ceil(float(fraction) * DATA_SYMBOLS_PER_FRAME))))
    mask = np.zeros(DATA_SYMBOLS_PER_FRAME, dtype=np.bool_)
    mask[order[:erase_count]] = True
    return mask, summary


def _resolve_branch_gain_branches(
    branches: Sequence[int] | str | None,
    *,
    interleaver_mode: str,
) -> tuple[int, ...] | None:
    if branches is None:
        return None
    if interleaver_mode == "none":
        return None
    spec = SYMBOL_INTERLEAVERS[interleaver_mode]
    if isinstance(branches, str):
        text = branches.strip().lower()
        if not text:
            return None
        if text == "all":
            return tuple(range(spec.branch_count))
        values = [int(token.strip()) for token in text.split(",") if token.strip()]
    else:
        values = [int(value) for value in branches]
    if not values:
        return None
    result = []
    seen = set()
    for value in values:
        if value < 0 or value >= spec.branch_count:
            raise ValueError(
                f"branch_gain_branches values must be in 0..{spec.branch_count - 1}"
            )
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _source_branches_for_post_deinterleaver_symbols(
    symbol_count: int,
    *,
    mode: str,
    phase: int,
) -> np.ndarray:
    count = max(0, int(symbol_count))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    selected_mode = str(mode).lower()
    if selected_mode == "none":
        return (np.arange(count, dtype=np.int64) % 52).astype(np.int32, copy=False)
    spec = SYMBOL_INTERLEAVERS[selected_mode]
    source_indices = convolutional_deinterleave_source_indices(
        count,
        mode=selected_mode,  # type: ignore[arg-type]
        phase=int(phase),
    )
    return ((source_indices % spec.branch_count) + int(phase)).astype(np.int32, copy=False)


def _apply_post_deinterleave_branch_gain_correction(
    symbols: np.ndarray,
    *,
    qam_mode: QamMode,
    interleaver_mode: str,
    interleaver_phase: int,
    selected_branches: Sequence[int] | str,
    reliability_threshold: float,
    min_symbols: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = _resolve_branch_gain_branches(
        selected_branches,
        interleaver_mode=interleaver_mode,
    )
    if not resolved:
        values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
        return values, _branch_gain_report(
            selected_branches=None,
            reliability_threshold=reliability_threshold,
            min_symbols=min_symbols,
        )
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1).copy()
    normalized = normalize_qam_symbols(values, mode=qam_mode)
    nearest = qam_nearest(normalized, mode=qam_mode)
    nearest_power = np.maximum(np.abs(nearest), 1e-6)
    relative_error = np.abs(normalized - nearest) / nearest_power
    reliable = relative_error <= float(reliability_threshold)
    branches = _source_branches_for_post_deinterleaver_symbols(
        int(values.size),
        mode=interleaver_mode,
        phase=interleaver_phase,
    )
    corrected: list[dict[str, Any]] = []
    for branch in resolved:
        mask = branches == int(branch)
        selected = mask & reliable
        reliable_count = int(np.count_nonzero(selected))
        if reliable_count < int(min_symbols):
            corrected.append(
                {
                    "branch": int(branch),
                    "reliable_symbols": reliable_count,
                    "symbols": int(np.count_nonzero(mask)),
                    "applied": False,
                    "reason": "insufficient_reliable_symbols",
                }
            )
            continue
        ref = nearest[selected]
        obs = normalized[selected]
        denominator = float(np.vdot(ref, ref).real)
        if denominator <= 1e-9:
            corrected.append(
                {
                    "branch": int(branch),
                    "reliable_symbols": reliable_count,
                    "symbols": int(np.count_nonzero(mask)),
                    "applied": False,
                    "reason": "zero_reference_power",
                }
            )
            continue
        gain = np.vdot(ref, obs) / denominator
        if abs(gain) <= 1e-9:
            corrected.append(
                {
                    "branch": int(branch),
                    "reliable_symbols": reliable_count,
                    "symbols": int(np.count_nonzero(mask)),
                    "applied": False,
                    "reason": "zero_gain",
                }
            )
            continue
        values[mask] = (values[mask] / gain).astype(np.complex64)
        corrected.append(
            {
                "branch": int(branch),
                "reliable_symbols": reliable_count,
                "symbols": int(np.count_nonzero(mask)),
                "applied": True,
                "gain_real": float(np.real(gain)),
                "gain_imag": float(np.imag(gain)),
                "gain_amplitude": float(abs(gain)),
                "cpe_rad": float(np.angle(gain)),
            }
        )
    return values, _branch_gain_report(
        selected_branches=resolved,
        reliability_threshold=reliability_threshold,
        min_symbols=min_symbols,
        corrected_branches=corrected,
    )


def _resolve_llr_plane_scales(
    plane_scales: Sequence[float] | str | None,
    *,
    bits_per_symbol: int,
) -> tuple[float, ...] | None:
    if plane_scales is None:
        return None
    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be positive")
    if isinstance(plane_scales, str):
        values = tuple(
            float(token.strip())
            for token in plane_scales.split(",")
            if token.strip()
        )
    else:
        values = tuple(float(value) for value in plane_scales)
    if len(values) != bits_per_symbol:
        raise ValueError(
            f"llr_plane_scales must contain {bits_per_symbol} values for this QAM mode"
        )
    if any((not np.isfinite(value)) or value < 0.0 for value in values):
        raise ValueError("llr_plane_scales values must be finite and non-negative")
    return values


def _qam_symbol_quality_report(
    symbols_before_deinterleave: np.ndarray,
    symbols_after_deinterleave: np.ndarray,
    *,
    qam_mode: str,
) -> dict[str, Any]:
    before = np.asarray(symbols_before_deinterleave, dtype=np.complex64).reshape(-1)
    after = np.asarray(symbols_after_deinterleave, dtype=np.complex64).reshape(-1)
    return {
        "qam_mode": qam_mode,
        "normalizer": "standard",
        "before_symbol_deinterleave": qam_symbol_quality(
            before,
            mode=qam_mode,  # type: ignore[arg-type]
            normalize=True,
        ),
        "after_symbol_deinterleave": qam_symbol_quality(
            after,
            mode=qam_mode,  # type: ignore[arg-type]
            normalize=True,
        ),
    }


def _hard_bit_balance_per_plane(
    hard_bits: np.ndarray,
    *,
    bits_per_symbol: int,
) -> list[float]:
    values = np.asarray(hard_bits, dtype=np.uint8).reshape(-1)
    if bits_per_symbol <= 0:
        return []
    usable = (values.size // bits_per_symbol) * bits_per_symbol
    if usable == 0:
        return []
    grouped = values[:usable].reshape(-1, bits_per_symbol)
    return [float(np.mean(grouped[:, index])) for index in range(bits_per_symbol)]


def _concat_bits(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate(chunks).astype(np.uint8, copy=False)


def _concat_symbols(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.complex64)
    return np.concatenate(chunks).astype(np.complex64, copy=False)


def _iter_frame_bits(bits: np.ndarray, *, bits_per_frame: int) -> list[np.ndarray]:
    if bits_per_frame <= 0:
        return []
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    frame_count = values.size // bits_per_frame
    return [
        values[index * bits_per_frame : (index + 1) * bits_per_frame]
        for index in range(frame_count)
    ]


def _iter_frame_llrs(llr: np.ndarray, *, llrs_per_frame: int) -> list[np.ndarray]:
    if llrs_per_frame <= 0:
        return []
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    frame_count = values.size // llrs_per_frame
    return [
        values[index * llrs_per_frame : (index + 1) * llrs_per_frame]
        for index in range(frame_count)
    ]


def _bits_per_symbol(mode: QamMode) -> int:
    if mode == "4qam":
        return 2
    if mode == "16qam":
        return 4
    if mode == "32qam":
        return 5
    if mode == "64qam":
        return 6
    raise ValueError("unsupported QAM mode")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _write_ts_bytes(output_path: str | Path, data: bytes) -> None:
    """Write TS bytes to a file path or stdout ('-')."""

    target = str(output_path)
    if target == "-":
        import sys

        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            # Downstream consumer closed its stdin after taking enough bytes
            # (ffmpeg often does this once it has a full TS program). Treat
            # that as a clean end-of-stream so the receiver reports success.
            try:
                sys.stdout.close()
            except Exception:
                pass
        return
    Path(target).write_bytes(data)


def _fec_rate_or_auto(value: str) -> int | Literal["auto"]:
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed not in (1, 2, 3):
        raise argparse.ArgumentTypeError("must be auto, 1, 2, or 3")
    return parsed


def _parse_pn_mmse(value: str | None) -> float | Literal["auto"] | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("off", "none", ""):
        return None
    if text == "auto":
        return "auto"
    parsed = float(text)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("--pn-mmse must be off, auto, or a non-negative float")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
