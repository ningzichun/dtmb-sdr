"""Stage: CI8 + sysinfo -> LLR stream (+ framing sidecar).

This stage runs the receiver front end (acquisition, timing, FFT, frame
body equalization, QAM soft demap, symbol deinterleaver) and writes the
post-deinterleaver LLR stream plus a small JSON sidecar describing how
the stream is framed. It deliberately stops before LDPC so a downstream
``decode_ldpc`` stage can consume the pair, regardless of which LDPC
backend is in use.

Inputs:
    <prefix>.ci8
    <prefix>.ci8.json
    <prefix>.sysinfo.json (optional; pass --sysinfo to use it)

Outputs:
    <prefix>.llr.f32          float32 little-endian LLR stream
    <prefix>.llr.f32.json     framing sidecar (qam_mode, fec_rate_index,
                              bits_per_frame, symbol_deinterleave, ...)
    <prefix>.demap.json       stage diagnostic (phase, CFO, Gate C)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from _common import ROOT, child_env, load_capture_sidecar, read_json
from gate_c import (
    pick_mode_from_gate_c,
    resolve_receiver_timing_policy,
    selected_system_info_index,
    sysinfo_oracle_quality,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-demap")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sysinfo", type=Path, default=None)
    parser.add_argument("--sysinfo-oracle", type=Path, default=None)
    parser.add_argument("--output-llr", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument(
        "--input-skip",
        type=int,
        default=0,
        help="Raw input samples to skip before receiver/demap analysis.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help="Complex frequency shift in Hz applied before receiver/demap analysis.",
    )
    parser.add_argument("--phase-offset", type=int, default=None)
    parser.add_argument("--body-window-offset", type=int, default=0)
    parser.add_argument("--fft-bin-shift", type=int, default=0)
    parser.add_argument(
        "--frequency-deinterleaver-direction",
        choices=("standard", "inverse"),
        default="standard",
    )
    parser.add_argument(
        "--data-carrier-order",
        choices=("normal", "reverse"),
        default="normal",
    )
    parser.add_argument("--carrier-permutation", default="identity")
    parser.add_argument("--logical-position-shift", type=int, default=0)
    parser.add_argument("--expected-si-index", type=int, default=23)
    parser.add_argument(
        "--expected-qam",
        choices=("4qam", "16qam", "32qam", "64qam"),
        default="64qam",
    )
    parser.add_argument(
        "--expected-fec-rate", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument(
        "--expected-interleaver", choices=("mode1", "mode2"), default="mode1"
    )
    parser.add_argument(
        "--equalizer",
        choices=(
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
        ),
        default="dd",
    )
    parser.add_argument(
        "--dd-max-hard-bit-bias",
        type=float,
        default=0.0,
        help=(
            "Pass-through opt-in DD QAM bit-balance guard. Default 0 keeps "
            "dtmb.receiver's current behavior."
        ),
    )
    parser.add_argument(
        "--llr-scale",
        type=float,
        default=1.0,
        help="Pass-through demapped LLR scale applied before LDPC-side artifacts.",
    )
    parser.add_argument(
        "--llr-clip",
        type=float,
        default=None,
        help="Optional pass-through demapped LLR clip magnitude.",
    )
    parser.add_argument(
        "--llr-erase-fraction",
        type=float,
        default=0.0,
        help="Optional pass-through low-confidence LLR erase fraction.",
    )
    parser.add_argument(
        "--llr-plane-scales",
        default=None,
        help="Optional pass-through comma-separated per-bit-plane LLR multipliers.",
    )
    parser.add_argument(
        "--no-per-frame-cpe",
        action="store_true",
        help="Disable per-frame CPE correction (enabled by default).",
    )
    parser.add_argument(
        "--synthetic-loopback",
        action="store_true",
        help=(
            "Assume synthetic CI8 produced by dtmb.tx_c3780: skip timing "
            "search, pin phase offset to 0, disable equalization."
        ),
    )
    parser.add_argument(
        "--force-demap",
        action="store_true",
        help=(
            "Run the full demap path even when Gate C failed. By default a "
            "weak Gate C writes an empty .llr.f32 and marks the sidecar as "
            "skipped, mirroring the receive stage's behaviour."
        ),
    )
    parser.add_argument(
        "--timing-policy",
        choices=("fixed", "windowed", "trajectory"),
        default="fixed",
        help="Frame slicing policy passed through to dtmb.receiver.",
    )
    parser.add_argument(
        "--timing-trajectory",
        type=Path,
        help="Optional timing trajectory JSON for windowed/trajectory slicing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress dtmb.receiver JSON on stdout.",
    )
    return parser


def _pick_mode(
    sysinfo_json: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, int, str, str, bool, dict[str, Any]]:
    """Return mode choice plus Gate C quality metadata."""

    return pick_mode_from_gate_c(
        sysinfo_json,
        expected_qam=args.expected_qam,
        expected_fec_rate=args.expected_fec_rate,
        expected_interleaver=args.expected_interleaver,
        synthetic_loopback=args.synthetic_loopback,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = load_capture_sidecar(args.capture)
    sample_rate = int(sidecar["sample_rate_sps"])
    byte_count = int(sidecar["byte_count"])
    input_skip = max(0, int(args.input_skip))
    remaining_samples = max(0, byte_count // 2 - input_skip)
    symbols_needed = args.frames * 4725 * 2
    samples_needed = int(symbols_needed * sample_rate / 7_560_000)
    max_samples = min(remaining_samples, max(samples_needed, 1_000_000))

    sysinfo_data: dict[str, Any] = {}
    if args.sysinfo is not None and args.sysinfo.exists():
        sysinfo_data = read_json(args.sysinfo)
    if args.sysinfo_oracle is not None and args.sysinfo_oracle.exists():
        oracle_data = read_json(args.sysinfo_oracle)
    else:
        oracle_data = None
    oracle_quality = sysinfo_oracle_quality(oracle_data)
    qam, fec_rate, interleaver, gate_c_note, fallback_active, gate_c = _pick_mode(
        sysinfo_data, args
    )
    receiver_system_info_index = selected_system_info_index(
        sysinfo_data,
        args.expected_si_index,
    )

    # Honour the same Gate-C-fallback skip the receive stage uses so the
    # demap + decode_ldpc split does not accidentally waste minutes of CPU
    # on a capture whose Gate C already failed upstream.
    oracle_blocked = oracle_quality["blocking"] is True
    if (fallback_active or oracle_blocked) and not args.force_demap:
        if oracle_blocked and not fallback_active:
            note = (
                "skipped_demap_due_to_sysinfo_oracle_gate; "
                "rerun with --force-demap to emit an LLR dump anyway."
            )
        else:
            note = (
                "skipped_demap_due_to_gate_c_fallback; "
                "rerun with --force-demap to emit an LLR dump anyway."
            )
        print(f"[demap] {note}", file=sys.stderr)
        args.output_llr.parent.mkdir(parents=True, exist_ok=True)
        args.output_llr.write_bytes(b"")
        sidecar_path = args.output_llr.with_suffix(args.output_llr.suffix + ".json")
        sidecar_path.write_text(
            json.dumps(
                {
                    "skipped": True,
                    "skip_reason": note,
                    "gate_c_note": gate_c_note,
                    "gate_c_quality": gate_c,
                    "sysinfo_oracle_quality": oracle_quality,
                    "picked_qam_mode": qam,
                    "picked_fec_rate_index": fec_rate,
                    "picked_symbol_deinterleave": interleaver,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "pipeline": {
                        "gate_c_note": gate_c_note,
                        "gate_c_quality": gate_c,
                        "sysinfo_oracle_path": str(args.sysinfo_oracle)
                        if args.sysinfo_oracle
                        else None,
                        "sysinfo_oracle_quality": oracle_quality,
                        "picked_qam_mode": qam,
                        "picked_fec_rate_index": fec_rate,
                        "picked_symbol_deinterleave": interleaver,
                        "picked_system_info_index": receiver_system_info_index,
                        "sysinfo_path": str(args.sysinfo) if args.sysinfo else None,
                        "skipped": True,
                        "skip_reason": note,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    args.output_llr.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    equalizer = "none" if args.synthetic_loopback else args.equalizer
    cmd = [
        sys.executable,
        "-m",
        "dtmb.receiver",
        str(args.capture),
        "--sample-rate",
        str(sample_rate),
        "--max-samples",
        str(max_samples),
        "--frequency-shift",
        str(args.frequency_shift),
        "--input-skip",
        str(input_skip),
        "--mode",
        "pn945",
        "--qam",
        qam,
        "--equalizer",
        equalizer,
        "--dd-max-hard-bit-bias",
        str(args.dd_max_hard_bit_bias),
        "--llr-scale",
        str(args.llr_scale),
        "--llr-erase-fraction",
        str(args.llr_erase_fraction),
        "--symbol-deinterleave",
        interleaver,
        "--fec-mode",
        "none",
        "--fec-rate",
        str(fec_rate),
        "--system-info-index",
        str(receiver_system_info_index),
        "--frames",
        str(args.frames),
        "--body-window-offset",
        str(args.body_window_offset),
        "--fft-bin-shift",
        str(args.fft_bin_shift),
        "--frequency-deinterleaver-direction",
        str(args.frequency_deinterleaver_direction),
        "--data-carrier-order",
        str(args.data_carrier_order),
        "--carrier-permutation",
        str(args.carrier_permutation),
        "--logical-position-shift",
        str(args.logical_position_shift),
        "--llr-output",
        str(args.output_llr),
        "--json",
        str(args.output_json),
        "--min-ts-packets",
        "1",
        "--min-ts-sync-ratio",
        "0.0",
        "--min-ts-valid-ratio",
        "0.0",
    ]
    if args.phase_offset is not None:
        cmd.extend(["--phase-offset", str(args.phase_offset)])
    if args.quiet:
        cmd.append("--quiet")
    if args.llr_clip is not None:
        cmd.extend(["--llr-clip", str(args.llr_clip)])
    if args.llr_plane_scales is not None:
        cmd.extend(["--llr-plane-scales", str(args.llr_plane_scales)])
    effective_timing_policy = resolve_receiver_timing_policy(
        args.timing_policy,
        args.timing_trajectory,
    )
    cmd.extend(["--timing-policy", effective_timing_policy])
    if args.timing_trajectory is not None:
        cmd.extend(["--timing-trajectory", str(args.timing_trajectory)])
    if args.synthetic_loopback:
        cmd.extend(["--phase-offset", "0", "--no-timing-search"])
    elif args.timing_trajectory is not None and effective_timing_policy in {
        "windowed",
        "trajectory",
    }:
        cmd.append("--no-timing-search")
    elif not args.no_per_frame_cpe:
        cmd.append("--correct-per-frame-cpe")
    if (
        not args.synthetic_loopback
        and not args.no_per_frame_cpe
        and "--correct-per-frame-cpe" not in cmd
    ):
        cmd.append("--correct-per-frame-cpe")
    print("[demap]", gate_c_note, " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())

    # Annotate the demap.json with the pipeline decision so later stages
    # know which sysinfo source picked the mode.
    if args.output_json.exists():
        data = read_json(args.output_json)
        data["pipeline"] = {
            "gate_c_note": gate_c_note,
            "gate_c_quality": gate_c,
            "sysinfo_oracle_path": str(args.sysinfo_oracle)
            if args.sysinfo_oracle
            else None,
            "sysinfo_oracle_quality": oracle_quality,
            "picked_qam_mode": qam,
            "picked_fec_rate_index": fec_rate,
            "picked_symbol_deinterleave": interleaver,
            "picked_system_info_index": receiver_system_info_index,
            "sysinfo_path": str(args.sysinfo) if args.sysinfo else None,
            "requested_timing_policy": args.timing_policy,
            "receiver_timing_policy": effective_timing_policy,
            "timing_trajectory_path": str(args.timing_trajectory)
            if args.timing_trajectory is not None
            else None,
            "skipped": False,
        }
        args.output_json.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    # rc=0 (TS locked) and rc=2 (no TS, diagnostic only) are both normal
    # stage outcomes; only surface genuine errors.
    return 0 if rc in (0, 2) else rc


if __name__ == "__main__":
    raise SystemExit(main())
