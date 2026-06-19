"""Stage 3: full offline DTMB receive to recovered MPEG-TS.

Inputs:
    <capture>.ci8
    <capture>.ci8.json
    <capture>.sysinfo.json  (optional; pass --sysinfo path to use it)

Outputs:
    <capture>.recovered.ts
    <capture>.receiver.json

The stage reads the sysinfo summary produced by Stage 2 when present to
decide whether Gate C is stable enough to proceed. When sysinfo is stable,
it uses the reported (qam_mode, fec_rate, interleaver) triplet. When the
sysinfo file is missing, empty, or Gate C is weak, it falls back to the
`--expected-si-index` value so the run still produces a diagnostic
artifact instead of aborting. The pipeline.receive JSON section records
which source drove the mode decision.
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
    gate_c_quality,
    pick_mode_from_gate_c,
    resolve_receiver_timing_policy,
    selected_system_info_index,
    sysinfo_oracle_quality,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-receive")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--sysinfo",
        type=Path,
        default=None,
        help="Optional Gate C JSON produced by sysinfo.py.",
    )
    parser.add_argument(
        "--sysinfo-oracle",
        type=Path,
        default=None,
        help="Optional independent raw system-information oracle JSON.",
    )
    parser.add_argument("--output-ts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument(
        "--input-skip",
        type=int,
        default=0,
        help="Raw input samples to skip before receiver analysis.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help="Complex frequency shift in Hz applied before receiver analysis.",
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
    parser.add_argument(
        "--expected-si-index",
        type=int,
        default=23,
        help="Fallback system-information index when Gate C is weak.",
    )
    parser.add_argument(
        "--expected-qam",
        choices=("4qam", "16qam", "32qam", "64qam"),
        default="64qam",
        help="Fallback QAM mode.",
    )
    parser.add_argument(
        "--expected-fec-rate",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Fallback FEC rate index (1/2/3 for rates 0.4/0.6/0.8).",
    )
    parser.add_argument(
        "--expected-interleaver",
        choices=("mode1", "mode2"),
        default="mode1",
        help="Fallback interleaver mode.",
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
        default="pn-dd",
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
        help="Pass-through demapped LLR scale applied before LDPC decode.",
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
            "search, pin phase offset to 0, disable equalization. Matches "
            "scripts/pipeline_end_to_end.py defaults."
        ),
    )
    parser.add_argument(
        "--force-receive",
        action="store_true",
        help=(
            "Run the full LDPC receive path even when Gate C failed. By "
            "default a weak Gate C skips receive and emits a diagnostic-only "
            "JSON because full DD+CPE FEC on 600 frames is very expensive."
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


def _gate_c_quality(sysinfo_json: dict[str, Any]) -> dict[str, Any]:
    """Summarise Gate C quality markers used to decide the receive budget."""

    return gate_c_quality(sysinfo_json)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = load_capture_sidecar(args.capture)
    sample_rate = int(sidecar["sample_rate_sps"])
    byte_count = int(sidecar["byte_count"])
    input_skip = max(0, int(args.input_skip))
    remaining_samples = max(0, byte_count // 2 - input_skip)
    # Budget just enough samples for --frames at the capture rate, with 2x
    # safety factor for acquisition/drain. For a 600-frame pass at 20 MSps
    # that is roughly 15M samples (30 MB), well under a typical capture.
    symbols_needed = args.frames * 4725 * 2
    samples_needed = int(symbols_needed * sample_rate / 7_560_000)
    max_samples = min(remaining_samples, max(samples_needed, 1_000_000))

    if args.sysinfo is not None and args.sysinfo.exists():
        sysinfo_data = read_json(args.sysinfo)
    else:
        sysinfo_data = {}
    if args.sysinfo_oracle is not None and args.sysinfo_oracle.exists():
        oracle_data = read_json(args.sysinfo_oracle)
    else:
        oracle_data = None
    oracle_quality = sysinfo_oracle_quality(oracle_data)

    qam, fec_rate, interleaver, gate_c_note, fallback_active, gate_c = (
        pick_mode_from_gate_c(
            sysinfo_data,
            expected_qam=args.expected_qam,
            expected_fec_rate=args.expected_fec_rate,
            expected_interleaver=args.expected_interleaver,
            synthetic_loopback=args.synthetic_loopback,
        )
    )
    receiver_system_info_index = selected_system_info_index(
        sysinfo_data,
        args.expected_si_index,
    )
    if interleaver == "auto":
        interleaver = args.expected_interleaver

    # When Gate C has failed on a real capture, running a full DD+CPE LDPC
    # pass across --frames is extremely expensive (tens of minutes on a
    # 600-frame wall capture) and yields no usable TS. Skip the receiver
    # entirely in that case and emit a diagnostic-only JSON so the
    # pipeline stays within a predictable time budget.
    oracle_blocked = oracle_quality["blocking"] is True
    if (fallback_active or oracle_blocked) and not args.force_receive:
        if oracle_blocked and not fallback_active:
            note = (
                "skipped_ldpc_receive_due_to_sysinfo_oracle_gate; "
                "rerun with --force-receive to decode regardless."
            )
        else:
            note = (
                "skipped_ldpc_receive_due_to_gate_c_fallback; "
                "rerun with --force-receive to decode regardless."
            )
        print(f"[receive] {note}", file=sys.stderr)
        args.output_ts.parent.mkdir(parents=True, exist_ok=True)
        args.output_ts.write_bytes(b"")
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
                    "ts": {"lock": None},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    args.output_ts.parent.mkdir(parents=True, exist_ok=True)
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
        "ldpc",
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
        "--min-ts-packets",
        "8",
        "--output",
        str(args.output_ts),
        "--json",
        str(args.output_json),
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
    print("[receive]", gate_c_note, " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())

    # Annotate the diagnostic JSON with the pipeline decision so later
    # stages (and humans) can tell whether Gate C drove the mode choice.
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

    # rc=0 is a TS lock, rc=2 is receiver ran with no TS lock. Both are
    # acceptable stage outcomes; the transcode stage decides what to do.
    return 0 if rc in (0, 2) else rc


if __name__ == "__main__":
    raise SystemExit(main())
