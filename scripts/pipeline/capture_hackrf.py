"""Stage 0: record a bounded CI8 capture from HackRF.

Inputs:
    --frequency Hz
    --sample-rate sps (default 20 MSps, use 7560000 for native symbol rate)
    --duration seconds
    --output path to .ci8

Outputs:
    <output>            raw CI8 bytes
    <output>.json       capture metadata sidecar

This is a thin wrapper over `python -m dtmb.capture` so Makefile targets
only need to know the output path and desired parameters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env


def _resolve_default_hackrf_bin() -> str:
    """Pick a hackrf_transfer binary that actually exists on this machine.

    Tries common install locations used by this project so the stage works
    out-of-the-box on developer boxes that don't have hackrf on PATH.
    """

    import os
    import shutil

    on_path = shutil.which("hackrf_transfer")
    if on_path:
        return on_path

    candidates = [
        os.path.expandvars(
            r"%USERPROFILE%\miniforge3\envs\gr-dtmb\Library\bin\hackrf_transfer.exe"
        ),
        os.path.expandvars(
            r"%USERPROFILE%\miniforge3\Library\bin\hackrf_transfer.exe"
        ),
        r"C:\Program Files\PothosSDR\bin\hackrf_transfer.exe",
        r"C:\Program Files (x86)\PothosSDR\bin\hackrf_transfer.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    # Fall back to bare name so the child process error surfaces clearly.
    return "hackrf_transfer"


DEFAULT_HACKRF_BIN = _resolve_default_hackrf_bin()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, default=20_000_000)
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=None,
        help=(
            "Baseband filter bandwidth; defaults to HackRF's anti-aliasing "
            "choice (<= 0.75 * sample rate), except native 7.56 MSps DTMB "
            "capture uses 8 MHz so the channel edges are not cut."
        ),
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--lna-gain", type=int, default=24)
    parser.add_argument("--vga-gain", type=int, default=18)
    parser.add_argument("--antenna", default="UHF antenna")
    parser.add_argument("--location", default="")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--hackrf-bin", default=DEFAULT_HACKRF_BIN)
    parser.add_argument(
        "--device-serial",
        help="Select the same HackRF serial used by clock preflight and streaming stages.",
    )
    parser.add_argument(
        "--clock-source",
        choices=("internal", "external-10mhz", "gpsdo-10mhz", "unknown"),
        default="internal",
    )
    parser.add_argument("--external-reference-hz", type=int, default=None)
    parser.add_argument("--hardware-trigger", action="store_true")
    parser.add_argument("--trigger-source", default=None)
    parser.add_argument("--trigger-time-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bandwidth = args.bandwidth
    if bandwidth is None:
        bandwidth = _snap_bandwidth(args.sample_rate)
    cmd = [
        sys.executable,
        "-m",
        "dtmb.capture",
        str(args.output),
        "--frequency",
        str(args.frequency),
        "--sample-rate",
        str(args.sample_rate),
        "--bandwidth",
        str(bandwidth),
        "--duration",
        str(args.duration),
        "--amp",
        str(args.amp),
        "--lna-gain",
        str(args.lna_gain),
        "--vga-gain",
        str(args.vga_gain),
        "--executable",
        args.hackrf_bin,
        "--clock-source",
        args.clock_source,
        "--antenna",
        args.antenna,
        "--location",
        args.location,
    ]
    if args.device_serial:
        cmd.extend(["--device-serial", args.device_serial])
    if args.external_reference_hz is not None:
        cmd.extend(["--external-reference-hz", str(args.external_reference_hz)])
    if args.hardware_trigger:
        cmd.append("--hardware-trigger")
    if args.trigger_source:
        cmd.extend(["--trigger-source", args.trigger_source])
    if args.trigger_time_utc:
        cmd.extend(["--trigger-time-utc", args.trigger_time_utc])
    if args.notes:
        cmd.extend(["--notes", args.notes])
    print("[capture]", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(ROOT), env=child_env())


def _snap_bandwidth(sample_rate: int) -> int:
    """Mirror HackRF's anti-aliasing default while preserving native DTMB."""

    valid = (
        1_750_000,
        2_500_000,
        3_500_000,
        5_000_000,
        5_500_000,
        6_000_000,
        7_000_000,
        8_000_000,
        9_000_000,
        10_000_000,
        12_000_000,
        14_000_000,
        15_000_000,
        20_000_000,
    )
    if sample_rate <= 8_000_000:
        return 8_000_000
    target = int(sample_rate * 0.75)
    eligible = [bw for bw in valid if bw <= target]
    if not eligible:
        return valid[0]
    return max(eligible)


if __name__ == "__main__":
    raise SystemExit(main())
