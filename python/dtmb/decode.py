"""Minimal vendor-neutral CI8 file/stdin to MPEG-TS pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import BinaryIO, Sequence


SYMBOL_RATE = 7_560_000
PROFILES = {
    19: (1, "mode1"),
    20: (1, "mode2"),
    21: (2, "mode1"),
    22: (2, "mode2"),
    23: (3, "mode1"),
    24: (3, "mode2"),
}


def _exe(name: str, bin_dir: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = (
        bin_dir / "Release" / f"{name}{suffix}",
        bin_dir / f"{name}{suffix}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing native executable: {name} in {bin_dir}")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.input_rate <= 0:
        raise ValueError("--input-rate must be positive")
    if args.system_info_index not in PROFILES:
        raise ValueError("--system-info-index must be 19..24")
    fec_rate, interleaver = PROFILES[args.system_info_index]
    source = args.input
    commands: list[list[str]] = []

    if args.input_rate != SYMBOL_RATE:
        commands.append(
            [
                str(_exe("dtmb_core_ci8_resample", args.bin_dir)),
                "--input-rate",
                str(args.input_rate),
                "--output-rate",
                str(SYMBOL_RATE),
                "--workers",
                str(args.workers),
                source,
                "-",
            ]
        )
        source = "-"

    commands.append(
        [
            str(_exe("dtmb_core_c3780_extract", args.bin_dir)),
            "--auto-sync",
            "--sync-frames",
            "300",
            "--acquisition-frames",
            "16",
            "--auto-phase-adjustment",
            "1" if args.input_rate != SYMBOL_RATE else "0",
            "--workers",
            str(args.workers),
            "--equalizer",
            "pn",
            "--pn-estimator",
            "wideband",
            "--pn-channel-taps",
            "8",
            "--pn-wideband-block-frames",
            "2",
            "--pn-mmse",
            "0.004",
            "--remove-dc",
            "--normalization",
            "qam64",
            "--system-info-index",
            str(args.system_info_index),
            source,
            "-",
        ]
    )
    commands.append(
        [
            str(_exe("dtmb_core_deinterleave_qam64", args.bin_dir)),
            "--mode",
            interleaver,
            "--phase",
            "0",
            "--workers",
            str(args.workers),
            "-",
            "-",
        ]
    )
    fec = [
        str(_exe("dtmb_core_ldpc_bch_decode", args.bin_dir)),
        "--fec-rate",
        str(fec_rate),
        "--alist",
        str((args.data_dir / f"dtmb_ldpc_rate{fec_rate}.alist").resolve()),
        "--codewords-per-frame",
        "3",
        "--workers",
        str(args.workers),
        "--ldpc-accel",
        args.acceleration,
        "--decode-batch-frames",
        str(args.decode_batch_frames),
        "--max-iterations",
        str(args.max_iterations),
        "--early-syndrome-reject-ratio",
        str(args.early_syndrome_reject_ratio),
        "--clean-frames-only",
    ]
    if args.error_policy == "continue":
        fec.extend(["--mark-discontinuities", "--insert-discontinuity-packets"])
    fec.extend(["-", args.output])
    commands.append(fec)
    return commands


def run(commands: list[list[str]]) -> int:
    processes: list[subprocess.Popen[bytes]] = []
    previous: BinaryIO | None = None
    try:
        for index, command in enumerate(commands):
            process = subprocess.Popen(
                command,
                stdin=previous,
                stdout=None if index == len(commands) - 1 else subprocess.PIPE,
            )
            if previous is not None:
                previous.close()
            previous = process.stdout
            processes.append(process)
        returncodes = [process.wait() for process in processes]
    except BaseException:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        raise
    failures = [code for code in returncodes if code != 0]
    return failures[-1] if failures else 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        prog="dtmb-decode",
        description="Decode a CI8 file or stdin stream into MPEG-TS.",
    )
    parser.add_argument("--input", default="-", help="CI8 path or - for stdin")
    parser.add_argument("--input-rate", type=int, required=True)
    parser.add_argument("--output", default="-", help="MPEG-TS path or - for stdout")
    parser.add_argument("--system-info-index", type=int, default=22)
    parser.add_argument("--acceleration", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--decode-batch-frames", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--early-syndrome-reject-ratio", type=float, default=0.46)
    parser.add_argument("--error-policy", choices=("fail", "continue"), default="fail")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--bin-dir", type=Path, default=root / "build" / "core-cpp")
    parser.add_argument("--data-dir", type=Path, default=root / "python" / "dtmb" / "data")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        commands = build_commands(args)
        if args.dry_run:
            for command in commands:
                print(subprocess.list2cmdline(command))
            return 0
        return run(commands)
    except (FileNotFoundError, ValueError) as exc:
        print(f"dtmb-decode: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
