"""Minimal vendor-neutral CI8 file/stdin to MPEG-TS pipeline."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import BinaryIO, Sequence
from .cuda import native_cuda_environment


SYMBOL_RATE = 7_560_000
PROFILES = {
    3: (3, "mode1"),
    4: (3, "mode2"),
    **{i: ((i-5)//2+1, "mode1" if i%2 else "mode2") for i in range(5, 11)},
    11: (1, "mode1"),
    12: (1, "mode2"),
    13: (2, "mode1"),
    14: (2, "mode2"),
    15: (3, "mode1"),
    16: (3, "mode2"),
    17: (3, "mode1"),
    18: (3, "mode2"),
    19: (1, "mode1"),
    20: (1, "mode2"),
    21: (2, "mode1"),
    22: (2, "mode2"),
    23: (3, "mode1"),
    24: (3, "mode2"),
}


def _exe(name: str, bin_dir: Path | None) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    package_bin = Path(__file__).resolve().parent / "bin"
    candidates: list[Path] = []
    if bin_dir is not None:
        candidates.extend(
            [
                bin_dir / "Release" / f"{name}{suffix}",
                bin_dir / f"{name}{suffix}",
            ]
        )
    candidates.append(package_bin / f"{name}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing native executable: {name}; searched {searched}")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.input_rate <= 0:
        raise ValueError("--input-rate must be positive")
    if args.system_info_index not in PROFILES:
        raise ValueError("--system-info-index must be 3/4 (4QAM-NR), 5..10 (4QAM), 11..16 (16QAM), 17..18 (32QAM) or 19..24 (64QAM)")
    if args.pn_mode not in ("pn420", "pn595", "pn945"):
        raise ValueError("--pn-mode must be pn420, pn595 or pn945")
    frame_body_mode = getattr(args, "frame_body_mode", "c3780")
    if frame_body_mode not in {"auto", "c1", "c3780"}:
        raise ValueError("--frame-body-mode must be auto, c1 or c3780")
    if frame_body_mode == "c1" and args.pn_mode != "pn595":
        raise ValueError("C=1 processing requires PN595")
    if args.pn_mode == "pn595" and args.pn_schedule_tracking:
        raise ValueError("PN595 uses fixed-sequence timing tracking, without cyclic PN schedule options")
    fec_rate, interleaver = PROFILES[args.system_info_index]
    qam = "4qam-nr" if args.system_info_index <= 4 else "4qam" if args.system_info_index <= 10 else "16qam" if args.system_info_index <= 16 else "32qam" if args.system_info_index <= 18 else "64qam"
    codewords_per_group = {"4qam-nr": 1, "4qam": 1, "16qam": 2, "32qam": 5, "64qam": 3}[qam]
    if args.qam32_frame_phase != "auto" and qam != "32qam":
        raise ValueError("--qam32-frame-phase requires a 32QAM profile")
    if args.nr_frame_phase != "auto" and qam != "4qam-nr":
        raise ValueError("--nr-frame-phase requires a 4QAM-NR profile")
    if not math.isfinite(args.resample_headroom_db) or not 0 <= args.resample_headroom_db <= 24:
        raise ValueError("--resample-headroom-db must be finite and in 0..24")
    if args.resample_headroom_db and args.input_rate == SYMBOL_RATE:
        raise ValueError("--resample-headroom-db requires input-rate conversion")
    if args.pipeline_buffer_mib < 0:
        raise ValueError("--pipeline-buffer-mib must be non-negative")
    input_format = getattr(args, "input_format", "ci8")
    precise = input_format not in {"ci8", "cs8"}
    source = args.input
    commands: list[list[str]] = []
    resample_gain = 10 ** (-args.resample_headroom_db / 20)

    if args.input_rate != SYMBOL_RATE or precise:
        resample = [
            str(_exe("dtmb_core_ci8_resample", args.bin_dir)),
            "--input-rate", str(args.input_rate),
            "--output-rate", str(SYMBOL_RATE),
            "--workers", str(args.workers),
        ]
        if precise:
            resample.extend(["--input-format", input_format, "--output-format", "cf32"])
        if args.resample_headroom_db:
            resample.extend(["--output-scale", format(resample_gain if precise else resample_gain / math.sqrt(45), ".9g")])
        resample.extend([source, "-"])
        commands.append(resample)
        source = "-"

    frontend = [
            str(_exe("dtmb_core_c3780_extract", args.bin_dir)),
            "--auto-sync",
            "--sync-frames", "300",
            "--acquisition-frames", "16",
            "--auto-phase-adjustment", "1" if args.input_rate == 20_000_000 and args.pn_mode == "pn945" and qam == "64qam" else "0",
            "--timing-search-radius", "2",
            "--timing-search-threshold", "0.45",
            "--timing-trajectory-interval-frames", "400",
            "--timing-trajectory-fit-points", "17",
            "--timing-trajectory-max-innovation-samples", "2",
            "--workers", str(args.workers),
            "--batch-frames", str(max(args.workers, min(args.workers * 8, 256))),
            "--equalizer",
            "pn",
            "--pn-estimator",
            "wideband",
            "--pn-channel-taps",
            "8",
            "--pn-wideband-block-frames",
            "2",
            "--pn-wideband-header-observation", "direct" if args.pn_mode == "pn595" else "core-postfix",
            "--pn-wideband-scale-estimator", "masked-frame-taps",
            "--pn-mmse",
            format(0.004 * resample_gain**2, ".9g"),
            "--remove-dc",
            "--pn-mode", args.pn_mode,
            "--qam", qam,
            "--normalization",
            "qam64" if qam == "64qam" else "qam",
            "--system-info-index",
            str(args.system_info_index),
    ]
    if frame_body_mode != "c3780":
        frontend.extend(["--frame-body-mode", frame_body_mode])
    if precise:
        frontend.extend(["--input-format", "cf32"])
    if args.pn_schedule_tracking:
        frontend.extend(["--pn-schedule-tracking", "--pn-current-header-tracking"])
    frontend.extend([source, "-"])
    commands.append(frontend)

    demap = [
            str(_exe("dtmb_core_deinterleave_qam64", args.bin_dir)),
            "--qam", qam,
            "--mode",
            interleaver,
            "--phase",
            "0",
            "--workers", str(args.workers),
            "--chunk-symbols", "1048576",
    ]
    if args.source_frame_confidence != "off":
        demap.extend(["--source-frame-confidence", args.source_frame_confidence])
    demap.extend(["-", "-"])
    commands.append(demap)
    fec = [
        str(_exe("dtmb_core_ldpc_bch_decode", args.bin_dir)),
        "--fec-rate",
        str(fec_rate),
        "--alist",
        str((args.data_dir / f"dtmb_ldpc_rate{fec_rate}.alist").resolve()),
        "--codewords-per-frame",
        str(codewords_per_group),
        "--workers",
        str(args.workers),
        "--ldpc-accel",
        args.acceleration,
        "--decode-batch-frames",
        str(args.decode_batch_frames),
        "--max-iterations",
        str(args.max_iterations),
        "--retry-max-iterations", "50",
        "--attenuation", "0.65",
        "--clean-frames-only",
        "--require-output",
    ]
    if qam == "32qam":
        fec.extend(["--qam", qam, "--qam32-frame-phase", args.qam32_frame_phase])
    elif qam == "4qam-nr":
        fec.extend(["--qam", qam, "--nr-frame-phase", args.nr_frame_phase])
    elif qam == "4qam":
        fec.extend(["--qam", qam])
    if args.error_policy == "continue":
        fec.append("--insert-discontinuity-packets")
    else:
        fec.append("--fail-on-unclean-frame")
    if args.early_syndrome_reject_ratio is not None:
        fec.extend(["--early-syndrome-reject-ratio", str(args.early_syndrome_reject_ratio)])
    fec.extend(["-", args.output])
    commands.append(fec)

    if args.pipeline_buffer_mib:
        buffer_bytes = args.pipeline_buffer_mib * 1024 * 1024
        buffer = str(_exe("dtmb_core_pipe_buffer", args.bin_dir))
        buffered: list[list[str]] = []
        for command in commands:
            buffered.append(command)
            if Path(command[0]).stem in {
                "dtmb_core_ci8_resample",
                "dtmb_core_c3780_extract",
                "dtmb_core_deinterleave_qam64",
            }:
                buffered.append([
                    buffer, "--buffer-bytes", str(buffer_bytes),
                    "--chunk-bytes", str(min(1024 * 1024, buffer_bytes)), "-", "-",
                ])
        commands = buffered
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
                env=native_cuda_environment() if "--ldpc-accel" in command
                and command[command.index("--ldpc-accel") + 1] == "cuda" else None,
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
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="dtmb-decode",
        description="Decode a CI8 file or stdin stream into MPEG-TS.",
    )
    parser.add_argument("--input", default="-", help="CI8 path or - for stdin")
    parser.add_argument("--input-format", choices=("ci8", "cs8", "cu8", "ci16", "sc16", "cs16", "cf32"), default="ci8")
    parser.add_argument("--input-rate", type=int, required=True)
    parser.add_argument("--output", default="-", help="MPEG-TS path or - for stdout")
    parser.add_argument("--pn-mode", choices=("pn420", "pn595", "pn945"), default="pn945")
    parser.add_argument("--frame-body-mode", choices=("auto", "c1", "c3780"), default="c3780",
                        help="Frame-body processing; Auto additionally checks PN595/C=1 normal 4QAM.")
    parser.add_argument("--system-info-index", type=int, choices=tuple(PROFILES), default=22,
                        help="3/4: 4QAM-NR; 5..10: 4QAM; 11..16: 16QAM; 17..18: 32QAM; 19..24: 64QAM (default 22)")
    parser.add_argument("--nr-frame-phase", choices=("auto", "0", "1"), default="auto",
                        help="4QAM-NR packing: five-frame LDPC/BCH detection, or skip zero/one signal frame.")
    parser.add_argument("--qam32-frame-phase", choices=("auto", "0", "1"), default="auto",
                        help="32QAM packing alignment: LDPC/BCH detection, or skip zero/one signal frame.")
    parser.add_argument("--acceleration", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--decode-batch-frames", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--early-syndrome-reject-ratio", type=float)
    parser.add_argument("--error-policy", choices=("fail", "continue"), default="fail")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--pn-schedule-tracking", action="store_true")
    parser.add_argument(
        "--source-frame-confidence", choices=("off", "inverse-mse"), default="off"
    )
    parser.add_argument("--resample-headroom-db", type=float, default=0.0)
    parser.add_argument(
        "--pipeline-buffer-mib", type=int, default=64,
        help="Bounded queue size after each bulk receiver stage (0 disables queues).",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        help="Use native executables from this build directory instead of packaged binaries.",
    )
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data")
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
