"""End-to-end HackRF -> DTMB receiver -> TS -> ffmpeg video pipeline.

This drives the full chain a consumer needs when the source is a HackRF SDR:

    hackrf_transfer (CI8 IQ) -> dtmb.receiver (TS bytes) -> ffmpeg (playable media)

Modes:

- ``live`` runs ``hackrf_transfer -r file`` into a bounded CI8 capture on
  disk, then runs the DTMB receiver and ffmpeg against that file. That is
  the path a DTMB viewer would take when tuning a real HackRF.
- ``loopback-file`` uses a pre-recorded CI8 file instead of HackRF. Good
  for rerunning the pipeline on a saved capture.
- ``loopback-synthetic`` synthesises a DTMB CI8 from a chosen MPEG-TS
  input through the in-tree transmitter. Good for end-to-end verification
  without RF hardware. The synthesised CI8 can be produced at either the
  native DTMB symbol rate (7.56 Msps) or a HackRF-style rate (20 Msps).

Every run prints sha256 digests and byte sizes of inspectable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
DTMB_SYMBOL_RATE_SPS = 7_560_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hackrf-to-video",
        description=(
            "Record or simulate a DTMB IQ capture and decode it to playable "
            "media via ffmpeg."
        ),
    )
    parser.add_argument(
        "mode",
        choices=("live", "loopback-file", "loopback-synthetic"),
        help="See module docstring for mode semantics.",
    )
    parser.add_argument(
        "--output-media",
        type=Path,
        required=True,
        help="Decoded media file written by ffmpeg after the RX stage.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=20_000_000,
        help="CI8 sample rate. 20 Msps matches HackRF DTMB captures.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Capture duration (seconds) for live or synthetic modes.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "Directory to place intermediate CI8 / recovered-TS files. "
            "Defaults to a sibling of --output-media."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Override the ffmpeg executable path.",
    )

    live = parser.add_argument_group("live mode")
    live.add_argument("--frequency", type=int, help="Center frequency in Hz.")
    live.add_argument(
        "--hackrf-bin",
        default="hackrf_transfer",
        help="hackrf_transfer executable path.",
    )
    live.add_argument("--amp", type=int, default=0)
    live.add_argument("--lna-gain", type=int, default=24)
    live.add_argument("--vga-gain", type=int, default=18)
    live.add_argument(
        "--bandwidth",
        type=int,
        default=10_000_000,
        help="Baseband filter bandwidth in Hz.",
    )

    loopback_file = parser.add_argument_group("loopback-file mode")
    loopback_file.add_argument(
        "--ci8-input",
        type=Path,
        help="CI8 file to feed the receiver instead of hackrf_transfer.",
    )

    loopback_syn = parser.add_argument_group("loopback-synthetic mode")
    loopback_syn.add_argument(
        "--input-ts",
        type=Path,
        help="MPEG-TS file synthesised into a DTMB CI8 stream.",
    )
    loopback_syn.add_argument(
        "--system-info-index",
        type=int,
        default=23,
    )
    loopback_syn.add_argument(
        "--amplitude",
        type=float,
        default=0.75,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_media.parent.mkdir(parents=True, exist_ok=True)
    work_dir = _resolve_work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)

    env = _child_env()

    if args.mode == "live":
        ci8_path = _run_hackrf_capture(args, work_dir=work_dir)
    elif args.mode == "loopback-file":
        if not args.ci8_input or not args.ci8_input.exists():
            raise SystemExit("loopback-file mode requires --ci8-input")
        ci8_path = args.ci8_input
    elif args.mode == "loopback-synthetic":
        if not args.input_ts or not args.input_ts.exists():
            raise SystemExit("loopback-synthetic mode requires --input-ts")
        ci8_path = _materialise_synthetic_capture(
            args,
            work_dir=work_dir,
            env=env,
        )
    else:
        raise SystemExit(f"unsupported mode {args.mode}")

    recovered_ts_path = work_dir / (ci8_path.stem + ".recovered.ts")
    diagnostics_path = work_dir / (ci8_path.stem + ".receiver.json")
    rc_rx = _run_receiver(
        ci8_path=ci8_path,
        recovered_ts_path=recovered_ts_path,
        diagnostics_path=diagnostics_path,
        sample_rate=args.sample_rate,
        env=env,
        mode=args.mode,
    )
    if rc_rx != 0 or not recovered_ts_path.exists():
        print(
            f"[pipeline] receiver failed rc={rc_rx}",
            file=sys.stderr,
        )
        _print_hash("input ci8", ci8_path)
        return 1

    rc_ff = _run_ffmpeg(
        recovered_ts_path=recovered_ts_path,
        output_media=args.output_media,
        ffmpeg_bin=args.ffmpeg_bin,
    )

    _print_hash("input ci8", ci8_path)
    _print_hash("recovered TS", recovered_ts_path)
    _print_hash("receiver JSON", diagnostics_path)
    _print_hash("decoded media", args.output_media)
    return rc_ff


def _resolve_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir is not None:
        return args.work_dir
    return args.output_media.parent / (args.output_media.stem + "_pipeline")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join((str(PYTHON_DIR), existing))
    else:
        env["PYTHONPATH"] = str(PYTHON_DIR)
    return env


def _run_hackrf_capture(
    args: argparse.Namespace,
    *,
    work_dir: Path,
) -> Path:
    if args.frequency is None:
        raise SystemExit("live mode requires --frequency")
    sample_count = int(args.duration * args.sample_rate)
    ci8_path = work_dir / (
        f"hackrf_{args.frequency}_{args.sample_rate}_{sample_count}.ci8"
    )
    cmd = [
        args.hackrf_bin,
        "-r",
        str(ci8_path),
        "-f",
        str(args.frequency),
        "-s",
        str(args.sample_rate),
        "-b",
        str(args.bandwidth),
        "-a",
        str(args.amp),
        "-l",
        str(args.lna_gain),
        "-g",
        str(args.vga_gain),
        "-n",
        str(sample_count),
    ]
    print("[pipeline] capturing:", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"hackrf_transfer failed with rc={rc}")
    return ci8_path


def _materialise_synthetic_capture(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    env: dict[str, str],
) -> Path:
    baseband_path = work_dir / (args.input_ts.stem + "_dtmb_baseband.ci8")
    tx_cmd = [
        sys.executable,
        "-m",
        "dtmb.tx_c3780",
        str(args.input_ts),
        "--output",
        str(baseband_path),
        "--system-info-index",
        str(args.system_info_index),
        "--amplitude",
        str(args.amplitude),
        "--pad-null-packets",
    ]
    rc = subprocess.call(tx_cmd, env=env)
    if rc != 0:
        raise SystemExit(f"dtmb-transmit failed with rc={rc}")

    if args.sample_rate == DTMB_SYMBOL_RATE_SPS:
        return baseband_path

    resampled_path = work_dir / (
        f"{args.input_ts.stem}_dtmb_{args.sample_rate}.ci8"
    )
    _pulse_shape_ci8_to_rate(
        src_path=baseband_path,
        dst_path=resampled_path,
        src_rate=DTMB_SYMBOL_RATE_SPS,
        dst_rate=args.sample_rate,
    )
    return resampled_path


def _pulse_shape_ci8_to_rate(
    *,
    src_path: Path,
    dst_path: Path,
    src_rate: int,
    dst_rate: int,
) -> None:
    """Turn a one-sample-per-symbol baseband CI8 into an SRRC-shaped CI8 at dst_rate."""

    from dtmb.pulse_shape import pulse_shape_to_rate

    raw = np.fromfile(src_path, dtype=np.int8)
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw.reshape(-1, 2).astype(np.float32) / 128.0
    symbols = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    shaped = pulse_shape_to_rate(
        symbols,
        symbol_rate_sps=src_rate,
        sample_rate_sps=dst_rate,
        roll_off=0.05,
        symbol_span=16,
        peak_amplitude=0.75,
    )
    ci8 = np.empty(shaped.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(shaped.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(shaped.imag * 127), -128, 127).astype(np.int8)
    dst_path.write_bytes(ci8.tobytes())


def _run_receiver(
    *,
    ci8_path: Path,
    recovered_ts_path: Path,
    diagnostics_path: Path,
    sample_rate: int,
    env: dict[str, str],
    mode: str,
) -> int:
    ci8_size = ci8_path.stat().st_size
    sample_count = ci8_size // 2
    max_samples = max(sample_count, 1)
    symbol_frames = max(1, int(sample_count * DTMB_SYMBOL_RATE_SPS / sample_rate / 4725))

    cmd = [
        sys.executable,
        "-m",
        "dtmb.receiver",
        str(ci8_path),
        "--sample-rate",
        str(sample_rate),
        "--max-samples",
        str(max_samples),
        "--mode",
        "pn945",
        "--qam",
        "auto",
        "--equalizer",
        "none" if sample_rate == DTMB_SYMBOL_RATE_SPS else "dd",
        "--symbol-deinterleave",
        "mode1",
        "--frames",
        str(symbol_frames),
        "--fec-mode",
        "ldpc",
        "--fec-rate",
        "auto",
        "--min-ts-packets",
        "8",
        "--output",
        str(recovered_ts_path),
        "--json",
        str(diagnostics_path),
        "--quiet",
    ]
    if sample_rate == DTMB_SYMBOL_RATE_SPS and mode == "loopback-synthetic":
        cmd.extend(["--phase-offset", "0", "--no-timing-search"])
    print("[pipeline] decoding:", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, env=env)


def _run_ffmpeg(
    *,
    recovered_ts_path: Path,
    output_media: Path,
    ffmpeg_bin: str,
) -> int:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(recovered_ts_path),
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(output_media),
    ]
    print("[pipeline] ffmpeg:", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


def _print_hash(label: str, path: Path) -> None:
    if not path.exists():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{label}: sha256={digest} size={path.stat().st_size}")


if __name__ == "__main__":
    raise SystemExit(main())
