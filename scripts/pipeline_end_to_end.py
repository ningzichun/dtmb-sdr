"""End-to-end DTMB pipeline demo: TS -> CI8 -> TS -> ffmpeg decode.

Drives the full pipeline a real-time DTMB receiver needs:

    ts_bytes --dtmb-transmit--> ci8_bytes --dtmb-receive--> ts_bytes --ffmpeg--> decoded

Two modes:

- **pipe** (default): runs TX | RX | ffmpeg as chained subprocesses with
  OS pipes. Closest to a live SDR + receiver + media player chain and
  keeps no intermediate files.
- **tap** (``--ci8-tap`` or ``--recovered-ts``): materialises the
  intermediate streams on disk so each stage can be audited by hash and
  replayed through ``ffprobe`` / ``ffmpeg``.

Both modes always print ``sha256`` and byte-size summaries of every
inspectable artifact. The exit status is non-zero if any stage fails.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-pipeline-demo",
        description=(
            "Pipe a real MPEG-TS file through DTMB TX -> DTMB RX -> ffmpeg."
        ),
    )
    parser.add_argument(
        "--input-ts",
        type=Path,
        required=True,
        help="Input MPEG-TS file to transmit through the DTMB pipeline.",
    )
    parser.add_argument(
        "--output-media",
        type=Path,
        required=True,
        help="Decoded media file written by ffmpeg after the RX stage.",
    )
    parser.add_argument(
        "--system-info-index",
        type=int,
        default=23,
        help="DTMB system-information index (23 = 64QAM rate-3 mode-1).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.75,
        help="TX peak amplitude before CI8 quantisation.",
    )
    parser.add_argument(
        "--ci8-tap",
        type=Path,
        help=(
            "Write intermediate CI8 bytes to this file. Enables the "
            "file-intermediate run mode where TX finishes before RX starts."
        ),
    )
    parser.add_argument(
        "--recovered-ts",
        type=Path,
        help=(
            "Write recovered TS bytes to this file. Enables the "
            "file-intermediate run mode where RX finishes before ffmpeg runs."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Override the ffmpeg executable path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=7_560_000,
        help="Sample rate reported to the DTMB receiver (matches TX baseband).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_ts.exists():
        raise SystemExit(f"input TS not found: {args.input_ts}")
    if args.output_media.parent:
        args.output_media.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join((str(PYTHON_DIR), existing))
    else:
        env["PYTHONPATH"] = str(PYTHON_DIR)

    tx_cmd = _build_tx_command(args)
    total_frames, max_samples = _plan_rx_budget(args.input_ts)
    rx_cmd = _build_rx_command(args, total_frames=total_frames, max_samples=max_samples)
    ffmpeg_cmd = _build_ffmpeg_command(args)

    use_taps = args.ci8_tap is not None or args.recovered_ts is not None
    if use_taps:
        return_code = _run_with_file_intermediates(
            args,
            env=env,
            tx_cmd=tx_cmd,
            rx_cmd=rx_cmd,
            ffmpeg_cmd=ffmpeg_cmd,
        )
    else:
        return_code = _run_with_pipes(
            env=env,
            tx_cmd=tx_cmd,
            rx_cmd=rx_cmd,
            ffmpeg_cmd=ffmpeg_cmd,
        )

    _print_hash("input TS", args.input_ts)
    if args.ci8_tap is not None:
        _print_hash("ci8 tap", args.ci8_tap)
    if args.recovered_ts is not None:
        _print_hash("recovered TS", args.recovered_ts)
    _print_hash("decoded media", args.output_media)
    return return_code


def _plan_rx_budget(input_ts: Path) -> tuple[int, int]:
    ts_bytes = input_ts.stat().st_size
    payload_frames = max(1, (ts_bytes + 2255) // 2256)
    total_frames = payload_frames + 170
    max_samples = total_frames * 4725
    return total_frames, max_samples


def _build_tx_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dtmb.tx_c3780",
        str(args.input_ts),
        "--output",
        "-",
        "--system-info-index",
        str(args.system_info_index),
        "--amplitude",
        str(args.amplitude),
        "--pad-null-packets",
    ]


def _build_rx_command(
    args: argparse.Namespace,
    *,
    total_frames: int,
    max_samples: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dtmb.receiver",
        "-",
        "--sample-rate",
        str(args.sample_rate),
        "--symbol-rate",
        str(args.sample_rate),
        "--max-samples",
        str(max_samples),
        "--mode",
        "pn945",
        "--phase-offset",
        "0",
        "--frames",
        str(total_frames),
        "--qam",
        "auto",
        "--equalizer",
        "none",
        "--symbol-deinterleave",
        "mode1",
        "--no-timing-search",
        "--fec-mode",
        "ldpc",
        "--fec-rate",
        "auto",
        "--min-ts-packets",
        "8",
        "--output",
        "-",
        "--quiet",
    ]


def _build_ffmpeg_command(args: argparse.Namespace) -> list[str]:
    return [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        "pipe:0",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(args.output_media),
    ]


def _run_with_pipes(
    *,
    env: dict[str, str],
    tx_cmd: list[str],
    rx_cmd: list[str],
    ffmpeg_cmd: list[str],
) -> int:
    """Stream TX | RX | ffmpeg with direct OS pipes."""

    tx_proc = subprocess.Popen(
        tx_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        env=env,
    )
    rx_proc = subprocess.Popen(
        rx_cmd,
        stdin=tx_proc.stdout,
        stdout=subprocess.PIPE,
        env=env,
    )
    # Let RX own the read end so TX can see SIGPIPE/EOF correctly.
    if tx_proc.stdout is not None:
        tx_proc.stdout.close()
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=rx_proc.stdout,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    if rx_proc.stdout is not None:
        rx_proc.stdout.close()

    tx_rc = tx_proc.wait()
    rx_rc = rx_proc.wait()
    ff_rc = ffmpeg_proc.wait()
    print(
        f"[pipeline] tx={tx_rc} rx={rx_rc} ffmpeg={ff_rc}",
        file=sys.stderr,
    )
    return tx_rc or rx_rc or ff_rc


def _run_with_file_intermediates(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    tx_cmd: list[str],
    rx_cmd: list[str],
    ffmpeg_cmd: list[str],
) -> int:
    """Run each stage sequentially, materialising the intermediate streams."""

    ci8_path = args.ci8_tap or (args.output_media.with_suffix(".pipe.ci8"))
    recovered_path = (
        args.recovered_ts
        or args.output_media.with_suffix(".pipe.recovered.ts")
    )
    ci8_path.parent.mkdir(parents=True, exist_ok=True)
    recovered_path.parent.mkdir(parents=True, exist_ok=True)

    with ci8_path.open("wb") as ci8_handle:
        tx_rc = subprocess.call(
            tx_cmd,
            stdin=subprocess.DEVNULL,
            stdout=ci8_handle,
            env=env,
        )
    print(f"[pipeline] tx={tx_rc}", file=sys.stderr)
    if tx_rc:
        return tx_rc

    with ci8_path.open("rb") as ci8_in, recovered_path.open("wb") as ts_out:
        rx_rc = subprocess.call(
            rx_cmd,
            stdin=ci8_in,
            stdout=ts_out,
            env=env,
        )
    print(f"[pipeline] rx={rx_rc}", file=sys.stderr)
    if rx_rc:
        return rx_rc

    with recovered_path.open("rb") as ts_in:
        ff_rc = subprocess.call(
            ffmpeg_cmd,
            stdin=ts_in,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
    print(f"[pipeline] ffmpeg={ff_rc}", file=sys.stderr)
    return ff_rc


def _print_hash(label: str, path: Path) -> None:
    if not path.exists():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{label}: sha256={digest} size={path.stat().st_size}")


if __name__ == "__main__":
    raise SystemExit(main())
