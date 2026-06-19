"""Run a native hackrf_sweep and rank DTMB-sized windows.

This stage keeps the spectrum measurement in hackrf_sweep's native format:
the CSV rows are written exactly by hackrf_sweep, including the reported
``hz_bin_width`` field controlled by ``-w``. The Python side only ranks that
CSV afterward; it does not reinterpret ``-w`` as capture bandwidth.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env
from dtmb.sweep import load_sweep_bins, rank_sweep_windows


def _resolve_default_hackrf_sweep_bin() -> str:
    on_path = shutil.which("hackrf_sweep")
    if on_path:
        return on_path
    candidates = [
        Path.home() / "miniforge3" / "envs" / "gr-dtmb" / "Library" / "bin" / "hackrf_sweep.exe",
        Path.home() / "miniforge3" / "Library" / "bin" / "hackrf_sweep.exe",
        Path(r"C:\Program Files\PothosSDR\bin\hackrf_sweep.exe"),
        Path(r"C:\Program Files (x86)\PothosSDR\bin\hackrf_sweep.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "hackrf_sweep"


DEFAULT_HACKRF_SWEEP_BIN = _resolve_default_hackrf_sweep_bin()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-sweep-hackrf")
    parser.add_argument(
        "--range-mhz",
        default="470:862",
        help="hackrf_sweep -f range in MHz, e.g. 470:862.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=1_000_000,
        help="hackrf_sweep -w FFT bin width in Hz.",
    )
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--lna-gain", type=int, default=16)
    parser.add_argument("--vga-gain", type=int, default=16)
    parser.add_argument("--antenna-power", type=int, choices=(0, 1), default=None)
    parser.add_argument("--device-serial", default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--window-bandwidth", type=int, default=8_000_000)
    parser.add_argument("--window-step", type=int, default=1_000_000)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--hackrf-sweep-bin", default=DEFAULT_HACKRF_SWEEP_BIN)
    return parser


def build_hackrf_sweep_command(args: argparse.Namespace) -> list[str]:
    if args.bin_width <= 0:
        raise ValueError("bin width must be positive")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    cmd = [
        args.hackrf_sweep_bin,
        "-f",
        args.range_mhz,
        "-w",
        str(args.bin_width),
        "-N",
        str(args.sweeps),
        "-a",
        str(args.amp),
        "-l",
        str(args.lna_gain),
        "-g",
        str(args.vga_gain),
        "-r",
        str(args.output_csv),
    ]
    if args.antenna_power is not None:
        cmd.extend(["-p", str(args.antenna_power)])
    if args.device_serial:
        cmd.extend(["-d", args.device_serial])
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_hackrf_sweep_command(args)
    print("[sweep]", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())
    if rc != 0:
        report = {
            "stage": "hackrf_sweep",
            "ok": False,
            "rc": rc,
            "command": cmd,
            "output_csv": str(args.output_csv),
        }
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return rc

    bins = load_sweep_bins(args.output_csv)
    windows = rank_sweep_windows(
        bins,
        bandwidth_hz=args.window_bandwidth,
        step_hz=args.window_step,
        top=args.top,
    )
    report = {
        "stage": "hackrf_sweep",
        "ok": True,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "command": cmd,
        "output_csv": str(args.output_csv),
        "range_mhz": args.range_mhz,
        "requested_bin_width_hz": int(args.bin_width),
        "native_bin_width_hz": sorted({float(item.bin_width_hz) for item in bins}),
        "native_sample_counts": sorted({int(item.sample_count) for item in bins}),
        "bin_count": len(bins),
        "window_bandwidth_hz": int(args.window_bandwidth),
        "window_step_hz": int(args.window_step),
        "windows": [window.to_dict() for window in windows],
    }
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
