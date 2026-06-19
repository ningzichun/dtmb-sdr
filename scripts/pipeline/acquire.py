"""Stage 1: PN945 Gate A diagnostic.

Inputs:
    <capture>.ci8
    <capture>.ci8.json  (sidecar metadata)

Outputs:
    <capture>.acquire.json

This stage runs `dtmb.detect` with `--pn-search` so it is seconds-fast even
on 240 MB captures. The generated JSON reports PN-family metric, coarse
CFO, and cyclic-extension phase, which are the Gate A acceptance signals.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env, load_capture_sidecar
from dtmb.gate_visuals import write_gate_a_visual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-acquire")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4_000_000,
        help="Complex samples consumed by the acquisition scan.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help="Complex frequency shift in Hz applied before PN search.",
    )
    parser.add_argument("--pn-max-symbols", type=int, default=250_000)
    parser.add_argument(
        "--visual",
        choices=("off", "auto", "required"),
        default="auto",
        help=(
            "Gate A visual rendering policy. 'auto' writes PNG/manifest when "
            "possible and records plotting failures without failing acquire."
        ),
    )
    parser.add_argument(
        "--visual-png",
        type=Path,
        help="Optional Gate A visual PNG output path.",
    )
    parser.add_argument(
        "--visual-json",
        type=Path,
        help="Optional Gate A visual manifest output path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = load_capture_sidecar(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "dtmb.detect",
        str(args.capture),
        "--sample-rate",
        str(int(sidecar["sample_rate_sps"])),
        "--max-samples",
        str(args.max_samples),
        "--frequency-shift",
        str(args.frequency_shift),
        "--pn-search",
        "--pn-max-symbols",
        str(args.pn_max_symbols),
        "--json",
        str(args.output),
    ]
    print("[acquire]", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())
    if rc == 0 and args.visual != "off":
        try:
            manifest = write_gate_a_visual(
                args.capture,
                acquire_json_path=args.output,
                png_path=args.visual_png,
                manifest_path=args.visual_json,
                sample_rate_sps=int(sidecar["sample_rate_sps"]),
                max_samples=args.max_samples,
                required=args.visual == "required",
            )
            print(
                "[acquire-visual] "
                f"{manifest.get('status')}: "
                f"{(manifest.get('artifacts') or {}).get('manifest')}",
                file=sys.stderr,
            )
            if args.visual == "required" and manifest.get("status") != "rendered":
                return 1
        except Exception as exc:
            print(
                f"[acquire-visual] error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.visual == "required":
                return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
