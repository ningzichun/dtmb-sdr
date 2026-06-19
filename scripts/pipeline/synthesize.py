"""Stage S: synthesize a DTMB CI8 from an MPEG-TS payload.

Inputs:
    --input-ts  path to real MPEG-TS bytes
    --output    path to .ci8 (with sidecar JSON written alongside)

Outputs:
    <output>                baseband CI8 at 7.56 Msps
    <output>.json           sidecar so downstream stages treat it like a capture
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-synthesize")
    parser.add_argument("--input-ts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-info-index", type=int, default=23)
    parser.add_argument("--amplitude", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "dtmb.tx_c3780",
        str(args.input_ts),
        "--output",
        str(args.output),
        "--system-info-index",
        str(args.system_info_index),
        "--amplitude",
        str(args.amplitude),
        "--pad-null-packets",
    ]
    print("[synthesize]", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())
    if rc != 0:
        return rc

    byte_count = args.output.stat().st_size
    sample_count = byte_count // 2
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    write_json(
        sidecar_path,
        {
            "format": "ci8",
            "sample_rate_sps": 7_560_000,
            "bandwidth_hz": 7_560_000,
            "frequency_hz": 0,
            "byte_count": byte_count,
            "sample_count": sample_count,
            "duration_s": sample_count / 7_560_000,
            "amp": 0,
            "lna_gain": 0,
            "vga_gain": 0,
            "antenna": "synthetic-loopback",
            "location": "synthesized",
            "capture_command": cmd,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
