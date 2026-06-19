"""Clipping safety gate for HackRF captures.

Reads the acquire-stage JSON (which already contains `iq.clip_count_i`,
`iq.clip_count_q`, and `iq.component_rms_dbfs`) and exits non-zero if the
capture has pushed the HackRF front-end too close to saturation. This is a
make-friendly hard gate so experiments cannot silently run with gains high
enough to clip the ADC or risk damage on a strong UHF signal.

Default thresholds:

- clip_ratio_max   (fraction of samples clipping either axis): 0.005 (0.5%)
- rms_dbfs_max     (component RMS in dBFS; -3 is the ceiling):  -3.0

Pass --write-verdict to persist a small JSON alongside the capture so make
targets can see the verdict independently of stderr/stdout.

Exit codes:
    0  Capture is within safe limits.
    1  Clipping above threshold - abort the pipeline for this capture.
       Missing/incomplete acquire diagnostics also fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-clipping-gate")
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Path to the .ci8 capture whose acquire.json will be inspected.",
    )
    parser.add_argument(
        "--acquire-json",
        type=Path,
        default=None,
        help="Override the path to the acquire JSON (defaults to <capture>.acquire.json).",
    )
    parser.add_argument(
        "--clip-ratio-max",
        type=float,
        default=0.005,
        help="Maximum allowed clipping fraction per axis (default 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--rms-dbfs-max",
        type=float,
        default=-3.0,
        help="Maximum allowed component RMS in dBFS (default -3 dBFS).",
    )
    parser.add_argument(
        "--write-verdict",
        type=Path,
        default=None,
        help="Optional path to persist the verdict JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    acquire_path = args.acquire_json or args.capture.with_suffix(".acquire.json")
    if not acquire_path.exists():
        print(
            f"[clipping-gate] no acquire JSON at {acquire_path}; refusing capture",
            file=sys.stderr,
        )
        _write_verdict(
            args.write_verdict,
            {
                "capture": str(args.capture),
                "acquire_json": str(acquire_path),
                "ok": False,
                "violations": ["missing acquire JSON"],
            },
        )
        return 1

    data = json.loads(acquire_path.read_text(encoding="utf-8"))
    iq = data.get("iq") or {}
    analyzed = data.get("analyzed") or {}
    analyzed_samples = int(
        analyzed.get("sample_count")
        or data.get("file", {}).get("sample_count")
        or 0
    )
    if analyzed_samples <= 0:
        print(
            "[clipping-gate] acquire JSON has no sample count; refusing capture",
            file=sys.stderr,
        )
        _write_verdict(
            args.write_verdict,
            {
                "capture": str(args.capture),
                "acquire_json": str(acquire_path),
                "ok": False,
                "violations": ["acquire JSON has no sample count"],
            },
        )
        return 1

    clip_i = int(iq.get("clip_count_i") or 0)
    clip_q = int(iq.get("clip_count_q") or 0)
    ratio_i = clip_i / analyzed_samples
    ratio_q = clip_q / analyzed_samples
    rms_dbfs = float(iq.get("component_rms_dbfs") or 0.0)
    rms_linear = float(iq.get("component_rms") or 0.0)

    verdict = {
        "capture": str(args.capture),
        "analyzed_samples": analyzed_samples,
        "clip_count_i": clip_i,
        "clip_count_q": clip_q,
        "clip_ratio_i": ratio_i,
        "clip_ratio_q": ratio_q,
        "clip_ratio_max_allowed": args.clip_ratio_max,
        "component_rms": rms_linear,
        "component_rms_dbfs": rms_dbfs,
        "rms_dbfs_max_allowed": args.rms_dbfs_max,
    }
    violations = []
    if ratio_i > args.clip_ratio_max or ratio_q > args.clip_ratio_max:
        violations.append(
            f"clipping above {args.clip_ratio_max:.3%} threshold: "
            f"I={ratio_i:.3%}, Q={ratio_q:.3%}"
        )
    if rms_dbfs > args.rms_dbfs_max:
        violations.append(
            f"component RMS {rms_dbfs:.2f} dBFS exceeds "
            f"{args.rms_dbfs_max:.2f} dBFS"
        )
    verdict["violations"] = violations
    verdict["ok"] = not violations

    _write_verdict(args.write_verdict, verdict)

    if violations:
        print(
            "[clipping-gate] UNSAFE capture: " + "; ".join(violations),
            file=sys.stderr,
        )
        print(
            "[clipping-gate] lower gains (e.g. -l 16 -g 16 -a 0) and recapture.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[clipping-gate] OK: clip I={ratio_i:.4%}, Q={ratio_q:.4%}, "
        f"RMS={rms_dbfs:.2f} dBFS",
        file=sys.stderr,
    )
    return 0


def _write_verdict(path: Path | None, verdict: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
