"""PN-acquisition scan: capture short CI8 files at candidate centers.

This is not a native spectrum sweep. Use ``scripts/pipeline/sweep_hackrf.py``
or ``make -f pipeline.mk sweep`` when the job is to run ``hackrf_sweep -w`` and
rank the resulting CSV bins. This PN scan is the next step after spectrum
discovery: it checks whether specific center frequencies have DTMB PN evidence
before committing to a full receiver capture.

Usage:
    python scripts/pipeline/scan_dtmb.py --duration 2 \
        --lna-gain 16 --vga-gain 16 --amp 0 \
        --output-dir captures/scan_YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env


DEFAULT_CENTERS = "602000000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-scan-dtmb")
    parser.add_argument(
        "--centers",
        default=DEFAULT_CENTERS,
        help="Comma-separated exact channel-center frequencies in Hz.",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=20_000_000)
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=8_000_000,
        help="Centered DTMB filter width; 8 MHz limits adjacent-channel ingress.",
    )
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--lna-gain", type=int, default=16)
    parser.add_argument("--vga-gain", type=int, default=16)
    parser.add_argument("--device-serial")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--antenna", default="UHF antenna")
    parser.add_argument("--location", default="")
    parser.add_argument(
        "--clock-source",
        choices=("internal", "external-10mhz", "gpsdo-10mhz", "unknown"),
        default="internal",
    )
    parser.add_argument("--external-reference-hz", type=int, default=None)
    parser.add_argument("--hardware-trigger", action="store_true")
    parser.add_argument("--trigger-source", default=None)
    parser.add_argument("--trigger-time-utc", default=None)
    parser.add_argument(
        "--keep-ci8",
        action="store_true",
        help="Retain CI8 files after scan (default deletes them).",
    )
    return parser


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(ROOT), env=child_env())


def _best_metrics(acquire_json: Path) -> dict:
    data = json.loads(acquire_json.read_text(encoding="utf-8"))
    pn_search = data.get("pn_search") or {}

    def best_in(group: str, mode: str) -> float:
        items = pn_search.get(group) or []
        best = 0.0
        for item in items:
            if item.get("mode") == mode:
                m = float(item.get("max_metric") or 0.0)
                if m > best:
                    best = m
        return best

    return {
        "pn945_cyclic_ext": best_in("cyclic_extension_trains", "pn945"),
        "pn945_frame": best_in("frame_trains", "pn945"),
        "pn945_family": best_in("phase_family_trains", "pn945"),
        "pn420_cyclic_ext": best_in("cyclic_extension_trains", "pn420"),
        "pn595_cyclic_ext": best_in("cyclic_extension_trains", "pn595"),
    }


def _capture_health(acquire_json: Path) -> dict:
    data = json.loads(acquire_json.read_text(encoding="utf-8"))
    iq = data.get("iq") or {}
    analyzed = data.get("analyzed") or {}
    return {
        "analyzed_samples": int(analyzed.get("sample_count") or 0),
        "clip_count_i": int(iq.get("clip_count_i") or 0),
        "clip_count_q": int(iq.get("clip_count_q") or 0),
        "component_rms_dbfs": iq.get("component_rms_dbfs"),
        "complex_power_db": iq.get("complex_power_db"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centers = [int(x.strip()) for x in args.centers.split(",") if x.strip()]

    results = []
    for freq in centers:
        name = f"scan_{freq}_amp{args.amp}_lna{args.lna_gain}_vga{args.vga_gain}"
        ci8 = args.output_dir / f"{name}.ci8"
        acquire_path = args.output_dir / f"{name}.acquire.json"
        clipping_path = args.output_dir / f"{name}.clipping.json"
        capture_cmd = [
            sys.executable,
            "scripts/pipeline/capture_hackrf.py",
            "--output",
            str(ci8),
            "--frequency",
            str(freq),
            "--sample-rate",
            str(args.sample_rate),
            "--duration",
            str(args.duration),
            "--amp",
            str(args.amp),
            "--lna-gain",
            str(args.lna_gain),
            "--vga-gain",
            str(args.vga_gain),
            "--clock-source",
            args.clock_source,
            "--antenna",
            args.antenna,
            "--location",
            args.location,
        ]
        capture_cmd.extend(["--bandwidth", str(args.bandwidth)])
        if args.external_reference_hz is not None:
            capture_cmd.extend(["--external-reference-hz", str(args.external_reference_hz)])
        if args.device_serial:
            capture_cmd.extend(["--device-serial", args.device_serial])
        if args.hardware_trigger:
            capture_cmd.append("--hardware-trigger")
        if args.trigger_source:
            capture_cmd.extend(["--trigger-source", args.trigger_source])
        if args.trigger_time_utc:
            capture_cmd.extend(["--trigger-time-utc", args.trigger_time_utc])
        rc = _run(capture_cmd)
        if rc != 0:
            results.append({"frequency_hz": freq, "error": f"capture rc={rc}"})
            continue
        acq_cmd = [
            sys.executable,
            "scripts/pipeline/acquire.py",
            "--capture",
            str(ci8),
            "--output",
            str(acquire_path),
        ]
        rc = _run(acq_cmd)
        if rc != 0:
            results.append(
                {"frequency_hz": freq, "error": f"acquire rc={rc}"}
            )
            continue
        health = _capture_health(acquire_path)
        clipping_cmd = [
            sys.executable,
            "scripts/pipeline/clipping_gate.py",
            "--capture",
            str(ci8),
            "--acquire-json",
            str(acquire_path),
            "--write-verdict",
            str(clipping_path),
        ]
        rc = _run(clipping_cmd)
        if rc != 0:
            results.append(
                {
                    "frequency_hz": freq,
                    "error": f"clipping gate rc={rc}",
                    "capture_health": health,
                    "acquire_json": str(acquire_path),
                    "clipping_json": str(clipping_path),
                }
            )
            continue
        metrics = _best_metrics(acquire_path)
        metrics["frequency_hz"] = freq
        metrics["ci8"] = str(ci8)
        metrics["acquire_json"] = str(acquire_path)
        metrics["clipping_json"] = str(clipping_path)
        metrics["capture_health"] = health
        results.append(metrics)
        if not args.keep_ci8:
            try:
                ci8.unlink(missing_ok=True)
                (ci8.with_suffix(ci8.suffix + ".json")).unlink(missing_ok=True)
            except OSError:
                pass

    results.sort(
        key=lambda r: float(r.get("pn945_cyclic_ext", 0.0)),
        reverse=True,
    )
    summary_path = args.output_dir / "scan_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 1 if any("error" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
