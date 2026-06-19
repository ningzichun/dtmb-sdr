"""One-shot real-time DTMB receive stage: capture -> decode -> verify video.

This is the canonical user-facing entry point for the question:

    "HackRF is connected. Give me an ffmpeg-verified video stream."

It combines every lower-level stage so a single invocation:

1. Records a bounded CI8 capture from the HackRF.
2. Runs Gate A (PN945 acquisition) and Gate C (system-information classifier).
3. Decides whether the capture is worth a full DD + LDPC decode.
4. Either emits a recovered MPEG-TS + MP4, or explains (with numbers)
   at which stage the real-RF path stopped.

The script is intentionally opinionated about defaults so it works out of
the box against real SDR captures on UHF DTMB channels. Override any of
`--frequency`, `--amp`, `--lna-gain`, `--vga-gain`, `--duration` on the
command line to match a different region or antenna setup.

Exit codes:

    0  Video MP4 produced and verified by ffprobe.
    3  Capture + acquisition OK, but Gate C failed (no TS produced).
       This is the documented real-capture blocker (see AGENTS.md).
    4  Capture succeeded but Gate A (PN acquisition) failed.
    5  HackRF capture itself failed.
    6  Clipping safety gate failed (ADC saturating; lower gains).
    2  ffprobe rejected the recovered TS (shouldn't happen when decode
       succeeded; reported for completeness).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from _common import ROOT, child_env, read_json, write_json
from gate_c import (
    gate_c_quality,
    resolve_receiver_timing_policy,
    sysinfo_oracle_quality,
)


GATE_A_PASS_THRESHOLD = 0.45
GATE_C_PASS_THRESHOLD = 0.75


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-realtime",
        description=(
            "HackRF -> DTMB receiver -> MPEG-TS -> ffmpeg, in a single call. "
            "Reports per-stage metrics and stops early when a gate fails."
        ),
    )
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=20_000_000,
        choices=(7_560_000, 20_000_000),
    )
    parser.add_argument("--bandwidth", type=int, default=None)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--lna-gain", type=int, default=16)
    parser.add_argument("--vga-gain", type=int, default=16)
    parser.add_argument("--antenna", default="UHF antenna")
    parser.add_argument("--location", default="")
    parser.add_argument(
        "--device-serial",
        help="Select the same HackRF serial used by clock preflight and capture.",
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
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help=(
            "Prefix used for analysis artifacts; e.g. captures/analysis/realtime_20260510. "
            "Produces <prefix>.acquire.json, <prefix>.sysinfo.json, "
            "<prefix>.recovered.ts, <prefix>.probe.json, <prefix>.mp4."
        ),
    )
    parser.add_argument(
        "--raw-prefix",
        type=Path,
        help=(
            "Prefix used for the raw CI8 and sidecar. Defaults to --output-prefix "
            "for backward compatibility."
        ),
    )
    parser.add_argument("--receive-frames", type=int, default=600)
    parser.add_argument("--sysinfo-frames", type=int, default=48)
    parser.add_argument("--sysinfo-oracle-frames", type=int, default=620)
    parser.add_argument(
        "--sysinfo-oracle-equalizer",
        choices=(
            "none",
            "sparse",
            "pn",
            "pn-sparse",
            "dd",
            "pn-dd",
            "dd-data",
            "pn-dd-data",
            "dd-raw",
            "pn-dd-raw",
        ),
        default="none",
        help=(
            "Equalizer for the long oracle source. raw SI is captured before "
            "equalization, so none is the fast default."
        ),
    )
    parser.add_argument("--sysinfo-index", type=int, default=23)
    parser.add_argument(
        "--force-receive",
        action="store_true",
        help="Run full LDPC decode even when Gate C fails (expensive; diagnostic only).",
    )
    parser.add_argument(
        "--timing-policy",
        choices=("fixed", "windowed", "trajectory"),
        default="fixed",
        help=(
            "Frame slicing policy passed through to Gate C and receive. "
            "Supplying --timing-trajectory promotes the default fixed policy "
            "to trajectory."
        ),
    )
    parser.add_argument(
        "--timing-trajectory",
        type=Path,
        help="Optional timing trajectory JSON for Gate C, oracle, and receive.",
    )
    parser.add_argument(
        "--keep-ci8",
        action="store_true",
        help="Retain the raw CI8 after the pipeline runs.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def _run(cmd: list[str], label: str) -> int:
    print(f"[{label}]", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(ROOT), env=child_env())


def _best_pn945_metric(acquire_json: dict[str, Any]) -> float:
    pn_search = acquire_json.get("pn_search") or {}
    best = 0.0
    for group in ("cyclic_extension_trains", "phase_family_trains", "frame_trains"):
        for item in pn_search.get(group) or []:
            if item.get("mode") == "pn945":
                m = float(item.get("max_metric") or 0.0)
                if m > best:
                    best = m
    return best


def _gate_c_summary(sysinfo_json: dict[str, Any]) -> dict[str, Any]:
    quality = gate_c_quality(sysinfo_json)
    si = sysinfo_json.get("system_info") or {}
    selected = si.get("selected")
    return {
        "selected_index": (selected or {}).get("index"),
        "agreement_ratio": quality["agreement_ratio"],
        "auto_eligible_frame_count": quality["auto_eligible_frame_count"],
        "expected_best_metric": quality["best_metric"],
        "expected_mean_metric": quality["mean_metric"],
        "expected_qualified_frame_count": quality["qualified_frame_count"],
        "timing_continuity_stable": quality["timing_continuity_stable"],
        "timing_continuity_verdict": quality["timing_continuity_verdict"],
        "timing_phase_span_symbols": quality["timing_phase_span_symbols"],
        "timing_max_adjacent_phase_step_symbols": quality[
            "timing_max_adjacent_phase_step_symbols"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prefix = args.output_prefix
    raw_prefix = args.raw_prefix or args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_prefix.parent.mkdir(parents=True, exist_ok=True)
    ci8 = raw_prefix.with_suffix(".ci8")
    acquire_path = prefix.with_suffix(".acquire.json")
    sysinfo_path = prefix.with_suffix(".sysinfo.json")
    sysinfo_oracle_source_path = prefix.with_suffix(".sysinfo_oracle.source.json")
    sysinfo_oracle_path = prefix.with_suffix(".sysinfo_oracle.raw.json")
    recovered_ts = prefix.with_suffix(".recovered.ts")
    receiver_json = prefix.with_suffix(".receiver.json")
    probe_path = prefix.with_suffix(".probe.json")
    mp4_path = prefix.with_suffix(".mp4")
    report_path = prefix.with_suffix(".realtime.json")
    effective_timing_policy = resolve_receiver_timing_policy(
        args.timing_policy,
        args.timing_trajectory,
    )
    timing_args = ["--timing-policy", effective_timing_policy]
    if args.timing_trajectory is not None:
        timing_args.extend(["--timing-trajectory", str(args.timing_trajectory)])

    report: dict[str, Any] = {
        "frequency_hz": args.frequency,
        "sample_rate_sps": args.sample_rate,
        "duration_s": args.duration,
        "amp": args.amp,
        "lna_gain": args.lna_gain,
        "vga_gain": args.vga_gain,
        "device_serial": args.device_serial,
        "clock_source": args.clock_source,
        "external_reference_hz": args.external_reference_hz,
        "hardware_trigger": args.hardware_trigger,
        "trigger_source": args.trigger_source,
        "trigger_time_utc": args.trigger_time_utc,
        "timing": {
            "requested_policy": args.timing_policy,
            "receiver_policy": effective_timing_policy,
            "trajectory_path": str(args.timing_trajectory)
            if args.timing_trajectory is not None
            else None,
        },
        "stages": {},
    }

    # -------------------------------------------------------------------
    # Stage 0: capture
    # -------------------------------------------------------------------
    capture_cmd = [
        sys.executable,
        "scripts/pipeline/capture_hackrf.py",
        "--output",
        str(ci8),
        "--frequency",
        str(args.frequency),
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
    if args.bandwidth:
        capture_cmd.extend(["--bandwidth", str(args.bandwidth)])
    if args.device_serial:
        capture_cmd.extend(["--device-serial", args.device_serial])
    if args.external_reference_hz is not None:
        capture_cmd.extend(["--external-reference-hz", str(args.external_reference_hz)])
    if args.hardware_trigger:
        capture_cmd.append("--hardware-trigger")
    if args.trigger_source:
        capture_cmd.extend(["--trigger-source", args.trigger_source])
    if args.trigger_time_utc:
        capture_cmd.extend(["--trigger-time-utc", args.trigger_time_utc])
    rc = _run(capture_cmd, "capture")
    report["stages"]["capture"] = {
        "rc": rc,
        "ci8": str(ci8),
        "bytes": ci8.stat().st_size if ci8.exists() else 0,
    }
    if rc != 0 or not ci8.exists():
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 5

    # -------------------------------------------------------------------
    # Stage 1: Gate A (PN acquisition)
    # -------------------------------------------------------------------
    rc = _run(
        [
            sys.executable,
            "scripts/pipeline/acquire.py",
            "--capture",
            str(ci8),
            "--output",
            str(acquire_path),
        ],
        "acquire",
    )
    best_pn945 = 0.0
    if acquire_path.exists():
        best_pn945 = _best_pn945_metric(read_json(acquire_path))
    report["stages"]["acquire"] = {
        "rc": rc,
        "pn945_best_metric": best_pn945,
        "gate_a_pass": best_pn945 >= GATE_A_PASS_THRESHOLD,
    }
    # Clipping safety gate: abort before committing CPU to sysinfo/receive
    # if the capture saturated the ADC. Keeps strong-signal experiments from
    # silently running with unsafe front-end gains.
    clipping_path = prefix.with_suffix(".clipping.json")
    rc_clip = _run(
        [
            sys.executable,
            "scripts/pipeline/clipping_gate.py",
            "--capture",
            str(ci8),
            "--acquire-json",
            str(acquire_path),
            "--write-verdict",
            str(clipping_path),
        ],
        "clipping-gate",
    )
    report["stages"]["clipping_gate"] = {
        "rc": rc_clip,
        "verdict": read_json(clipping_path) if clipping_path.exists() else None,
    }
    if rc_clip != 0:
        report["verdict"] = "unsafe_clipping"
        report["hint"] = (
            "HackRF ADC clipping above threshold. Lower gains (-l 16 -g 16 -a 0) "
            "and recapture. See <prefix>.clipping.json for numeric detail."
        )
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 6
    if best_pn945 < GATE_A_PASS_THRESHOLD:
        report["verdict"] = "gate_a_failed_no_dtmb_signal"
        report["hint"] = (
            "PN945 metric below 0.45: DTMB likely not on this frequency. "
            "Run scripts/pipeline/scan_dtmb.py to find an active channel."
        )
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4

    # -------------------------------------------------------------------
    # Stage 2: Gate C (system information)
    # -------------------------------------------------------------------
    rc = _run(
        [
            sys.executable,
            "scripts/pipeline/sysinfo.py",
            "--capture",
            str(ci8),
            "--output",
            str(sysinfo_path),
            "--frames",
            str(args.sysinfo_frames),
            "--system-info-index",
            str(args.sysinfo_index),
            *timing_args,
        ],
        "sysinfo",
    )
    gate_c = {}
    if sysinfo_path.exists():
        gate_c = _gate_c_summary(read_json(sysinfo_path))
    gate_c_pass = (
        gate_c.get("auto_eligible_frame_count", 0) > 0
        and gate_c.get("expected_best_metric", 0.0) >= GATE_C_PASS_THRESHOLD
        and gate_c.get("timing_continuity_stable", True)
    )
    report["stages"]["sysinfo"] = {"rc": rc, **gate_c, "gate_c_pass": gate_c_pass}

    # -------------------------------------------------------------------
    # Stage 2b: independent raw system-information oracle
    # -------------------------------------------------------------------
    rc_oracle_source = _run(
        [
            sys.executable,
            "scripts/pipeline/sysinfo.py",
            "--capture",
            str(ci8),
            "--output",
            str(sysinfo_oracle_source_path),
            "--frames",
            str(args.sysinfo_oracle_frames),
            "--system-info-index",
            str(args.sysinfo_index),
            "--equalizer",
            args.sysinfo_oracle_equalizer,
            *timing_args,
            "--no-timing-continuity",
        ],
        "sysinfo-oracle-source",
    )
    rc_oracle = 1
    if sysinfo_oracle_source_path.exists():
        rc_oracle = _run(
            [
                sys.executable,
                "scripts/probe_sysinfo_oracle.py",
                str(sysinfo_oracle_source_path),
                "--iq-source",
                "raw",
                "--omit-frames",
                "--json",
                str(sysinfo_oracle_path),
            ],
            "sysinfo-oracle",
        )
    oracle_report = read_json(sysinfo_oracle_path) if sysinfo_oracle_path.exists() else None
    oracle_quality = sysinfo_oracle_quality(oracle_report)
    oracle_ready = bool(oracle_report is not None and oracle_quality.get("ok"))
    report["stages"]["sysinfo_oracle"] = {
        "source_rc": rc_oracle_source,
        "rc": rc_oracle,
        "source_json": str(sysinfo_oracle_source_path),
        "oracle_json": str(sysinfo_oracle_path),
        "quality": oracle_quality,
        "gate_c_oracle_pass": oracle_ready,
    }

    # -------------------------------------------------------------------
    # Stage 3: full receive
    # -------------------------------------------------------------------
    receive_cmd = [
        sys.executable,
        "scripts/pipeline/receive.py",
        "--capture",
        str(ci8),
        "--sysinfo",
        str(sysinfo_path),
        "--sysinfo-oracle",
        str(sysinfo_oracle_path),
        "--output-ts",
        str(recovered_ts),
        "--output-json",
        str(receiver_json),
        "--frames",
        str(args.receive_frames),
        *timing_args,
    ]
    if args.force_receive:
        receive_cmd.append("--force-receive")
    if not oracle_ready and not args.force_receive:
        recovered_ts.parent.mkdir(parents=True, exist_ok=True)
        recovered_ts.write_bytes(b"")
        write_json(
            receiver_json,
            {
                "pipeline": {
                    "gate_c_note": "sysinfo_oracle_gate",
                    "sysinfo_oracle_path": str(sysinfo_oracle_path),
                    "sysinfo_oracle_quality": oracle_quality,
                    "requested_timing_policy": args.timing_policy,
                    "receiver_timing_policy": effective_timing_policy,
                    "timing_trajectory_path": str(args.timing_trajectory)
                    if args.timing_trajectory is not None
                    else None,
                    "skipped": True,
                    "skip_reason": (
                        "skipped_realtime_receive_due_to_sysinfo_oracle_gate; "
                        "rerun with --force-receive to decode regardless."
                    ),
                },
                "ts": {"lock": None},
            },
        )
        rc = 0
    else:
        rc = _run(receive_cmd, "receive")
    report["stages"]["receive"] = {
        "rc": rc,
        "skipped": bool(not oracle_ready and not args.force_receive),
        "ts_bytes": recovered_ts.stat().st_size if recovered_ts.exists() else 0,
    }

    # -------------------------------------------------------------------
    # Stage 4: ffprobe + ffmpeg gate
    # -------------------------------------------------------------------
    transcode_cmd = [
        sys.executable,
        "scripts/pipeline/transcode.py",
        "--input-ts",
        str(recovered_ts),
        "--probe-json",
        str(probe_path),
        "--output-mp4",
        str(mp4_path),
        "--ffmpeg-bin",
        args.ffmpeg_bin,
        "--ffprobe-bin",
        args.ffprobe_bin,
    ]
    rc = _run(transcode_cmd, "transcode")
    probe = read_json(probe_path) if probe_path.exists() else {"ok": False}
    report["stages"]["transcode"] = {
        "rc": rc,
        "probe_ok": bool(probe.get("ok")),
        "mp4_bytes": mp4_path.stat().st_size if mp4_path.exists() else 0,
        "streams": [
            {
                "codec_name": s.get("codec_name"),
                "codec_type": s.get("codec_type"),
                "width": s.get("width"),
                "height": s.get("height"),
                "sample_rate": s.get("sample_rate"),
            }
            for s in (probe.get("streams") or [])
        ],
    }

    if probe.get("ok") and mp4_path.exists() and mp4_path.stat().st_size > 0:
        report["verdict"] = "video_produced_and_verified"
        exit_code = 0
    elif not oracle_ready:
        report["verdict"] = "gate_c_oracle_failed_no_ts"
        report["hint"] = (
            "Short Gate C is not enough for TS recovery. The independent raw "
            "system-information oracle did not pass beyond deinterleaver "
            "latency; capture a longer/stabler block or use --force-receive "
            "only for diagnostics."
        )
        exit_code = 3
    elif gate_c_pass:
        report["verdict"] = "gate_c_ok_but_decode_failed"
        exit_code = 2
    else:
        report["verdict"] = "gate_c_failed_no_ts"
        report["hint"] = (
            "Gate A (PN945) passed but Gate C (system-info classifier) did "
            "not lock a standard vector. This is the documented real-capture "
            "blocker: the offline receiver's channel estimation/equalization "
            "does not yet yield stable sysinfo on fresh captures. See "
            "docs/capture-observations-20260509.md."
        )
        exit_code = 3

    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.keep_ci8 and ci8.exists():
        # CI8 is the largest artifact; keep it by default if decode failed so
        # the user can rerun stages, otherwise it's safe to drop.
        if report["verdict"] == "video_produced_and_verified":
            try:
                ci8.unlink()
                (ci8.with_suffix(ci8.suffix + ".json")).unlink(missing_ok=True)
            except OSError:
                pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
