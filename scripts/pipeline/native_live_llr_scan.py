"""Scan live HackRF centers by native post-deinterleaver LLR health.

This is a cheaper live diagnostic than a full LDPC+BCH decode. Each center is
captured through the native front end and QAM64 deinterleaver/demapper. A
bounded Python prefix provides descriptive LLR health while the streaming
native rolling-H gate scores the complete bounded capture. The output ranks
centers by LDPC-input health rather than PN acquisition or wideband-channel
metrics alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _common import ROOT, child_env, write_json
from dtmb.llr_health import analyze_llr_health

import check_hackrf_clock


DATA_SYMBOLS_PER_C3780_FRAME = 3744
BITS_PER_QAM64_SYMBOL = 6
BITS_PER_QAM64_FRAME = DATA_SYMBOLS_PER_C3780_FRAME * BITS_PER_QAM64_SYMBOL
DEFAULT_CENTERS = "602000000"
AVERAGE_POWER_RE = re.compile(r"average power ([+-]?\d+(?:\.\d+)?) dBfs")
TRANSFER_RATE_RE = re.compile(r"=\s+([+-]?\d+(?:\.\d+)?) MB/second")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-native-live-llr-scan")
    parser.add_argument("--centers", default=DEFAULT_CENTERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--sample-rate", type=int, default=20_000_000)
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=8_000_000,
        help="Centered DTMB filter width; 8 MHz limits adjacent-channel ingress.",
    )
    parser.add_argument("--samples", type=int, default=25_000_000)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Maximum frontend frames; zero consumes the complete bounded capture.",
    )
    parser.add_argument("--sync-frames", type=int, default=300)
    parser.add_argument("--acquisition-frames", type=int, default=16)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--lna-gain", type=int, default=8)
    parser.add_argument("--vga-gain", type=int, default=24)
    parser.add_argument(
        "--clock-source",
        choices=("internal", "external-10mhz", "gpsdo-10mhz", "unknown"),
        default="internal",
    )
    parser.add_argument("--external-reference-hz", type=int)
    parser.add_argument("--hackrf-clock", default="hackrf_clock")
    parser.add_argument("--hackrf-transfer", default="hackrf_transfer")
    parser.add_argument(
        "--hackrf-acquire",
        default="build/core-cpp/dtmb_core_hackrf_acquire",
        help="Native bounded HackRF CI8 acquisition tool.",
    )
    parser.add_argument(
        "--capture-source",
        choices=("file", "hackrf_transfer", "native_acquire"),
        default="hackrf_transfer",
        help="CI8 source for each center scan.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="CI8 file to replay when --capture-source=file.",
    )
    parser.add_argument(
        "--max-queued-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Native acquisition queue limit when --capture-source=native_acquire.",
    )
    parser.add_argument("--device-serial")
    parser.add_argument("--hackrf-force", action="store_true")
    parser.add_argument("--ci8-stats", default="build/core-cpp/dtmb_core_ci8_stats")
    parser.add_argument("--resampler", default="build/core-cpp/dtmb_core_ci8_resample")
    parser.add_argument("--extractor", default="build/core-cpp/dtmb_core_c3780_extract")
    parser.add_argument(
        "--demapper",
        default="build/core-cpp/dtmb_core_deinterleave_qam64",
    )
    parser.add_argument(
        "--h-scorer",
        default="build/core-cpp/dtmb_core_ldpc_h_gate",
        help="Native hard-H scorer used for the calibrated rolling shortlist.",
    )
    parser.add_argument(
        "--full-h-scorer",
        default="build/core-cpp/dtmb_core_ldpc_h_score",
        help="Native rolling-H scorer that writes the full window list.",
    )
    parser.add_argument(
        "--h-score-mode",
        choices=("gate", "full"),
        default="gate",
        help="Use the compact gate scorer or full rolling-H window-list scorer.",
    )
    parser.add_argument(
        "--source-score",
        action="store_true",
        help=(
            "Run the bounded native pre-LDPC source provenance scorer for each "
            "center after the rolling-H gate."
        ),
    )
    parser.add_argument(
        "--source-scorer",
        default="build/core-cpp/dtmb_core_pre_ldpc_source_score",
        help="Native bounded pre-LDPC source provenance scorer.",
    )
    parser.add_argument("--source-score-top", type=int, default=12)
    parser.add_argument("--source-score-frame-group-cap", type=int, default=4096)
    parser.add_argument("--source-score-weak-llr-threshold", type=float, default=1.0)
    parser.add_argument("--source-score-codewords-per-frame", type=int, default=3)
    parser.add_argument(
        "--source-score-fec-frame-range",
        action="append",
        default=[],
        help="Restrict source scoring to a 1-based inclusive FEC frame range FIRST:LAST.",
    )
    parser.add_argument(
        "--source-score-track-ldpc-variable",
        type=int,
        action="append",
        default=[],
        help="Track one LDPC variable's source-tag breakdown in the source scorer.",
    )
    parser.add_argument(
        "--alist",
        type=Path,
        help="LDPC alist; defaults to the selected DTMB FEC-rate alist.",
    )
    parser.add_argument("--resample-workers", type=int, default=8)
    parser.add_argument("--frontend-workers", type=int, default=8)
    parser.add_argument("--demap-workers", type=int, default=8)
    parser.add_argument("--frequency-shift-hz", type=float, default=0.0)
    parser.add_argument("--system-info-index", type=int, default=22)
    parser.add_argument("--fec-rate", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--interleaver-mode", choices=("mode1", "mode2"), default="mode2")
    parser.add_argument("--interleaver-phase", type=int, default=0)
    parser.add_argument("--branch-gain-branches", default="")
    parser.add_argument("--branch-gain-reliability-threshold", type=float, default=0.55)
    parser.add_argument("--branch-gain-min-symbols", type=int, default=32)
    parser.add_argument(
        "--branch-gain-source-frame-range",
        action="append",
        default=[],
        metavar="FIRST:LAST",
        help="Restrict branch-gain diagnostics to a source-frame range.",
    )
    parser.add_argument(
        "--branch-gain-diagnostics",
        action="store_true",
        help="Write per-source-frame branch-gain diagnostics beside each LLR.",
    )
    parser.add_argument(
        "--source-frame-llr-scale",
        action="append",
        default=[],
        metavar="BRANCH:FIRST:LAST:SCALE",
        help="Scale post-demap LLR confidence for a source-frame/branch range.",
    )
    parser.add_argument(
        "--source-carrier-llr-scale",
        action="append",
        default=[],
        metavar="CARRIER:FIRST:LAST:SCALE",
        help="Scale post-demap LLR confidence for a source-carrier/frame range.",
    )
    parser.add_argument(
        "--source-frame-symbol-gain",
        action="append",
        default=[],
        metavar="BRANCH:FIRST:LAST",
        help="Apply decision-directed pre-demap symbol gain correction for a source-frame/branch range.",
    )
    parser.add_argument(
        "--source-frame-axis-affine",
        action="append",
        default=[],
        metavar="BRANCH:FIRST:LAST",
        help="Apply decision-directed independent I/Q affine correction for a source-frame/branch range.",
    )
    parser.add_argument("--timing-search-radius", type=int, default=0)
    parser.add_argument("--timing-search-threshold", type=float, default=0.45)
    parser.add_argument("--timing-trajectory-interval-frames", type=int, default=0)
    parser.add_argument("--timing-trajectory-fit-points", type=int, default=17)
    parser.add_argument(
        "--timing-trajectory-max-innovation-samples",
        type=float,
        default=2.0,
    )
    parser.add_argument("--timing-trajectory-local-search", action="store_true")
    parser.add_argument(
        "--timing-trajectory-local-search-min-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument("--timing-diagnostics", action="store_true")
    parser.add_argument("--frame-residual-diagnostics", action="store_true")
    parser.add_argument("--auto-phase-adjustment", type=int)
    parser.add_argument("--pn-estimator", choices=("compact", "wideband"), default="wideband")
    parser.add_argument("--pn-wideband-block-frames", type=int, default=16)
    parser.add_argument("--pn-mmse", default="0.05")
    parser.add_argument("--normalization", choices=("qam64", "system-info", "none"), default="qam64")
    parser.add_argument("--no-remove-dc", action="store_true")
    parser.add_argument("--no-wideband-diagnostics", action="store_true")
    parser.add_argument("--sample-health-frames", type=int, default=16)
    parser.add_argument("--h-window-codewords", type=int, default=24)
    parser.add_argument("--h-window-step-codewords", type=int, default=3)
    parser.add_argument("--h-window-threshold", type=float, default=0.44)
    parser.add_argument(
        "--stream-h-score",
        action="store_true",
        help=(
            "Pipe deinterleaved LLR directly into the native rolling-H scorer "
            "instead of writing a full .llr.f32 artifact. This skips Python "
            "LLR-health and source-score diagnostics."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned commands and summary without touching hardware.",
    )
    return parser


def parse_centers(text: str) -> list[int]:
    centers = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not centers:
        raise ValueError("at least one center frequency is required")
    for center in centers:
        if center <= 0:
            raise ValueError("center frequencies must be positive")
    return centers


def build_pipeline_command(args: argparse.Namespace, center_hz: int, prefix: Path) -> str:
    if args.stream_h_score and args.source_score:
        raise ValueError("--stream-h-score cannot be combined with --source-score")
    llr_path = prefix.with_suffix(".llr.f32")
    parts: list[str] = [
        _build_capture_source_command(args, center_hz, prefix),
        _join([args.ci8_stats, "--passthrough", "-"])
        + f" 2>{shlex.quote(str(prefix.with_suffix('.ci8.log')))}",
    ]
    if args.sample_rate != 7_560_000:
        parts.append(
            _join(
                [
                    args.resampler,
                    "--input-rate",
                    str(args.sample_rate),
                    "--workers",
                    str(args.resample_workers),
                    "-",
                    "-",
                ]
            )
            + f" 2>{shlex.quote(str(prefix.with_suffix('.resample.log')))}"
        )
    extract_cmd = [
        args.extractor,
        "--auto-sync",
        "--sync-frames",
        str(args.sync_frames),
        "--acquisition-frames",
        str(args.acquisition_frames),
        "--auto-phase-adjustment",
        str(_auto_phase_adjustment(args)),
    ]
    if args.timing_search_radius:
        extract_cmd.extend(
            [
                "--timing-search-radius",
                str(args.timing_search_radius),
                "--timing-search-threshold",
                str(args.timing_search_threshold),
            ]
        )
    if args.timing_trajectory_interval_frames:
        if not args.timing_search_radius:
            raise ValueError(
                "timing-trajectory-interval-frames requires timing-search-radius"
            )
        extract_cmd.extend(
            [
                "--timing-trajectory-interval-frames",
                str(args.timing_trajectory_interval_frames),
                "--timing-trajectory-fit-points",
                str(args.timing_trajectory_fit_points),
                "--timing-trajectory-max-innovation-samples",
                _format_float(args.timing_trajectory_max_innovation_samples),
            ]
        )
        if args.timing_trajectory_local_search:
            extract_cmd.extend(
                [
                    "--timing-trajectory-local-search",
                    "--timing-trajectory-local-search-min-improvement",
                    _format_float(
                        args.timing_trajectory_local_search_min_improvement
                    ),
                ]
            )
        elif args.timing_trajectory_local_search_min_improvement != 0.0:
            raise ValueError(
                "timing-trajectory-local-search-min-improvement requires "
                "timing-trajectory-local-search"
            )
    elif args.timing_trajectory_local_search:
        raise ValueError(
            "timing-trajectory-local-search requires timing-trajectory-interval-frames"
        )
    elif args.timing_trajectory_local_search_min_improvement != 0.0:
        raise ValueError(
            "timing-trajectory-local-search-min-improvement requires "
            "timing-trajectory-local-search"
        )
    if args.timing_diagnostics:
        extract_cmd.extend(
            [
                "--timing-diagnostics",
                str(prefix.with_suffix(".timing.csv")),
            ]
        )
    if args.frame_residual_diagnostics:
        extract_cmd.extend(
            [
                "--frame-residual-diagnostics",
                str(prefix.with_suffix(".frame_residuals.csv")),
            ]
        )
    extract_cmd.extend(
        [
            "--max-frames",
            str(args.frames),
            "--workers",
            str(args.frontend_workers),
            "--frequency-shift-hz",
            _format_float(args.frequency_shift_hz),
            "--equalizer",
            "pn",
            "--pn-estimator",
            args.pn_estimator,
            "--pn-wideband-block-frames",
            str(args.pn_wideband_block_frames),
        ]
    )
    if args.pn_estimator == "wideband" and not args.no_wideband_diagnostics:
        extract_cmd.extend(
            [
                "--pn-wideband-diagnostics",
                str(prefix.with_suffix(".wideband.csv")),
            ]
        )
    extract_cmd.extend(["--pn-mmse", str(args.pn_mmse)])
    if not args.no_remove_dc:
        extract_cmd.append("--remove-dc")
    extract_cmd.extend(
        [
            "--normalization",
            args.normalization,
            "--system-info-index",
            str(args.system_info_index),
            "-",
            "-",
        ]
    )
    parts.append(
        _join(extract_cmd)
        + f" 2>{shlex.quote(str(prefix.with_suffix('.extract.log')))}"
    )
    demap_cmd = [
        args.demapper,
        "--mode",
        args.interleaver_mode,
        "--phase",
        str(args.interleaver_phase),
        "--workers",
        str(args.demap_workers),
    ]
    if args.branch_gain_branches:
        demap_cmd.extend(
            [
                "--branch-gain-branches",
                args.branch_gain_branches,
                "--branch-gain-reliability-threshold",
                _format_float(args.branch_gain_reliability_threshold),
                "--branch-gain-min-symbols",
                str(args.branch_gain_min_symbols),
            ]
        )
        for frame_range in args.branch_gain_source_frame_range:
            demap_cmd.extend(["--branch-gain-source-frame-range", frame_range])
        if args.branch_gain_diagnostics:
            demap_cmd.extend(
                [
                    "--branch-gain-diagnostics-out",
                    str(prefix.with_suffix(".branch_gain_frames.json")),
                ]
            )
    for scale_rule in args.source_frame_llr_scale:
        demap_cmd.extend(["--source-frame-llr-scale", scale_rule])
    for carrier_rule in args.source_carrier_llr_scale:
        demap_cmd.extend(["--source-carrier-llr-scale", carrier_rule])
    for gain_rule in args.source_frame_symbol_gain:
        demap_cmd.extend(["--source-frame-symbol-gain", gain_rule])
    for affine_rule in args.source_frame_axis_affine:
        demap_cmd.extend(["--source-frame-axis-affine", affine_rule])
    demap_output = "-" if args.stream_h_score else str(llr_path)
    demap_cmd.extend(["-", demap_output])
    parts.append(
        _join(demap_cmd)
        + f" 2>{shlex.quote(str(prefix.with_suffix('.demap.log')))}"
    )
    if args.stream_h_score:
        parts.append(
            _join(
                _h_score_command(
                    args,
                    "-",
                    prefix.with_suffix(".h_score.json"),
                )
            )
            + " > /dev/null"
            + f" 2>{shlex.quote(str(prefix.with_suffix('.h_score.log')))}"
        )
    return " | ".join(parts)


def scan_center(args: argparse.Namespace, center_hz: int) -> dict[str, Any]:
    prefix = args.output_dir / f"live_{center_hz}"
    report: dict[str, Any] = {
        "frequency_hz": center_hz,
        "prefix": str(prefix),
        "llr_path": None if args.stream_h_score else str(prefix.with_suffix(".llr.f32")),
        "llr_health_json": (
            None if args.stream_h_score else str(prefix.with_suffix(".llr_health.json"))
        ),
        "h_score_json": str(prefix.with_suffix(".h_score.json")),
        "source_score_json": (
            str(prefix.with_suffix(".source_score.json")) if args.source_score else None
        ),
        "capture_source": args.capture_source,
        "capture_log": str(_capture_log_path(args, prefix)),
        "transfer_log": str(prefix.with_suffix(".transfer.log")),
        "acquire_log": str(prefix.with_suffix(".acquire.log")),
        "extract_log": str(prefix.with_suffix(".extract.log")),
        "demap_log": str(prefix.with_suffix(".demap.log")),
    }
    if (
        args.capture_source != "file"
        and args.clock_source in check_hackrf_clock.EXTERNAL_CLOCK_SOURCES
    ):
        try:
            report["clock_check"] = check_hackrf_clock.check_clock(
                clock_source=args.clock_source,
                external_reference_hz=args.external_reference_hz,
                hackrf_clock=args.hackrf_clock,
                device_serial=args.device_serial,
                log_path=prefix.with_suffix(".clkin.log"),
            )
        except (RuntimeError, ValueError) as exc:
            report.update({"ok": False, "stage": "clock", "error": str(exc)})
            return report

    command = build_pipeline_command(args, center_hz, prefix)
    report["command"] = command
    if args.dry_run:
        report.update({"ok": True, "stage": "dry_run"})
        return report

    completed = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=str(ROOT),
        env=child_env(),
        check=False,
    )
    report["pipeline_returncode"] = completed.returncode
    if completed.returncode not in {0, 141}:
        report.update({"ok": False, "stage": "pipeline"})
        return report

    llr_path = prefix.with_suffix(".llr.f32")
    if args.stream_h_score:
        try:
            h_score = json.loads(
                prefix.with_suffix(".h_score.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            report.update({"ok": False, "stage": "h_score", "error": str(exc)})
            return report
        health = {
            "available": False,
            "reason": "stream_h_score_no_llr_artifact",
        }
        source_score = None
        extract = _read_key_value_log(prefix.with_suffix(".extract.log"))
        report.update(
            {
                "llr_bytes": 0,
                "ok": True,
                "stage": "rolling_h_gate",
                "transfer": _summarize_transfer_log(
                    _capture_log_path(args, prefix),
                    sample_rate=args.sample_rate,
                ),
                "extract": extract,
                "demap": _read_key_value_log(prefix.with_suffix(".demap.log")),
                "wideband": _summarize_wideband(prefix.with_suffix(".wideband.csv")),
                "llr_health": health,
                "h_score": _compact_h_score(h_score),
                "source_score": source_score,
                "frontend_coverage": _frontend_coverage(args.frames, extract.get("frames")),
            }
        )
        if args.capture_source == "file":
            report["transfer"] = {
                "available": False,
                "source": "file",
                "input": str(args.input) if args.input is not None else None,
                "meets_realtime_rate": None,
            }
        return report

    if not llr_path.exists():
        report.update({"ok": False, "stage": "llr_missing"})
        return report
    report["llr_bytes"] = llr_path.stat().st_size
    sidecar = _llr_sidecar(args, center_hz, prefix, command)
    write_json(prefix.with_suffix(".llr.f32.json"), sidecar)

    llr, health_scope = _read_health_sample(
        llr_path,
        bits_per_frame=BITS_PER_QAM64_FRAME,
        sample_frames=args.sample_health_frames,
    )
    health = analyze_llr_health(
        llr,
        bits_per_frame=BITS_PER_QAM64_FRAME,
        fec_rate_index=args.fec_rate,
        qam_mode="64qam",
        metadata={"live_llr_scan": sidecar, "sample_scope": health_scope},
        sample_frames=args.sample_health_frames,
    )
    health.update(health_scope)
    write_json(prefix.with_suffix(".llr_health.json"), health)
    try:
        h_score = _run_h_score(args, llr_path, prefix.with_suffix(".h_score.json"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report.update(
            {
                "ok": False,
                "stage": "h_score",
                "error": str(exc),
                "llr_health": _compact_health(health),
            }
        )
        return report
    source_score: dict[str, Any] | None = None
    if args.source_score:
        try:
            source_score = _run_source_score(
                args,
                llr_path,
                prefix.with_suffix(".source_score.json"),
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report.update(
                {
                    "ok": False,
                    "stage": "source_score",
                    "error": str(exc),
                    "llr_health": _compact_health(health),
                    "h_score": _compact_h_score(h_score),
                }
            )
            return report
    extract = _read_key_value_log(prefix.with_suffix(".extract.log"))
    report.update(
        {
            "ok": True,
            "stage": "rolling_h_gate",
            "transfer": _summarize_transfer_log(
                _capture_log_path(args, prefix),
                sample_rate=args.sample_rate,
            ),
            "extract": extract,
            "demap": _read_key_value_log(prefix.with_suffix(".demap.log")),
            "wideband": _summarize_wideband(prefix.with_suffix(".wideband.csv")),
            "llr_health": _compact_health(health),
            "h_score": _compact_h_score(h_score),
            "source_score": (
                _compact_source_score(source_score) if source_score is not None else None
            ),
            "frontend_coverage": _frontend_coverage(args.frames, extract.get("frames")),
        }
    )
    if args.capture_source == "file":
        report["transfer"] = {
            "available": False,
            "source": "file",
            "input": str(args.input) if args.input is not None else None,
            "meets_realtime_rate": None,
        }
    return report


def rank_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(reports, key=_ranking_key)
    for report in ranked:
        report["ranking"] = _ranking_summary(report)
    return ranked


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        centers = parse_centers(args.centers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or (args.output_dir / "llr_scan_summary.json")
    reports = [scan_center(args, center) for center in centers]
    ranked = rank_reports(reports)
    summary = {
        "stage": "native_live_llr_scan",
        "ok": all(report.get("ok") for report in reports),
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "centers": centers,
        "sample_rate_sps": args.sample_rate,
        "bandwidth_hz": args.bandwidth,
        "samples_per_center": args.samples,
        "frames_per_center": args.frames,
        "frontend_frames_requested_per_center": args.frames,
        "sample_limited_candidate_count": sum(
            1
            for report in ranked
            if ((report.get("frontend_coverage") or {}).get("status") == "sample_limited")
        ),
        "clock_source": args.clock_source,
        "external_reference_hz": args.external_reference_hz,
        "device_serial": args.device_serial,
        "actionable_candidate_count": sum(
            1
            for report in ranked
            if _report_actionable(report)
        ),
        "best_candidate_frequency_hz": ranked[0]["frequency_hz"] if ranked else None,
        "best_candidate_verdict": (
            ((ranked[0].get("h_score") or {}).get("gate_verdict")) if ranked else None
        ),
        "best_candidate_actionable": _report_actionable(ranked[0]) if ranked else False,
        "selection_note": _selection_note(ranked[0]) if ranked else None,
        "ranked": ranked,
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


def _auto_phase_adjustment(args: argparse.Namespace) -> int:
    if args.auto_phase_adjustment is not None:
        return int(args.auto_phase_adjustment)
    return 0 if args.sample_rate == 7_560_000 else 1


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _llr_sidecar(
    args: argparse.Namespace,
    center_hz: int,
    prefix: Path,
    command: str,
) -> dict[str, Any]:
    return {
        "stage": "native_live_llr_scan",
        "frequency_hz": center_hz,
        "sample_rate_sps": args.sample_rate,
        "bandwidth_hz": args.bandwidth,
        "samples": args.samples,
        "frames_requested": args.frames,
        "capture_source": args.capture_source,
        "input": str(args.input) if args.input is not None else None,
        "hackrf_force": bool(args.hackrf_force),
        "device_serial": args.device_serial,
        "qam_mode": "64qam",
        "fec_rate_index": args.fec_rate,
        "bits_per_frame": BITS_PER_QAM64_FRAME,
        "symbol_deinterleave": args.interleaver_mode,
        "symbol_deinterleave_phase": args.interleaver_phase,
        "system_info_index": args.system_info_index,
        "pn_estimator": args.pn_estimator,
        "pn_wideband_block_frames": args.pn_wideband_block_frames,
        "timing_search_radius": args.timing_search_radius,
        "timing_search_threshold": args.timing_search_threshold,
        "timing_trajectory_interval_frames": args.timing_trajectory_interval_frames,
        "timing_trajectory_fit_points": args.timing_trajectory_fit_points,
        "timing_trajectory_max_innovation_samples": (
            args.timing_trajectory_max_innovation_samples
        ),
        "timing_diagnostics": bool(args.timing_diagnostics),
        "timing_diagnostics_path": (
            str(prefix.with_suffix(".timing.csv"))
            if args.timing_diagnostics
            else None
        ),
        "frame_residual_diagnostics": bool(args.frame_residual_diagnostics),
        "frame_residual_diagnostics_path": (
            str(prefix.with_suffix(".frame_residuals.csv"))
            if args.frame_residual_diagnostics
            else None
        ),
        "llr_path": str(prefix.with_suffix(".llr.f32")),
        "command": command,
    }


def _build_capture_source_command(
    args: argparse.Namespace,
    center_hz: int,
    prefix: Path,
) -> str:
    if args.capture_source == "file":
        if args.input is None:
            raise ValueError("--input is required when --capture-source=file")
        return (
            _join(["cat", args.input])
            + f" 2>{shlex.quote(str(prefix.with_suffix('.file.log')))}"
        )
    if args.capture_source == "native_acquire":
        return (
            _join(
                [
                    args.hackrf_acquire,
                    "--frequency",
                    str(center_hz),
                    "--sample-rate",
                    str(args.sample_rate),
                    "--bandwidth",
                    str(args.bandwidth),
                    "--amp",
                    str(args.amp),
                    "--lna-gain",
                    str(args.lna_gain),
                    "--vga-gain",
                    str(args.vga_gain),
                    *(["--serial", args.device_serial] if args.device_serial else []),
                    "--samples",
                    str(args.samples),
                    "--max-queued-bytes",
                    str(args.max_queued_bytes),
                    "--output",
                    "-",
                ]
            )
            + f" 2>{shlex.quote(str(prefix.with_suffix('.acquire.log')))}"
        )
    return (
        _join(
            [
                args.hackrf_transfer,
                "-r",
                "-",
                "-f",
                str(center_hz),
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
                *(["-d", args.device_serial] if args.device_serial else []),
                *(["-F"] if args.hackrf_force else []),
                "-n",
                str(args.samples),
            ]
        )
        + f" 2>{shlex.quote(str(prefix.with_suffix('.transfer.log')))}"
    )


def _capture_log_path(args: argparse.Namespace, prefix: Path) -> Path:
    if args.capture_source == "file":
        return prefix.with_suffix(".file.log")
    if args.capture_source == "native_acquire":
        return prefix.with_suffix(".acquire.log")
    return prefix.with_suffix(".transfer.log")


def _run_h_score(args: argparse.Namespace, llr_path: Path, output_path: Path) -> dict[str, Any]:
    command = _h_score_command(args, llr_path, output_path)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=child_env(),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise RuntimeError(
            f"native H scorer failed with rc={completed.returncode}: {detail}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _h_score_command(
    args: argparse.Namespace,
    llr_path: str | Path,
    output_path: Path,
) -> list[str]:
    if args.h_window_codewords <= 0:
        raise ValueError("h-window-codewords must be positive")
    if args.h_window_step_codewords <= 0:
        raise ValueError("h-window-step-codewords must be positive")
    if not 0.0 <= args.h_window_threshold <= 1.0:
        raise ValueError("h-window-threshold must be between zero and one")
    alist = args.alist or (
        ROOT / "python" / "dtmb" / "data" / f"dtmb_ldpc_rate{args.fec_rate}.alist"
    )
    if args.h_score_mode == "full":
        return [
            args.full_h_scorer,
            "--fec-rate",
            str(args.fec_rate),
            "--alist",
            str(alist),
            "--input-format",
            "llr-f32",
            "--workers",
            str(args.demap_workers),
            "--window-codewords",
            str(args.h_window_codewords),
            "--window-step-codewords",
            str(args.h_window_step_codewords),
            "--window-threshold",
            _format_float(args.h_window_threshold),
            "--json-out",
            str(output_path),
            str(llr_path),
        ]
    command = [
        args.h_scorer,
        "--fec-rate",
        str(args.fec_rate),
        "--alist",
        str(alist),
        "--window-codewords",
        str(args.h_window_codewords),
        "--window-step-codewords",
        str(args.h_window_step_codewords),
        "--window-threshold",
        _format_float(args.h_window_threshold),
        "--output-mode",
        "passthrough",
        "--diagnostics-format",
        "json",
        "--diagnostics-out",
        str(output_path),
        str(llr_path),
    ]
    return command


def _run_source_score(
    args: argparse.Namespace,
    llr_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if args.source_score_top < 0:
        raise ValueError("source-score-top must be non-negative")
    if args.source_score_frame_group_cap <= 0:
        raise ValueError("source-score-frame-group-cap must be positive")
    if args.source_score_codewords_per_frame <= 0:
        raise ValueError("source-score-codewords-per-frame must be positive")
    if any(variable < 0 for variable in args.source_score_track_ldpc_variable):
        raise ValueError("source-score-track-ldpc-variable must be non-negative")
    if not np.isfinite(args.source_score_weak_llr_threshold) or (
        args.source_score_weak_llr_threshold < 0
    ):
        raise ValueError("source-score-weak-llr-threshold must be non-negative")
    alist = args.alist or (
        ROOT / "python" / "dtmb" / "data" / f"dtmb_ldpc_rate{args.fec_rate}.alist"
    )
    command = [
        args.source_scorer,
        "--fec-rate",
        str(args.fec_rate),
        "--alist",
        str(alist),
        "--interleaver-mode",
        args.interleaver_mode,
        "--interleaver-phase",
        str(args.interleaver_phase),
        "--top",
        str(args.source_score_top),
        "--source-frame-group-cap",
        str(args.source_score_frame_group_cap),
        "--weak-llr-threshold",
        _format_float(args.source_score_weak_llr_threshold),
        "--codewords-per-frame",
        str(args.source_score_codewords_per_frame),
        "--diagnostics-out",
        str(output_path),
    ]
    for frame_range in args.source_score_fec_frame_range:
        command.extend(["--fec-frame-range", frame_range])
    for variable in args.source_score_track_ldpc_variable:
        command.extend(["--track-ldpc-variable", str(variable)])
    command.append(str(llr_path))
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=child_env(),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise RuntimeError(
            f"native source scorer failed with rc={completed.returncode}: {detail}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _read_health_sample(
    llr_path: Path,
    *,
    bits_per_frame: int,
    sample_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bits_per_frame <= 0:
        raise ValueError("bits_per_frame must be positive")
    if sample_frames < 0:
        raise ValueError("sample_frames must be non-negative")
    byte_count = llr_path.stat().st_size
    float_bytes = np.dtype("<f4").itemsize
    if byte_count % float_bytes:
        raise ValueError("LLR input byte count must be float32-aligned")
    artifact_llrs = byte_count // float_bytes
    requested_llrs = sample_frames * bits_per_frame
    sampled_llrs = min(artifact_llrs, requested_llrs)
    values = np.fromfile(llr_path, dtype="<f4", count=sampled_llrs)
    return values, {
        "artifact_llr_count": artifact_llrs,
        "artifact_complete_frames": artifact_llrs // bits_per_frame,
        "artifact_unused_llrs": artifact_llrs % bits_per_frame,
        "sampled_prefix_llrs": int(values.size),
        "sampled_prefix_frames": int(values.size // bits_per_frame),
        "sampled_prefix_bytes": int(values.nbytes),
        "sample_scope": "bounded_prefix_for_descriptive_health",
        "full_capture_score": "native_streaming_rolling_h_gate",
    }


def _compact_h_score(report: dict[str, Any]) -> dict[str, Any]:
    schema = report.get("schema")
    if schema == "dtmb.ldpc_h_gate.v1":
        gate_verdict = report.get("gate_verdict")
        compact_verdict = "pass" if gate_verdict == "pass" else "reject"
        best_window = report.get("best_window")
        threshold = _safe_float(report.get("window_threshold"))
        return {
            "schema": schema,
            "gate_verdict": compact_verdict,
            "native_gate_verdict": gate_verdict,
            "status": report.get("status"),
            "verdict": report.get("verdict"),
            "input_byte_aligned": report.get("input_byte_aligned"),
            "input_codeword_aligned": report.get("input_codeword_aligned"),
            "dirty_codewords": report.get("dirty_codewords"),
            "nonfinite_llrs": report.get("nonfinite_llrs"),
            "full_mean_syndrome_ratio": None,
            "full_min_syndrome_ratio": None,
            "codewords": report.get("complete_codewords"),
            "window_codewords": report.get("window_codewords"),
            "window_step_codewords": report.get("window_step_codewords"),
            "window_threshold": threshold,
            "window_count": report.get("scored_windows"),
            "passing_window_count": report.get("pass_windows"),
            "best_window": best_window,
        }

    best = report.get("best") or {}
    windows = report.get("windows") or []
    threshold = _safe_float(report.get("window_threshold"))
    passing = [
        row
        for row in windows
        if threshold is not None
        and _safe_float(row.get("mean_syndrome_ratio")) is not None
        and float(row["mean_syndrome_ratio"]) <= threshold
    ]
    best_window = min(
        windows,
        key=lambda row: float(row.get("mean_syndrome_ratio", 1.0)),
        default=None,
    )
    return {
        "schema": schema,
        "gate_verdict": "pass" if passing else "reject",
        "full_mean_syndrome_ratio": _safe_float(best.get("mean_syndrome_ratio")),
        "full_min_syndrome_ratio": _safe_float(best.get("min_syndrome_ratio")),
        "codewords": best.get("codewords"),
        "window_codewords": report.get("window_codewords"),
        "window_step_codewords": report.get("window_step_codewords"),
        "window_threshold": threshold,
        "window_count": report.get("window_count"),
        "passing_window_count": len(passing),
        "best_window": best_window,
    }


def _compact_source_score(report: dict[str, Any]) -> dict[str, Any]:
    dimensions = report.get("dimensions") or {}
    hard_h = report.get("hard_h") or {}
    summary = report.get("summary") or {}
    return {
        "schema": report.get("schema"),
        "status": report.get("status"),
        "verdict": report.get("verdict"),
        "codewords": hard_h.get("codewords") or summary.get("complete_codewords"),
        "mean_syndrome_ratio": _safe_float(hard_h.get("mean_syndrome_ratio")),
        "failed_check_edge_ratio": _safe_float(
            hard_h.get("failed_check_edge_ratio")
        ),
        "observed_check_edges": hard_h.get("observed_check_edges"),
        "failed_check_edges": hard_h.get("failed_check_edges"),
        "dimensions": {
            name: _compact_source_score_dimension(dimensions.get(name) or {})
            for name in (
                "codeword_slot",
                "ldpc_variable_mod_127",
                "ldpc_variable",
                "qam_plane",
                "source_branch",
                "source_carrier",
                "source_frame",
            )
        },
        "tracked_ldpc_variables": [
            _compact_tracked_ldpc_variable(variable)
            for variable in (report.get("tracked_ldpc_variables") or [])
        ],
    }


def _compact_tracked_ldpc_variable(report: dict[str, Any]) -> dict[str, Any]:
    dimensions = report.get("dimensions") or {}
    summary = report.get("summary") or {}
    return {
        "variable": report.get("variable"),
        "bit_count": summary.get("bit_count"),
        "llr_mean_abs": _safe_float(summary.get("llr_mean_abs")),
        "weak_llr_fraction": _safe_float(summary.get("weak_llr_fraction")),
        "hard_h_failed_check_edge_ratio": _safe_float(
            summary.get("hard_h_failed_check_edge_ratio")
        ),
        "dimensions": {
            name: _compact_source_score_dimension(dimensions.get(name) or {})
            for name in (
                "codeword_slot",
                "qam_plane",
                "source_branch",
                "source_carrier",
                "source_frame",
            )
        },
    }


def _compact_source_score_dimension(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_count": report.get("group_count"),
        "capped": report.get("capped"),
        "overflow_bit_count": report.get("overflow_bit_count"),
        "hard_h_failed_check_edge_ratio": (
            (report.get("summary") or {}).get("hard_h_failed_check_edge_ratio")
        ),
        "weakest_llr": _compact_source_score_rows(report.get("weakest_llr") or []),
        "worst_hard_h": _compact_source_score_rows(report.get("worst_hard_h") or []),
    }


def _compact_source_score_rows(rows: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in list(rows)[:limit]:
        compact.append(
            {
                "id": row.get("id"),
                "bit_count": row.get("bit_count"),
                "llr_mean_abs": _safe_float(row.get("llr_mean_abs")),
                "weak_llr_fraction": _safe_float(row.get("weak_llr_fraction")),
                "hard_h_failed_check_edge_ratio": _safe_float(
                    row.get("hard_h_failed_check_edge_ratio")
                ),
            }
        )
    return compact


def _read_key_value_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = _coerce_scalar(value)
    return result


def _summarize_wideband(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, Any] = {
        "available": True,
        "model_count": len(rows),
    }
    for key in (
        "span_symbols",
        "significant_taps",
        "noise_variance",
        "per_frame_noise_tap_power",
        "truncated_energy_fraction",
    ):
        values = [_safe_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[key] = {
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "max": max(values),
            }
    return summary


def _summarize_transfer_log(path: Path, *, sample_rate: int | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    powers = [float(match.group(1)) for match in AVERAGE_POWER_RE.finditer(text)]
    rates = [float(match.group(1)) for match in TRANSFER_RATE_RE.finditer(text)]
    native_reports = _parse_native_acquire_reports(text)
    native_rates = [
        float(report["throughput_mbps"])
        for report in native_reports
        if isinstance(report.get("throughput_mbps"), (int, float))
        and report.get("status") == "running"
    ]
    if not rates and native_rates:
        rates = native_rates
    summary: dict[str, Any] = {
        "available": True,
        "status_lines": len(text.splitlines()),
    }
    if native_reports:
        complete = next(
            (report for report in reversed(native_reports) if report.get("status") == "complete"),
            None,
        )
        summary["source"] = "dtmb_core_hackrf_acquire"
        if complete:
            summary["complete"] = {
                "stop_reason": complete.get("stop_reason"),
                "total_elapsed_s": complete.get("total_elapsed_s"),
                "capture_elapsed_s": complete.get("capture_elapsed_s"),
                "expected_capture_s": complete.get("expected_capture_s"),
                "capture_realtime_ratio": complete.get("capture_realtime_ratio"),
                "samples_seen": complete.get("samples_seen"),
                "max_observed_queued_bytes": complete.get("max_observed_queued_bytes"),
                "bytes_written": complete.get("bytes_written"),
                "samples_written": complete.get("samples_written"),
                "throughput_mbps": complete.get("throughput_mbps"),
                "dropped_bytes": complete.get("dropped_bytes"),
                "dropped_samples": complete.get("dropped_samples"),
            }
            summary["dropped_bytes"] = complete.get("dropped_bytes")
            summary["dropped_samples"] = complete.get("dropped_samples")
    if powers:
        summary["average_power_dbfs"] = {
            "min": min(powers),
            "median": statistics.median(powers),
            "mean": statistics.mean(powers),
            "max": max(powers),
            "samples": len(powers),
        }
    if rates:
        summary["throughput_mb_per_second"] = {
            "min": min(rates),
            "median": statistics.median(rates),
            "mean": statistics.mean(rates),
            "max": max(rates),
            "samples": len(rates),
        }
    if sample_rate is not None:
        required = sample_rate * 2 / 1_000_000
        median_rate = statistics.median(rates) if rates else None
        capture_realtime_ratio = None
        if native_reports and "complete" in summary:
            ratio = summary["complete"].get("capture_realtime_ratio")
            if isinstance(ratio, (int, float)):
                capture_realtime_ratio = float(ratio)
        summary["required_ci8_mb_per_second"] = required
        summary["realtime_ratio_median"] = (
            median_rate / required if median_rate is not None else None
        )
        if capture_realtime_ratio is not None:
            summary["capture_realtime_ratio"] = capture_realtime_ratio
            summary["meets_realtime_rate"] = capture_realtime_ratio >= 0.90
        else:
            summary["meets_realtime_rate"] = bool(
                median_rate is not None and median_rate >= required * 0.90
            )
    return summary


def _parse_native_acquire_reports(text: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and "status" in report:
            reports.append(report)
    return reports


def _compact_health(report: dict[str, Any]) -> dict[str, Any]:
    parity = report.get("ldpc_parity_check", {}) or {}
    appendix = parity.get("appendix_b_syndrome", {}) or {}
    llr = report.get("llr", {}) or {}
    hard_bits = report.get("hard_bits", {}) or {}
    return {
        "verdict": report.get("verdict"),
        "complete_frames": report.get("complete_frames"),
        "llr_count": report.get("llr_count"),
        "llr_mean_abs": llr.get("mean_abs"),
        "llr_median_abs": llr.get("median_abs"),
        "hard_bit_one_ratio": hard_bits.get("one_ratio"),
        "codewords": parity.get("codewords"),
        "mean_mismatch_ratio": parity.get("mean_mismatch_ratio"),
        "min_mismatch_ratio": parity.get("min_mismatch_ratio"),
        "zero_mismatch_codewords": parity.get("zero_mismatch_codewords"),
        "mean_syndrome_ratio": appendix.get("mean_syndrome_ratio"),
        "min_syndrome_ratio": appendix.get("min_syndrome_ratio"),
        "zero_syndrome_codewords": appendix.get("zero_syndrome_codewords"),
        "interpretation": parity.get("interpretation"),
    }


def _ranking_key(report: dict[str, Any]) -> tuple[float, float, float, float, int]:
    if not report.get("ok"):
        return (9.0, 9.0, 9.0, 9.0, int(report.get("frequency_hz", 0) or 0))
    h_score = report.get("h_score", {}) or {}
    health = report.get("llr_health", {}) or {}
    wideband = report.get("wideband", {}) or {}
    gate_verdict = str(h_score.get("gate_verdict") or "unknown")
    passing_windows = int(h_score.get("passing_window_count", 0) or 0)
    best_window = _safe_float((h_score.get("best_window") or {}).get("mean_syndrome_ratio"))
    full_h_mean = _safe_float(h_score.get("full_mean_syndrome_ratio"))
    syndrome = health.get("mean_syndrome_ratio")
    mismatch = health.get("mean_mismatch_ratio")
    one_ratio = health.get("hard_bit_one_ratio")
    verdict = str(health.get("verdict") or "unknown")
    noise_mean = _safe_float((wideband.get("noise_variance") or {}).get("mean"))
    tap_noise_mean = _safe_float((wideband.get("per_frame_noise_tap_power") or {}).get("mean"))
    balance_error = abs(float(one_ratio) - 0.5) if one_ratio is not None else 1.0
    frequency_hz = int(report.get("frequency_hz", 0) or 0)
    if gate_verdict == "pass":
        return (
            0.0,
            -float(passing_windows),
            best_window if best_window is not None else 1.0,
            full_h_mean if full_h_mean is not None else 1.0,
            frequency_hz,
        )
    if gate_verdict == "reject":
        return (
            1.0,
            best_window if best_window is not None else 1.0,
            full_h_mean if full_h_mean is not None else 1.0,
            noise_mean if noise_mean is not None else 1.0,
            frequency_hz,
        )
    if verdict == "ldpc_consistent":
        return (
            2.0,
            float(syndrome) if syndrome is not None else 1.0,
            float(mismatch) if mismatch is not None else 1.0,
            balance_error,
            frequency_hz,
        )
    if verdict == "partial_consistency_investigate":
        return (
            3.0,
            float(syndrome) if syndrome is not None else 1.0,
            float(mismatch) if mismatch is not None else 1.0,
            balance_error,
            frequency_hz,
        )
    if verdict == "hard_parity_uninformative_before_ldpc":
        return (
            4.0,
            noise_mean if noise_mean is not None else 1.0,
            tap_noise_mean if tap_noise_mean is not None else 1.0,
            balance_error,
            frequency_hz,
        )
    if verdict == "no_codewords":
        return (5.0, 1.0, 1.0, 1.0, frequency_hz)
    return (
        6.0,
        float(syndrome) if syndrome is not None else 1.0,
        float(mismatch) if mismatch is not None else 1.0,
        balance_error,
        frequency_hz,
    )


def _ranking_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("ok"):
        return {"basis": "pipeline_failure"}
    h_score = report.get("h_score", {}) or {}
    health = report.get("llr_health", {}) or {}
    wideband = report.get("wideband", {}) or {}
    verdict = str(health.get("verdict") or "unknown")
    gate_verdict = str(h_score.get("gate_verdict") or "unknown")
    if gate_verdict in {"pass", "reject"}:
        return {
            "basis": "calibrated_rolling_h_gate",
            "gate_verdict": gate_verdict,
            "passing_window_count": h_score.get("passing_window_count"),
            "window_count": h_score.get("window_count"),
            "best_window_mean_syndrome_ratio": _safe_float(
                (h_score.get("best_window") or {}).get("mean_syndrome_ratio")
            ),
            "full_mean_syndrome_ratio": _safe_float(
                h_score.get("full_mean_syndrome_ratio")
            ),
        }
    if verdict == "hard_parity_uninformative_before_ldpc":
        noise_mean = _safe_float((wideband.get("noise_variance") or {}).get("mean"))
        tap_noise_mean = _safe_float(
            (wideband.get("per_frame_noise_tap_power") or {}).get("mean")
        )
        return {
            "basis": (
                "wideband_noise_after_uninformative_parity"
                if noise_mean is not None or tap_noise_mean is not None
                else "hard_bit_balance_after_uninformative_parity"
            ),
            "wideband_noise_mean": noise_mean,
            "per_frame_noise_tap_power_mean": tap_noise_mean,
            "hard_bit_balance_error": _hard_bit_balance_error(health),
        }
    return {
        "basis": "ldpc_parity_then_balance",
        "mean_syndrome_ratio": _safe_float(health.get("mean_syndrome_ratio")),
        "mean_mismatch_ratio": _safe_float(health.get("mean_mismatch_ratio")),
        "hard_bit_balance_error": _hard_bit_balance_error(health),
        "wideband_noise_mean": _safe_float((wideband.get("noise_variance") or {}).get("mean")),
    }


def _selection_note(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        return "no_successful_scan_result"
    coverage = report.get("frontend_coverage") or {}
    if coverage.get("status") == "sample_limited":
        return (
            "best candidate did not reach the requested frontend frame coverage; "
            "increase samples or set frames=0 before promotion"
        )
    if report.get("capture_source") == "file":
        gate_verdict = str(((report.get("h_score") or {}).get("gate_verdict")) or "unknown")
        if gate_verdict == "pass":
            return (
                "offline file replay passes the calibrated rolling hard-H shortlist; "
                "use it for parameter comparison, not live promotion"
            )
        return (
            "offline file replay does not pass the calibrated rolling hard-H "
            "shortlist; keep this parameter out of live promotion"
        )
    transfer = report.get("transfer") or {}
    if transfer.get("meets_realtime_rate") is not True:
        return (
            "best candidate did not sustain verified real-time CI8 transfer "
            "throughput; skip promotion"
        )
    gate_verdict = str(((report.get("h_score") or {}).get("gate_verdict")) or "unknown")
    if gate_verdict == "pass":
        return "best candidate passes the calibrated rolling hard-H shortlist before soft decode"
    if gate_verdict == "reject":
        return "all current candidates fail the calibrated rolling hard-H shortlist; skip soft decode"
    verdict = str(((report.get("llr_health") or {}).get("verdict")) or "unknown")
    if verdict == "hard_parity_uninformative_before_ldpc":
        return (
            "all current candidates remain random-like before LDPC; ranking falls "
            "back to wideband noise and hard-bit balance"
        )
    if verdict == "ldpc_consistent":
        return "best candidate is already LDPC-consistent before decode"
    return "best candidate is ranked by LDPC parity and hard-bit balance"


def _frontend_coverage(requested_frames: int, observed_frames: Any) -> dict[str, Any]:
    observed = _safe_int(observed_frames)
    if requested_frames == 0:
        return {
            "requested_frames": 0,
            "observed_frames": observed,
            "complete": observed is not None,
            "status": "complete_bounded_capture" if observed is not None else "unknown",
        }
    complete = observed is not None and observed >= requested_frames
    return {
        "requested_frames": requested_frames,
        "observed_frames": observed,
        "complete": complete,
        "status": "complete" if complete else "sample_limited",
    }


def _report_actionable(report: dict[str, Any]) -> bool:
    coverage = report.get("frontend_coverage") or {}
    return bool(
        report.get("ok")
        and coverage.get("complete")
        and ((report.get("transfer") or {}).get("meets_realtime_rate") is True)
        and ((report.get("h_score") or {}).get("gate_verdict") == "pass")
    )


def _hard_bit_balance_error(health: dict[str, Any]) -> float | None:
    ratio = _safe_float(health.get("hard_bit_one_ratio"))
    if ratio is None:
        return None
    return abs(ratio - 0.5)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text in {"true", "false"}:
        return text == "true"
    try:
        if any(marker in text for marker in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


if __name__ == "__main__":
    raise SystemExit(main())
