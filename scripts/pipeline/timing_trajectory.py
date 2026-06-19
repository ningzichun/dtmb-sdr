"""Stage: CI8 -> receiver-consumable timing trajectory JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _common import load_capture_sidecar, write_json

from dtmb.frame_drift import (
    analyze_capture_frame_drift,
    analyze_capture_frame_start_trajectory_body_bias,
    bias_frame_start_trajectory,
)
from dtmb.timing import analyze_capture_timing_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-timing-trajectory")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("gate-c", "pn-drift"),
        default="gate-c",
        help=(
            "Trajectory source. gate-c searches C=3780 body timing with the "
            "system-info classifier; pn-drift tracks the PN header clock and "
            "optionally applies --body-phase-bias."
        ),
    )
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument(
        "--input-skip",
        type=int,
        default=0,
        help="Raw input samples to skip before timing analysis.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help="Complex frequency shift in Hz applied before timing analysis.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Maximum raw input samples to analyze after --input-skip. By "
            "default the stage budgets enough samples for --frames plus "
            "acquisition headroom."
        ),
    )
    parser.add_argument("--system-info-index", type=int, default=23)
    parser.add_argument("--window-frames", type=int, default=48)
    parser.add_argument("--window-step", type=int, default=48)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--radius", type=int, default=512)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--max-phase-step", type=int, default=64)
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=8,
        help="Candidate phases retained per timing window in diagnostics.",
    )
    parser.add_argument(
        "--continuity-candidates",
        type=int,
        default=8,
        help=(
            "Top candidate phases per window always considered for smooth "
            "trajectory search; lower-ranked bridge candidates may also be "
            "retained when --max-phase-step is set."
        ),
    )
    parser.add_argument(
        "--continuity-bridge-candidates",
        type=int,
        help=(
            "Maximum candidate rank eligible as a lower-ranked continuity "
            "bridge. Defaults to twice --continuity-candidates."
        ),
    )
    parser.add_argument(
        "--smoothness-penalty",
        type=float,
        default=0.25,
        help="Score penalty per symbol of adjacent timing motion.",
    )
    parser.add_argument(
        "--per-candidate-cfo",
        action="store_true",
        help=(
            "Estimate PN cyclic-extension CFO independently for every timing "
            "candidate instead of reusing the window-center CFO."
        ),
    )
    parser.add_argument(
        "--body-phase-bias",
        type=int,
        default=0,
        help=(
            "When --source pn-drift is used, shift every PN-derived receiver "
            "phase by this many symbols to probe C=3780 body timing bias."
        ),
    )
    parser.add_argument(
        "--search-body-phase-bias",
        action="store_true",
        help=(
            "When --source pn-drift is used, rank a bounded grid of global "
            "body-phase biases with Gate C metrics and apply the best bias to "
            "the emitted trajectory."
        ),
    )
    parser.add_argument("--body-phase-bias-min", type=int, default=-256)
    parser.add_argument("--body-phase-bias-max", type=int, default=256)
    parser.add_argument("--body-phase-bias-step", type=int, default=4)
    parser.add_argument(
        "--drift-search-radius",
        type=int,
        default=192,
        help="PN cyclic peak search radius in symbols for --source pn-drift.",
    )
    parser.add_argument(
        "--drift-hit-threshold",
        type=float,
        default=0.35,
        help="Minimum PN cyclic metric retained in the PN-drift trajectory.",
    )
    parser.add_argument(
        "--drift-track-selected-phase",
        action="store_true",
        help=(
            "Track around the receiver-selected phase instead of the raw PN "
            "cyclic phase for --source pn-drift."
        ),
    )
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = load_capture_sidecar(args.capture)
    sample_rate = int(sidecar["sample_rate_sps"])
    byte_count = int(sidecar["byte_count"])
    input_skip = max(0, int(args.input_skip))
    remaining_samples = max(0, byte_count // 2 - input_skip)
    symbols_needed = args.frames * 4725 * 2
    samples_needed = int(symbols_needed * sample_rate / 7_560_000)
    max_samples = min(remaining_samples, max(samples_needed, 1_000_000))
    if args.max_samples is not None:
        max_samples = min(remaining_samples, max(1, int(args.max_samples)))
    def _pn_drift_report():
        if args.search_body_phase_bias and int(args.body_phase_bias) != 0:
            raise ValueError(
                "--search-body-phase-bias cannot be combined with a non-zero "
                "--body-phase-bias"
            )
        drift = analyze_capture_frame_drift(
            args.capture,
            sample_rate_sps=sample_rate,
            max_samples=max_samples,
            frequency_shift_hz=float(args.frequency_shift),
            input_skip_samples=input_skip,
            mode="pn945",
            max_frames=args.frames,
            search_radius_symbols=args.drift_search_radius,
            hit_threshold=args.drift_hit_threshold,
            timing_search=False,
            expected_system_info_index=args.system_info_index,
            expected_frame_body_mode="C3780",
            track_selected_phase=args.drift_track_selected_phase,
        )
        trajectory = drift["trajectory"]
        body_phase_bias_search = None
        if args.search_body_phase_bias:
            body_phase_bias_search = analyze_capture_frame_start_trajectory_body_bias(
                args.capture,
                sample_rate_sps=sample_rate,
                max_samples=max_samples,
                frequency_shift_hz=float(args.frequency_shift),
                input_skip_samples=input_skip,
                mode="pn945",
                trajectory=trajectory,
                max_frames=int(args.frames),
                bias_min_symbols=int(args.body_phase_bias_min),
                bias_max_symbols=int(args.body_phase_bias_max),
                bias_step_symbols=int(args.body_phase_bias_step),
                expected_system_info_index=int(args.system_info_index),
                expected_frame_body_mode="C3780",
            )
            selected = (body_phase_bias_search.get("search") or {}).get("selected") or {}
            selected_bias = int(selected.get("body_phase_bias_symbols") or 0)
            if selected_bias:
                trajectory = bias_frame_start_trajectory(
                    trajectory,
                    body_phase_bias_symbols=selected_bias,
                    source="pn-cyclic-drift-plus-body-bias-search",
                )
        elif args.body_phase_bias:
            trajectory = bias_frame_start_trajectory(
                trajectory,
                body_phase_bias_symbols=args.body_phase_bias,
            )
        report = {
            "stage": "timing_trajectory",
            "trajectory_source": "pn-drift",
            "capture_path": str(args.capture),
            "input_sample_rate_sps": int(sample_rate),
            "symbol_rate_sps": int(drift["symbol_rate_sps"]),
            "resample_up": int(drift["resample_up"]),
            "resample_down": int(drift["resample_down"]),
            "frequency_shift_hz": float(args.frequency_shift),
            "input_skip_samples": int(input_skip),
            "analyzed_symbols": int(drift["analyzed_symbols"]),
            "acquisition": drift["acquisition"],
            "drift": drift["drift"],
            "trajectory": trajectory,
            "pn_drift": {
                "tracking_phase_offset": drift["tracking_phase_offset"],
                "tracking_phase_source": drift["tracking_phase_source"],
                "search_radius_symbols": drift["search_radius_symbols"],
                "observation_count": len(drift.get("observations") or []),
                "hit_threshold": float(args.drift_hit_threshold),
                "body_phase_bias_symbols": int(args.body_phase_bias),
                "search_body_phase_bias": bool(args.search_body_phase_bias),
            },
        }
        if body_phase_bias_search is not None:
            report["body_phase_bias_search"] = body_phase_bias_search
        return report

    if args.source == "pn-drift":
        report = _pn_drift_report()
    else:
        report = analyze_capture_timing_trajectory(
            args.capture,
            sample_rate_sps=sample_rate,
            max_samples=max_samples,
            frequency_shift_hz=float(args.frequency_shift),
            input_skip_samples=input_skip,
            mode="pn945",
            window_frames=args.window_frames,
            window_step_frames=args.window_step,
            max_windows=args.max_windows,
            search_radius_symbols=args.radius,
            step_symbols=args.step,
            expected_system_info_index=args.system_info_index,
            expected_frame_body_mode="C3780",
            top_candidates=args.top_candidates,
            continuity_candidate_limit=args.continuity_candidates,
            continuity_bridge_candidate_limit=args.continuity_bridge_candidates,
            smoothness_penalty_per_symbol=args.smoothness_penalty,
            per_candidate_cfo=args.per_candidate_cfo,
            max_phase_step_symbols=args.max_phase_step,
            jobs=args.jobs,
            max_frames=args.frames,
        )
        report["trajectory_source"] = "gate-c"
        if not (report.get("trajectory") or {}).get("available"):
            # gate-c windowed continuity (e.g. no_continuous_path on real
            # multipath) is unusable; fall back to the robust PN-header clock.
            gate_c_reason = (report.get("trajectory") or {}).get("reason")
            report = _pn_drift_report()
            report["trajectory_source"] = "pn-drift-fallback"
            report["gate_c_fallback_reason"] = gate_c_reason
    write_json(args.output, report)
    trajectory = report["trajectory"]
    verdict = "available" if trajectory.get("available") else trajectory.get("reason")
    print(
        "[timing-trajectory] "
        f"{args.source}:{verdict}; segments={len(trajectory.get('segments') or [])} "
        f"phase_span={trajectory.get('phase_span_symbols')} "
        f"max_step={trajectory.get('max_adjacent_phase_step_symbols')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
