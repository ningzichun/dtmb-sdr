"""Stage 2: Gate C system-information classifier summary.

Inputs:
    <capture>.ci8
    <capture>.ci8.json
    <capture>.acquire.json

Outputs:
    <capture>.sysinfo.json  -- full receiver diagnostics JSON, with
                               `system_info` summary driving downstream stages.

The stage runs the offline receiver with `--fec-mode none`, `--uncoded`, and
no symbol deinterleaving so it is cheap: it only needs the PN acquisition +
C=3780 frame body + system-info classification path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from _common import ROOT, child_env, load_capture_sidecar, read_json, write_json
from gate_c import (
    DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS,
    DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
    compact_timing_tracking,
    gate_c_quality,
    resolve_receiver_timing_policy,
    timing_continuity_verdict,
    timing_trajectory_verdict,
)
from dtmb.body_timing_heatmap import analyze_capture_body_timing_heatmap
from dtmb.timing import analyze_capture_timing_windows


BODY_WINDOW_FALLBACK_OFFSET_MIN = -256
BODY_WINDOW_FALLBACK_OFFSET_MAX = 256
BODY_WINDOW_FALLBACK_OFFSET_STEP = 4
BODY_WINDOW_FALLBACK_SEGMENT_FRAMES = 8
BODY_WINDOW_FALLBACK_MAX_SEGMENTS = 4
BODY_WINDOW_FALLBACK_TOP_OFFSETS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-sysinfo")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        default=48,
        help="Number of signal frames to classify (Gate C is noisy enough per-frame).",
    )
    parser.add_argument(
        "--input-skip",
        type=int,
        default=0,
        help="Raw input samples to skip before receiver/sysinfo analysis.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help="Complex frequency shift in Hz applied before receiver analysis.",
    )
    parser.add_argument("--phase-offset", type=int, default=None)
    parser.add_argument("--body-window-offset", type=int, default=0)
    parser.add_argument("--fft-bin-shift", type=int, default=0)
    parser.add_argument(
        "--frequency-deinterleaver-direction",
        choices=("standard", "inverse"),
        default="standard",
    )
    parser.add_argument(
        "--data-carrier-order",
        choices=("normal", "reverse"),
        default="normal",
    )
    parser.add_argument("--carrier-permutation", default="identity")
    parser.add_argument("--logical-position-shift", type=int, default=0)
    parser.set_defaults(body_window_fallback=False)
    parser.add_argument(
        "--body-window-fallback",
        action="store_true",
        dest="body_window_fallback",
        help=(
            "When Gate C fails with the default body window, run a bounded "
            "body-window heatmap and retry the receiver once with the best "
            "aggregate offset."
        ),
    )
    parser.add_argument(
        "--no-body-window-fallback",
        action="store_false",
        dest="body_window_fallback",
        help="Disable the bounded Gate C body-window fallback retry.",
    )
    parser.add_argument(
        "--equalizer",
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
        default="dd",
        help=(
            "Equalizer used to sharpen the system-info constellation. "
            "`dd` is the Gate C default on real RF: decision-directed refinement "
            "produces the cleanest system-info constellations on real captures "
            "even though it is ~20x slower than `sparse`. Use `sparse` for a "
            "cheap screening pass."
        ),
    )
    parser.add_argument(
        "--system-info-min-metric",
        type=float,
        default=0.55,
        help=(
            "Per-frame match threshold. 0.75 is the synthetic-clean default; "
            "real RF through DD equalization typically peaks 0.5-0.7."
        ),
    )
    parser.add_argument(
        "--timing-search",
        action="store_true",
        default=True,
        help=(
            "Run the coarse C=3780 timing search before classifying. Enabled "
            "by default because real RF generally needs it. Pass "
            "--no-timing-search to skip it on known-good captures."
        ),
    )
    parser.add_argument(
        "--no-timing-search",
        dest="timing_search",
        action="store_false",
        help="Disable timing search (cheaper, only use on known-good fixtures).",
    )
    parser.add_argument("--system-info-index", type=int, default=23)
    parser.add_argument(
        "--timing-continuity",
        action="store_true",
        default=True,
        help=(
            "Run a compact windowed timing continuity preflight and record it "
            "in the sysinfo JSON. Enabled by default for real-RF Gate C."
        ),
    )
    parser.add_argument(
        "--no-timing-continuity",
        dest="timing_continuity",
        action="store_false",
        help="Skip the windowed timing-continuity preflight.",
    )
    parser.add_argument("--timing-window-frames", type=int, default=24)
    parser.add_argument("--timing-window-step", type=int, default=24)
    parser.add_argument("--timing-max-windows", type=int, default=4)
    parser.add_argument(
        "--timing-max-phase-span",
        type=int,
        default=DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
    )
    parser.add_argument(
        "--timing-max-adjacent-step",
        type=int,
        default=DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS,
    )
    parser.add_argument(
        "--timing-per-candidate-cfo",
        action="store_true",
        help=(
            "Estimate PN cyclic-extension CFO independently for every timing "
            "candidate in both the receiver timing search and continuity "
            "preflight."
        ),
    )
    parser.add_argument(
        "--timing-policy",
        choices=("fixed", "windowed", "trajectory"),
        default="fixed",
        help=(
            "Frame slicing policy passed through to dtmb.receiver. Use with "
            "--timing-trajectory to run Gate C on windowed/trajectory timing "
            "instead of a single fixed phase."
        ),
    )
    parser.add_argument(
        "--timing-trajectory",
        type=Path,
        help="Optional timing trajectory JSON for windowed/trajectory Gate C.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress dtmb.receiver JSON on stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = load_capture_sidecar(args.capture)
    sample_rate = int(sidecar["sample_rate_sps"])
    byte_count = int(sidecar["byte_count"])
    input_skip = max(0, int(args.input_skip))
    remaining_samples = max(0, byte_count // 2 - input_skip)
    # Budget just enough samples to cover --frames plus acquisition headroom.
    # Each signal frame is 4725 symbols at 7.56 MSps; we add a 2x safety factor
    # and convert to raw sample count at the capture rate.
    symbols_needed = args.frames * 4725 * 2
    samples_needed = int(symbols_needed * sample_rate / 7_560_000)
    max_samples = min(remaining_samples, max(samples_needed, 500_000))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    effective_timing_policy = resolve_receiver_timing_policy(
        args.timing_policy,
        args.timing_trajectory,
    )
    rc = _run_receiver(
        args,
        output=args.output,
        sample_rate_sps=sample_rate,
        max_samples=max_samples,
        input_skip_samples=input_skip,
        timing_policy=effective_timing_policy,
    )
    if rc not in (0, 2):
        return rc
    if not args.output.exists():
        return rc
    data = _load_and_annotate_output(
        args.output,
        capture=args.capture,
        sample_rate_sps=sample_rate,
        max_samples=max_samples,
        args=args,
    )
    data = _maybe_retry_body_window_fallback(
        data,
        capture=args.capture,
        output=args.output,
        sample_rate_sps=sample_rate,
        max_samples=max_samples,
        input_skip_samples=input_skip,
        args=args,
        timing_policy=effective_timing_policy,
    )
    write_json(args.output, data)
    return 0


def _run_receiver(
    args: argparse.Namespace,
    *,
    output: Path,
    sample_rate_sps: int,
    max_samples: int,
    input_skip_samples: int,
    timing_policy: str,
    body_window_offset: int | None = None,
) -> int:
    cmd = _build_receiver_cmd(
        args,
        output=output,
        sample_rate_sps=sample_rate_sps,
        max_samples=max_samples,
        input_skip_samples=input_skip_samples,
        timing_policy=timing_policy,
        body_window_offset=body_window_offset,
    )
    print("[sysinfo]", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(ROOT), env=child_env())


def _build_receiver_cmd(
    args: argparse.Namespace,
    *,
    output: Path,
    sample_rate_sps: int,
    max_samples: int,
    input_skip_samples: int,
    timing_policy: str,
    body_window_offset: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dtmb.receiver",
        str(args.capture),
        "--sample-rate",
        str(sample_rate_sps),
        "--max-samples",
        str(max_samples),
        "--frequency-shift",
        str(args.frequency_shift),
        "--input-skip",
        str(input_skip_samples),
        "--mode",
        "pn945",
        "--qam",
        "64qam",
        "--equalizer",
        args.equalizer,
        "--fec-mode",
        "none",
        "--uncoded",
        "--frames",
        str(args.frames),
        "--system-info-index",
        str(args.system_info_index),
        "--body-window-offset",
        str(
            int(args.body_window_offset)
            if body_window_offset is None
            else int(body_window_offset)
        ),
        "--fft-bin-shift",
        str(args.fft_bin_shift),
        "--frequency-deinterleaver-direction",
        str(args.frequency_deinterleaver_direction),
        "--data-carrier-order",
        str(args.data_carrier_order),
        "--carrier-permutation",
        str(args.carrier_permutation),
        "--logical-position-shift",
        str(args.logical_position_shift),
        "--system-info-min-metric",
        str(args.system_info_min_metric),
        "--min-ts-packets",
        "1",
        "--min-ts-sync-ratio",
        "0.0",
        "--min-ts-valid-ratio",
        "0.0",
        "--json",
        str(output),
    ]
    if args.phase_offset is not None:
        cmd.extend(["--phase-offset", str(args.phase_offset)])
    if args.quiet:
        cmd.append("--quiet")
    cmd.extend(["--timing-policy", timing_policy])
    if args.timing_trajectory is not None:
        cmd.extend(["--timing-trajectory", str(args.timing_trajectory)])
    if not args.timing_search:
        cmd.append("--no-timing-search")
    elif args.timing_trajectory is not None and timing_policy in {"windowed", "trajectory"}:
        cmd.append("--no-timing-search")
    if args.timing_per_candidate_cfo:
        cmd.append("--timing-per-candidate-cfo")
    return cmd


def _load_and_annotate_output(
    output: Path,
    *,
    capture: Path,
    sample_rate_sps: int,
    max_samples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = read_json(output)
    if args.timing_continuity:
        try:
            _annotate_timing_continuity(
                data,
                capture=capture,
                sample_rate_sps=sample_rate_sps,
                max_samples=max_samples,
                args=args,
            )
        except Exception as exc:  # pragma: no cover - pipeline artifact boundary
            data.setdefault("pipeline", {})["timing_continuity"] = {
                "stable": False,
                "verdict": "timing_continuity_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return data


def _maybe_retry_body_window_fallback(
    data: dict[str, Any],
    *,
    capture: Path,
    output: Path,
    sample_rate_sps: int,
    max_samples: int,
    input_skip_samples: int,
    args: argparse.Namespace,
    timing_policy: str,
) -> dict[str, Any]:
    if not bool(getattr(args, "body_window_fallback", True)):
        return data
    if int(getattr(args, "body_window_offset", 0)) != 0:
        return data
    if getattr(args, "timing_trajectory", None) is not None:
        return data
    baseline_quality = gate_c_quality(data)
    if baseline_quality["auto_eligible_frame_count"] > 0:
        return data

    acquisition = data.get("acquisition") or {}
    center_phase_offset = acquisition.get("phase_offset")
    if center_phase_offset is None:
        return data

    heatmap_report = analyze_capture_body_timing_heatmap(
        capture,
        sample_rate_sps=sample_rate_sps,
        max_samples=max_samples,
        frequency_shift_hz=float(getattr(args, "frequency_shift", 0.0)),
        input_skip_samples=input_skip_samples,
        mode="pn945",
        center_phase_offset=int(center_phase_offset),
        segment_frames=min(BODY_WINDOW_FALLBACK_SEGMENT_FRAMES, max(1, int(args.frames))),
        segment_step_frames=min(
            BODY_WINDOW_FALLBACK_SEGMENT_FRAMES,
            max(1, int(args.frames)),
        ),
        max_segments=BODY_WINDOW_FALLBACK_MAX_SEGMENTS,
        offset_min_symbols=BODY_WINDOW_FALLBACK_OFFSET_MIN,
        offset_max_symbols=BODY_WINDOW_FALLBACK_OFFSET_MAX,
        offset_step_symbols=BODY_WINDOW_FALLBACK_OFFSET_STEP,
        expected_system_info_index=int(args.system_info_index),
        expected_frame_body_mode="C3780",
        estimate_cfo=True,
        per_candidate_cfo=bool(args.timing_per_candidate_cfo),
        rank_policy="expected",
        top_offsets=BODY_WINDOW_FALLBACK_TOP_OFFSETS,
    )
    ranked_offsets = _rank_heatmap_offsets(heatmap_report)
    fallback_info: dict[str, Any] = {
        "attempted": True,
        "accepted": False,
        "selected_offset_symbols": None,
        "baseline_quality": baseline_quality,
        "heatmap_summary": (heatmap_report.get("heatmap") or {}).get("summary") or {},
        "heatmap_top_offsets": ranked_offsets[:BODY_WINDOW_FALLBACK_TOP_OFFSETS],
    }
    heatmap_artifact = output.with_name(f"{output.stem}.body_window_fallback.json")
    write_json(heatmap_artifact, heatmap_report)
    fallback_info["heatmap_artifact"] = str(heatmap_artifact)
    if not ranked_offsets:
        fallback_info["reason"] = "no_heatmap_offsets"
        data.setdefault("pipeline", {})["body_window_fallback"] = fallback_info
        return data
    selected_offset = int(ranked_offsets[0]["body_window_offset_symbols"])
    fallback_info["selected_offset_symbols"] = selected_offset
    if selected_offset == 0:
        fallback_info["reason"] = "heatmap_selected_zero_offset"
        data.setdefault("pipeline", {})["body_window_fallback"] = fallback_info
        return data

    retry_output = output.with_name(f"{output.stem}.body_window_retry.json")
    rc = _run_receiver(
        args,
        output=retry_output,
        sample_rate_sps=sample_rate_sps,
        max_samples=max_samples,
        input_skip_samples=input_skip_samples,
        timing_policy=timing_policy,
        body_window_offset=selected_offset,
    )
    if rc not in (0, 2) or not retry_output.exists():
        fallback_info["reason"] = "retry_run_failed"
        fallback_info["retry_rc"] = int(rc)
        data.setdefault("pipeline", {})["body_window_fallback"] = fallback_info
        return data

    retry_data = _load_and_annotate_output(
        retry_output,
        capture=capture,
        sample_rate_sps=sample_rate_sps,
        max_samples=max_samples,
        args=args,
    )
    retry_quality = gate_c_quality(retry_data)
    fallback_info["retry_quality"] = retry_quality
    baseline_key = _gate_c_quality_sort_key(baseline_quality)
    retry_key = _gate_c_quality_sort_key(retry_quality)
    if retry_key > baseline_key:
        fallback_info["accepted"] = True
        fallback_info["reason"] = "retry_quality_improved"
        retry_data.setdefault("pipeline", {})["body_window_fallback"] = fallback_info
        return retry_data

    fallback_info["reason"] = "retry_quality_not_better"
    data.setdefault("pipeline", {})["body_window_fallback"] = fallback_info
    return data


def _rank_heatmap_offsets(report: dict[str, Any]) -> list[dict[str, Any]]:
    heatmap = report.get("heatmap") if isinstance(report, dict) else None
    segments = heatmap.get("segments") if isinstance(heatmap, dict) else None
    if not isinstance(segments, list):
        return []
    aggregated: dict[int, dict[str, Any]] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        selected = segment.get("selected") or {}
        selected_offset = selected.get("body_window_offset_symbols")
        for row in segment.get("offset_scores") or []:
            if not isinstance(row, dict):
                continue
            offset = int(row.get("body_window_offset_symbols") or 0)
            aggregate = aggregated.setdefault(
                offset,
                {
                    "body_window_offset_symbols": offset,
                    "segment_wins": 0,
                    "segments_scored": 0,
                    "frame_count": 0,
                    "expected_top_count": 0,
                    "expected_top3_count": 0,
                    "score_sum": 0.0,
                    "best_expected_metric": 0.0,
                    "median_expected_metric_sum": 0.0,
                },
            )
            aggregate["segments_scored"] += 1
            aggregate["frame_count"] += int(row.get("frame_count") or 0)
            aggregate["expected_top_count"] += int(row.get("expected_top_count") or 0)
            aggregate["expected_top3_count"] += int(
                row.get("expected_top3_count") or 0
            )
            aggregate["score_sum"] += float(row.get("score") or 0.0)
            aggregate["best_expected_metric"] = max(
                float(aggregate["best_expected_metric"]),
                float(row.get("best_expected_metric") or 0.0),
            )
            aggregate["median_expected_metric_sum"] += float(
                row.get("median_expected_metric") or 0.0
            )
            if selected_offset is not None and int(selected_offset) == offset:
                aggregate["segment_wins"] += 1
    ranked = list(aggregated.values())
    for row in ranked:
        segments_scored = max(1, int(row["segments_scored"]))
        row["mean_median_expected_metric"] = float(
            row["median_expected_metric_sum"] / segments_scored
        )
        del row["median_expected_metric_sum"]
    ranked.sort(
        key=lambda row: (
            int(row["expected_top_count"]),
            int(row["expected_top3_count"]),
            int(row["segment_wins"]),
            float(row["score_sum"]),
            float(row["best_expected_metric"]),
            -abs(int(row["body_window_offset_symbols"])),
        ),
        reverse=True,
    )
    return ranked


def _gate_c_quality_sort_key(quality: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(int(quality.get("auto_eligible_frame_count", 0) > 0)),
        float(quality.get("auto_eligible_frame_count") or 0),
        float(int(bool(quality.get("timing_continuity_stable", True)))),
        float(quality.get("qualified_frame_count") or 0),
        float(quality.get("agreement_ratio") or 0.0),
        float(quality.get("best_metric") or 0.0),
        float(quality.get("mean_metric") or 0.0),
    )


def _annotate_timing_continuity(
    data: dict,
    *,
    capture: Path,
    sample_rate_sps: int,
    max_samples: int,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "timing_trajectory", None) is not None:
        report = read_json(args.timing_trajectory)
        requested_timing_policy = getattr(args, "timing_policy", "fixed")
        receiver_timing_policy = resolve_receiver_timing_policy(
            requested_timing_policy,
            args.timing_trajectory,
        )
        verdict = timing_trajectory_verdict(
            report,
            max_phase_span_symbols=int(args.timing_max_phase_span),
            max_adjacent_phase_step_symbols=int(args.timing_max_adjacent_step),
        )
        data.setdefault("pipeline", {})["timing_continuity"] = {
            **verdict,
            "requested_timing_policy": requested_timing_policy,
            "receiver_timing_policy": receiver_timing_policy,
            "tracking": {
                "source": "timing_trajectory",
                "trajectory": _compact_timing_trajectory_report(report),
            },
        }
        return

    acquisition = data.get("acquisition") or {}
    phase = acquisition.get("phase_offset")
    center_phase = int(phase) if phase is not None else None
    window_frames = min(max(1, int(args.timing_window_frames)), max(1, args.frames))
    window_step = min(max(1, int(args.timing_window_step)), max(1, args.frames))
    diagnostics = analyze_capture_timing_windows(
        capture,
        sample_rate_sps=sample_rate_sps,
        max_samples=max_samples,
        frequency_shift_hz=float(getattr(args, "frequency_shift", 0.0)),
        input_skip_samples=max(0, int(getattr(args, "input_skip", 0))),
        mode="pn945",
        center_phase_offset=center_phase,
        window_frames=window_frames,
        window_step_frames=window_step,
        max_windows=max(1, int(args.timing_max_windows)),
        search_radius_symbols=512,
        step_symbols=4,
        expected_system_info_index=int(args.system_info_index),
        expected_frame_body_mode="C3780",
        top_candidates=4,
        continuity_tracking=True,
        continuity_candidate_limit=8,
        per_candidate_cfo=bool(args.timing_per_candidate_cfo),
        max_phase_step_symbols=int(args.timing_max_adjacent_step),
    )
    tracking = diagnostics.get("tracking") or diagnostics
    verdict = timing_continuity_verdict(
        tracking,
        max_phase_span_symbols=int(args.timing_max_phase_span),
        max_adjacent_phase_step_symbols=int(args.timing_max_adjacent_step),
    )
    data.setdefault("pipeline", {})["timing_continuity"] = {
        **verdict,
        "tracking": compact_timing_tracking(tracking),
    }


def _compact_timing_trajectory_report(report: dict) -> dict:
    trajectory = report.get("trajectory") if isinstance(report, dict) else None
    if not isinstance(trajectory, dict):
        trajectory = report if isinstance(report, dict) else {}
    return {
        "available": trajectory.get("available"),
        "reason": trajectory.get("reason"),
        "source": trajectory.get("source"),
        "policy": trajectory.get("policy"),
        "body_phase_bias_symbols": trajectory.get("body_phase_bias_symbols"),
        "phase_offsets": trajectory.get("phase_offsets"),
        "phase_span_symbols": trajectory.get("phase_span_symbols"),
        "max_adjacent_phase_step_symbols": trajectory.get(
            "max_adjacent_phase_step_symbols"
        ),
        "segment_count": len(trajectory.get("segments") or []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
