"""Real-RF video promotion verdict for the DTMB pipeline.

This stage consumes existing stage artifacts and answers one question: is this
capture ready to spend CPU on real video recovery, or did an earlier gate
already prove it is blocked?

The script intentionally does not run DSP itself. Make targets build the
artifacts first:

    acquire -> clipping -> sysinfo -> demap -> llr-health

Then this script writes one compact JSON verdict. A blocked verdict is still a
successful command exit by default so batch runs can continue across channels.
Pass ``--fail-on-blocked`` when a caller wants a non-zero exit for CI-style
promotion checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from _common import read_json, write_json
from gate_c import (
    DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS,
    DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
    gate_c_quality,
    sysinfo_oracle_quality,
)


GATE_A_PASS_THRESHOLD = 0.45
GATE_C_PASS_THRESHOLD = 0.75
LDPC_CONSISTENT_THRESHOLD = 0.05
LDPC_RANDOM_THRESHOLD = 0.30
QAM_MAX_HARD_BIT_BIAS_THRESHOLD = 0.10
QAM_MAX_AXIS_FRACTION_ERROR_THRESHOLD = 0.10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-video-preflight")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--acquire-json", type=Path)
    parser.add_argument("--clipping-json", type=Path)
    parser.add_argument("--sysinfo-json", type=Path)
    parser.add_argument("--sysinfo-oracle-json", type=Path)
    parser.add_argument("--timing-trajectory-json", type=Path)
    parser.add_argument("--channel-axis-json", type=Path)
    parser.add_argument("--demap-json", type=Path)
    parser.add_argument("--llr-health-json", type=Path)
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--gate-a-threshold", type=float, default=GATE_A_PASS_THRESHOLD)
    parser.add_argument("--gate-c-threshold", type=float, default=GATE_C_PASS_THRESHOLD)
    parser.add_argument(
        "--ldpc-consistent-threshold",
        type=float,
        default=LDPC_CONSISTENT_THRESHOLD,
    )
    parser.add_argument(
        "--ldpc-random-threshold",
        type=float,
        default=LDPC_RANDOM_THRESHOLD,
    )
    parser.add_argument(
        "--sysinfo-oracle-near-clean-threshold",
        type=float,
        default=0.95,
        help=(
            "Required near-clean raw system-information frame ratio when an "
            "oracle JSON is present."
        ),
    )
    parser.add_argument(
        "--qam-max-hard-bit-bias",
        type=float,
        default=QAM_MAX_HARD_BIT_BIAS_THRESHOLD,
        help=(
            "Maximum allowed absolute hard-bit-plane bias in the demap QAM "
            "quality report before blocking LDPC promotion."
        ),
    )
    parser.add_argument(
        "--qam-max-axis-fraction-error",
        type=float,
        default=QAM_MAX_AXIS_FRACTION_ERROR_THRESHOLD,
        help=(
            "Maximum allowed 64QAM axis-level occupancy error in the demap "
            "QAM quality report before blocking LDPC promotion."
        ),
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit 2 unless the capture is promoted for video work.",
    )
    return parser


def evaluate_video_preflight(
    *,
    capture: str | None = None,
    acquire: dict[str, Any] | None = None,
    clipping: dict[str, Any] | None = None,
    sysinfo: dict[str, Any] | None = None,
    sysinfo_oracle: dict[str, Any] | None = None,
    timing_trajectory: dict[str, Any] | None = None,
    channel_axis: dict[str, Any] | None = None,
    demap: dict[str, Any] | None = None,
    llr_health: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    gate_a_threshold: float = GATE_A_PASS_THRESHOLD,
    gate_c_threshold: float = GATE_C_PASS_THRESHOLD,
    ldpc_consistent_threshold: float = LDPC_CONSISTENT_THRESHOLD,
    ldpc_random_threshold: float = LDPC_RANDOM_THRESHOLD,
    sysinfo_oracle_near_clean_threshold: float = 0.95,
    qam_max_hard_bit_bias: float = QAM_MAX_HARD_BIT_BIAS_THRESHOLD,
    qam_max_axis_fraction_error: float = QAM_MAX_AXIS_FRACTION_ERROR_THRESHOLD,
) -> dict[str, Any]:
    """Return a compact promotion verdict from pipeline stage dictionaries."""

    gates: dict[str, Any] = {}
    thresholds = {
        "gate_a_pn945_metric": float(gate_a_threshold),
        "gate_c_expected_best_metric": float(gate_c_threshold),
        "ldpc_consistent_mean_ratio": float(ldpc_consistent_threshold),
        "ldpc_random_like_mean_ratio": float(ldpc_random_threshold),
        "sysinfo_oracle_near_clean_ratio": float(
            sysinfo_oracle_near_clean_threshold
        ),
        "qam_max_hard_bit_bias": float(qam_max_hard_bit_bias),
        "qam_max_axis_fraction_error": float(qam_max_axis_fraction_error),
    }

    clipping_ok = True
    if clipping is None:
        gates["clipping"] = {"status": "missing", "ok": None}
    else:
        clipping_ok = bool(clipping.get("ok", True))
        gates["clipping"] = {
            "status": "pass" if clipping_ok else "fail",
            "ok": clipping_ok,
            "component_rms_dbfs": clipping.get("component_rms_dbfs"),
            "clip_ratio_i": clipping.get("clip_ratio_i"),
            "clip_ratio_q": clipping.get("clip_ratio_q"),
            "violations": clipping.get("violations") or [],
        }

    pn945_metric = _best_pn945_metric(acquire or {})
    gate_a_ok = acquire is not None and pn945_metric >= gate_a_threshold
    gates["gate_a"] = {
        "status": "pass" if gate_a_ok else ("fail" if acquire is not None else "missing"),
        "pn945_best_metric": pn945_metric if acquire is not None else None,
        "threshold": float(gate_a_threshold),
    }

    gate_c_ok = False
    gate_c_metric_ok = False
    trajectory_quality = _timing_trajectory_quality(timing_trajectory)
    gates["timing_trajectory"] = trajectory_quality
    oracle_quality = sysinfo_oracle_quality(
        sysinfo_oracle,
        near_clean_threshold=sysinfo_oracle_near_clean_threshold,
    )
    gates["sysinfo_oracle"] = oracle_quality
    if sysinfo is None:
        gates["gate_c"] = {"status": "missing"}
    else:
        quality = gate_c_quality(sysinfo)
        timing = (sysinfo.get("pipeline") or {}).get("timing_continuity") or {}
        timing_present = "stable" in timing
        timing_source = "sysinfo"
        if trajectory_quality["status"] != "missing":
            timing_present = True
            timing_source = "timing_trajectory"
            quality["timing_continuity_stable"] = bool(
                trajectory_quality["stable"]
            )
            quality["timing_continuity_verdict"] = trajectory_quality["verdict"]
            quality["timing_phase_span_symbols"] = int(
                trajectory_quality["phase_span_symbols"] or 0
            )
            quality["timing_max_adjacent_phase_step_symbols"] = int(
                trajectory_quality["max_adjacent_phase_step_symbols"] or 0
            )
        elif not timing_present:
            quality["timing_continuity_stable"] = False
            quality["timing_continuity_verdict"] = "missing_timing_continuity"
        gate_c_metric_ok = (
            quality["auto_eligible_frame_count"] > 0
            and quality["best_metric"] >= gate_c_threshold
            and quality["timing_continuity_stable"]
        )
        gate_c_ok = gate_c_metric_ok and oracle_quality["blocking"] is not True
        gates["gate_c"] = {
            "status": "pass" if gate_c_ok else "fail",
            "metric_status": "pass" if gate_c_metric_ok else "fail",
            "oracle_blocked": oracle_quality["blocking"] is True,
            "threshold": float(gate_c_threshold),
            "timing_continuity_present": bool(timing_present),
            "timing_source": timing_source,
            **quality,
        }

    channel_consistency = _channel_axis_quality(channel_axis)
    gates["channel_consistency"] = channel_consistency

    qam_shape = _qam_shape_quality(
        demap,
        max_hard_bit_bias=qam_max_hard_bit_bias,
        max_axis_fraction_error=qam_max_axis_fraction_error,
    )
    gates["qam_shape"] = qam_shape

    ldpc_health = _ldpc_health_quality(
        llr_health,
        demap,
        ldpc_consistent_threshold=ldpc_consistent_threshold,
        ldpc_random_threshold=ldpc_random_threshold,
    )
    ldpc_ok = bool(ldpc_health.get("ok"))
    gates["ldpc_health"] = ldpc_health

    probe_ok = bool((probe or {}).get("ok"))
    if probe is None:
        gates["probe"] = {"status": "missing"}
    else:
        gates["probe"] = {
            "status": "pass" if probe_ok else "fail",
            "ok": probe_ok,
            "reason": probe.get("reason"),
            "input": probe.get("input"),
        }

    verdict, next_step = _verdict_and_next_step(
        clipping_ok=clipping_ok,
        acquire_present=acquire is not None,
        gate_a_ok=gate_a_ok,
        sysinfo_present=sysinfo is not None,
        sysinfo_oracle_failed=(
            gate_c_metric_ok and oracle_quality["blocking"] is True
        ),
        gate_c_ok=gate_c_ok,
        channel_consistency_failed=channel_consistency["status"] == "fail",
        qam_shape_failed=qam_shape["status"] == "fail",
        ldpc_health_present=ldpc_health["status"] not in {"missing", "skipped"},
        ldpc_ok=ldpc_ok,
        probe_present=probe is not None,
        probe_ok=probe_ok,
        ldpc_status=gates["ldpc_health"]["status"],
    )

    ready_for_receive = gate_a_ok and gate_c_ok
    ready_for_video_attempt = ready_for_receive and ldpc_ok
    ready_for_video_stream = probe_ok
    return {
        "stage": "video_preflight",
        "capture": capture,
        "inputs": inputs or {},
        "thresholds": thresholds,
        "gates": gates,
        "ready_for_receive": bool(ready_for_receive),
        "ready_for_video_attempt": bool(ready_for_video_attempt),
        "ready_for_video_stream": bool(ready_for_video_stream),
        "verdict": verdict,
        "next": next_step,
    }


def _verdict_and_next_step(
    *,
    clipping_ok: bool,
    acquire_present: bool,
    gate_a_ok: bool,
    sysinfo_present: bool,
    sysinfo_oracle_failed: bool,
    gate_c_ok: bool,
    channel_consistency_failed: bool,
    qam_shape_failed: bool,
    ldpc_health_present: bool,
    ldpc_ok: bool,
    probe_present: bool,
    probe_ok: bool,
    ldpc_status: str,
) -> tuple[str, str]:
    if not clipping_ok:
        return (
            "unsafe_clipping",
            "lower HackRF gains and recapture before running receiver stages",
        )
    if probe_present and probe_ok:
        return ("video_verified", "use the same mode path for live-stream or UDP preview")
    if not acquire_present:
        return ("missing_acquire", "run make -f pipeline.mk acquire CAPTURE=<capture>")
    if not gate_a_ok:
        return (
            "gate_a_failed_no_dtmb_signal",
            "scan or tune another RF center before trying system information",
        )
    if not sysinfo_present:
        return ("missing_sysinfo", "run make -f pipeline.mk sysinfo CAPTURE=<capture>")
    if sysinfo_oracle_failed:
        return (
            "gate_c_oracle_failed_no_ldpc",
            "run independent raw SI oracle across deinterleaver latency before LDPC",
        )
    if not gate_c_ok:
        return (
            "gate_c_failed_no_ldpc",
            "fix timing/CFO/channel tracking or acquire a capture with stable Gate C",
        )
    if ldpc_health_present and ldpc_status not in {"missing", "skipped"}:
        if not ldpc_ok:
            return (
                "ldpc_health_failed_no_video",
                "keep real-RF TS receive blocked until LDPC health is below random-like mismatch",
            )
    elif channel_consistency_failed:
        return (
            "channel_consistency_failed_before_ldpc_health",
            "resolve SI-pilot/data-channel conflict before spending CPU on LDPC or TS recovery",
        )
    elif qam_shape_failed:
        return (
            "qam_shape_failed_before_ldpc_health",
            "fix channel/equalizer output or run llr-health before spending CPU on full LDPC or ffmpeg",
        )
    else:
        return (
            "gate_c_ok_run_llr_health",
            "run make -f pipeline.mk llr-health CAPTURE=<capture>",
        )
    return (
        "ldpc_health_ok_run_receive",
        "run make -f pipeline.mk receive probe CAPTURE=<capture>",
    )


def _best_pn945_metric(acquire_json: dict[str, Any]) -> float:
    pn_search = acquire_json.get("pn_search") or {}
    best = 0.0
    for group in ("cyclic_extension_trains", "phase_family_trains", "frame_trains"):
        for item in pn_search.get(group) or []:
            if item.get("mode") == "pn945":
                best = max(best, float(item.get("max_metric") or 0.0))
    return best


def _qam_shape_quality(
    demap: dict[str, Any] | None,
    *,
    max_hard_bit_bias: float,
    max_axis_fraction_error: float,
) -> dict[str, Any]:
    """Summarise whether demapped QAM symbols look plausible before LDPC."""

    if demap is None:
        return {"status": "missing"}
    if (demap.get("pipeline") or {}).get("skipped"):
        return {
            "status": "skipped",
            "skip_reason": (demap.get("pipeline") or {}).get("skip_reason"),
        }
    quality_root = demap.get("qam_symbol_quality")
    quality = None
    if isinstance(quality_root, dict):
        after = quality_root.get("after_symbol_deinterleave")
        before = quality_root.get("before_symbol_deinterleave")
        if isinstance(after, dict):
            quality = after
        elif isinstance(before, dict):
            quality = before
        else:
            quality = quality_root
    if not isinstance(quality, dict):
        return {"status": "missing", "reason": "missing_qam_symbol_quality"}

    axis = quality.get("axis_level_occupancy")
    axis_error = None
    inner_fraction_mean = None
    if isinstance(axis, dict):
        axis_error = _optional_float(axis.get("max_abs_fraction_error_from_uniform"))
        i_axis = axis.get("i") if isinstance(axis.get("i"), dict) else {}
        q_axis = axis.get("q") if isinstance(axis.get("q"), dict) else {}
        inner_values = [
            value
            for value in (
                _optional_float(i_axis.get("inner_fraction")),
                _optional_float(q_axis.get("inner_fraction")),
            )
            if value is not None
        ]
        if inner_values:
            inner_fraction_mean = float(sum(inner_values) / len(inner_values))

    max_bias = _optional_float(quality.get("max_abs_hard_bit_bias"))
    grid_evm = _optional_float(quality.get("grid_evm_rms"))
    violations = []
    if max_bias is not None and max_bias > float(max_hard_bit_bias):
        violations.append("hard_bit_bias")
    if axis_error is not None and axis_error > float(max_axis_fraction_error):
        violations.append("axis_level_occupancy")

    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "grid_evm_rms": grid_evm,
        "max_abs_hard_bit_bias": max_bias,
        "max_hard_bit_bias": float(max_hard_bit_bias),
        "axis_max_abs_fraction_error_from_uniform": axis_error,
        "max_axis_fraction_error": float(max_axis_fraction_error),
        "axis_inner_fraction_mean": inner_fraction_mean,
    }


def _ldpc_health_quality(
    llr_health: dict[str, Any] | None,
    demap: dict[str, Any] | None,
    *,
    ldpc_consistent_threshold: float,
    ldpc_random_threshold: float,
) -> dict[str, Any]:
    """Summarise the strongest available LDPC parity/syndrome evidence."""

    source = "llr_health"
    report = llr_health
    parity = (llr_health or {}).get("ldpc_parity_check") if llr_health else None
    if llr_health is None:
        source = "demap"
        report = demap
        parity = _demap_ldpc_parity_check(demap)

    if report is None:
        return {"status": "missing", "source": None, "ok": False}
    if (report.get("pipeline") or {}).get("skipped") or report.get("skipped"):
        return {
            "status": "skipped",
            "source": source,
            "ok": False,
            "verdict": report.get("verdict"),
            "skip_reason": report.get("skip_reason")
            or (report.get("pipeline") or {}).get("skip_reason"),
            "gate_c_note": report.get("gate_c_note")
            or (report.get("pipeline") or {}).get("gate_c_note"),
        }
    if not isinstance(parity, dict):
        return {
            "status": "missing",
            "source": source,
            "ok": False,
            "reason": "missing_ldpc_parity_check",
        }

    appendix = parity.get("appendix_b_syndrome") or {}
    generator_mean = parity.get("mean_mismatch_ratio")
    syndrome_mean = appendix.get("mean_syndrome_ratio")
    codewords = int(parity.get("codewords") or 0)
    ok = (
        codewords > 0
        and generator_mean is not None
        and syndrome_mean is not None
        and float(generator_mean) <= float(ldpc_consistent_threshold)
        and float(syndrome_mean) <= float(ldpc_consistent_threshold)
    )
    random_like = (
        generator_mean is not None
        and syndrome_mean is not None
        and float(generator_mean) >= float(ldpc_random_threshold)
        and float(syndrome_mean) >= float(ldpc_random_threshold)
    )
    return {
        "status": "pass" if ok else "fail",
        "source": source,
        "ok": bool(ok),
        "verdict": report.get("verdict")
        or (report.get("fec") or {}).get("mode")
        or "embedded_ldpc_parity_check",
        "codewords": codewords,
        "mean_mismatch_ratio": generator_mean,
        "mean_syndrome_ratio": syndrome_mean,
        "zero_mismatch_codewords": parity.get("zero_mismatch_codewords"),
        "zero_syndrome_codewords": appendix.get("zero_syndrome_codewords"),
        "random_like": bool(random_like),
    }


def _demap_ldpc_parity_check(demap: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(demap, dict):
        return None
    parity = demap.get("ldpc_parity_check")
    if isinstance(parity, dict):
        return parity
    fec = demap.get("fec")
    if isinstance(fec, dict) and isinstance(fec.get("ldpc_parity_check"), dict):
        return fec["ldpc_parity_check"]
    return None


def _channel_axis_quality(report: dict[str, Any] | None) -> dict[str, Any]:
    """Summarise optional PN/sparse/DD channel-axis consistency evidence."""

    if report is None:
        return {"status": "missing"}
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        return {"status": "fail", "reason": "missing_summary"}

    interpretation = str(summary.get("interpretation") or "unknown")
    overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
    pilot_vs_data = summary.get("pilot_vs_data")
    pilot = {}
    data = {}
    if isinstance(pilot_vs_data, dict):
        if isinstance(pilot_vs_data.get("pilot"), dict):
            pilot = pilot_vs_data["pilot"]
        if isinstance(pilot_vs_data.get("data"), dict):
            data = pilot_vs_data["data"]
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    ok = interpretation == "pn_and_equalizer_channel_responses_agree"
    return {
        "status": "pass" if ok else "fail",
        "ok": bool(ok),
        "interpretation": interpretation,
        "reason": None if ok else interpretation,
        "reference": metadata.get("reference"),
        "equalizer": metadata.get("equalizer"),
        "overall_relative_rms": _optional_float(overall.get("relative_rms")),
        "pilot_relative_rms": _optional_float(pilot.get("relative_rms")),
        "data_relative_rms": _optional_float(data.get("relative_rms")),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timing_trajectory_quality(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    if report is None:
        return {
            "status": "missing",
            "stable": None,
            "verdict": None,
            "available": None,
            "reason": None,
            "phase_span_symbols": None,
            "max_adjacent_phase_step_symbols": None,
            "segment_count": 0,
        }
    trajectory = report.get("trajectory") if isinstance(report, dict) else None
    if not isinstance(trajectory, dict):
        trajectory = report if isinstance(report, dict) else {}
    available = bool(trajectory.get("available"))
    phase_span = int(trajectory.get("phase_span_symbols") or 0)
    max_adjacent = int(trajectory.get("max_adjacent_phase_step_symbols") or 0)
    stable = (
        available
        and phase_span <= DEFAULT_MAX_PHASE_SPAN_SYMBOLS
        and max_adjacent <= DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS
    )
    if stable:
        verdict = "stable"
    elif not available:
        verdict = str(trajectory.get("reason") or "trajectory_unavailable")
    else:
        verdict = "unstable_phase_jumps"
    return {
        "status": "pass" if stable else "fail",
        "stable": bool(stable),
        "verdict": verdict,
        "available": available,
        "reason": trajectory.get("reason"),
        "phase_span_symbols": phase_span,
        "max_adjacent_phase_step_symbols": max_adjacent,
        "max_phase_span_symbols": DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
        "max_allowed_adjacent_phase_step_symbols": (
            DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS
        ),
        "segment_count": len(trajectory.get("segments") or []),
        "source": trajectory.get("source"),
    }


def _default_paths(capture: Path) -> dict[str, Path]:
    return {
        "acquire": capture.with_suffix(".acquire.json"),
        "clipping": capture.with_suffix(".clipping.json"),
        "sysinfo": capture.with_suffix(".sysinfo.json"),
        "sysinfo_oracle": capture.with_suffix(".sysinfo_oracle.raw.json"),
        "timing_trajectory": capture.with_suffix(".timing_trajectory.json"),
        "channel_axis": capture.with_suffix(".channel_axis.json"),
        "demap": capture.with_suffix(".demap.json"),
        "llr_health": capture.with_suffix(".llr_health.json"),
        "probe": capture.with_suffix(".probe.json"),
    }


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = _default_paths(args.capture)
    paths = {
        "acquire": args.acquire_json or defaults["acquire"],
        "clipping": args.clipping_json or defaults["clipping"],
        "sysinfo": args.sysinfo_json or defaults["sysinfo"],
        "sysinfo_oracle": args.sysinfo_oracle_json or defaults["sysinfo_oracle"],
        "timing_trajectory": args.timing_trajectory_json
        or defaults["timing_trajectory"],
        "channel_axis": args.channel_axis_json or defaults["channel_axis"],
        "demap": args.demap_json or defaults["demap"],
        "llr_health": args.llr_health_json or defaults["llr_health"],
        "probe": args.probe_json or defaults["probe"],
    }
    inputs = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in paths.items()
    }
    report = evaluate_video_preflight(
        capture=str(args.capture),
        acquire=_read_optional(paths["acquire"]),
        clipping=_read_optional(paths["clipping"]),
        sysinfo=_read_optional(paths["sysinfo"]),
        sysinfo_oracle=_read_optional(paths["sysinfo_oracle"]),
        timing_trajectory=_read_optional(paths["timing_trajectory"]),
        channel_axis=_read_optional(paths["channel_axis"]),
        demap=_read_optional(paths["demap"]),
        llr_health=_read_optional(paths["llr_health"]),
        probe=_read_optional(paths["probe"]),
        inputs=inputs,
        gate_a_threshold=args.gate_a_threshold,
        gate_c_threshold=args.gate_c_threshold,
        ldpc_consistent_threshold=args.ldpc_consistent_threshold,
        ldpc_random_threshold=args.ldpc_random_threshold,
        sysinfo_oracle_near_clean_threshold=(
            args.sysinfo_oracle_near_clean_threshold
        ),
        qam_max_hard_bit_bias=args.qam_max_hard_bit_bias,
        qam_max_axis_fraction_error=args.qam_max_axis_fraction_error,
    )
    write_json(args.output_json, report)
    print(f"[video-preflight] {report['verdict']}: {report['next']}")
    if args.fail_on_blocked and not (
        report["ready_for_video_attempt"] or report["ready_for_video_stream"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
