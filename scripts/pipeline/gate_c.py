"""Shared Gate C quality helpers for pipeline stages."""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_PHASE_SPAN_SYMBOLS = 128
DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS = 64


def resolve_receiver_timing_policy(
    timing_policy: str,
    timing_trajectory: Any,
) -> str:
    """Return the receiver timing policy implied by pipeline arguments.

    Supplying a trajectory artifact is an explicit request to consume it.
    Keeping the receiver at the CLI default ``fixed`` while annotating Gate C
    from that trajectory makes the artifact self-contradictory, so pipeline
    wrappers promote the effective policy to ``trajectory`` unless the caller
    explicitly requested ``windowed``.
    """

    policy = str(timing_policy)
    if policy not in {"fixed", "windowed", "trajectory"}:
        raise ValueError("timing_policy must be fixed, windowed, or trajectory")
    if timing_trajectory is not None and policy == "fixed":
        return "trajectory"
    return policy


def timing_continuity_verdict(
    tracking: dict[str, Any],
    *,
    max_phase_span_symbols: int = DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
    max_adjacent_phase_step_symbols: int = DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS,
) -> dict[str, Any]:
    """Classify whether windowed timing stayed continuous enough for Gate C."""

    summary = tracking.get("summary") or {}
    trajectory = tracking.get("trajectory")
    if isinstance(trajectory, dict) and trajectory.get("available"):
        metric_source = "trajectory"
        trajectory_phases = trajectory.get("phase_offsets") or []
        selected_window_count = len(trajectory_phases)
        phase_span = _trajectory_metric(
            trajectory,
            "phase_span_symbols",
            trajectory_phases,
            metric="span",
        )
        max_adjacent = _trajectory_metric(
            trajectory,
            "max_adjacent_phase_step_symbols",
            trajectory_phases,
            metric="max_adjacent",
        )
    else:
        metric_source = "selected"
        selected_window_count = int(summary.get("selected_window_count") or 0)
        phase_span = int(summary.get("phase_span_symbols") or 0)
        max_adjacent = int(summary.get("max_adjacent_phase_step_symbols") or 0)
    window_count = int(tracking.get("window_count") or 0)
    expected_top_ratio = float(summary.get("expected_top_ratio") or 0.0)
    expected_top3_ratio = float(summary.get("expected_top3_ratio") or 0.0)
    sufficient_windows = window_count >= 2 and selected_window_count >= 2
    stable = False
    verdict = "insufficient_windows"
    if sufficient_windows:
        stable = (
            phase_span <= max_phase_span_symbols
            and max_adjacent <= max_adjacent_phase_step_symbols
        )
        verdict = "stable" if stable else "unstable_phase_jumps"
    return {
        "stable": bool(stable),
        "verdict": verdict,
        "window_count": window_count,
        "selected_window_count": selected_window_count,
        "phase_span_symbols": phase_span,
        "max_adjacent_phase_step_symbols": max_adjacent,
        "max_phase_span_symbols": int(max_phase_span_symbols),
        "max_allowed_adjacent_phase_step_symbols": int(
            max_adjacent_phase_step_symbols
        ),
        "metric_source": metric_source,
        "expected_top_ratio": expected_top_ratio,
        "expected_top3_ratio": expected_top3_ratio,
    }


def timing_trajectory_verdict(
    report: dict[str, Any],
    *,
    max_phase_span_symbols: int = DEFAULT_MAX_PHASE_SPAN_SYMBOLS,
    max_adjacent_phase_step_symbols: int = DEFAULT_MAX_ADJACENT_PHASE_STEP_SYMBOLS,
) -> dict[str, Any]:
    """Classify an explicit receiver trajectory as the Gate C timing source."""

    trajectory = report.get("trajectory") if isinstance(report, dict) else None
    if not isinstance(trajectory, dict):
        trajectory = report if isinstance(report, dict) else {}
    phases = trajectory.get("phase_offsets") or [
        segment.get("phase_offset")
        for segment in trajectory.get("segments") or []
        if isinstance(segment, dict) and segment.get("phase_offset") is not None
    ]
    phase_span = _trajectory_metric(
        trajectory,
        "phase_span_symbols",
        phases,
        metric="span",
    )
    max_adjacent = _trajectory_metric(
        trajectory,
        "max_adjacent_phase_step_symbols",
        phases,
        metric="max_adjacent",
    )
    available = bool(trajectory.get("available"))
    stable = (
        available
        and phase_span <= max_phase_span_symbols
        and max_adjacent <= max_adjacent_phase_step_symbols
        and bool(phases or trajectory.get("segments"))
    )
    if stable:
        verdict = "stable"
    elif not available:
        verdict = str(trajectory.get("reason") or "trajectory_unavailable")
    elif not phases and not trajectory.get("segments"):
        verdict = "no_trajectory_segments"
    else:
        verdict = "unstable_phase_jumps"
    return {
        "stable": bool(stable),
        "verdict": verdict,
        "window_count": int(trajectory.get("window_count") or 0),
        "selected_window_count": len(phases),
        "phase_span_symbols": phase_span,
        "max_adjacent_phase_step_symbols": max_adjacent,
        "max_phase_span_symbols": int(max_phase_span_symbols),
        "max_allowed_adjacent_phase_step_symbols": int(
            max_adjacent_phase_step_symbols
        ),
        "metric_source": "timing_trajectory",
        "trajectory_available": available,
        "trajectory_source": trajectory.get("source"),
        "expected_top_ratio": 0.0,
        "expected_top3_ratio": 0.0,
    }


def _trajectory_metric(
    trajectory: dict[str, Any],
    key: str,
    phase_offsets: list[Any],
    *,
    metric: str,
) -> int:
    value = trajectory.get(key)
    if value is not None:
        return int(value)
    phases = [int(phase) for phase in phase_offsets]
    if not phases:
        return 0
    if metric == "span":
        return max(phases) - min(phases)
    if metric == "max_adjacent":
        if len(phases) < 2:
            return 0
        return max(abs(b - a) for a, b in zip(phases, phases[1:], strict=False))
    raise ValueError(f"unknown trajectory metric: {metric}")


def compact_timing_tracking(tracking: dict[str, Any]) -> dict[str, Any]:
    """Keep the timing preflight JSON small enough for pipeline artifacts."""

    compact_windows = []
    for window in tracking.get("windows") or []:
        selected = window.get("selected") or {}
        continuous = window.get("continuous_selected") or {}
        compact_windows.append(
            {
                "start_frame_index": window.get("start_frame_index"),
                "analysis_start_symbol": window.get("analysis_start_symbol"),
                "center_phase_offset": window.get("center_phase_offset"),
                "selected_phase_offset": selected.get("phase_offset"),
                "selected_relative_offset_symbols": selected.get(
                    "relative_offset_symbols"
                ),
                "selected_score": selected.get("score"),
                "selected_expected_top_count": selected.get("expected_top_count"),
                "selected_expected_top3_count": selected.get("expected_top3_count"),
                "continuous_phase_offset": continuous.get("phase_offset"),
                "continuous_relative_offset_symbols": continuous.get(
                    "relative_offset_symbols"
                ),
            }
        )
    trajectory = tracking.get("trajectory") or {}
    return {
        "mode": tracking.get("mode"),
        "center_phase_offset": tracking.get("center_phase_offset"),
        "window_frames": tracking.get("window_frames"),
        "window_step_frames": tracking.get("window_step_frames"),
        "max_windows": tracking.get("max_windows"),
        "search_radius_symbols": tracking.get("search_radius_symbols"),
        "step_symbols": tracking.get("step_symbols"),
        "expected_system_info_index": tracking.get("expected_system_info_index"),
        "expected_frame_body_mode": tracking.get("expected_frame_body_mode"),
        "continuity_tracking": tracking.get("continuity_tracking"),
        "continuity_candidate_limit": tracking.get("continuity_candidate_limit"),
        "continuity_bridge_candidate_limit": tracking.get(
            "continuity_bridge_candidate_limit"
        ),
        "smoothness_penalty_per_symbol": tracking.get("smoothness_penalty_per_symbol"),
        "window_count": tracking.get("window_count"),
        "summary": tracking.get("summary") or {},
        "trajectory": {
            "available": trajectory.get("available"),
            "reason": trajectory.get("reason"),
            "phase_offsets": trajectory.get("phase_offsets"),
            "relative_offsets_symbols": trajectory.get("relative_offsets_symbols"),
            "unwrapped_relative_offsets_symbols": trajectory.get(
                "unwrapped_relative_offsets_symbols"
            ),
            "phase_span_symbols": trajectory.get("phase_span_symbols"),
            "max_adjacent_phase_step_symbols": trajectory.get(
                "max_adjacent_phase_step_symbols"
            ),
        }
        if trajectory
        else None,
        "windows": compact_windows,
    }


def gate_c_quality(sysinfo_json: dict[str, Any]) -> dict[str, Any]:
    """Summarise Gate C quality markers used by pipeline budget gates."""

    system_info = sysinfo_json.get("system_info") or {}
    expected = system_info.get("expected") or {}
    pipeline = sysinfo_json.get("pipeline") or {}
    timing = pipeline.get("timing_continuity") or {}
    timing_stable = bool(timing.get("stable", True))
    return {
        "agreement_ratio": float(system_info.get("agreement_ratio") or 0.0),
        "auto_eligible_frame_count": int(
            system_info.get("auto_eligible_frame_count") or 0
        ),
        "best_metric": float(expected.get("best_metric") or 0.0),
        "mean_metric": float(expected.get("mean_metric") or 0.0),
        "qualified_frame_count": int(expected.get("qualified_frame_count") or 0),
        "timing_continuity_stable": timing_stable,
        "timing_continuity_verdict": timing.get("verdict"),
        "timing_phase_span_symbols": int(timing.get("phase_span_symbols") or 0),
        "timing_max_adjacent_phase_step_symbols": int(
            timing.get("max_adjacent_phase_step_symbols") or 0
        ),
    }


def selected_system_info_index(
    sysinfo_json: dict[str, Any],
    fallback_index: int,
) -> int:
    """Return the selected system-info index when Gate C is stable."""

    system_info = sysinfo_json.get("system_info") or {}
    selected = system_info.get("selected") or {}
    selected_index = selected.get("index")
    if (
        gate_c_quality(sysinfo_json)["timing_continuity_stable"]
        and selected_index is not None
    ):
        return int(selected_index)
    return int(fallback_index)


def sysinfo_oracle_quality(
    report: dict[str, Any] | None,
    *,
    near_clean_threshold: float = 0.95,
) -> dict[str, Any]:
    """Classify independent raw system-information oracle evidence.

    Missing oracle reports are non-blocking so older deterministic fixtures and
    explicit force runs keep working. A present but failing oracle blocks real-RF
    demap/receive promotion unless the caller explicitly overrides it.
    """

    if report is None:
        return {
            "status": "missing",
            "blocking": False,
            "ok": None,
            "verdict": None,
        }

    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        return {
            "status": "fail",
            "blocking": True,
            "ok": False,
            "verdict": "missing_summary",
        }

    frames = int(summary.get("frames_analyzed") or 0)
    near = int(summary.get("expected_near_clean_frames_le2") or 0)
    independent = int(summary.get("system_info_iq_independent_frames") or 0)
    trained = int(summary.get("system_info_iq_trained_frames") or 0)
    expected_trained = int(summary.get("system_info_iq_expected_trained_frames") or 0)
    warmup = int(summary.get("deinterleaver_warmup_frames") or 0)
    post_latency = int(summary.get("post_latency_frames") or 0)
    post_latency_near = int(
        summary.get("post_latency_expected_near_clean_frames_le2") or 0
    )
    near_ratio = float(near / frames) if frames else 0.0
    post_latency_near_ratio = (
        float(post_latency_near / post_latency) if post_latency else None
    )
    independent_ok = frames > 0 and independent == frames and trained == 0
    independent_ok = independent_ok and expected_trained == 0
    near_ok = frames > 0 and near_ratio >= float(near_clean_threshold)
    if warmup > 0:
        latency_ok = (
            post_latency > 0
            and post_latency_near_ratio is not None
            and post_latency_near_ratio >= float(near_clean_threshold)
        )
    else:
        latency_ok = True
    ok = independent_ok and near_ok and latency_ok

    if ok:
        verdict = "pass"
    elif frames <= 0:
        verdict = "no_system_info_iq_frames"
    elif not independent_ok:
        verdict = "system_info_oracle_not_independent"
    elif not near_ok:
        verdict = "system_info_oracle_not_near_clean"
    elif warmup > 0 and post_latency <= 0:
        verdict = "system_info_oracle_short_of_deinterleaver_latency"
    else:
        verdict = "system_info_oracle_post_latency_not_near_clean"

    return {
        "status": "pass" if ok else "fail",
        "blocking": not ok,
        "ok": bool(ok),
        "verdict": verdict,
        "interpretation": report.get("interpretation"),
        "iq_source": (report.get("metadata") or {}).get("iq_source"),
        "frames_analyzed": frames,
        "independent_frames": independent,
        "trained_frames": trained,
        "expected_trained_frames": expected_trained,
        "near_clean_frames_le2": near,
        "near_clean_ratio_le2": near_ratio,
        "near_clean_threshold": float(near_clean_threshold),
        "deinterleaver_warmup_frames": warmup,
        "post_latency_frames": post_latency,
        "post_latency_near_clean_frames_le2": post_latency_near,
        "post_latency_near_clean_ratio_le2": post_latency_near_ratio,
    }


def pick_mode_from_gate_c(
    sysinfo_json: dict[str, Any],
    *,
    expected_qam: str,
    expected_fec_rate: int,
    expected_interleaver: str,
    synthetic_loopback: bool = False,
) -> tuple[str, int, str, str, bool, dict[str, Any]]:
    """Return mode choice plus Gate C decision metadata.

    The returned tuple is ``(qam, fec_rate, interleaver, note, fallback_active,
    quality)``. ``fallback_active`` means real-RF downstream work should skip
    expensive LDPC/demap unless explicitly forced.
    """

    quality = gate_c_quality(sysinfo_json)
    system_info = sysinfo_json.get("system_info") or {}
    selected = system_info.get("selected") or None
    if selected is None:
        note = "synthetic_loopback_pinned" if synthetic_loopback else "gate_c_fallback"
        return (
            expected_qam,
            int(expected_fec_rate),
            expected_interleaver,
            note,
            not synthetic_loopback,
            quality,
        )
    if not quality["timing_continuity_stable"] and not synthetic_loopback:
        return (
            expected_qam,
            int(expected_fec_rate),
            expected_interleaver,
            "gate_c_timing_unstable",
            True,
            quality,
        )
    params = selected.get("parameters", {})
    qam = params.get("qam_mode") or expected_qam
    fec = int(params.get("fec_rate_index") or expected_fec_rate)
    interleaver = params.get("interleaver_mode") or expected_interleaver
    return qam, fec, interleaver, "from_sysinfo", False, quality
