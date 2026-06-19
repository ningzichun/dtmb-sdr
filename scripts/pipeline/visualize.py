"""Render best-effort gate visual artifacts for a pipeline prefix.

This wrapper intentionally stays thin: gate-specific modules own extraction and
plotting, while the pipeline layer only resolves the conventional artifact
paths and records which visuals were rendered or skipped.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from _common import write_json


VisualWriter = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-visualize")
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Source capture path. The .ci8 suffix is replaced to find sidecars.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional summary JSON path. Defaults to <capture>.visuals.json.",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="Promote render errors from gate visual writers into failures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = (
        args.summary_json
        if args.summary_json is not None
        else args.capture.with_suffix(".visuals.json")
    )
    prefix = _visual_prefix(args.capture, summary_path)
    reports = render_pipeline_visuals(prefix, required=args.required)
    write_json(
        summary_path,
        {
            "stage": "pipeline_visuals",
            "capture_path": str(args.capture),
            "prefix": str(prefix),
            "required": bool(args.required),
            "visuals": reports,
        },
    )
    if args.required and any(item.get("status") == "error" for item in reports):
        return 1
    return 0


def _visual_prefix(capture_path: Path, summary_path: Path) -> Path:
    if summary_path.name.endswith(".visuals.json"):
        return summary_path.with_name(summary_path.name[: -len(".visuals.json")])
    return capture_path.with_suffix("")


def render_pipeline_visuals(prefix: str | Path, *, required: bool = False) -> list[dict[str, Any]]:
    """Render all conventional visuals whose inputs are present."""

    base = Path(prefix)
    receiver = _resolve_related_artifact(base, ".receiver.json")
    receiver_prefix = _artifact_prefix(receiver, ".receiver.json")
    sysinfo = _resolve_related_artifact(base, ".sysinfo.json", anchor=receiver_prefix)
    timing = _resolve_related_artifact(base, ".timing_trajectory.json", anchor=receiver_prefix)
    llr_health = _resolve_related_artifact(base, ".llr_health.json", anchor=receiver_prefix)
    demap = _resolve_related_artifact(base, ".demap.json", anchor=receiver_prefix)
    probe = _resolve_related_artifact(base, ".probe.json", anchor=receiver_prefix)
    recovered_ts = _resolve_related_artifact(base, ".recovered.ts", anchor=receiver_prefix)
    pn_cir = _resolve_related_artifact(base, ".pn_cir.json", anchor=receiver_prefix)
    reports: list[dict[str, Any]] = []
    reports.append(
        _maybe_render(
            gate="B",
            module_name="dtmb.capture_visuals",
            function_name="write_gate_b_visual",
            input_paths=[timing, sysinfo],
            kwargs={
                "png_path": _append_suffix(base, ".gate_b.png"),
                "manifest_path": _append_suffix(base, ".gate_b.visual.json"),
                "required": required,
            },
        )
    )
    reports.append(
        _maybe_render(
            gate="C",
            module_name="dtmb.pipeline_visuals",
            function_name="write_gate_c_visual",
            input_paths=[sysinfo],
            kwargs={
                "png_path": _append_suffix(base, ".gate_c.png"),
                "manifest_path": _append_suffix(base, ".gate_c.visual.json"),
                "required": required,
            },
        )
    )
    reports.append(
        _maybe_render(
            gate="D",
            module_name="dtmb.capture_visuals",
            function_name="write_gate_d_visual",
            input_paths=[
                receiver,
                llr_health,
                demap,
            ],
            kwargs={
                "receiver_json_path": receiver,
                "llr_health_json_path": llr_health,
                "demap_json_path": demap,
                "png_path": _append_suffix(base, ".gate_d.png"),
                "manifest_path": _append_suffix(base, ".gate_d.visual.json"),
                "required": required,
            },
            positional_input=False,
        )
    )
    reports.append(
        _maybe_render(
            gate="E",
            module_name="dtmb.pipeline_visuals",
            function_name="write_gate_e_visual",
            input_paths=[receiver, llr_health],
            kwargs={
                "receiver_json_path": receiver,
                "llr_health_json_path": llr_health,
                "png_path": _append_suffix(base, ".gate_e.png"),
                "manifest_path": _append_suffix(base, ".gate_e.visual.json"),
                "required": required,
            },
            positional_input=False,
        )
    )
    reports.append(
        _maybe_render(
            gate="F",
            module_name="dtmb.pipeline_visuals",
            function_name="write_gate_f_visual",
            input_paths=[probe, recovered_ts],
            kwargs={
                "probe_json_path": probe,
                "ts_path": recovered_ts,
                "png_path": _append_suffix(base, ".gate_f.png"),
                "manifest_path": _append_suffix(base, ".gate_f.visual.json"),
                "required": required,
            },
            positional_input=False,
        )
    )
    reports.append(
        _maybe_render(
            gate="echo",
            module_name="dtmb.capture_visuals",
            function_name="write_echo_visual",
            input_paths=[pn_cir],
            kwargs={
                "png_path": _append_suffix(base, ".echo.png"),
                "manifest_path": _append_suffix(base, ".echo.visual.json"),
                "required": required,
            },
        )
    )
    return reports


def _resolve_related_artifact(
    base: Path,
    suffix: str,
    *,
    anchor: Path | None = None,
) -> Path:
    """Resolve an exact or variant pipeline artifact related to ``base``."""

    candidates: list[Path] = []
    for prefix in _prefix_candidates(anchor or base, stop=base):
        candidates.append(_append_suffix(prefix, suffix))
    if anchor is not None:
        for prefix in _prefix_candidates(base, stop=base):
            candidates.append(_append_suffix(prefix, suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    variants = list(base.parent.glob(f"{base.name}.*{suffix}"))
    if variants:
        return max(variants, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[0] if candidates else _append_suffix(base, suffix)


def _artifact_prefix(path: Path, suffix: str) -> Path | None:
    if not path.exists() or not path.name.endswith(suffix):
        return None
    return path.with_name(path.name[: -len(suffix)])


def _prefix_candidates(prefix: Path, *, stop: Path) -> list[Path]:
    candidates = [prefix]
    while prefix.parent == stop.parent and "." in prefix.name:
        prefix = prefix.with_name(prefix.name.rsplit(".", 1)[0])
        candidates.append(prefix)
    return candidates


def _append_suffix(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}{suffix}")


def _maybe_render(
    *,
    gate: str,
    module_name: str,
    function_name: str,
    input_paths: Sequence[Path],
    kwargs: dict[str, Any],
    positional_input: bool = True,
) -> dict[str, Any]:
    existing = [path for path in input_paths if path.exists()]
    if not existing:
        manifest_path = Path(kwargs["manifest_path"])
        png_path = Path(kwargs["png_path"])
        _write_gate_status_manifest(
            manifest_path,
            gate=gate,
            status="skipped",
            reason="missing_inputs",
            image_path=png_path,
            inputs=input_paths,
        )
        return {
            "gate": gate,
            "status": "skipped",
            "reason": "missing_inputs",
            "manifest": str(manifest_path),
            "image": str(png_path),
            "inputs": [str(path) for path in input_paths],
        }
    try:
        writer = _load_writer(module_name, function_name)
        if positional_input:
            manifest = writer(existing[0], **kwargs)
        else:
            manifest = writer(**kwargs)
        return {
            "gate": gate,
            "status": manifest.get("status", "unknown"),
            "manifest": manifest.get("artifacts", {}).get("manifest"),
            "image": manifest.get("artifacts", {}).get("image"),
            "input": str(existing[0]),
        }
    except Exception as exc:
        manifest_path = Path(kwargs["manifest_path"])
        png_path = Path(kwargs["png_path"])
        _write_gate_status_manifest(
            manifest_path,
            gate=gate,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            image_path=png_path,
            inputs=input_paths,
        )
        return {
            "gate": gate,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "manifest": str(manifest_path),
            "image": str(png_path),
            "inputs": [str(path) for path in input_paths],
        }


def _load_writer(module_name: str, function_name: str) -> VisualWriter:
    module = importlib.import_module(module_name)
    writer = getattr(module, function_name)
    if not callable(writer):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return writer


def _write_gate_status_manifest(
    path: Path,
    *,
    gate: str,
    status: str,
    reason: str,
    image_path: Path,
    inputs: Sequence[Path],
) -> None:
    write_json(
        path,
        {
            "schema": "dtmb.pipeline_visuals.gate_status.v1",
            "gate": gate,
            "title": f"Gate {gate} visual",
            "status": status,
            "error": None if status == "skipped" else reason,
            "reason": reason,
            "inputs": [str(item) for item in inputs],
            "artifacts": {
                "image": str(image_path),
                "manifest": str(path),
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
