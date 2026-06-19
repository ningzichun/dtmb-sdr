"""Parse the root ``make dtmb pipeline ...`` shorthand and run pipeline.mk.

The root Makefile exists for operator ergonomics only. This script turns a
compact positional command such as:

    make dtmb pipeline 602mhz 24 14 0 5s

into the existing staged pipeline invocation:

    python scripts/profile_run.py -- make -f pipeline.mk wall-7msps \
        FREQUENCY=602000000 LNA_GAIN=24 VGA_GAIN=14 AMP=0 DURATION=5
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = "wall-7msps"

FORWARDED_KEYS = {
    "BANDWIDTH",
    "CAPTURE_TUNE_OFFSET_HZ",
    "CAPTURE_ANALYSIS_DIR",
    "CAPTURE_GROUP",
    "CAPTURE_NAME",
    "CAPTURE_TAG",
    "CAPTURES_DIR",
    "CROP_LOWPASS_CUTOFF_HZ",
    "CROP_LOWPASS_TRANSITION_HZ",
    "CROP_OUTPUT_BANDWIDTH_HZ",
    "CROP_OUTPUT_SAMPLE_RATE",
    "CROP_WIDEBAND_CAPTURE",
    "DD_MAX_HARD_BIT_BIAS",
    "DTMB_TARGET",
    "FFT_BIN_SHIFT",
    "FREQUENCY_DEINTERLEAVER_DIRECTION",
    "DATA_CARRIER_ORDER",
    "CARRIER_PERMUTATION",
    "LOGICAL_POSITION_SHIFT",
    "FFMPEG",
    "FFPROBE",
    "FORCE_DEMAP",
    "FORCE_RECEIVE",
    "FREQUENCY_SHIFT",
    "MAKE_JOBS",
    "BODY_WINDOW_FALLBACK",
    "BODY_WINDOW_OFFSET",
    "PHASE_OFFSET",
    "PROFILE",
    "RAW_CAPTURES_DIR",
    "RECEIVE_FRAMES",
    "SYSINFO_FRAMES",
    "SYSINFO_INDEX",
    "SYSINFO_ORACLE_EQUALIZER",
    "SYSINFO_ORACLE_FRAMES",
    "TARGET",
    "TIMING_BODY_PHASE_BIAS",
    "TIMING_BODY_PHASE_BIAS_SEARCH",
    "TIMING_BODY_PHASE_BIAS_MIN",
    "TIMING_BODY_PHASE_BIAS_MAX",
    "TIMING_BODY_PHASE_BIAS_STEP",
    "TIMING_MAX_SAMPLES",
    "TIMING_POLICY",
    "TIMING_TRAJECTORY",
    "TIMING_TRAJECTORY_SOURCE",
}


@dataclass(frozen=True)
class PipelineRequest:
    target: str
    variables: dict[str, str]
    jobs: int
    profile: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make dtmb pipeline",
        description=(
            "Run the staged HackRF -> DTMB -> TS pipeline from a compact "
            "make command. Positional order: frequency LNA VGA amp duration."
        ),
    )
    parser.add_argument("--make-bin", default="make")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Make variable forwarded from the root Makefile.",
    )
    parser.add_argument("tokens", nargs="*")
    return parser


def parse_request(
    tokens: Sequence[str],
    forwarded: dict[str, str] | None = None,
    *,
    now: datetime | None = None,
) -> PipelineRequest:
    forwarded = dict(forwarded or {})
    explicit_capture_dir = "CAPTURES_DIR" in forwarded
    explicit_capture_group = "CAPTURE_GROUP" in forwarded
    explicit_capture_tag = "CAPTURE_TAG" in forwarded
    capture_name = forwarded.pop("CAPTURE_NAME", None)
    explicit: dict[str, str] = {}
    positionals: list[str] = []
    target = _normalize_target(
        forwarded.pop("DTMB_TARGET", forwarded.pop("TARGET", DEFAULT_TARGET))
    )
    jobs = _parse_jobs(forwarded.pop("MAKE_JOBS", "auto"))
    profile = _truthy(forwarded.pop("PROFILE", "1"))

    for token in tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in {"7msps", "7.56msps", "7p56msps", "native"}:
            target = "wall-7msps"
            continue
        if lowered in {"20msps", "20m", "wide"}:
            target = "wall-20msps"
            continue
        if lowered in {"force", "force-receive"}:
            explicit["FORCE_RECEIVE"] = "1"
            continue
        if lowered in {"timing-auto", "trajectory-auto"}:
            explicit["TIMING_TRAJECTORY"] = "auto"
            continue
        bw_value = _optional_prefixed_value(cleaned, ("bw", "bandwidth"))
        if bw_value is not None:
            explicit["BANDWIDTH"] = str(parse_hz(bw_value, default_unit="mhz"))
            continue
        positionals.append(cleaned)

    if len(positionals) > 5:
        extra = " ".join(positionals[5:])
        raise ValueError(f"too many positional arguments after duration: {extra}")

    variables = dict(forwarded)
    variables.update(explicit)
    names = ("FREQUENCY", "LNA_GAIN", "VGA_GAIN", "AMP", "DURATION")
    parsers = (
        lambda value: str(parse_hz(value, default_unit="mhz")),
        lambda value: str(parse_int(value, "LNA gain")),
        lambda value: str(parse_int(value, "VGA gain")),
        parse_amp,
        lambda value: _format_number(parse_seconds(value)),
    )
    for name, parser_fn, value in zip(names, parsers, positionals):
        variables[name] = parser_fn(value)

    variables.setdefault("FREQUENCY", str(parse_hz("602mhz", default_unit="mhz")))
    variables.setdefault("LNA_GAIN", "24")
    variables.setdefault("VGA_GAIN", "14")
    variables.setdefault("AMP", "0")
    variables.setdefault("DURATION", "5")

    if "BANDWIDTH" in variables:
        variables["BANDWIDTH"] = str(parse_hz(variables["BANDWIDTH"], default_unit="hz"))

    variables.setdefault("PYTHON", sys.executable.replace("\\", "/"))
    _apply_default_capture_layout(
        variables,
        target=target,
        capture_name=capture_name,
        explicit_capture_dir=explicit_capture_dir,
        explicit_capture_group=explicit_capture_group,
        explicit_capture_tag=explicit_capture_tag,
        now=now,
    )
    return PipelineRequest(
        target=target,
        variables=variables,
        jobs=jobs,
        profile=profile,
    )


def make_command(request: PipelineRequest, make_bin: str) -> list[str]:
    resolved_make = resolve_make_bin(make_bin)
    make_args = [
        resolved_make,
        f"-j{request.jobs}",
        "-f",
        "pipeline.mk",
        request.target,
        *[f"{key}={value}" for key, value in sorted(request.variables.items())],
    ]
    if not request.profile:
        return make_args
    return [sys.executable, "scripts/profile_run.py", "--", *make_args]


def resolve_make_bin(make_bin: str) -> str:
    """Return a Windows-safe executable path for child subprocesses."""

    candidate = Path(make_bin)
    if candidate.parent != Path(".") or candidate.suffix:
        if candidate.exists():
            return str(candidate).replace("\\", "/")
    resolved = shutil.which(make_bin)
    if resolved:
        return resolved.replace("\\", "/")
    fallback = Path(sys.executable).resolve().parent / "Library" / "bin" / "make.exe"
    if fallback.exists():
        return str(fallback).replace("\\", "/")
    return make_bin


def parse_forwarded(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in FORWARDED_KEYS:
            raise ValueError(f"unsupported forwarded variable: {key}")
        if value != "":
            parsed[key] = value
    return parsed


def _apply_default_capture_layout(
    variables: dict[str, str],
    *,
    target: str,
    capture_name: str | None,
    explicit_capture_dir: bool,
    explicit_capture_group: bool,
    explicit_capture_tag: bool,
    now: datetime | None,
) -> None:
    run_time = now or datetime.now(timezone.utc)
    if run_time.tzinfo is None:
        run_time = run_time.replace(tzinfo=timezone.utc)
    run_time = run_time.astimezone(timezone.utc)

    if not explicit_capture_tag:
        variables["CAPTURE_TAG"] = run_time.strftime("%Y%m%dT%H%M%SZ")

    variables.setdefault("CAPTURES_DIR", "captures")
    if explicit_capture_group:
        return
    if explicit_capture_dir:
        return

    tag = variables["CAPTURE_TAG"]
    month = _month_from_tag(tag) or run_time.strftime("%Y%m")
    name = capture_name or _default_capture_name(variables, target=target, tag=tag)
    variables["CAPTURE_GROUP"] = f"{month}/{_sanitize_path_component(name)}"


def artifact_paths(request: PipelineRequest) -> dict[str, Path]:
    raw_prefix, analysis_prefix = _capture_prefixes(request)
    return {
        "raw_dir": raw_prefix.parent,
        "analysis_dir": analysis_prefix.parent,
        "capture": raw_prefix.with_suffix(".ci8"),
        "capture_sidecar": Path(str(raw_prefix.with_suffix(".ci8")) + ".json"),
        "gate_a": analysis_prefix.with_suffix(".acquire.json"),
        "clipping": analysis_prefix.with_suffix(".clipping.json"),
        "gate_c": analysis_prefix.with_suffix(".sysinfo.json"),
        "gate_c_oracle": analysis_prefix.with_suffix(".sysinfo_oracle.raw.json"),
        "receiver_json": analysis_prefix.with_suffix(".receiver.json"),
        "receiver_ts": analysis_prefix.with_suffix(".recovered.ts"),
        "probe": analysis_prefix.with_suffix(".probe.json"),
        "mp4": analysis_prefix.with_suffix(".mp4"),
        "demap_json": analysis_prefix.with_suffix(".demap.json"),
        "llr": analysis_prefix.with_suffix(".llr.f32"),
        "llr_sidecar": analysis_prefix.with_suffix(".llr.f32.json"),
        "llr_health": analysis_prefix.with_suffix(".llr_health.json"),
        "decoded_ts": analysis_prefix.with_suffix(".decoded.ts"),
        "decode_ldpc": analysis_prefix.with_suffix(".decode_ldpc.json"),
        "visuals": analysis_prefix.with_suffix(".visuals.json"),
    }


def _capture_prefixes(request: PipelineRequest) -> tuple[Path, Path]:
    variables = request.variables
    rate_suffix = "7p56msps" if request.target == "wall-7msps" else "20msps"
    stem = (
        f"wall_{variables['FREQUENCY']}_amp{variables['AMP']}"
        f"_lna{variables['LNA_GAIN']}_vga{variables['VGA_GAIN']}"
        f"_{variables['CAPTURE_TAG']}_{rate_suffix}"
    )
    captures_dir = Path(variables.get("CAPTURES_DIR", "captures"))
    raw_dir = Path(variables.get("RAW_CAPTURES_DIR", str(captures_dir / "raw")))
    analysis_dir = Path(
        variables.get("CAPTURE_ANALYSIS_DIR", str(captures_dir / "analysis"))
    )
    group = variables.get("CAPTURE_GROUP")
    if group:
        raw_dir = raw_dir / group
        analysis_dir = analysis_dir / group
    return raw_dir / stem, analysis_dir / stem


def print_artifact_plan(request: PipelineRequest) -> None:
    paths = artifact_paths(request)
    print("[dtmb-pipeline] artifacts:", file=sys.stderr)
    print(f"  raw dir        {paths['raw_dir']}", file=sys.stderr)
    print(f"  analysis dir   {paths['analysis_dir']}", file=sys.stderr)
    print(f"  capture        {paths['capture']}", file=sys.stderr)
    print(f"  direct TS      {paths['receiver_ts']}", file=sys.stderr)
    print(f"  direct report  {paths['receiver_json']}", file=sys.stderr)
    print(f"  demap LLR      {paths['llr']}", file=sys.stderr)
    print(f"  demap report   {paths['demap_json']}", file=sys.stderr)
    print(f"  LLR health     {paths['llr_health']}", file=sys.stderr)
    print(f"  LDPC TS        {paths['decoded_ts']}", file=sys.stderr)
    print(f"  LDPC report    {paths['decode_ldpc']}", file=sys.stderr)
    print(f"  TS probe       {paths['probe']}", file=sys.stderr)
    print(f"  visuals        {paths['visuals']}", file=sys.stderr)


def print_artifact_result(request: PipelineRequest) -> None:
    paths = artifact_paths(request)
    print("[dtmb-pipeline] result artifacts:", file=sys.stderr)
    for label in (
        "capture",
        "gate_a",
        "clipping",
        "gate_c",
        "gate_c_oracle",
        "receiver_ts",
        "receiver_json",
        "probe",
        "mp4",
        "demap_json",
        "llr",
        "llr_sidecar",
        "llr_health",
        "decoded_ts",
        "decode_ldpc",
        "visuals",
    ):
        path = paths[label]
        status = _path_status(path)
        print(f"  {label:14} {status} {path}", file=sys.stderr)


def _path_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return f"{path.stat().st_size}B"
    return "dir"


def _default_capture_name(variables: dict[str, str], *, target: str, tag: str) -> str:
    rate = "7p56msps" if target == "wall-7msps" else "20msps"
    return "_".join(
        [
            "wall",
            _frequency_label(int(variables["FREQUENCY"])),
            f"amp{variables['AMP']}",
            f"lna{variables['LNA_GAIN']}",
            f"vga{variables['VGA_GAIN']}",
            _duration_label(variables["DURATION"]),
            rate,
            tag,
        ]
    )


def _month_from_tag(tag: str) -> str | None:
    match = re.match(r"^(\d{6})\d{2}T", tag)
    if not match:
        return None
    return match.group(1)


def _frequency_label(hz: int) -> str:
    mhz = hz / 1_000_000.0
    text = f"{mhz:g}".replace(".", "p")
    return f"{text}mhz"


def _duration_label(value: str) -> str:
    text = _format_number(parse_seconds(value)).replace(".", "p")
    return f"{text}s"


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError("capture name must contain at least one safe character")
    return cleaned


def parse_hz(value: str, *, default_unit: str) -> int:
    text = value.strip().lower().replace("_", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-z]*)", text)
    if not match:
        raise ValueError(f"invalid frequency/bandwidth value: {value!r}")
    magnitude = float(match.group(1))
    unit = match.group(2) or default_unit
    multipliers = {
        "hz": 1.0,
        "k": 1_000.0,
        "khz": 1_000.0,
        "m": 1_000_000.0,
        "mhz": 1_000_000.0,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported frequency/bandwidth unit in {value!r}")
    return int(round(magnitude * multipliers[unit]))


def parse_seconds(value: str) -> float:
    text = value.strip().lower().replace("_", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s)?", text)
    if not match:
        raise ValueError(f"invalid duration value: {value!r}")
    seconds = float(match.group(1))
    if match.group(2) == "ms":
        seconds /= 1000.0
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


def parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def parse_amp(value: str) -> str:
    parsed = parse_int(value, "amp")
    if parsed not in {0, 1}:
        raise ValueError("amp must be 0 or 1")
    return str(parsed)


def _normalize_target(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "7": "wall-7msps",
        "7msps": "wall-7msps",
        "7.56msps": "wall-7msps",
        "7p56msps": "wall-7msps",
        "native": "wall-7msps",
        "wall-7msps": "wall-7msps",
        "20": "wall-20msps",
        "20msps": "wall-20msps",
        "wide": "wall-20msps",
        "wall-20msps": "wall-20msps",
    }
    if lowered not in aliases:
        raise ValueError(f"unsupported DTMB pipeline target: {value!r}")
    return aliases[lowered]


def _optional_prefixed_value(token: str, prefixes: tuple[str, ...]) -> str | None:
    lowered = token.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix + "="):
            return token.split("=", 1)[1]
        if lowered.startswith(prefix + "-"):
            return token[len(prefix) + 1 :]
        if lowered.startswith(prefix) and len(token) > len(prefix):
            suffix = token[len(prefix) :]
            if suffix[:1].isdigit():
                return suffix
    return None


def _truthy(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_jobs(value: str) -> int:
    text = value.strip().lower()
    if text in {"", "auto"}:
        return max(1, os.cpu_count() or 1)
    try:
        parsed = int(text, 10)
    except ValueError as exc:
        raise ValueError(
            f"MAKE_JOBS must be a positive integer or auto: {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError("MAKE_JOBS must be positive")
    return parsed


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        forwarded = parse_forwarded(args.sets)
        request = parse_request(args.tokens, forwarded)
        cmd = make_command(request, args.make_bin)
    except ValueError as exc:
        print(f"dtmb pipeline: {exc}", file=sys.stderr)
        return 2

    display = " ".join(cmd)
    print(f"[dtmb-pipeline] {display}", file=sys.stderr)
    print_artifact_plan(request)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())
    print_artifact_result(request)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
