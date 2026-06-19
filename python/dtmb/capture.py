"""HackRF capture wrapper for bounded CI8 recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Sequence


DEFAULT_SAMPLE_RATE_SPS = 20_000_000
DEFAULT_BANDWIDTH_HZ = 10_000_000
UTC_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|\+00:00)$"
)


@dataclass(frozen=True)
class HackrfCaptureConfig:
    """Configuration for one bounded HackRF CI8 capture."""

    output_path: Path
    frequency_hz: int
    sample_rate_sps: int = DEFAULT_SAMPLE_RATE_SPS
    bandwidth_hz: int = DEFAULT_BANDWIDTH_HZ
    amp: int = 0
    lna_gain: int = 16
    vga_gain: int = 20
    sample_count: int | None = None
    device_serial: str | None = None
    executable: str = "hackrf_transfer"
    clock_executable: str | None = None
    command_prefix: tuple[str, ...] = ()
    clock_source: str = "internal"
    external_reference_hz: int | None = None
    verify_external_clock: bool = True
    hardware_trigger: bool = False
    trigger_source: str | None = None
    trigger_time_utc: str | None = None
    antenna: str = "unknown"
    location: str = "not recorded"
    notes: str | None = None

    @classmethod
    def from_duration(
        cls,
        *,
        output_path: str | Path,
        frequency_hz: int,
        duration_s: float,
        sample_rate_sps: int = DEFAULT_SAMPLE_RATE_SPS,
        bandwidth_hz: int = DEFAULT_BANDWIDTH_HZ,
        amp: int = 0,
        lna_gain: int = 16,
        vga_gain: int = 20,
        device_serial: str | None = None,
        executable: str = "hackrf_transfer",
        clock_executable: str | None = None,
        command_prefix: Sequence[str] = (),
        clock_source: str = "internal",
        external_reference_hz: int | None = None,
        verify_external_clock: bool = True,
        hardware_trigger: bool = False,
        trigger_source: str | None = None,
        trigger_time_utc: str | None = None,
        antenna: str = "unknown",
        location: str = "not recorded",
        notes: str | None = None,
    ) -> "HackrfCaptureConfig":
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        sample_count = int(round(duration_s * sample_rate_sps))
        return cls(
            output_path=Path(output_path),
            frequency_hz=frequency_hz,
            sample_rate_sps=sample_rate_sps,
            bandwidth_hz=bandwidth_hz,
            amp=amp,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            sample_count=sample_count,
            device_serial=device_serial,
            executable=executable,
            clock_executable=clock_executable,
            command_prefix=tuple(command_prefix),
            clock_source=clock_source,
            external_reference_hz=external_reference_hz,
            verify_external_clock=verify_external_clock,
            hardware_trigger=hardware_trigger,
            trigger_source=trigger_source,
            trigger_time_utc=trigger_time_utc,
            antenna=antenna,
            location=location,
            notes=notes,
        )


@dataclass(frozen=True)
class CaptureTiming:
    """Host-side timing bracket for one `hackrf_transfer` subprocess run."""

    start_unix_ns: int
    end_unix_ns: int
    start_monotonic_ns: int
    end_monotonic_ns: int

    @property
    def elapsed_s(self) -> float:
        return (self.end_monotonic_ns - self.start_monotonic_ns) / 1_000_000_000.0


def build_hackrf_transfer_command(config: HackrfCaptureConfig) -> list[str]:
    """Return the `hackrf_transfer` command for a bounded CI8 capture."""

    _validate_capture_config(config)
    command = [
        *config.command_prefix,
        config.executable,
        "-r",
        str(config.output_path),
        "-f",
        str(config.frequency_hz),
        "-s",
        str(config.sample_rate_sps),
        "-b",
        str(config.bandwidth_hz),
        "-a",
        str(config.amp),
        "-l",
        str(config.lna_gain),
        "-g",
        str(config.vga_gain),
    ]
    if config.device_serial:
        command.extend(["-d", config.device_serial])
    if config.hardware_trigger:
        command.append("-H")
    if config.sample_count is not None:
        command.extend(["-n", str(config.sample_count)])
    return command


def capture_metadata(
    config: HackrfCaptureConfig,
    *,
    byte_count: int,
    created_utc: str | None = None,
    command: Sequence[str] | None = None,
    timing: CaptureTiming | None = None,
    clock_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build sidecar metadata for a completed HackRF capture."""

    _validate_capture_config(config)
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    sample_count = byte_count // 2
    duration_s = sample_count / config.sample_rate_sps
    created = created_utc or (
        utc_timestamp_ns(timing.start_unix_ns) if timing is not None else utc_timestamp()
    )
    metadata: dict[str, Any] = {
        "created_utc": created,
        "format": "ci8",
        "frequency_hz": config.frequency_hz,
        "sample_rate_sps": config.sample_rate_sps,
        "bandwidth_hz": config.bandwidth_hz,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "amp": config.amp,
        "lna_gain": config.lna_gain,
        "vga_gain": config.vga_gain,
        "antenna": config.antenna,
        "location": config.location,
        "byte_count": byte_count,
        "hackrf_tools_source": _tools_source(config),
        "timestamp_source": _timestamp_source(config),
        "hackrf_clock": {
            "source": config.clock_source,
            "external_reference_hz": config.external_reference_hz,
            "sample_timestamp_supported": False,
            "absolute_time_supported": False,
            "note": (
                "HackRF CI8 streams do not carry hardware time-of-day sample "
                "timestamps; CLKIN/external references improve sample-clock "
                "frequency accuracy, not absolute time."
            ),
        },
        "hardware_trigger": {
            "enabled": config.hardware_trigger,
            "source": config.trigger_source,
            "trigger_time_utc": config.trigger_time_utc,
        },
    }
    if clock_status is not None:
        metadata["hackrf_clock"]["clkin"] = clock_status
    if timing is not None:
        metadata.update(
            _timing_metadata(timing, sample_count, config.sample_rate_sps)
        )
    if config.hardware_trigger and config.trigger_time_utc:
        host_bounds = metadata.get("sample_time_bounds_utc_unix_ns")
        if host_bounds is not None:
            metadata["host_process_sample_time_bounds_utc_unix_ns"] = host_bounds
        metadata.update(_trigger_time_metadata(config, sample_count))
    if config.notes:
        metadata["notes"] = config.notes
    if config.device_serial:
        metadata["device_selector"] = "specified"
    if command is not None:
        metadata["capture_command"] = _sanitized_command(command)
    return metadata


def run_capture(
    config: HackrfCaptureConfig,
    *,
    metadata_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[list[str], Path | None]:
    """Run `hackrf_transfer` and write a metadata sidecar after success."""

    command = build_hackrf_transfer_command(config)
    if dry_run:
        return command, None

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    clock_status = _checked_external_clock_status(config)
    start_unix_ns = time.time_ns()
    start_monotonic_ns = time.monotonic_ns()
    subprocess.run(command, check=True)
    end_monotonic_ns = time.monotonic_ns()
    end_unix_ns = time.time_ns()
    byte_count = config.output_path.stat().st_size
    timing = CaptureTiming(
        start_unix_ns=start_unix_ns,
        end_unix_ns=end_unix_ns,
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=end_monotonic_ns,
    )
    metadata = capture_metadata(
        config,
        byte_count=byte_count,
        command=command,
        timing=timing,
        clock_status=clock_status,
    )
    sidecar = Path(metadata_path) if metadata_path else default_metadata_path(config.output_path)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return command, sidecar


def default_metadata_path(capture_path: str | Path) -> Path:
    """Return the preferred project sidecar path for a capture."""

    path = Path(capture_path)
    return path.with_name(path.name + ".json")


def utc_timestamp() -> str:
    """Return a UTC timestamp suitable for capture metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp_ns(unix_ns: int | None = None) -> str:
    """Return an RFC 3339 UTC timestamp with nanosecond fractional seconds."""

    if unix_ns is None:
        unix_ns = time.time_ns()
    seconds, nanoseconds = divmod(int(unix_ns), 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{nanoseconds:09d}Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-capture",
        description="Record a bounded raw CI8 capture with hackrf_transfer.",
    )
    parser.add_argument("output", type=Path, help="Output .ci8 capture path")
    parser.add_argument("--frequency", type=_positive_int, required=True, help="Center frequency in Hz")
    parser.add_argument("--sample-rate", type=_positive_int, default=DEFAULT_SAMPLE_RATE_SPS)
    parser.add_argument("--bandwidth", type=_positive_int, default=DEFAULT_BANDWIDTH_HZ)
    parser.add_argument("--duration", type=_positive_float, help="Capture duration in seconds")
    parser.add_argument("--samples", type=_positive_int, help="Number of complex samples to capture")
    parser.add_argument("--amp", type=int, default=0, help="RF amplifier state, 0 or 1")
    parser.add_argument("--lna-gain", type=int, default=16, help="RX LNA gain in dB, 0..40 step 8")
    parser.add_argument("--vga-gain", type=int, default=20, help="RX VGA gain in dB, 0..62 step 2")
    parser.add_argument("--device-serial", help="HackRF serial selector; not recorded in metadata")
    parser.add_argument("--executable", default="hackrf_transfer", help="hackrf_transfer executable")
    parser.add_argument(
        "--clock-executable",
        help=(
            "hackrf_clock executable for CLKIN verification; defaults to a "
            "hackrf_clock binary next to --executable when possible."
        ),
    )
    parser.add_argument("--mamba-env", help="Run hackrf_transfer through `mamba run -n ENV`")
    parser.add_argument(
        "--clock-source",
        choices=("internal", "external-10mhz", "gpsdo-10mhz", "unknown"),
        default="internal",
        help=(
            "HackRF sample-clock reference. This records metadata only; HackRF "
            "streams do not include absolute hardware timestamps."
        ),
    )
    parser.add_argument(
        "--external-reference-hz",
        type=_positive_int,
        help="External reference frequency connected to HackRF CLKIN, usually 10000000.",
    )
    parser.add_argument(
        "--no-clock-check",
        action="store_true",
        help="Do not verify CLKIN with hackrf_clock before external/GPSDO-clocked captures.",
    )
    parser.add_argument(
        "--hardware-trigger",
        action="store_true",
        help="Pass -H to hackrf_transfer and record that capture waited for hardware trigger.",
    )
    parser.add_argument(
        "--trigger-source",
        help="Human-readable external trigger source, e.g. gpsdo-pps or lab-sync.",
    )
    parser.add_argument(
        "--trigger-time-utc",
        help="Known UTC time of the trigger edge, if supplied by external timing gear.",
    )
    parser.add_argument("--antenna", default="unknown")
    parser.add_argument("--location", default="not recorded")
    parser.add_argument("--notes")
    parser.add_argument("--metadata", type=Path, help="Sidecar metadata path")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without recording")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.duration is None) == (args.samples is None):
        parser.error("exactly one of --duration or --samples is required")

    if args.duration is not None:
        prefix = ("mamba", "run", "-n", args.mamba_env) if args.mamba_env else ()
        config = HackrfCaptureConfig.from_duration(
            output_path=args.output,
            frequency_hz=args.frequency,
            duration_s=args.duration,
            sample_rate_sps=args.sample_rate,
            bandwidth_hz=args.bandwidth,
            amp=args.amp,
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            device_serial=args.device_serial,
            executable=args.executable,
            clock_executable=args.clock_executable,
            command_prefix=prefix,
            clock_source=args.clock_source,
            external_reference_hz=args.external_reference_hz,
            verify_external_clock=not args.no_clock_check,
            hardware_trigger=args.hardware_trigger,
            trigger_source=args.trigger_source,
            trigger_time_utc=args.trigger_time_utc,
            antenna=args.antenna,
            location=args.location,
            notes=args.notes,
        )
    else:
        prefix = ("mamba", "run", "-n", args.mamba_env) if args.mamba_env else ()
        config = HackrfCaptureConfig(
            output_path=args.output,
            frequency_hz=args.frequency,
            sample_rate_sps=args.sample_rate,
            bandwidth_hz=args.bandwidth,
            amp=args.amp,
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            sample_count=args.samples,
            device_serial=args.device_serial,
            executable=args.executable,
            clock_executable=args.clock_executable,
            command_prefix=prefix,
            clock_source=args.clock_source,
            external_reference_hz=args.external_reference_hz,
            verify_external_clock=not args.no_clock_check,
            hardware_trigger=args.hardware_trigger,
            trigger_source=args.trigger_source,
            trigger_time_utc=args.trigger_time_utc,
            antenna=args.antenna,
            location=args.location,
            notes=args.notes,
        )

    try:
        command, sidecar = run_capture(config, metadata_path=args.metadata, dry_run=args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    print(" ".join(_quote_part(part) for part in _sanitized_command(command)))
    if sidecar is not None:
        print(f"metadata: {sidecar}")
    return 0


def _validate_capture_config(config: HackrfCaptureConfig) -> None:
    if config.frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if config.sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if config.amp not in {0, 1}:
        raise ValueError("amp must be 0 or 1")
    if config.lna_gain < 0 or config.lna_gain > 40 or config.lna_gain % 8 != 0:
        raise ValueError("lna_gain must be 0..40 dB in 8 dB steps")
    if config.vga_gain < 0 or config.vga_gain > 62 or config.vga_gain % 2 != 0:
        raise ValueError("vga_gain must be 0..62 dB in 2 dB steps")
    if config.sample_count is None or config.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if config.clock_source not in {"internal", "external-10mhz", "gpsdo-10mhz", "unknown"}:
        raise ValueError("clock_source must be internal, external-10mhz, gpsdo-10mhz, or unknown")
    if config.external_reference_hz is not None and config.external_reference_hz <= 0:
        raise ValueError("external_reference_hz must be positive")
    if config.clock_source in {"external-10mhz", "gpsdo-10mhz"}:
        if config.external_reference_hz is not None and config.external_reference_hz != 10_000_000:
            raise ValueError("HackRF CLKIN expects a 10 MHz external reference")
    if config.verify_external_clock and config.clock_source in {"external-10mhz", "gpsdo-10mhz"}:
        if config.external_reference_hz != 10_000_000:
            raise ValueError("verified HackRF external clock captures require external_reference_hz=10000000")
    if config.trigger_time_utc and not config.hardware_trigger:
        raise ValueError("trigger_time_utc requires hardware_trigger")
    if config.trigger_time_utc:
        _parse_utc_timestamp_ns(config.trigger_time_utc)


def _sanitized_command(command: Sequence[str]) -> list[str]:
    sanitized = list(command)
    for index, part in enumerate(sanitized[:-1]):
        if part == "-d":
            sanitized[index + 1] = "<redacted>"
    return sanitized


def _timestamp_source(config: HackrfCaptureConfig) -> str:
    if config.hardware_trigger and config.trigger_time_utc:
        return "external_trigger_time"
    return "host_process_time_bounds"


def _timing_metadata(
    timing: CaptureTiming, sample_count: int, sample_rate_sps: int
) -> dict[str, Any]:
    span_ns = _sample_span_ns(sample_count, sample_rate_sps)
    host_elapsed_ns = max(0, timing.end_monotonic_ns - timing.start_monotonic_ns)
    first_latest = max(timing.start_unix_ns, timing.end_unix_ns - span_ns)
    last_earliest = min(timing.end_unix_ns, timing.start_unix_ns + span_ns)
    uncertainty_ns = max(0, host_elapsed_ns - span_ns)
    return {
        "host_capture_start_utc": utc_timestamp_ns(timing.start_unix_ns),
        "host_capture_end_utc": utc_timestamp_ns(timing.end_unix_ns),
        "host_capture_start_unix_ns": timing.start_unix_ns,
        "host_capture_end_unix_ns": timing.end_unix_ns,
        "host_capture_start_monotonic_ns": timing.start_monotonic_ns,
        "host_capture_end_monotonic_ns": timing.end_monotonic_ns,
        "host_capture_elapsed_s": timing.elapsed_s,
        "sample_time_bounds_utc_unix_ns": {
            "first_sample_earliest": timing.start_unix_ns,
            "first_sample_latest": first_latest,
            "last_sample_earliest": last_earliest,
            "last_sample_latest": timing.end_unix_ns,
        },
        "host_timestamp_uncertainty_s": uncertainty_ns / 1_000_000_000.0,
    }


def _trigger_time_metadata(
    config: HackrfCaptureConfig, sample_count: int
) -> dict[str, Any]:
    if not config.trigger_time_utc:
        raise ValueError("trigger_time_utc is required")
    trigger_ns = _parse_utc_timestamp_ns(config.trigger_time_utc)
    sample_period_ns = _sample_period_bound_ns(config.sample_rate_sps)
    span_ns = _sample_span_ns(sample_count, config.sample_rate_sps)
    first_earliest = trigger_ns - sample_period_ns
    first_latest = trigger_ns + sample_period_ns
    return {
        "sample_time_bounds_utc_unix_ns": {
            "first_sample_earliest": first_earliest,
            "first_sample_latest": first_latest,
            "last_sample_earliest": first_earliest + span_ns,
            "last_sample_latest": first_latest + span_ns,
        },
        "sample_time_reference": {
            "source": "external_trigger_time",
            "trigger_time_utc": utc_timestamp_ns(trigger_ns),
            "trigger_time_unix_ns": trigger_ns,
            "first_sample_error_bound_s": sample_period_ns / 1_000_000_000.0,
            "note": (
                "HackRF hardware triggering synchronizes the sample stream to "
                "the trigger edge within less than one sample period; UTC "
                "accuracy comes from the external trigger source."
            ),
        },
    }


def _sample_span_ns(sample_count: int, sample_rate_sps: int) -> int:
    if sample_count <= 1:
        return 0
    return int(round((sample_count - 1) * 1_000_000_000 / sample_rate_sps))


def _sample_period_bound_ns(sample_rate_sps: int) -> int:
    return max(1, (1_000_000_000 + sample_rate_sps - 1) // sample_rate_sps)


def _parse_utc_timestamp_ns(value: str) -> int:
    match = UTC_TIMESTAMP_RE.match(value)
    if not match:
        raise ValueError(
            "trigger_time_utc must be an RFC 3339 UTC timestamp ending in Z or +00:00"
        )
    base, fractional, _zone = match.groups()
    seconds = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    nanoseconds = int((fractional or "").ljust(9, "0"))
    return int(seconds.timestamp()) * 1_000_000_000 + nanoseconds


def _checked_external_clock_status(config: HackrfCaptureConfig) -> dict[str, Any] | None:
    if config.clock_source not in {"external-10mhz", "gpsdo-10mhz"}:
        return None
    if not config.verify_external_clock:
        return {
            "checked": False,
            "status": "not_checked",
            "note": "CLKIN verification disabled by capture option.",
        }

    command = _hackrf_clock_command(config)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    status = _parse_clkin_status(output)
    if completed.returncode != 0 or status != "detected":
        rendered = " ".join(_quote_part(part) for part in _sanitized_command(command))
        detail = output if output else f"return code {completed.returncode}"
        raise ValueError(
            "HackRF external clock requested but CLKIN was not detected: "
            f"{rendered}: {detail}"
        )
    return {
        "checked": True,
        "checked_utc": utc_timestamp_ns(),
        "status": status,
        "tool": Path(_hackrf_clock_executable(config)).name,
        "output": output,
    }


def _hackrf_clock_command(config: HackrfCaptureConfig) -> list[str]:
    executable = _hackrf_clock_executable(config)
    command = [*config.command_prefix, executable]
    if config.device_serial:
        command.extend(["-d", config.device_serial])
    command.append("-i")
    return command


def _hackrf_clock_executable(config: HackrfCaptureConfig) -> str:
    return config.clock_executable or _derive_hackrf_clock_executable(config.executable)


def _derive_hackrf_clock_executable(hackrf_transfer_executable: str) -> str:
    path = Path(hackrf_transfer_executable)
    name = path.name
    if name in {"hackrf_transfer", "hackrf_transfer.exe"}:
        clock_name = "hackrf_clock.exe" if name.endswith(".exe") else "hackrf_clock"
        if path.parent == Path("."):
            return clock_name
        return str(path.with_name(clock_name))
    return "hackrf_clock"


def _parse_clkin_status(output: str) -> str:
    normalized = output.lower()
    if "no clock signal" in normalized or "not detected" in normalized:
        return "missing"
    if "clock signal detected" in normalized:
        return "detected"
    return "unknown"


def _tools_source(config: HackrfCaptureConfig) -> str:
    prefix = config.command_prefix
    if len(prefix) >= 4 and prefix[:3] == ("mamba", "run", "-n"):
        return f"mamba:{prefix[3]}"
    return "path"


def _quote_part(part: str) -> str:
    if not part or any(character.isspace() for character in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
