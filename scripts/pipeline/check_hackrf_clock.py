"""Preflight HackRF CLKIN before live native runs.

The bounded capture wrapper records CLKIN status in the CI8 sidecar. Native
live targets do not have a sidecar, so this stage provides the same fail-fast
check before starting an unbounded `hackrf_transfer` pipe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any, Sequence


EXTERNAL_CLOCK_SOURCES = {"external-10mhz", "gpsdo-10mhz"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-check-hackrf-clock")
    parser.add_argument(
        "--clock-source",
        choices=("internal", "external-10mhz", "gpsdo-10mhz", "unknown"),
        default="internal",
    )
    parser.add_argument("--external-reference-hz", type=int)
    parser.add_argument("--hackrf-clock", default="hackrf_clock")
    parser.add_argument("--device-serial")
    parser.add_argument("--log", type=Path)
    return parser


def check_clock(
    *,
    clock_source: str,
    external_reference_hz: int | None,
    hackrf_clock: str,
    device_serial: str | None = None,
    log_path: Path | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Verify CLKIN when an external HackRF reference was requested."""

    if clock_source not in {"internal", "external-10mhz", "gpsdo-10mhz", "unknown"}:
        raise ValueError("clock_source must be internal, external-10mhz, gpsdo-10mhz, or unknown")
    if clock_source not in EXTERNAL_CLOCK_SOURCES:
        return {
            "required": False,
            "clock_source": clock_source,
            "status": "not_required",
        }
    if external_reference_hz != 10_000_000:
        raise ValueError(f"{clock_source} requires external_reference_hz=10000000")

    command = [hackrf_clock]
    if device_serial:
        command.extend(["-d", device_serial])
    command.append("-i")
    completed = runner(command, check=False, capture_output=True, text=True)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output + ("\n" if output else ""), encoding="utf-8")
    status = _parse_clkin_status(output)
    report = {
        "required": True,
        "clock_source": clock_source,
        "external_reference_hz": external_reference_hz,
        "status": status,
        "returncode": completed.returncode,
        "output": output,
    }
    if completed.returncode != 0 or status != "detected":
        detail = output if output else f"return code {completed.returncode}"
        raise RuntimeError(f"HackRF external clock requested but CLKIN was not detected: {detail}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = check_clock(
            clock_source=args.clock_source,
            external_reference_hz=args.external_reference_hz,
            hackrf_clock=args.hackrf_clock,
            device_serial=args.device_serial,
            log_path=args.log,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[clock] {exc}", flush=True)
        return 2
    if report["required"]:
        print(f"[clock] {report['output']}", flush=True)
    return 0


def _parse_clkin_status(output: str) -> str:
    normalized = output.lower()
    if "no clock signal" in normalized or "not detected" in normalized:
        return "missing"
    if "clock signal detected" in normalized:
        return "detected"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
