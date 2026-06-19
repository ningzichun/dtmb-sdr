"""Utilities for ranking HackRF sweep captures by DTMB-sized windows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SweepBin:
    """One power bin from a `hackrf_sweep` CSV row."""

    frequency_hz: float
    power_db: float
    bin_width_hz: float
    row_start_hz: float
    row_stop_hz: float
    sample_count: int


@dataclass(frozen=True)
class SweepWindow:
    """Power summary for one candidate RF channel window."""

    start_hz: int
    stop_hz: int
    center_hz: int
    mean_db: float
    max_db: float
    min_db: float
    spread_db: float
    bin_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "center_hz": self.center_hz,
            "mean_db": self.mean_db,
            "max_db": self.max_db,
            "min_db": self.min_db,
            "spread_db": self.spread_db,
            "bin_count": self.bin_count,
        }


def load_sweep_bins(path: str | Path) -> list[SweepBin]:
    """Read HackRF sweep CSV output into frequency/power bins."""

    bins: list[SweepBin] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 7:
                raise ValueError(f"sweep row {row_number} has too few fields")
            try:
                start_hz = float(row[2])
                stop_hz = float(row[3])
                bin_width_hz = float(row[4])
                sample_count = int(float(row[5]))
                powers = [float(value) for value in row[6:]]
            except ValueError as exc:
                raise ValueError(f"sweep row {row_number} contains non-numeric fields") from exc
            if bin_width_hz <= 0 or stop_hz <= start_hz:
                raise ValueError(f"sweep row {row_number} has invalid frequency range")
            for index, power_db in enumerate(powers):
                frequency_hz = start_hz + (index + 0.5) * bin_width_hz
                if frequency_hz < stop_hz:
                    bins.append(
                        SweepBin(
                            frequency_hz=frequency_hz,
                            power_db=power_db,
                            bin_width_hz=bin_width_hz,
                            row_start_hz=start_hz,
                            row_stop_hz=stop_hz,
                            sample_count=sample_count,
                        )
                    )
    return sorted(bins, key=lambda item: item.frequency_hz)


def rank_sweep_windows(
    bins: Sequence[SweepBin],
    *,
    bandwidth_hz: int = 8_000_000,
    step_hz: int = 1_000_000,
    top: int = 10,
) -> list[SweepWindow]:
    """Rank candidate channel windows by mean sweep power."""

    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if step_hz <= 0:
        raise ValueError("step_hz must be positive")
    if top <= 0:
        raise ValueError("top must be positive")
    if not bins:
        return []

    frequencies = np.asarray([item.frequency_hz for item in bins], dtype=np.float64)
    powers = np.asarray([item.power_db for item in bins], dtype=np.float64)
    start = int(np.floor(float(np.min(frequencies)) / step_hz) * step_hz)
    stop = int(np.ceil(float(np.max(frequencies)) / step_hz) * step_hz)

    windows: list[SweepWindow] = []
    for window_start in range(start, stop - bandwidth_hz + step_hz, step_hz):
        window_stop = window_start + bandwidth_hz
        mask = (frequencies >= window_start) & (frequencies < window_stop)
        if not np.any(mask):
            continue
        values = powers[mask]
        windows.append(
            SweepWindow(
                start_hz=window_start,
                stop_hz=window_stop,
                center_hz=window_start + bandwidth_hz // 2,
                mean_db=float(np.mean(values)),
                max_db=float(np.max(values)),
                min_db=float(np.min(values)),
                spread_db=float(np.max(values) - np.min(values)),
                bin_count=int(values.size),
            )
        )
    return sorted(windows, key=lambda item: (item.mean_db, -item.spread_db), reverse=True)[:top]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-sweep-report",
        description="Rank HackRF sweep CSV data by DTMB-sized RF windows.",
    )
    parser.add_argument("sweep_csv", type=Path, help="CSV produced by hackrf_sweep")
    parser.add_argument("--bandwidth", type=_positive_int, default=8_000_000)
    parser.add_argument("--step", type=_positive_int, default=1_000_000)
    parser.add_argument("--top", type=_positive_int, default=10)
    parser.add_argument("--json", type=Path, help="Optional path to write report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bins = load_sweep_bins(args.sweep_csv)
    windows = rank_sweep_windows(
        bins,
        bandwidth_hz=args.bandwidth,
        step_hz=args.step,
        top=args.top,
    )
    report = {
        "sweep_csv": str(args.sweep_csv),
        "bin_count": len(bins),
        "native_bin_width_hz": sorted(
            {float(item.bin_width_hz) for item in bins}
        ),
        "native_sample_counts": sorted({int(item.sample_count) for item in bins}),
        "bandwidth_hz": args.bandwidth,
        "step_hz": args.step,
        "windows": [window.to_dict() for window in windows],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
