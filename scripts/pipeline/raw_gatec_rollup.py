"""Scan raw C=3780 system-information stability over a long capture."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from dtmb.ci8 import read_ci8
from dtmb.conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from dtmb.frame_sync import DTMB_SYMBOL_RATE_SPS
from dtmb.frames import iter_signal_frames
from dtmb.frequency import (
    frame_body_fft,
    frequency_deinterleave,
    split_system_info_and_data,
)
from dtmb.system_info import classify_system_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raw-gatec-rollup")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--symbol-rate", type=int, default=DTMB_SYMBOL_RATE_SPS)
    parser.add_argument("--cfo-hz", type=float, default=0.0)
    parser.add_argument("--phase", type=int, action="append", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-length", type=int, action="append", default=[])
    parser.add_argument("--target-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    window_lengths = args.window_length or [130, 240, 300, 510, 750]
    target_indexes = args.target_index or [23, 24]
    symbols = remove_dc(read_ci8(args.capture))
    if args.sample_rate != args.symbol_rate:
        symbols, _up, _down = resample_to_symbol_rate(
            symbols,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
        )
    if args.cfo_hz:
        symbols = frequency_shift(
            symbols,
            sample_rate_sps=args.symbol_rate,
            shift_hz=-args.cfo_hz,
        )

    rows = [
        _scan_phase(
            symbols,
            phase_offset=phase,
            max_frames=args.max_frames,
            window_lengths=window_lengths,
            target_indexes=target_indexes,
        )
        for phase in args.phase
    ]
    report = {
        "capture": str(args.capture),
        "sample_rate_sps": int(args.sample_rate),
        "symbol_rate_sps": int(args.symbol_rate),
        "cfo_hz": float(args.cfo_hz),
        "phases": [int(phase) for phase in args.phase],
        "window_lengths": [int(length) for length in window_lengths],
        "target_indexes": [int(index) for index in target_indexes],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _scan_phase(
    symbols,
    *,
    phase_offset: int,
    max_frames: int | None,
    window_lengths: list[int],
    target_indexes: list[int],
) -> dict[str, Any]:
    top_indexes: list[int | None] = []
    for frame in iter_signal_frames(
        symbols,
        mode="pn945",
        phase_offset=phase_offset,
        max_frames=max_frames,
    ):
        spectrum = frame_body_fft(frame.body)
        deinterleaved = frequency_deinterleave(spectrum)
        system_info, _data = split_system_info_and_data(deinterleaved)
        matches = classify_system_info(system_info, frame_body_modes=("C3780",))
        top_indexes.append(int(matches[0].index) if matches else None)

    summary: dict[str, Any] = {
        "phase_offset": int(phase_offset),
        "frame_count": int(len(top_indexes)),
        "top_counts": [
            {"index": index, "count": count}
            for index, count in Counter(top_indexes).most_common(10)
        ],
    }
    for length in window_lengths:
        for index in target_indexes:
            count, start = _best_rolling_count(top_indexes, index=index, length=length)
            summary[f"best_idx{index}_len{length}_count"] = int(count)
            summary[f"best_idx{index}_len{length}_start"] = (
                int(start) if start is not None else None
            )
    return summary


def _best_rolling_count(
    values: list[int | None],
    *,
    index: int,
    length: int,
) -> tuple[int, int | None]:
    if length <= 0:
        raise ValueError("window length must be positive")
    if len(values) < length:
        return 0, None
    count = sum(1 for value in values[:length] if value == index)
    best_count = count
    best_start = 0
    for start in range(1, len(values) - length + 1):
        if values[start - 1] == index:
            count -= 1
        if values[start + length - 1] == index:
            count += 1
        if count > best_count:
            best_count = count
            best_start = start
    return best_count, best_start


if __name__ == "__main__":
    raise SystemExit(main())
