"""Summarize calibrated rolling-H pass islands and their source-frame spans."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dtmb.frequency import DATA_SYMBOLS_PER_FRAME
from dtmb.ldpc import dtmb_ldpc_profile
from dtmb.qam import QAM_DEFINITIONS
from dtmb.symbol_interleaver import convolutional_deinterleave_source_indices


def analyze_clean_islands(
    h_score_path: str | Path,
    *,
    threshold: float | None = None,
    fec_rate_index: int = 2,
    qam_mode: str = "64qam",
    interleaver_mode: str = "mode2",
    interleaver_phase: int = 0,
    codewords_per_fec_frame: int = 3,
    wideband_csv: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(h_score_path)
    h_score = json.loads(source.read_text(encoding="utf-8"))
    windows = h_score.get("windows")
    if not isinstance(windows, list):
        raise ValueError("H-score report must contain a windows array")
    selected_threshold = float(
        threshold if threshold is not None else h_score.get("window_threshold", 0.44)
    )
    if not math.isfinite(selected_threshold) or not 0.0 <= selected_threshold <= 1.0:
        raise ValueError("threshold must be finite and within 0..1")
    if codewords_per_fec_frame <= 0:
        raise ValueError("codewords_per_fec_frame must be positive")
    profile = dtmb_ldpc_profile(fec_rate_index)
    bits_per_symbol = int(QAM_DEFINITIONS[qam_mode].bits_per_symbol)
    if profile.codeword_bits % bits_per_symbol:
        raise ValueError("LDPC codeword bits must contain whole QAM symbols")

    passing = sorted(
        (
            {
                "start_codeword": int(row["start_codeword"]),
                "end_codeword": int(row["end_codeword"]),
                "mean_syndrome_ratio": float(row["mean_syndrome_ratio"]),
            }
            for row in windows
            if float(row["mean_syndrome_ratio"]) <= selected_threshold
        ),
        key=lambda row: row["start_codeword"],
    )
    step = int(h_score.get("window_step_codewords") or _infer_step(passing) or 1)
    raw_islands: list[list[dict[str, Any]]] = []
    for window in passing:
        if (
            not raw_islands
            or window["start_codeword"]
            != raw_islands[-1][-1]["start_codeword"] + step
        ):
            raw_islands.append([window])
        else:
            raw_islands[-1].append(window)

    wideband_rows = _read_wideband_rows(wideband_csv)
    islands = [
        _island_report(
            rows,
            profile_codeword_bits=int(profile.codeword_bits),
            bits_per_symbol=bits_per_symbol,
            interleaver_mode=interleaver_mode,
            interleaver_phase=interleaver_phase,
            codewords_per_fec_frame=codewords_per_fec_frame,
            wideband_rows=wideband_rows,
        )
        for rows in raw_islands
    ]
    return {
        "schema": "dtmb.clean_h_islands.v1",
        "stage": "clean_h_island_analysis",
        "input": str(source),
        "threshold": selected_threshold,
        "window_codewords": h_score.get("window_codewords"),
        "window_step_codewords": step,
        "total_windows": len(windows),
        "passing_windows": len(passing),
        "island_count": len(islands),
        "max_island_windows": max(
            (int(row["passing_windows"]) for row in islands),
            default=0,
        ),
        "fec_rate_index": fec_rate_index,
        "qam_mode": qam_mode,
        "interleaver_mode": interleaver_mode,
        "interleaver_phase": interleaver_phase,
        "codewords_per_fec_frame": codewords_per_fec_frame,
        "wideband_csv": str(wideband_csv) if wideband_csv is not None else None,
        "islands": islands,
        "verdict": "clean_islands_found" if islands else "no_clean_islands",
    }


def _island_report(
    windows: list[dict[str, Any]],
    *,
    profile_codeword_bits: int,
    bits_per_symbol: int,
    interleaver_mode: str,
    interleaver_phase: int,
    codewords_per_fec_frame: int,
    wideband_rows: list[dict[str, float]],
) -> dict[str, Any]:
    start_codeword = int(windows[0]["start_codeword"])
    end_codeword = int(windows[-1]["end_codeword"])
    symbols_per_codeword = profile_codeword_bits // bits_per_symbol
    source_symbols = convolutional_deinterleave_source_indices(
        (end_codeword - start_codeword) * symbols_per_codeword,
        mode=interleaver_mode,  # type: ignore[arg-type]
        phase=interleaver_phase,
        output_start=start_codeword * symbols_per_codeword,
        after_latency_discard=True,
    )
    valid = source_symbols[source_symbols >= 0]
    source_frames = np.unique(valid // DATA_SYMBOLS_PER_FRAME)
    source_min = int(source_frames.min()) if source_frames.size else None
    source_max = int(source_frames.max()) if source_frames.size else None
    overlapping_models = [
        row
        for row in wideband_rows
        if source_min is not None
        and source_max is not None
        and row["first_output_frame"] <= source_max
        and row["first_output_frame"] + row["output_frames"] > source_min
    ]
    return {
        "start_codeword": start_codeword,
        "end_codeword": end_codeword,
        "passing_windows": len(windows),
        "best_mean_syndrome_ratio": min(
            float(row["mean_syndrome_ratio"]) for row in windows
        ),
        "output_fec_frame_start": start_codeword // codewords_per_fec_frame,
        "output_fec_frame_end": math.ceil(end_codeword / codewords_per_fec_frame),
        "source_frame_min": source_min,
        "source_frame_max": source_max,
        "unique_source_frames": int(source_frames.size),
        "wideband": _wideband_summary(overlapping_models),
    }


def _infer_step(windows: list[dict[str, Any]]) -> int | None:
    differences = [
        int(right["start_codeword"]) - int(left["start_codeword"])
        for left, right in zip(windows, windows[1:])
        if int(right["start_codeword"]) > int(left["start_codeword"])
    ]
    return min(differences) if differences else None


def _read_wideband_rows(path: str | Path | None) -> list[dict[str, float]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value not in (None, "")}
            for row in csv.DictReader(handle)
        ]


def _wideband_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    def distribution(key: str) -> dict[str, float | int | None]:
        values = np.asarray(
            [row[key] for row in rows if key in row and math.isfinite(row[key])],
            dtype=np.float64,
        )
        if values.size == 0:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    return {
        "model_count": len(rows),
        "noise_variance": distribution("noise_variance"),
        "span_symbols": distribution("span_symbols"),
        "significant_taps": distribution("significant_taps"),
        "rotation_symbols": distribution("rotation_symbols"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze-clean-islands")
    parser.add_argument("h_score_json", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--fec-rate", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--qam", choices=tuple(QAM_DEFINITIONS), default="64qam")
    parser.add_argument("--interleaver-mode", choices=("mode1", "mode2"), default="mode2")
    parser.add_argument("--interleaver-phase", type=int, default=0)
    parser.add_argument("--codewords-per-fec-frame", type=int, default=3)
    parser.add_argument("--wideband-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_clean_islands(
        args.h_score_json,
        threshold=args.threshold,
        fec_rate_index=args.fec_rate,
        qam_mode=args.qam,
        interleaver_mode=args.interleaver_mode,
        interleaver_phase=args.interleaver_phase,
        codewords_per_fec_frame=args.codewords_per_fec_frame,
        wideband_csv=args.wideband_csv,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
