"""DTMB acquisition refinement diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    detect_pn_cyclic_extension_trains,
    detect_pn_frame_trains,
    estimate_cfo_from_pn_cyclic_extension,
    score_pn_family_delay_train,
)
from .pn import PnMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-acquire",
        description="Refine DTMB PN acquisition with delay-tolerant PN-family scoring.",
    )
    parser.add_argument("capture", type=Path, help="Path to raw CI8 capture")
    parser.add_argument("--sample-rate", type=_positive_int, default=20_000_000)
    parser.add_argument("--symbol-rate", type=_positive_int, default=DTMB_SYMBOL_RATE_SPS)
    parser.add_argument("--max-samples", type=_positive_int, default=3_000_000)
    parser.add_argument("--frequency-shift", type=float, default=0.0)
    parser.add_argument(
        "--frequency-shift-sweep",
        help="Optional START:STOP:STEP frequency-shift sweep in Hz, inclusive.",
    )
    parser.add_argument(
        "--input-skip",
        type=_non_negative_int,
        default=0,
        help="Drop this many raw input samples before conditioning.",
    )
    parser.add_argument(
        "--input-skip-sweep",
        help="Optional START:STOP:STEP raw-sample skip sweep, inclusive.",
    )
    parser.add_argument(
        "--mode",
        choices=("pn420", "pn595", "pn945", "auto"),
        default="auto",
    )
    parser.add_argument("--cyclic-trains", type=_positive_int, default=3)
    parser.add_argument("--family-frames", type=_positive_int, default=12)
    parser.add_argument("--max-delay", type=_non_negative_int, default=256)
    parser.add_argument("--json", type=Path, help="Optional path to write diagnostic JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frequency_shift_sweep or args.input_skip_sweep:
        diagnostics = sweep_acquisition(
            args.capture,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
            max_samples=args.max_samples,
            frequency_shifts_hz=_parse_float_sweep(args.frequency_shift_sweep)
            if args.frequency_shift_sweep
            else (args.frequency_shift,),
            input_skip_samples=_parse_int_sweep(args.input_skip_sweep)
            if args.input_skip_sweep
            else (args.input_skip,),
            mode=args.mode,
            cyclic_trains=args.cyclic_trains,
            family_frames=args.family_frames,
            max_delay_symbols=args.max_delay,
        )
    else:
        diagnostics = refine_acquisition(
            args.capture,
            sample_rate_sps=args.sample_rate,
            symbol_rate_sps=args.symbol_rate,
            max_samples=args.max_samples,
            frequency_shift_hz=args.frequency_shift,
            input_skip_samples=args.input_skip,
            mode=args.mode,
            cyclic_trains=args.cyclic_trains,
            family_frames=args.family_frames,
            max_delay_symbols=args.max_delay,
        )
    encoded = json.dumps(diagnostics, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def refine_acquisition(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 3_000_000,
    frequency_shift_hz: float = 0.0,
    input_skip_samples: int = 0,
    mode: PnMode | str = "auto",
    cyclic_trains: int = 3,
    family_frames: int = 12,
    max_delay_symbols: int = 256,
) -> dict[str, Any]:
    """Return cyclic-extension and delay-tolerant PN-family acquisition scores."""

    if cyclic_trains <= 0:
        raise ValueError("cyclic_trains must be positive")
    if input_skip_samples < 0:
        raise ValueError("input_skip_samples must be non-negative")
    samples = read_ci8(
        capture_path,
        max_samples=max_samples,
        skip_samples=input_skip_samples,
    )
    samples = remove_dc(samples)
    if frequency_shift_hz:
        samples = frequency_shift(
            samples,
            sample_rate_sps=sample_rate_sps,
            shift_hz=frequency_shift_hz,
        )
    if sample_rate_sps == symbol_rate_sps:
        symbols = samples
        up = 1
        down = 1
    else:
        symbols, up, down = resample_to_symbol_rate(
            samples,
            sample_rate_sps=sample_rate_sps,
            symbol_rate_sps=symbol_rate_sps,
        )
    modes = _modes(mode)
    cyclic_modes = tuple(mode for mode in modes if mode in ("pn420", "pn945"))
    direct_modes = tuple(mode for mode in modes if mode == "pn595")
    cyclic = (
        detect_pn_cyclic_extension_trains(
            symbols,
            modes=cyclic_modes,
            max_trains_per_mode=cyclic_trains,
        )
        if cyclic_modes
        else []
    )
    direct = (
        detect_pn_frame_trains(
            symbols,
            modes=direct_modes,
            max_trains_per_mode=cyclic_trains,
        )
        if direct_modes
        else []
    )

    candidates: list[dict[str, Any]] = []
    for train in cyclic:
        cfo = estimate_cfo_from_pn_cyclic_extension(
            symbols,
            mode=train.mode,
            phase_offset=train.phase_offset,
            symbol_rate_sps=symbol_rate_sps,
        )
        corrected = symbols
        if cfo is not None:
            corrected = frequency_shift(
                corrected,
                sample_rate_sps=symbol_rate_sps,
                shift_hz=-cfo,
            )
        family = score_pn_family_delay_train(
            corrected,
            mode=train.mode,
            phase_offset=train.phase_offset,
            max_frames=family_frames,
            max_delay_symbols=max_delay_symbols,
        )
        data = train.to_dict()
        data["acquisition_kind"] = "cyclic-extension"
        data["coarse_cfo_hz"] = cfo
        data["delay_family_train"] = family.to_dict()
        candidates.append(data)
    for train in direct:
        data = train.to_dict()
        data["acquisition_kind"] = "direct-pn-correlation"
        data["coarse_cfo_hz"] = None
        data["delay_family_train"] = {
            "mode": train.mode,
            "phase_offset": train.phase_offset,
            "frame_symbols": train.frame_symbols,
            "header_symbols": train.header_symbols,
            "mean_metric": 0.0,
            "max_metric": 0.0,
            "hit_count": 0,
            "observed_frames": 0,
            "median_delay_symbols": 0.0,
            "delay_mad_symbols": 0.0,
            "dominant_pn_phase": None,
            "dominant_pn_phase_count": 0,
            "frame_matches": [],
        }
        candidates.append(data)

    candidates.sort(
        key=lambda item: (
            item["delay_family_train"]["hit_count"],
            item["delay_family_train"]["mean_metric"],
            item["hit_count"],
            item["mean_metric"],
        ),
        reverse=True,
    )
    return {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "resample_up": up,
        "resample_down": down,
        "frequency_shift_hz": frequency_shift_hz,
        "input_skip_samples": input_skip_samples,
        "analyzed_symbols": int(symbols.size),
        "mode": mode,
        "max_delay_symbols": max_delay_symbols,
        "candidates": candidates,
        "selected": candidates[0] if candidates else None,
    }


def sweep_acquisition(
    capture_path: str | Path,
    *,
    sample_rate_sps: int,
    symbol_rate_sps: int = DTMB_SYMBOL_RATE_SPS,
    max_samples: int = 3_000_000,
    frequency_shifts_hz: Sequence[float] = (0.0,),
    input_skip_samples: Sequence[int] = (0,),
    mode: PnMode | str = "auto",
    cyclic_trains: int = 3,
    family_frames: int = 12,
    max_delay_symbols: int = 256,
) -> dict[str, Any]:
    """Run acquisition refinement across frequency-shift and input-skip grids."""

    results = []
    for skip in input_skip_samples:
        for shift in frequency_shifts_hz:
            result = refine_acquisition(
                capture_path,
                sample_rate_sps=sample_rate_sps,
                symbol_rate_sps=symbol_rate_sps,
                max_samples=max_samples,
                frequency_shift_hz=float(shift),
                input_skip_samples=int(skip),
                mode=mode,
                cyclic_trains=cyclic_trains,
                family_frames=family_frames,
                max_delay_symbols=max_delay_symbols,
            )
            results.append(result)

    results.sort(key=_sweep_sort_key, reverse=True)
    return {
        "capture_path": str(capture_path),
        "input_sample_rate_sps": sample_rate_sps,
        "symbol_rate_sps": symbol_rate_sps,
        "frequency_shifts_hz": [float(value) for value in frequency_shifts_hz],
        "input_skip_samples": [int(value) for value in input_skip_samples],
        "results": results,
        "selected": results[0] if results else None,
    }


def _modes(mode: PnMode | str) -> tuple[PnMode, ...]:
    if mode == "auto":
        return ("pn420", "pn595", "pn945")
    return (mode,)  # type: ignore[return-value]


def _sweep_sort_key(result: dict[str, Any]) -> tuple[float, ...]:
    selected = result.get("selected") or {}
    family = selected.get("delay_family_train") or {}
    return (
        float(family.get("hit_count") or 0),
        float(family.get("mean_metric") or 0.0),
        float(selected.get("hit_count") or 0),
        float(selected.get("mean_metric") or 0.0),
    )


def _parse_float_sweep(spec: str) -> tuple[float, ...]:
    start, stop, step = _parse_sweep_parts(spec, float)
    values = []
    current = start
    epsilon = abs(step) * 1e-9
    if step > 0:
        while current <= stop + epsilon:
            values.append(float(current))
            current += step
    else:
        while current >= stop - epsilon:
            values.append(float(current))
            current += step
    return tuple(values)


def _parse_int_sweep(spec: str) -> tuple[int, ...]:
    start, stop, step = _parse_sweep_parts(spec, int)
    if step == 0:
        raise argparse.ArgumentTypeError("sweep step must not be zero")
    values = []
    current = start
    if step > 0:
        while current <= stop:
            values.append(int(current))
            current += step
    else:
        while current >= stop:
            values.append(int(current))
            current += step
    return tuple(values)


def _parse_sweep_parts(spec: str, parser: Any) -> tuple[Any, Any, Any]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("sweep must be START:STOP:STEP")
    start, stop, step = (parser(part) for part in parts)
    if step == 0:
        raise argparse.ArgumentTypeError("sweep step must not be zero")
    if (stop - start) * step < 0:
        raise argparse.ArgumentTypeError("sweep step sign does not reach stop")
    return start, stop, step


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
