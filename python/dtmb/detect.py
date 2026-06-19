"""Command line entry point for offline DTMB capture diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analysis import analyze_ci8_capture
from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    detect_pn_frame_trains,
    detect_pn_cyclic_extension_trains,
    detect_pn_phase_family_trains,
    detect_pn_headers,
    estimate_cfo_from_pn_cyclic_extension,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-detect",
        description="Analyze a raw HackRF CI8 capture before DTMB demodulation.",
    )
    parser.add_argument("capture", type=Path, help="Path to a raw CI8 capture")
    parser.add_argument(
        "--sample-rate",
        type=_positive_int,
        help="Capture sample rate in samples/second. Optional if metadata exists.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to sidecar JSON metadata. Defaults to <capture>.json if present.",
    )
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=2_000_000,
        help="Maximum complex samples to analyze for quick diagnostics.",
    )
    parser.add_argument(
        "--fft-size",
        type=_positive_int,
        default=16_384,
        help="FFT size for coarse spectrum diagnostics.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write diagnostic JSON.",
    )
    parser.add_argument(
        "--pn-search",
        action="store_true",
        help="Run PN420/PN595/PN945 correlation after resampling to symbol rate.",
    )
    parser.add_argument(
        "--frequency-shift",
        type=float,
        default=0.0,
        help=(
            "Complex frequency shift in Hz applied before PN search. "
            "Use a negative value to move a channel above center down to baseband."
        ),
    )
    parser.add_argument(
        "--no-dc-removal",
        action="store_true",
        help="Do not remove complex DC before PN search.",
    )
    parser.add_argument(
        "--symbol-rate",
        type=_positive_int,
        default=DTMB_SYMBOL_RATE_SPS,
        help="Symbol rate used for PN search.",
    )
    parser.add_argument(
        "--pn-max-symbols",
        type=_positive_int,
        default=250_000,
        help="Maximum symbol-rate samples to use for PN search.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    diagnostics = analyze_ci8_capture(
        args.capture,
        sample_rate_sps=args.sample_rate,
        metadata_path=args.metadata,
        max_samples=args.max_samples,
        fft_size=args.fft_size,
    )
    if args.pn_search:
        sample_rate = int(diagnostics["sample_rate_sps"])
        samples = read_ci8(args.capture, max_samples=args.max_samples)
        dc_removed = not args.no_dc_removal
        if dc_removed:
            samples = remove_dc(samples)
        if args.frequency_shift:
            samples = frequency_shift(
                samples,
                sample_rate_sps=sample_rate,
                shift_hz=args.frequency_shift,
            )
        resampled = False
        up = 1
        down = 1
        if sample_rate != args.symbol_rate:
            samples, up, down = resample_to_symbol_rate(
                samples,
                sample_rate_sps=sample_rate,
                symbol_rate_sps=args.symbol_rate,
            )
            resampled = True
        symbols = samples[: args.pn_max_symbols]
        peaks = detect_pn_headers(symbols)
        trains = detect_pn_frame_trains(symbols)
        extension_trains = detect_pn_cyclic_extension_trains(symbols)
        family_trains = detect_pn_phase_family_trains(symbols)
        extension_train_dicts = []
        for train in extension_trains:
            data = train.to_dict()
            data["coarse_cfo_hz"] = estimate_cfo_from_pn_cyclic_extension(
                symbols,
                mode=train.mode,
                phase_offset=train.phase_offset,
                symbol_rate_sps=args.symbol_rate,
            )
            extension_train_dicts.append(data)
        diagnostics["pn_search"] = {
            "input_sample_rate_sps": sample_rate,
            "symbol_rate_sps": args.symbol_rate,
            "frequency_shift_hz": args.frequency_shift,
            "dc_removed": dc_removed,
            "resampled": resampled,
            "resample_up": up,
            "resample_down": down,
            "analyzed_symbols": int(symbols.size),
            "candidates": [peak.to_dict() for peak in peaks],
            "frame_trains": [train.to_dict() for train in trains],
            "cyclic_extension_trains": extension_train_dicts,
            "phase_family_trains": [train.to_dict() for train in family_trains],
        }
    encoded = json.dumps(diagnostics, indent=2, sort_keys=True)
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
