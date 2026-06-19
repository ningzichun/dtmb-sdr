"""Stage 0.5: digitally shift/crop/resample a wideband CI8 capture.

Inputs:
    <input>.ci8
    <input>.ci8.json

Outputs:
    <output>.ci8
    <output>.ci8.json
    analysis JSON describing the crop pipeline

This stage is intentionally file-domain and uses GNU Radio stock blocks so a
20 MHz wide HackRF capture can be:

1. mixed away from hardware DC,
2. low-pass cropped to the wanted DTMB channel,
3. resampled to a cheaper downstream working rate,
4. written back as repository-standard CI8.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from _common import load_capture_sidecar, read_json, sha256_of, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-crop-capture")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--target-frequency-hz", type=int, required=True)
    parser.add_argument("--tuner-frequency-hz", type=int, required=True)
    parser.add_argument("--frequency-shift-hz", type=float, required=True)
    parser.add_argument("--output-sample-rate", type=int, required=True)
    parser.add_argument("--output-bandwidth-hz", type=int, default=8_000_000)
    parser.add_argument("--lowpass-cutoff-hz", type=float, default=4_200_000.0)
    parser.add_argument("--lowpass-transition-hz", type=float, default=800_000.0)
    parser.add_argument("--fractional-bw", type=float, default=0.45)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_sidecar = load_capture_sidecar(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)

    output_rate = int(args.output_sample_rate)
    input_rate = int(input_sidecar["sample_rate_sps"])
    ratio = Fraction(output_rate, input_rate).limit_denominator(10_000)
    _run_gnuradio_crop(
        input_path=args.input,
        output_path=args.output,
        input_sample_rate_sps=input_rate,
        frequency_shift_hz=float(args.frequency_shift_hz),
        lowpass_cutoff_hz=float(args.lowpass_cutoff_hz),
        lowpass_transition_hz=float(args.lowpass_transition_hz),
        interpolation=ratio.numerator,
        decimation=ratio.denominator,
        fractional_bw=float(args.fractional_bw),
    )

    output_sidecar = args.output.with_name(args.output.name + ".json")
    metadata = build_crop_metadata(
        input_metadata=input_sidecar,
        input_path=args.input,
        output_path=args.output,
        target_frequency_hz=int(args.target_frequency_hz),
        tuner_frequency_hz=int(args.tuner_frequency_hz),
        frequency_shift_hz=float(args.frequency_shift_hz),
        output_sample_rate_sps=output_rate,
        output_bandwidth_hz=int(args.output_bandwidth_hz),
        lowpass_cutoff_hz=float(args.lowpass_cutoff_hz),
        lowpass_transition_hz=float(args.lowpass_transition_hz),
        interpolation=ratio.numerator,
        decimation=ratio.denominator,
    )
    write_json(output_sidecar, metadata)

    analysis = {
        "schema": "dtmb.pipeline.crop_capture.v1",
        "input_capture_path": str(args.input),
        "input_capture_sidecar_path": str(args.input.with_name(args.input.name + ".json")),
        "output_capture_path": str(args.output),
        "output_capture_sidecar_path": str(output_sidecar),
        "input_sha256": sha256_of(args.input),
        "output_sha256": sha256_of(args.output),
        "input_byte_count": int(Path(args.input).stat().st_size),
        "output_byte_count": int(Path(args.output).stat().st_size),
        "input_sample_rate_sps": input_rate,
        "output_sample_rate_sps": output_rate,
        "target_frequency_hz": int(args.target_frequency_hz),
        "tuner_frequency_hz": int(args.tuner_frequency_hz),
        "frequency_shift_hz": float(args.frequency_shift_hz),
        "lowpass_cutoff_hz": float(args.lowpass_cutoff_hz),
        "lowpass_transition_hz": float(args.lowpass_transition_hz),
        "interpolation": int(ratio.numerator),
        "decimation": int(ratio.denominator),
        "fractional_bw": float(args.fractional_bw),
    }
    write_json(args.analysis_json, analysis)
    return 0


def build_crop_metadata(
    *,
    input_metadata: dict[str, Any],
    input_path: Path,
    output_path: Path,
    target_frequency_hz: int,
    tuner_frequency_hz: int,
    frequency_shift_hz: float,
    output_sample_rate_sps: int,
    output_bandwidth_hz: int,
    lowpass_cutoff_hz: float,
    lowpass_transition_hz: float,
    interpolation: int,
    decimation: int,
) -> dict[str, Any]:
    output_byte_count = int(output_path.stat().st_size)
    output_sample_count = output_byte_count // 2
    inherited = {
        key: input_metadata[key]
        for key in (
            "amp",
            "lna_gain",
            "vga_gain",
            "antenna",
            "location",
        )
        if key in input_metadata
    }
    notes = str(input_metadata.get("notes") or "").strip()
    crop_note = (
        "wideband_capture_crop"
        f" tuner_frequency_hz={tuner_frequency_hz}"
        f" target_frequency_hz={target_frequency_hz}"
        f" frequency_shift_hz={frequency_shift_hz}"
        f" output_sample_rate_sps={output_sample_rate_sps}"
    )
    if notes:
        notes = f"{notes}; {crop_note}"
    else:
        notes = crop_note

    metadata: dict[str, Any] = {
        "format": "ci8",
        "created_utc": input_metadata.get("created_utc"),
        "frequency_hz": int(target_frequency_hz),
        "sample_rate_sps": int(output_sample_rate_sps),
        "bandwidth_hz": int(output_bandwidth_hz),
        "duration_s": float(output_sample_count) / float(output_sample_rate_sps),
        "sample_count": int(output_sample_count),
        "byte_count": int(output_byte_count),
        "notes": notes,
        "source_capture_path": str(input_path),
        "source_capture_sidecar_path": str(input_path.with_name(input_path.name + ".json")),
        "source_frequency_hz": input_metadata.get("frequency_hz"),
        "source_sample_rate_sps": input_metadata.get("sample_rate_sps"),
        "capture_tuner_frequency_hz": int(tuner_frequency_hz),
        "capture_target_frequency_hz": int(target_frequency_hz),
        "capture_frequency_shift_hz": float(frequency_shift_hz),
        "capture_lowpass_cutoff_hz": float(lowpass_cutoff_hz),
        "capture_lowpass_transition_hz": float(lowpass_transition_hz),
        "capture_resample_interpolation": int(interpolation),
        "capture_resample_decimation": int(decimation),
        "capture_workflow": "wideband_capture_crop",
        "hackrf_tools_source": input_metadata.get("hackrf_tools_source"),
    }
    metadata.update(inherited)
    return metadata


def _run_gnuradio_crop(
    *,
    input_path: Path,
    output_path: Path,
    input_sample_rate_sps: int,
    frequency_shift_hz: float,
    lowpass_cutoff_hz: float,
    lowpass_transition_hz: float,
    interpolation: int,
    decimation: int,
    fractional_bw: float,
) -> None:
    from gnuradio import blocks, filter, gr
    from gnuradio.filter import firdes
    from gnuradio.fft import window

    top_block = gr.top_block("dtmb_wideband_crop")
    source = blocks.file_source(gr.sizeof_char, str(input_path), False)
    to_complex = blocks.interleaved_char_to_complex(False, 128.0)
    phase_inc = (2.0 * 3.141592653589793 * float(frequency_shift_hz)) / float(
        input_sample_rate_sps
    )
    rotator = blocks.rotator_cc(phase_inc, False)
    taps = firdes.low_pass(
        1.0,
        float(input_sample_rate_sps),
        float(lowpass_cutoff_hz),
        float(lowpass_transition_hz),
        window.WIN_HAMMING,
        6.76,
    )
    lowpass = filter.fir_filter_ccf(1, taps)
    resampler = filter.rational_resampler_ccf(
        interpolation=int(interpolation),
        decimation=int(decimation),
        taps=[],
        fractional_bw=float(fractional_bw),
    )
    to_ci8 = blocks.complex_to_interleaved_char(False, 127.0)
    sink = blocks.file_sink(gr.sizeof_char, str(output_path), False)
    sink.set_unbuffered(True)

    top_block.connect(source, to_complex)
    top_block.connect(to_complex, rotator)
    top_block.connect(rotator, lowpass)
    top_block.connect(lowpass, resampler)
    top_block.connect(resampler, to_ci8)
    top_block.connect(to_ci8, sink)
    top_block.run()


if __name__ == "__main__":
    raise SystemExit(main())
