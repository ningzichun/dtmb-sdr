"""Stage: native post-deinterleaver LLR stream -> tagged pre-LDPC NPZ.

The native live pipeline writes a compact float32 LLR stream plus a JSON
sidecar.  This stage adds deterministic symbol-deinterleaver provenance without
rerunning acquisition, equalization, or demapping.  Every output bit is tagged
with its source frame, data-carrier index, interleaver branch, and QAM plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _common import read_json
from dtmb.frequency import DATA_SYMBOLS_PER_FRAME
from dtmb.ldpc import dtmb_ldpc_profile
from dtmb.qam import QAM_DEFINITIONS
from dtmb.symbol_interleaver import (
    SYMBOL_INTERLEAVERS,
    convolutional_deinterleave_source_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-tag-pre-ldpc",
        description=(
            "Create a tagged pre-LDPC NPZ from a native post-deinterleaver "
            "float32 LLR stream and its framing sidecar."
        ),
    )
    parser.add_argument("--llr", type=Path, required=True)
    parser.add_argument("--llr-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--qam", choices=tuple(QAM_DEFINITIONS))
    parser.add_argument("--fec-rate", type=int, choices=(1, 2, 3))
    parser.add_argument("--interleaver-mode", choices=("none", "mode1", "mode2"))
    parser.add_argument("--interleaver-phase", type=int)
    parser.add_argument("--bits-per-frame", type=int)
    return parser


def tag_pre_ldpc_llr(
    llr_path: str | Path,
    *,
    output_path: str | Path,
    sidecar_path: str | Path | None = None,
    source_capture: str | Path | None = None,
    qam_mode: str | None = None,
    fec_rate_index: int | None = None,
    interleaver_mode: str | None = None,
    interleaver_phase: int | None = None,
    bits_per_frame: int | None = None,
) -> dict[str, Any]:
    """Write a standard pre-LDPC dump with deterministic per-bit provenance."""

    source_path = Path(llr_path)
    output = Path(output_path)
    sidecar = Path(sidecar_path) if sidecar_path is not None else None
    default_sidecar = source_path.with_suffix(source_path.suffix + ".json")
    if sidecar is None and default_sidecar.is_file():
        sidecar = default_sidecar
    if not source_path.is_file():
        raise FileNotFoundError(f"LLR input missing: {source_path}")
    if source_path.stat().st_size % np.dtype("<f4").itemsize:
        raise ValueError("LLR input byte count must be float32-aligned")

    source_metadata = read_json(sidecar) if sidecar is not None else {}
    if sidecar is None and (
        qam_mode is None or fec_rate_index is None or interleaver_mode is None
    ):
        raise FileNotFoundError(
            "LLR sidecar missing; provide --qam, --fec-rate, and --interleaver-mode"
        )
    selected_qam_mode = str(
        qam_mode
        or source_metadata.get("qam_mode")
        or source_metadata.get("chosen_qam")
        or "64qam"
    )
    if selected_qam_mode not in QAM_DEFINITIONS:
        raise ValueError(f"unsupported QAM mode: {selected_qam_mode}")
    bits_per_symbol = int(QAM_DEFINITIONS[selected_qam_mode].bits_per_symbol)
    fec_rate = int(
        fec_rate_index
        or source_metadata.get("fec_rate_index")
        or source_metadata.get("chosen_fec_rate")
        or 3
    )
    profile = dtmb_ldpc_profile(fec_rate)
    selected_interleaver_mode = str(
        interleaver_mode
        or source_metadata.get("symbol_deinterleave")
        or source_metadata.get("chosen_interleaver_mode")
        or "none"
    ).lower()
    phase = int(
        interleaver_phase
        if interleaver_phase is not None
        else source_metadata.get("symbol_deinterleave_phase")
        or source_metadata.get("interleaver_phase")
        or 0
    )
    if selected_interleaver_mode not in ("none", *SYMBOL_INTERLEAVERS):
        raise ValueError(f"unsupported symbol deinterleaver mode: {selected_interleaver_mode}")
    if selected_interleaver_mode != "none":
        branch_count = int(SYMBOL_INTERLEAVERS[selected_interleaver_mode].branch_count)  # type: ignore[index]
        if phase < 0 or phase >= branch_count:
            raise ValueError(f"symbol deinterleaver phase must be in 0..{branch_count - 1}")

    llr = np.fromfile(source_path, dtype="<f4")
    if llr.size == 0:
        raise ValueError("LLR input is empty")
    if llr.size % bits_per_symbol:
        raise ValueError("LLR input must contain whole QAM symbols")
    if not bool(np.all(np.isfinite(llr))):
        raise ValueError("LLR input contains non-finite values")

    hard_bits = (llr < 0.0).astype(np.uint8)
    symbol_count = int(llr.size // bits_per_symbol)
    source_symbol_indices = _source_symbol_indices(
        symbol_count,
        interleaver_mode=selected_interleaver_mode,
        phase=phase,
    )
    source_frame_indices = (
        source_symbol_indices // DATA_SYMBOLS_PER_FRAME
    ).astype(np.int32, copy=False)
    source_carrier_indices = (
        source_symbol_indices % DATA_SYMBOLS_PER_FRAME
    ).astype(np.int16, copy=False)
    if selected_interleaver_mode == "none":
        source_branch_indices = np.full(symbol_count, -1, dtype=np.int8)
    else:
        branch_count = int(SYMBOL_INTERLEAVERS[selected_interleaver_mode].branch_count)  # type: ignore[index]
        source_branch_indices = (
            (source_symbol_indices % branch_count + phase) % branch_count
        ).astype(np.int8, copy=False)

    selected_bits_per_frame = int(
        bits_per_frame
        or source_metadata.get("bits_per_frame")
        or DATA_SYMBOLS_PER_FRAME * bits_per_symbol
    )
    frame_indices, codeword_indices, bit_indices_in_codeword = _output_bit_tags(
        int(llr.size),
        bits_per_frame=selected_bits_per_frame,
        codeword_bits=int(profile.codeword_bits),
    )
    bit_source_frame_indices = np.repeat(
        source_frame_indices, bits_per_symbol
    ).astype(np.int32, copy=False)
    bit_source_carrier_indices = np.repeat(
        source_carrier_indices, bits_per_symbol
    ).astype(np.int16, copy=False)
    bit_source_branch_indices = np.repeat(
        source_branch_indices, bits_per_symbol
    ).astype(np.int8, copy=False)
    bit_qam_plane_indices = np.tile(
        np.arange(bits_per_symbol, dtype=np.uint8),
        symbol_count,
    )

    output_sidecar = output.with_suffix(".json")
    metadata = {
        **source_metadata,
        "stage": "tagged_pre_ldpc",
        "source_llr_path": str(source_path),
        "source_llr_sidecar_path": str(sidecar) if sidecar is not None else None,
        "capture_path": str(source_capture)
        if source_capture is not None
        else source_metadata.get("capture_path"),
        "npz_file": str(output),
        "json_file": str(output_sidecar),
        "chosen_qam": selected_qam_mode,
        "chosen_fec_rate": fec_rate,
        "chosen_interleaver_mode": selected_interleaver_mode,
        "symbol_deinterleave_phase": phase,
        "bits_per_symbol": bits_per_symbol,
        "bits_per_frame": selected_bits_per_frame,
        "codeword_bits": int(profile.codeword_bits),
        "hard_bit_count": int(hard_bits.size),
        "llr_count": int(llr.size),
        "codewords": int(llr.size // profile.codeword_bits),
        "unused_bits": int(llr.size % profile.codeword_bits),
        "tagged_source_symbols": symbol_count,
        "tagged_source_frames": int(np.unique(source_frame_indices).size),
        "tagged_source_frame_min": int(source_frame_indices.min()),
        "tagged_source_frame_max": int(source_frame_indices.max()),
        "tagged_source_branches": (
            int(np.unique(source_branch_indices).size)
            if selected_interleaver_mode != "none"
            else 0
        ),
        "tag_contract": (
            "post-deinterleaver output bits traced to receiver-input data "
            "symbols using the configured convolutional-deinterleaver schedule"
        ),
        "arrays": {
            "hard_bits": "uint8 flat pre-LDPC hard decisions",
            "llr": "float32 flat pre-LDPC LLRs; positive means bit 0",
            "frame_indices": "int32 output-frame index per hard bit",
            "codeword_indices": "int32 LDPC codeword index per hard bit, -1 for tail",
            "bit_indices_in_codeword": "int32 bit position inside LDPC codeword, -1 for tail",
            "data_symbol_source_indices_after_deinterleave": "int64 pre-deinterleaver source-symbol index per output symbol",
            "data_symbol_source_frame_indices_after_deinterleave": "int32 source frame index per output symbol",
            "data_symbol_source_carrier_indices_after_deinterleave": "int16 source data-carrier index 0..3743 per output symbol",
            "data_symbol_source_branch_indices_after_deinterleave": "int8 physical interleaver branch per output symbol, -1 without interleaving",
            "bit_source_frame_indices": "int32 source frame index per pre-LDPC bit",
            "bit_source_carrier_indices": "int16 source data-carrier index per pre-LDPC bit",
            "bit_source_branch_indices": "int8 physical interleaver branch per pre-LDPC bit, -1 without interleaving",
            "bit_qam_plane_indices": "uint8 QAM bit-plane index per pre-LDPC bit",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            hard_bits=hard_bits,
            llr=llr,
            frame_indices=frame_indices,
            codeword_indices=codeword_indices,
            bit_indices_in_codeword=bit_indices_in_codeword,
            data_symbol_source_indices_after_deinterleave=source_symbol_indices,
            data_symbol_source_frame_indices_after_deinterleave=source_frame_indices,
            data_symbol_source_carrier_indices_after_deinterleave=source_carrier_indices,
            data_symbol_source_branch_indices_after_deinterleave=source_branch_indices,
            bit_source_frame_indices=bit_source_frame_indices,
            bit_source_carrier_indices=bit_source_carrier_indices,
            bit_source_branch_indices=bit_source_branch_indices,
            bit_qam_plane_indices=bit_qam_plane_indices,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    output_sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _source_symbol_indices(
    symbol_count: int,
    *,
    interleaver_mode: str,
    phase: int,
) -> np.ndarray:
    if interleaver_mode == "none":
        return np.arange(symbol_count, dtype=np.int64)
    return convolutional_deinterleave_source_indices(
        symbol_count,
        mode=interleaver_mode,  # type: ignore[arg-type]
        phase=phase,
        after_latency_discard=True,
    )


def _output_bit_tags(
    bit_count: int,
    *,
    bits_per_frame: int,
    codeword_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(bit_count, dtype=np.int64)
    frame_indices = (indices // bits_per_frame).astype(np.int32)
    codeword_indices = np.full(bit_count, -1, dtype=np.int32)
    bit_indices = np.full(bit_count, -1, dtype=np.int32)
    usable = (bit_count // codeword_bits) * codeword_bits
    if usable:
        codeword_indices[:usable] = (indices[:usable] // codeword_bits).astype(np.int32)
        bit_indices[:usable] = (indices[:usable] % codeword_bits).astype(np.int32)
    return frame_indices, codeword_indices, bit_indices


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = tag_pre_ldpc_llr(
        args.llr,
        output_path=args.output,
        sidecar_path=args.llr_sidecar,
        source_capture=args.source_capture,
        qam_mode=args.qam,
        fec_rate_index=args.fec_rate,
        interleaver_mode=args.interleaver_mode,
        interleaver_phase=args.interleaver_phase,
        bits_per_frame=args.bits_per_frame,
    )
    print(
        json.dumps(
            {
                "stage": metadata["stage"],
                "output": metadata["npz_file"],
                "hard_bit_count": metadata["hard_bit_count"],
                "tagged_source_frames": metadata["tagged_source_frames"],
                "tagged_source_branches": metadata["tagged_source_branches"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
