"""Stage: LLR stream -> MPEG-TS via LDPC + BCH + descrambler.

Takes the post-deinterleaver LLR dump produced by ``dtmb.receiver
--llr-output`` and runs only the FEC + descrambling + TS-alignment
portion of the pipeline. This is the "take apart the existing suite"
seam:

  - Upstream stages (acquire, sysinfo, demap via ``dtmb.receiver``)
    write ``<prefix>.llr.f32`` plus ``<prefix>.llr.f32.json``.
  - This stage reads that pair and produces ``<prefix>.recovered.ts``
    and ``<prefix>.decode_ldpc.json`` without re-running any DSP.
  - ``--backend cpp`` streams the LLR file through the portable native
    ``dtmb_core_ldpc_bch_decode`` executable.
  - Other alternate LDPC backends can use the same sidecar and TS contract.

The stage is deliberately small. It does not re-decide the mode (that
is already locked in by the sidecar) and does not re-run acquisition
(that already happened before the LLR dump was written). What it
proves is that the LDPC stage is decoupled from the DSP stages behind
a stable file-level interface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _common import ROOT, read_json, write_json

from dtmb.receiver import decode_llr_stream_to_transport_bytes  # noqa: E402
from dtmb.ts import (  # noqa: E402
    analyze_ts_lock_candidates,
    extract_ts_packets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-decode-ldpc",
        description=(
            "Decode an LLR stream produced by `dtmb.receiver --llr-output` "
            "through LDPC + BCH + descrambler into MPEG-TS, using the "
            "sidecar JSON for framing metadata. The LDPC backend is the "
            "in-tree normalized min-sum decoder by default."
        ),
    )
    parser.add_argument(
        "--llr",
        type=Path,
        required=True,
        help="Path to the float32 LE LLR file produced by --llr-output.",
    )
    parser.add_argument(
        "--llr-sidecar",
        type=Path,
        default=None,
        help=(
            "Path to the .llr.f32.json sidecar. Defaults to <llr>.json."
        ),
    )
    parser.add_argument(
        "--output-ts",
        type=Path,
        required=True,
        help="Output recovered MPEG-TS path (may be empty on no-TS-lock).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output decode diagnostics JSON.",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "cpp"),
        default="python",
        help="LDPC+BCH backend. The cpp backend streams the LLR file to the native core.",
    )
    parser.add_argument(
        "--cpp-decoder",
        type=Path,
        default=ROOT / "build" / "core-cpp" / (
            "dtmb_core_ldpc_bch_decode.exe"
            if sys.platform == "win32"
            else "dtmb_core_ldpc_bch_decode"
        ),
        help="Path to dtmb_core_ldpc_bch_decode when --backend=cpp.",
    )
    parser.add_argument(
        "--cpp-workers",
        type=int,
        default=0,
        help="Native LDPC worker count; 0 uses hardware concurrency.",
    )
    parser.add_argument(
        "--cpp-max-iterations",
        type=int,
        default=50,
        help="Native normalized-min-sum iteration limit.",
    )
    parser.add_argument(
        "--fec-mode",
        choices=("ldpc", "bch-assume-ldpc", "uncoded", "none"),
        default="ldpc",
        help=(
            "FEC stage to run over the LLR stream. `uncoded` packs the "
            "hard bits directly as bytes (synthetic/diagnostic use)."
        ),
    )
    parser.add_argument(
        "--min-ts-packets",
        type=int,
        default=8,
        help="Minimum TS packets required to claim a lock.",
    )
    parser.add_argument(
        "--min-ts-sync-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min-ts-valid-ratio",
        type=float,
        default=0.8,
    )
    return parser


def _load_sidecar(llr_path: Path, explicit_sidecar: Path | None) -> dict[str, Any]:
    if explicit_sidecar is not None:
        sidecar_path = explicit_sidecar
    else:
        sidecar_path = llr_path.with_suffix(llr_path.suffix + ".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"LLR sidecar missing: {sidecar_path}; run the receiver with "
            "--llr-output to produce the .f32 file and its sidecar together."
        )
    return read_json(sidecar_path)


def _write_ts(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _parse_native_stats(stderr: str) -> dict[str, int | str]:
    stats: dict[str, int | str] = {}
    for line in stderr.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        try:
            stats[key] = int(value)
        except ValueError:
            stats[key] = value
    return stats


def _decode_with_cpp(
    args: argparse.Namespace,
    *,
    bits_per_frame: int,
    fec_rate_index: int,
) -> tuple[bytes, dict[str, int | str]]:
    if args.fec_mode != "ldpc":
        raise ValueError("--backend=cpp currently requires --fec-mode=ldpc")
    if not args.cpp_decoder.is_file():
        raise FileNotFoundError(
            f"native decoder missing: {args.cpp_decoder}; build with "
            "`cmake -S core/cpp -B build/core-cpp && cmake --build build/core-cpp`"
        )
    profile_codeword_bits = 7488
    if bits_per_frame % profile_codeword_bits:
        raise ValueError(
            "sidecar bits_per_frame must contain whole 7488-bit LDPC codewords "
            "for the native backend"
        )
    codewords_per_frame = bits_per_frame // profile_codeword_bits
    alist = (
        ROOT
        / "python"
        / "dtmb"
        / "data"
        / f"dtmb_ldpc_rate{fec_rate_index}.alist"
    )
    cmd = [
        str(args.cpp_decoder),
        "--fec-rate",
        str(fec_rate_index),
        "--alist",
        str(alist),
        "--codewords-per-frame",
        str(codewords_per_frame),
        "--workers",
        str(args.cpp_workers),
        "--max-iterations",
        str(args.cpp_max_iterations),
        str(args.llr),
        "-",
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"native LDPC decoder exited with rc={completed.returncode}: {stderr[-2000:]}"
        )
    return completed.stdout, _parse_native_stats(stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = _load_sidecar(args.llr, args.llr_sidecar)
    if sidecar.get("skipped"):
        _write_ts(args.output_ts, b"")
        write_json(
            args.output_json,
            {
                "stage": "decode_ldpc",
                "input_llr": str(args.llr),
                "input_sidecar": str(
                    args.llr_sidecar
                    or args.llr.with_suffix(args.llr.suffix + ".json")
                ),
                "skipped": True,
                "skip_reason": sidecar.get("skip_reason"),
                "gate_c_note": sidecar.get("gate_c_note"),
                "gate_c_quality": sidecar.get("gate_c_quality"),
                "llr_count": 0,
                "payload_bytes": 0,
                "recovered_ts_bytes": 0,
                "ts_lock": None,
                "ts_stream": None,
            },
        )
        print(
            f"[decode_ldpc] skipped upstream demap: {sidecar.get('skip_reason')}",
            file=sys.stderr,
        )
        return 0
    bits_per_frame = int(sidecar["bits_per_frame"])
    fec_rate_index = int(sidecar["fec_rate_index"])
    ldpc_parity_position = str(sidecar.get("ldpc_parity_position", "front"))

    llr_count = args.llr.stat().st_size // np.dtype("<f4").itemsize
    if llr_count == 0:
        raise ValueError(f"LLR file is empty: {args.llr}")
    if args.llr.stat().st_size % np.dtype("<f4").itemsize:
        raise ValueError(f"LLR file byte count is not float32-aligned: {args.llr}")
    if bits_per_frame <= 0:
        raise ValueError(
            f"sidecar bits_per_frame must be positive, got {bits_per_frame}"
        )

    backend_diagnostics: dict[str, Any] = {}
    if args.backend == "cpp":
        payload_bytes, backend_diagnostics = _decode_with_cpp(
            args,
            bits_per_frame=bits_per_frame,
            fec_rate_index=fec_rate_index,
        )
        ldpc_parity = {
            "backend": "cpp",
            "computed": False,
            "reason": "native streaming backend does not run the Python hard-parity diagnostic",
        }
        frame_reports: list[dict[str, Any]] = []
    else:
        llr = np.fromfile(args.llr, dtype="<f4")
        fec_mode = "none" if args.fec_mode == "uncoded" else args.fec_mode
        fec_outcome = decode_llr_stream_to_transport_bytes(
            llr,
            bits_per_frame=bits_per_frame,
            fec_mode=fec_mode if args.fec_mode != "uncoded" else "none",
            fec_rate_index=fec_rate_index,
            ldpc_parity_position=ldpc_parity_position,
            uncoded_payload=(args.fec_mode == "uncoded"),
        )
        payload_bytes = fec_outcome["payload_bytes"]
        ldpc_parity = fec_outcome["ldpc_parity_check"]
        frame_reports = fec_outcome["frame_reports"]

    ts_candidates = (
        analyze_ts_lock_candidates(payload_bytes, min_packets=args.min_ts_packets)
        if payload_bytes
        else []
    )
    selected = next(
        (
            candidate
            for candidate in ts_candidates
            if candidate.lock.sync_ratio >= args.min_ts_sync_ratio
            and candidate.valid_packet_ratio >= args.min_ts_valid_ratio
        ),
        None,
    )
    recovered = b""
    if selected is not None:
        packets = extract_ts_packets(payload_bytes, selected.lock)
        recovered = b"".join(packets)
    _write_ts(args.output_ts, recovered)

    report = {
        "stage": "decode_ldpc",
        "input_llr": str(args.llr),
        "input_sidecar": str(
            args.llr_sidecar
            or args.llr.with_suffix(args.llr.suffix + ".json")
        ),
        "llr_count": int(llr_count),
        "bits_per_frame": bits_per_frame,
        "fec_rate_index": fec_rate_index,
        "fec_mode_requested": args.fec_mode,
        "backend": args.backend,
        "backend_diagnostics": backend_diagnostics,
        "ldpc_parity_position": ldpc_parity_position,
        "frames_decoded": int(backend_diagnostics.get("frames", len(frame_reports))),
        "payload_bytes": len(payload_bytes),
        "recovered_ts_bytes": len(recovered),
        "ts_lock": selected.lock.to_dict() if selected is not None else None,
        "ts_stream": selected.stream.to_dict() if selected is not None else None,
        "ldpc_parity_check": ldpc_parity,
        "frame_reports": frame_reports[:16],
        "frame_reports_truncated": len(frame_reports) > 16,
    }
    write_json(args.output_json, report)
    if selected is None:
        print(
            f"[decode_ldpc] no TS lock; recovered_ts_bytes=0 payload={len(payload_bytes)}",
            file=sys.stderr,
        )
        return 0
    print(
        f"[decode_ldpc] backend={args.backend} TS lock packet_size={selected.lock.packet_size} "
        f"packets={selected.stream.valid_packet_count} "
        f"sync_ratio={selected.lock.sync_ratio:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
