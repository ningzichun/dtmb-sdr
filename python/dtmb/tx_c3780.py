"""Streaming DTMB C=3780 transmitter: TS bytes in, CI8 IQ out.

This is the TX-side half of the live pipeline. It consumes a whole-number
multiple of frame payloads from the input TS stream (default one C=3780
frame at a time), encodes each frame through scrambler -> BCH -> LDPC ->
QAM -> convolutional interleaver -> C=3780 frame body -> PN945 header,
and writes the resulting signed 8-bit I/Q samples to the output.

The interleaver runs in continuous (non-flushing) mode so the stream can
be arbitrarily long. 170 extra drain frames are appended after the
payload so the receiver's symbol deinterleaver can flush without losing
the last frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .continuous_stream import (
    ContinuousStreamConfig,
    DATA_SYMBOLS_PER_FRAME,
    _system_info_index_to_parameters,
    encode_continuous_symbol_stream,
)
from .frequency import frequency_interleave
from .pn import pn_header_symbols_for_body_power
from .symbol_interleaver import SYMBOL_INTERLEAVERS
from .system_info import SYSTEM_INFO_VECTORS, system_info_symbols


PN_MODE_CHOICES = ("pn945",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-transmit",
        description=(
            "Encode an MPEG-TS byte stream into a DTMB C=3780 CI8 capture. "
            "Use '-' for stdin on input or stdout on output to stream."
        ),
    )
    parser.add_argument(
        "input",
        help="Input TS path, or '-' to read from stdin",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CI8 path, or '-' to write to stdout",
    )
    parser.add_argument(
        "--system-info-index",
        type=int,
        default=23,
        help="System-information index (23 = 64QAM, rate 3, mode 1)",
    )
    parser.add_argument(
        "--pn-mode",
        choices=PN_MODE_CHOICES,
        default="pn945",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.75,
        help="Peak output amplitude (0..1) before CI8 quantisation.",
    )
    parser.add_argument(
        "--pad-null-packets",
        action="store_true",
        help=(
            "Pad the last partial frame with MPEG-TS null packets "
            "(PID 0x1FFF, sync 0x47) so one C=3780 frame is always encoded."
        ),
    )
    parser.add_argument(
        "--trailing-drain-frames",
        type=int,
        default=None,
        help=(
            "Append this many extra drain frames after the payload so the "
            "RX deinterleaver can flush the final payload frame. Defaults "
            "to the symbol-interleaver latency (170 frames for mode 1)."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for the drain-frame filler payload.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qam_mode, fec_rate_index, interleaver_mode = _system_info_index_to_parameters(
        args.system_info_index
    )
    config = ContinuousStreamConfig(
        qam_mode=qam_mode,
        fec_rate_index=fec_rate_index,
        interleaver_mode=interleaver_mode,  # type: ignore[arg-type]
        interleaver_phase=0,
    )
    frame_bytes = config.transport_bytes_per_frame
    if args.trailing_drain_frames is None:
        trailing_drain = config.pre_roll_frames
    else:
        trailing_drain = max(0, int(args.trailing_drain_frames))

    payload = _read_input(args.input)
    if not payload:
        raise SystemExit("tx: input TS is empty")

    if len(payload) % frame_bytes:
        if not args.pad_null_packets:
            raise SystemExit(
                f"tx: input TS length {len(payload)} is not a multiple of "
                f"{frame_bytes} bytes; pass --pad-null-packets to allow padding"
            )
        padding = frame_bytes - (len(payload) % frame_bytes)
        if padding % 188:
            raise SystemExit(
                "tx: cannot pad with whole TS null packets to reach a full frame"
            )
        payload = payload + _make_null_packets(padding // 188)

    capture = _encode_capture(
        payload=payload,
        config=config,
        system_info_index=args.system_info_index,
        pn_mode=args.pn_mode,
        trailing_drain_frames=trailing_drain,
        amplitude=args.amplitude,
        rng_seed=args.rng_seed,
    )
    _write_output(args.output, capture)
    return 0


def _read_input(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def _write_output(path: str, capture: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(capture)
        sys.stdout.buffer.flush()
        return
    Path(path).write_bytes(capture)


def _make_null_packets(count: int) -> bytes:
    packet = bytes([0x47, 0x1F, 0xFF, 0x10]) + b"\xff" * 184
    return packet * count


def _encode_capture(
    *,
    payload: bytes,
    config: ContinuousStreamConfig,
    system_info_index: int,
    pn_mode: str,
    trailing_drain_frames: int,
    amplitude: float,
    rng_seed: int,
) -> bytes:
    frame_bytes = config.transport_bytes_per_frame
    payload_frames = len(payload) // frame_bytes
    rng = np.random.default_rng(rng_seed)
    drain_bytes = trailing_drain_frames * frame_bytes
    drain = (
        rng.integers(0, 256, size=drain_bytes, dtype=np.uint8).tobytes()
        if drain_bytes
        else b""
    )
    joined = payload + drain
    total_frames = payload_frames + trailing_drain_frames
    frame_chunks = [
        joined[index * frame_bytes : (index + 1) * frame_bytes]
        for index in range(total_frames)
    ]
    interleaved = encode_continuous_symbol_stream(frame_chunks, config=config)
    si = system_info_symbols(
        SYSTEM_INFO_VECTORS[system_info_index],
        frame_body_mode="C3780",
    )
    bodies: list[np.ndarray] = []
    for index in range(total_frames):
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = si
        logical[36:] = interleaved[
            index * DATA_SYMBOLS_PER_FRAME : (index + 1) * DATA_SYMBOLS_PER_FRAME
        ]
        bodies.append(np.fft.ifft(frequency_interleave(logical)).astype(np.complex64))
    body_power = float(np.mean([np.mean(np.abs(body) ** 2) for body in bodies]))
    pn_header = pn_header_symbols_for_body_power(pn_mode, body_power=body_power)
    frame_samples = [np.concatenate((pn_header, body)) for body in bodies]
    capture = np.concatenate(frame_samples)
    peak = float(np.max(np.abs(capture)))
    if peak > 0:
        capture = capture / peak * float(amplitude)
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    return ci8.tobytes()


if __name__ == "__main__":
    raise SystemExit(main())
