"""Generate a 64QAM rate-3 mode-1 synthetic DTMB capture for end-to-end TS recovery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from dtmb.fec import encode_frame_bch_ldpc_codewords
from dtmb.frequency import frequency_interleave
from dtmb.pn import pn_header_symbols_for_body_power
from dtmb.qam import qam_modulate
from dtmb.symbol_interleaver import convolutional_interleave
from dtmb.system_info import SYSTEM_INFO_VECTORS, system_info_symbols


BYTES_PER_CODEWORD = 752
CODEWORDS_PER_FRAME = 3  # 64QAM rate-3 payload = 3 LDPC codewords per C=3780 frame
FRAME_PAYLOAD_BYTES = BYTES_PER_CODEWORD * CODEWORDS_PER_FRAME  # 2256 bytes = 12 TS packets


def _make_ts_packets(packet_count: int, pid: int = 0x0100) -> bytes:
    out = bytearray()
    for index in range(packet_count):
        header = bytes(
            (
                0x47,
                (0x40 if index == 0 else 0x00) | ((pid >> 8) & 0x1F),
                pid & 0xFF,
                0x10 | (index & 0x0F),
            )
        )
        payload = bytes((pid + index + k) & 0xFF for k in range(184))
        out.extend(header + payload)
    return bytes(out)


def write_synthetic_capture(
    ci8_path: Path,
    ts_oracle_path: Path,
    *,
    frames: int,
    symbol_interleave: str = "mode1",
    system_info_index: int = 23,
) -> None:
    if frames < 171:
        raise ValueError("need at least 171 frames to cover mode-1 deinterleaver latency")

    # One C=3780 frame carries 12 transport packets' worth of bytes through rate-3 64QAM.
    ts_bytes = _make_ts_packets(CODEWORDS_PER_FRAME * 4)  # 12 packets (4 per codeword)
    assert len(ts_bytes) == FRAME_PAYLOAD_BYTES
    ts_oracle_path.write_bytes(ts_bytes)

    # Encode the frame's payload through scrambler+BCH+LDPC, map to 64QAM symbols.
    first_frame_symbols = qam_modulate(
        encode_frame_bch_ldpc_codewords(ts_bytes, fec_rate_index=3),
        mode="64qam",
    )
    data_chunks: list[np.ndarray] = [first_frame_symbols]
    data_chunks.extend(np.zeros(3744, dtype=np.complex64) for _ in range(frames - 1))

    data_stream = convolutional_interleave(
        np.concatenate(data_chunks),
        mode=symbol_interleave,
    )

    si_bits = SYSTEM_INFO_VECTORS[system_info_index]
    bodies = []
    for frame_index in range(frames):
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info_symbols(si_bits, frame_body_mode="C3780")
        logical[36:] = data_stream[frame_index * 3744 : (frame_index + 1) * 3744]
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        bodies.append(body)
    body_power = float(np.mean([np.mean(np.abs(body) ** 2) for body in bodies]))
    pn_header = pn_header_symbols_for_body_power("pn945", body_power=body_power)
    frame_samples = [np.concatenate((pn_header, body)) for body in bodies]
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    ci8_path.write_bytes(ci8.tobytes())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "captures"
    write_synthetic_capture(
        out_dir / "synthetic_64qam_r3_m1.ci8",
        out_dir / "synthetic_64qam_r3_m1.oracle.ts",
        frames=171,
    )
    print(
        "Generated "
        f"{out_dir / 'synthetic_64qam_r3_m1.ci8'} and "
        f"{out_dir / 'synthetic_64qam_r3_m1.oracle.ts'}"
    )
