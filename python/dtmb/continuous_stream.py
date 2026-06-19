"""Continuous-stream DTMB FEC/interleaver encode-decode fixture.

This module builds the full offline bit path end to end without the
zero-flush shortcut used by earlier fixtures:

    transport bytes
      -> scrambler (inside encode_frame_bch_ldpc_codewords)
      -> BCH(762, 752)
      -> QC-LDPC encode (DTMB erased-parity layout)
      -> QAM modulate
      -> convolutional_interleave(flush=False)

and the matching receive path:

    interleaved symbol stream
      -> convolutional_deinterleave(flush=False)
      -> drop (B-1)*M*B pre-roll symbols
      -> qam_soft_demodulate
      -> decode_frame_bch_descramble_from_ldpc_llr
      -> BCH/descramble

No PN header, FFT, frequency interleave, or CI8 round-trip is involved.
The purpose is to isolate the QAM-demap -> symbol-deinterleave -> LDPC
bit-order mismatch that real captures have been hitting on
``wall_522mhz_amp0_lna32_vga24_20260429_225951`` and similar.

See ``AGENTS.md`` section "P1 - Continuous-Stream FEC Fixture" for the
motivating diagnostic: LDPC parity-check mismatch ratio sits at exactly
0.50 on real captures across every equalizer and deinterleaver phase,
which rules out noise and pushes suspicion onto the bit-order chain.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .fec import (
    decode_frame_bch_descramble_from_ldpc_llr,
    encode_frame_bch_ldpc_codewords,
)
from .ldpc import dtmb_ldpc_parity_mismatch_counts, dtmb_ldpc_profile
from .qam import (
    QAM_DEFINITIONS,
    QamMode,
    qam_modulate,
    qam_soft_demodulate,
    qam_symbol_quality,
)
from .symbol_interleaver import (
    SYMBOL_INTERLEAVERS,
    convolutional_deinterleave,
    convolutional_interleave,
)


DATA_SYMBOLS_PER_FRAME = 3744


PhaseInterleaverMode = Literal["mode1", "mode2"]


@dataclass(frozen=True)
class ContinuousStreamConfig:
    """Parameters for a continuous-stream FEC fixture run."""

    qam_mode: QamMode
    fec_rate_index: int
    interleaver_mode: PhaseInterleaverMode | None
    interleaver_phase: int = 0

    @property
    def bits_per_symbol(self) -> int:
        return QAM_DEFINITIONS[self.qam_mode].bits_per_symbol

    @property
    def bits_per_frame(self) -> int:
        return DATA_SYMBOLS_PER_FRAME * self.bits_per_symbol

    @property
    def codewords_per_frame(self) -> int:
        profile = dtmb_ldpc_profile(self.fec_rate_index)
        if self.bits_per_frame % profile.codeword_bits:
            raise ValueError(
                "QAM mode does not divide the DTMB codeword into whole frames"
            )
        return self.bits_per_frame // profile.codeword_bits

    @property
    def transport_bytes_per_frame(self) -> int:
        profile = dtmb_ldpc_profile(self.fec_rate_index)
        return self.codewords_per_frame * profile.bch_blocks_per_codeword * 94

    @property
    def pre_roll_frames(self) -> int:
        """C=3780 frames required to fully drain the deinterleaver."""

        if self.interleaver_mode is None:
            return 0
        spec = SYMBOL_INTERLEAVERS[self.interleaver_mode]
        latency = spec.full_stream_latency_symbols
        if latency % DATA_SYMBOLS_PER_FRAME:
            raise ValueError(
                "interleaver latency does not align to 3744 data symbols"
            )
        return latency // DATA_SYMBOLS_PER_FRAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "qam_mode": self.qam_mode,
            "fec_rate_index": self.fec_rate_index,
            "interleaver_mode": self.interleaver_mode,
            "interleaver_phase": self.interleaver_phase,
            "bits_per_symbol": self.bits_per_symbol,
            "bits_per_frame": self.bits_per_frame,
            "codewords_per_frame": self.codewords_per_frame,
            "transport_bytes_per_frame": self.transport_bytes_per_frame,
            "pre_roll_frames": self.pre_roll_frames,
            "deinterleaver_latency_symbols": (
                SYMBOL_INTERLEAVERS[self.interleaver_mode].full_stream_latency_symbols
                if self.interleaver_mode
                else 0
            ),
        }


@dataclass(frozen=True)
class ContinuousStreamDiagnostics:
    """Diagnostics from one continuous-stream TX+RX pass."""

    config: ContinuousStreamConfig
    payload_frames: int
    payload_bytes: int
    pre_roll_frames: int
    total_frames: int
    total_symbols: int
    awgn_sigma: float
    pre_deinterleave_hard_errors: int
    pre_deinterleave_total_bits: int
    post_deinterleave_hard_errors: int
    post_deinterleave_total_bits: int
    ldpc_parity_mismatch_counts: tuple[int, ...]
    ldpc_syndrome_weights: tuple[int, ...]
    ldpc_iterations: tuple[int, ...]
    ldpc_converged_codewords: int
    bch_blocks: int
    bch_unclean_blocks: int
    bch_corrected_errors: int
    transport_match: bool
    transport_error_bytes: int
    pre_ldpc_alignment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "payload_frames": self.payload_frames,
            "payload_bytes": self.payload_bytes,
            "pre_roll_frames": self.pre_roll_frames,
            "total_frames": self.total_frames,
            "total_symbols": self.total_symbols,
            "awgn_sigma": self.awgn_sigma,
            "pre_deinterleave_hard_errors": self.pre_deinterleave_hard_errors,
            "pre_deinterleave_total_bits": self.pre_deinterleave_total_bits,
            "post_deinterleave_hard_errors": self.post_deinterleave_hard_errors,
            "post_deinterleave_total_bits": self.post_deinterleave_total_bits,
            "ldpc_parity_mismatch_counts": list(self.ldpc_parity_mismatch_counts),
            "ldpc_syndrome_weights": list(self.ldpc_syndrome_weights),
            "ldpc_iterations": list(self.ldpc_iterations),
            "ldpc_converged_codewords": self.ldpc_converged_codewords,
            "bch_blocks": self.bch_blocks,
            "bch_unclean_blocks": self.bch_unclean_blocks,
            "bch_corrected_errors": self.bch_corrected_errors,
            "transport_match": self.transport_match,
            "transport_error_bytes": self.transport_error_bytes,
            "pre_ldpc_alignment": self.pre_ldpc_alignment,
        }


def continuous_stream_bit_index_trace(
    *,
    config: ContinuousStreamConfig,
    payload_frames: int = 1,
) -> dict[str, Any]:
    """Trace RX pre-LDPC bits back to TX symbol/codeword positions.

    GB 20600-2006 section 4.4.4 states that the first symbol of each
    3744-symbol basic data block is synchronized to interleaver branch 0 and
    that the interleave/deinterleave pair has latency ``M*(B-1)*B`` symbols.
    This helper sends integer symbol labels through the same continuous
    interleave/deinterleave pair used by the fixture, discards that latency, and
    expands the recovered symbol labels to the QAM bit positions that feed LDPC.

    The returned arrays describe the receiver-visible pre-LDPC stream after the
    latency discard. A correct no-flush TX/RX pair should map output bit ``i``
    back to source bit ``i`` for all supported non-NR modes.
    """

    if payload_frames <= 0:
        raise ValueError("payload_frames must be positive")

    pre_roll_frames = config.pre_roll_frames
    total_frames = pre_roll_frames + int(payload_frames)
    source_symbols = np.arange(
        total_frames * DATA_SYMBOLS_PER_FRAME,
        dtype=np.float32,
    ).astype(np.complex64)
    if config.interleaver_mode is None:
        recovered = source_symbols
        latency = 0
        discarded = 0
    else:
        spec = SYMBOL_INTERLEAVERS[config.interleaver_mode]
        latency = spec.full_stream_latency_symbols
        interleaved = convolutional_interleave(
            source_symbols,
            mode=config.interleaver_mode,
            flush=False,
            fill=-1 + 0j,
            phase=config.interleaver_phase,
        )
        deinterleaved = convolutional_deinterleave(
            interleaved,
            mode=config.interleaver_mode,
            flush=False,
            fill=-1 + 0j,
            phase=config.interleaver_phase,
        )
        discarded = min(latency, int(deinterleaved.size))
        recovered = deinterleaved[discarded:]

    payload_symbols = min(
        int(recovered.size),
        int(payload_frames) * DATA_SYMBOLS_PER_FRAME,
    )
    source_symbol_indices = np.rint(
        np.asarray(recovered[:payload_symbols].real, dtype=np.float32)
    ).astype(np.int64)
    invalid_symbols = source_symbol_indices < 0
    bits_per_symbol = config.bits_per_symbol
    profile = dtmb_ldpc_profile(config.fec_rate_index)
    bits_per_frame = config.bits_per_frame
    bit_planes = np.tile(
        np.arange(bits_per_symbol, dtype=np.int64),
        source_symbol_indices.size,
    )
    repeated_symbols = np.repeat(source_symbol_indices, bits_per_symbol)
    source_bit_indices = repeated_symbols * bits_per_symbol + bit_planes
    source_bit_indices[repeated_symbols < 0] = -1

    source_frame_indices = source_symbol_indices // DATA_SYMBOLS_PER_FRAME
    source_symbol_in_frame = source_symbol_indices % DATA_SYMBOLS_PER_FRAME
    source_frame_indices[invalid_symbols] = -1
    source_symbol_in_frame[invalid_symbols] = -1

    source_codeword_indices = source_bit_indices // profile.codeword_bits
    source_bit_in_codeword = source_bit_indices % profile.codeword_bits
    source_codeword_in_frame = (
        (source_bit_indices % bits_per_frame) // profile.codeword_bits
    )
    invalid_bits = source_bit_indices < 0
    source_codeword_indices[invalid_bits] = -1
    source_bit_in_codeword[invalid_bits] = -1
    source_codeword_in_frame[invalid_bits] = -1

    output_bit_indices = np.arange(source_bit_indices.size, dtype=np.int64)
    return {
        "config": config.to_dict(),
        "payload_frames": int(payload_frames),
        "pre_roll_frames": int(pre_roll_frames),
        "total_frames": int(total_frames),
        "latency_symbols": int(latency),
        "discarded_symbols": int(discarded),
        "output_symbols": int(source_symbol_indices.size),
        "output_bits": int(source_bit_indices.size),
        "source_symbol_indices": source_symbol_indices,
        "source_frame_indices": source_frame_indices,
        "source_symbol_indices_in_frame": source_symbol_in_frame,
        "source_bit_indices": source_bit_indices,
        "source_codeword_indices": source_codeword_indices,
        "source_codeword_indices_in_frame": source_codeword_in_frame,
        "source_bit_indices_in_codeword": source_bit_in_codeword,
        "output_bit_indices": output_bit_indices,
        "output_frame_indices": output_bit_indices // bits_per_frame,
        "output_codeword_indices": output_bit_indices // profile.codeword_bits,
        "output_bit_indices_in_codeword": output_bit_indices % profile.codeword_bits,
        "bit_plane_indices": bit_planes,
    }


def encode_continuous_symbol_stream(
    frame_payloads: Sequence[bytes],
    *,
    config: ContinuousStreamConfig,
) -> np.ndarray:
    """Encode a list of whole-frame transport payloads into one symbol stream.

    Every entry of ``frame_payloads`` must have length
    ``config.transport_bytes_per_frame`` so one BCH/LDPC codeword block maps
    cleanly into one C=3780 data slot. Each frame is encoded through its own
    scrambler/BCH/LDPC invocation to match the receiver, which decodes one
    C=3780 signal-frame payload at a time (``decode_frame_bch_descramble_from_ldpc_llr``
    in ``dtmb.receiver``). Treating the stream as a single encoder run would
    carry scrambler state across frames and break decode on every frame after
    the first.

    After per-frame encoding the bit stream is QAM-modulated in one call and
    fed through the convolutional interleaver with ``flush=False`` so the TX
    side matches a real DTMB modulator driving a continuous symbol stream.
    """

    _validate_payload_shape(frame_payloads, config=config)
    coded_bits_per_frame = config.bits_per_frame
    coded_chunks = []
    for payload in frame_payloads:
        codewords = encode_frame_bch_ldpc_codewords(
            payload,
            fec_rate_index=config.fec_rate_index,
        )
        if codewords.size != coded_bits_per_frame:
            raise ValueError(
                "encoded LDPC codewords do not pack into a whole C=3780 frame"
            )
        coded_chunks.append(codewords.astype(np.uint8, copy=False))
    coded_bits = np.concatenate(coded_chunks).astype(np.uint8)
    symbols = qam_modulate(coded_bits, mode=config.qam_mode)
    if symbols.size != len(frame_payloads) * DATA_SYMBOLS_PER_FRAME:
        raise ValueError("QAM modulation did not yield whole-frame symbols")
    if config.interleaver_mode is None:
        return symbols
    spec = SYMBOL_INTERLEAVERS[config.interleaver_mode]
    if config.interleaver_phase < 0 or config.interleaver_phase >= spec.branch_count:
        raise ValueError("interleaver_phase must be 0..branch_count-1")
    return convolutional_interleave(
        symbols,
        mode=config.interleaver_mode,
        flush=False,
        phase=config.interleaver_phase,
    )


def decode_continuous_symbol_stream(
    symbols: np.ndarray,
    *,
    config: ContinuousStreamConfig,
    noise_variance: float = 1.0,
    normalize_qam: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the RX bit path on a continuous symbol stream.

    Returns ``(payload_llr, payload_bits, trace)`` where ``payload_llr`` is the
    LLR stream after the deinterleaver pre-roll has been discarded and
    ``payload_bits`` is ``(payload_llr < 0)``. ``trace`` reports the intermediate
    stream sizes so diagnostics can verify the discard arithmetic.
    """

    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    trace: dict[str, Any] = {
        "input_symbols": int(values.size),
        "deinterleaver_latency_symbols": 0,
        "discarded_symbols": 0,
    }
    if config.interleaver_mode is None:
        data_stream = values
    else:
        spec = SYMBOL_INTERLEAVERS[config.interleaver_mode]
        if config.interleaver_phase < 0 or config.interleaver_phase >= spec.branch_count:
            raise ValueError("interleaver_phase must be 0..branch_count-1")
        trace["deinterleaver_latency_symbols"] = spec.full_stream_latency_symbols
        deinterleaved = convolutional_deinterleave(
            values,
            mode=config.interleaver_mode,
            flush=False,
            phase=config.interleaver_phase,
        )
        discard = min(spec.full_stream_latency_symbols, int(deinterleaved.size))
        trace["discarded_symbols"] = int(discard)
        data_stream = deinterleaved[discard:]
    trace["payload_symbols"] = int(data_stream.size)
    llr = qam_soft_demodulate(
        data_stream,
        mode=config.qam_mode,
        noise_variance=noise_variance,
        normalize=normalize_qam,
    ).astype(np.float32, copy=False)
    bits = (llr < 0).astype(np.uint8)
    trace["payload_bits"] = int(bits.size)
    return llr, bits, trace


def pre_ldpc_bit_alignment_report(
    reference_bits: np.ndarray,
    candidate_bits: np.ndarray,
    *,
    bits_per_symbol: int,
    max_shift_bits: int = 0,
) -> dict[str, Any]:
    """Score whether RX pre-LDPC hard bits match transmitted LDPC codewords.

    ``reference_bits`` is the TX LDPC transmitted-codeword layout. ``candidate``
    is the RX hard bit stream after QAM demap and symbol deinterleaving, before
    LDPC decoding or BCH descrambling. A noise-free continuous fixture should
    report zero exact errors. If exact errors sit near 0.5, the shift and
    per-symbol bit-plane scores make common convention failures visible before
    spending CPU on LDPC iterations.
    """

    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be positive")
    tx = _as_binary(reference_bits)
    rx = _as_binary(candidate_bits)
    compared = min(int(tx.size), int(rx.size))
    exact_errors = _hamming_errors(tx[:compared], rx[:compared])
    inverted_errors = _hamming_errors(tx[:compared], rx[:compared] ^ 1)
    best_shift = _best_shift_alignment(tx, rx, max_shift_bits=max_shift_bits)
    bit_plane = _best_bit_plane_alignment(
        tx[:compared],
        rx[:compared],
        bits_per_symbol=bits_per_symbol,
    )
    return {
        "reference_bits": int(tx.size),
        "candidate_bits": int(rx.size),
        "compared_bits": int(compared),
        "exact_hard_errors": int(exact_errors),
        "exact_hamming_ratio": _ratio(exact_errors, compared),
        "inverted_hard_errors": int(inverted_errors),
        "inverted_hamming_ratio": _ratio(inverted_errors, compared),
        "best_shift": best_shift,
        "best_bit_plane_permutation": bit_plane,
    }


def run_continuous_fec_fixture(
    *,
    config: ContinuousStreamConfig,
    payload_frames: int = 3,
    seed: int = 20260509,
    awgn_sigma: float = 0.0,
    max_ldpc_iterations: int = 40,
    ldpc_attenuation: float = 0.75,
    override_payload: bytes | None = None,
) -> ContinuousStreamDiagnostics:
    """Encode+transmit+decode a deterministic continuous FEC fixture.

    The TX side generates ``pre_roll_frames + payload_frames`` whole-frame
    payloads, where ``pre_roll_frames`` drains the deinterleaver so the
    ``payload_frames`` that follow come out of RX aligned to the first
    decoded codeword. If ``awgn_sigma > 0`` complex AWGN is added to the
    transmitted symbols before soft demapping so the LDPC decoder sees
    realistic LLR magnitudes; otherwise the symbols go through the
    demapper untouched.
    """

    if payload_frames <= 0:
        raise ValueError("payload_frames must be positive")
    rng = np.random.default_rng(seed)

    if override_payload is not None:
        if len(override_payload) != payload_frames * config.transport_bytes_per_frame:
            raise ValueError(
                "override_payload size must match payload_frames * "
                "transport_bytes_per_frame"
            )
        payload = override_payload
    else:
        payload = rng.integers(
            0,
            256,
            size=payload_frames * config.transport_bytes_per_frame,
            dtype=np.uint8,
        ).tobytes()

    pre_roll_frames = config.pre_roll_frames
    pre_roll_bytes = pre_roll_frames * config.transport_bytes_per_frame
    pre_roll_payload = (
        rng.integers(0, 256, size=pre_roll_bytes, dtype=np.uint8).tobytes()
        if pre_roll_bytes
        else b""
    )

    total_frames = pre_roll_frames + payload_frames
    frame_payloads: list[bytes] = []
    # Payload frames go first in the TX input because the continuous
    # interleave/deinterleave pair maps input position i to RX position
    # (i + (B-1)*M*B). After dropping that latency on RX we recover the
    # first (total_frames - pre_roll_frames) input frames, so the payload
    # has to sit at the front. The pre-roll frames at the tail get
    # consumed by the deinterleaver and never emerge.
    for index in range(payload_frames):
        frame_payloads.append(
            payload[
                index * config.transport_bytes_per_frame :
                (index + 1) * config.transport_bytes_per_frame
            ]
        )
    for index in range(pre_roll_frames):
        frame_payloads.append(
            pre_roll_payload[
                index * config.transport_bytes_per_frame :
                (index + 1) * config.transport_bytes_per_frame
            ]
        )

    expected_tx_bits = np.concatenate(
        [
            encode_frame_bch_ldpc_codewords(
                chunk,
                fec_rate_index=config.fec_rate_index,
            )
            for chunk in frame_payloads[:payload_frames]
        ]
    ).astype(np.uint8)

    symbols = encode_continuous_symbol_stream(
        frame_payloads,
        config=config,
    )
    noisy_symbols = symbols
    noise_variance = 1.0
    if awgn_sigma > 0:
        noise_variance = float(awgn_sigma) ** 2
        noise = (
            rng.normal(0.0, awgn_sigma, symbols.size)
            + 1j * rng.normal(0.0, awgn_sigma, symbols.size)
        )
        noisy_symbols = (symbols + noise).astype(np.complex64)

    llr, bits, _trace = decode_continuous_symbol_stream(
        noisy_symbols,
        config=config,
        noise_variance=noise_variance,
    )

    usable_bits = payload_frames * config.bits_per_frame
    if bits.size < usable_bits:
        raise RuntimeError(
            "continuous-stream RX produced fewer bits than expected payload frames"
        )
    payload_llr = llr[:usable_bits]
    payload_bits = bits[:usable_bits]

    pre_errors = int(np.count_nonzero(payload_bits != expected_tx_bits))
    alignment_report = pre_ldpc_bit_alignment_report(
        expected_tx_bits,
        payload_bits,
        bits_per_symbol=config.bits_per_symbol,
        max_shift_bits=min(config.bits_per_symbol * 256, expected_tx_bits.size - 1),
    )
    mismatch_counts = dtmb_ldpc_parity_mismatch_counts(
        payload_bits,
        fec_rate_index=config.fec_rate_index,
    )

    # Decode one frame at a time because the DTMB descrambler restarts on
    # every signal-frame boundary. The receiver's main entry point does the
    # same (_iter_frame_llrs calls decode_frame_bch_descramble_from_ldpc_llr
    # per frame), so the fixture mirrors that behaviour here.
    frame_syndrome_weights: list[int] = []
    frame_iterations: list[int] = []
    converged_total = 0
    bch_blocks_total = 0
    bch_unclean_total = 0
    bch_corrected_total = 0
    decoded_frames: list[bytes] = []
    frame_llr_size = config.bits_per_frame
    for frame_index in range(payload_frames):
        frame_llr = payload_llr[
            frame_index * frame_llr_size : (frame_index + 1) * frame_llr_size
        ]
        frame_decoded = decode_frame_bch_descramble_from_ldpc_llr(
            frame_llr,
            fec_rate_index=config.fec_rate_index,
            max_ldpc_iterations=max_ldpc_iterations,
            ldpc_attenuation=ldpc_attenuation,
        )
        frame_syndrome_weights.extend(frame_decoded.ldpc_syndrome_weights)
        frame_iterations.extend(frame_decoded.ldpc_iterations)
        converged_total += frame_decoded.ldpc_converged_codewords
        bch_blocks_total += frame_decoded.bch_blocks
        bch_unclean_total += frame_decoded.bch_unclean_blocks
        bch_corrected_total += frame_decoded.bch_corrected_errors
        decoded_frames.append(frame_decoded.transport_bytes)
    decoded_bytes = b"".join(decoded_frames)
    decoded_pad = decoded_bytes.ljust(len(payload), b"\x00")[: len(payload)]
    error_bytes = int(
        sum(1 for lhs, rhs in zip(payload, decoded_pad) if lhs != rhs)
    ) + max(0, len(payload) - len(decoded_bytes))
    transport_match = decoded_bytes == payload

    return ContinuousStreamDiagnostics(
        config=config,
        payload_frames=payload_frames,
        payload_bytes=len(payload),
        pre_roll_frames=pre_roll_frames,
        total_frames=total_frames,
        total_symbols=int(symbols.size),
        awgn_sigma=float(awgn_sigma),
        pre_deinterleave_hard_errors=pre_errors,
        pre_deinterleave_total_bits=int(payload_bits.size),
        post_deinterleave_hard_errors=pre_errors,
        post_deinterleave_total_bits=int(payload_bits.size),
        ldpc_parity_mismatch_counts=tuple(int(value) for value in mismatch_counts),
        ldpc_syndrome_weights=tuple(frame_syndrome_weights),
        ldpc_iterations=tuple(frame_iterations),
        ldpc_converged_codewords=converged_total,
        bch_blocks=bch_blocks_total,
        bch_unclean_blocks=bch_unclean_total,
        bch_corrected_errors=bch_corrected_total,
        transport_match=transport_match,
        transport_error_bytes=error_bytes,
        pre_ldpc_alignment=alignment_report,
    )


def _validate_payload_shape(
    frame_payloads: Sequence[bytes],
    *,
    config: ContinuousStreamConfig,
) -> None:
    if not frame_payloads:
        raise ValueError("frame_payloads must contain at least one frame")
    expected = config.transport_bytes_per_frame
    for index, payload in enumerate(frame_payloads):
        if len(payload) != expected:
            raise ValueError(
                f"frame_payloads[{index}] is {len(payload)} bytes, expected {expected}"
            )


def _as_binary(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return values


def _hamming_errors(lhs: np.ndarray, rhs: np.ndarray) -> int:
    if lhs.size != rhs.size:
        raise ValueError("hamming inputs must have equal length")
    return int(np.count_nonzero(lhs != rhs))


def _ratio(errors: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(errors) / float(total)


def _best_shift_alignment(
    reference_bits: np.ndarray,
    candidate_bits: np.ndarray,
    *,
    max_shift_bits: int,
) -> dict[str, Any]:
    max_shift = max(0, int(max_shift_bits))
    max_shift = min(max_shift, max(0, int(reference_bits.size) - 1))
    max_shift = min(max_shift, max(0, int(candidate_bits.size) - 1))
    best: dict[str, Any] | None = None
    for shift in range(-max_shift, max_shift + 1):
        ref, cand = _shifted_overlap(reference_bits, candidate_bits, shift)
        if ref.size == 0:
            continue
        errors = _hamming_errors(ref, cand)
        row = {
            "shift_bits": int(shift),
            "compared_bits": int(ref.size),
            "hard_errors": int(errors),
            "hamming_ratio": _ratio(errors, int(ref.size)),
        }
        if best is None or (
            row["hard_errors"],
            abs(row["shift_bits"]),
            -row["compared_bits"],
        ) < (
            best["hard_errors"],
            abs(best["shift_bits"]),
            -best["compared_bits"],
        ):
            best = row
    if best is None:
        return {
            "shift_bits": 0,
            "compared_bits": 0,
            "hard_errors": 0,
            "hamming_ratio": None,
        }
    return best


def _shifted_overlap(
    reference_bits: np.ndarray,
    candidate_bits: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    if shift >= 0:
        compared = min(reference_bits.size, candidate_bits.size - shift)
        if compared <= 0:
            return reference_bits[:0], candidate_bits[:0]
        return reference_bits[:compared], candidate_bits[shift : shift + compared]
    offset = -shift
    compared = min(reference_bits.size - offset, candidate_bits.size)
    if compared <= 0:
        return reference_bits[:0], candidate_bits[:0]
    return reference_bits[offset : offset + compared], candidate_bits[:compared]


def _best_bit_plane_alignment(
    reference_bits: np.ndarray,
    candidate_bits: np.ndarray,
    *,
    bits_per_symbol: int,
) -> dict[str, Any]:
    symbols = min(reference_bits.size, candidate_bits.size) // bits_per_symbol
    compared = symbols * bits_per_symbol
    if symbols == 0:
        return {
            "compared_bits": 0,
            "best_permutation": list(range(bits_per_symbol)),
            "hard_errors": 0,
            "hamming_ratio": None,
            "bit_reversal_hard_errors": None,
            "bit_reversal_hamming_ratio": None,
            "iq_swap_hard_errors": None,
            "iq_swap_hamming_ratio": None,
        }
    ref = reference_bits[:compared].reshape(symbols, bits_per_symbol)
    cand = candidate_bits[:compared].reshape(symbols, bits_per_symbol)
    best_perm: tuple[int, ...] | None = None
    best_errors: int | None = None
    for perm in permutations(range(bits_per_symbol)):
        errors = int(np.count_nonzero(ref != cand[:, perm]))
        if best_errors is None or errors < best_errors:
            best_errors = errors
            best_perm = perm
    assert best_perm is not None and best_errors is not None

    reversal = tuple(reversed(range(bits_per_symbol)))
    reversal_errors = int(np.count_nonzero(ref != cand[:, reversal]))
    iq_swap_errors: int | None = None
    if bits_per_symbol % 2 == 0:
        half = bits_per_symbol // 2
        iq_swap = tuple(range(half, bits_per_symbol)) + tuple(range(half))
        iq_swap_errors = int(np.count_nonzero(ref != cand[:, iq_swap]))
    return {
        "compared_bits": int(compared),
        "best_permutation": [int(value) for value in best_perm],
        "hard_errors": int(best_errors),
        "hamming_ratio": _ratio(best_errors, compared),
        "bit_reversal_hard_errors": int(reversal_errors),
        "bit_reversal_hamming_ratio": _ratio(reversal_errors, compared),
        "iq_swap_hard_errors": None if iq_swap_errors is None else int(iq_swap_errors),
        "iq_swap_hamming_ratio": (
            None if iq_swap_errors is None else _ratio(iq_swap_errors, compared)
        ),
    }



def _fixture_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker entry point for ``run_continuous_fec_fixture_parallel``."""

    config_payload = dict(kwargs.pop("config"))
    config = ContinuousStreamConfig(
        qam_mode=config_payload["qam_mode"],
        fec_rate_index=int(config_payload["fec_rate_index"]),
        interleaver_mode=config_payload["interleaver_mode"],
        interleaver_phase=int(config_payload.get("interleaver_phase", 0)),
    )
    return run_continuous_fec_fixture(config=config, **kwargs).to_dict()


def run_continuous_fec_fixture_parallel(
    runs: Sequence[dict[str, Any]],
    *,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run multiple continuous-stream fixtures in parallel worker processes.

    Each entry in ``runs`` must be a dictionary with a ``config`` key (which can
    be a ``ContinuousStreamConfig`` instance or a mapping convertible to one)
    plus any keyword arguments accepted by :func:`run_continuous_fec_fixture`.
    The helper returns the diagnostics dictionaries in the original order.

    The fixture is CPU-bound (LDPC matmul, QAM demap, soft-decoding), so
    ``ProcessPoolExecutor`` is the right tool for scanning many
    (rate, QAM, interleaver mode, interleaver phase) combinations at once.
    Use ``max_workers=1`` for deterministic serial execution in tests.
    """

    prepared: list[dict[str, Any]] = []
    for entry in runs:
        payload = dict(entry)
        config = payload.get("config")
        if isinstance(config, ContinuousStreamConfig):
            payload["config"] = {
                "qam_mode": config.qam_mode,
                "fec_rate_index": config.fec_rate_index,
                "interleaver_mode": config.interleaver_mode,
                "interleaver_phase": config.interleaver_phase,
            }
        elif isinstance(config, dict):
            payload["config"] = dict(config)
        else:
            raise TypeError("each run must define a config mapping or object")
        prepared.append(payload)

    if max_workers is not None and max_workers <= 1:
        return [_fixture_worker(dict(payload)) for payload in prepared]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_fixture_worker, (dict(p) for p in prepared)))



def _system_info_index_to_parameters(system_info_index: int) -> tuple[QamMode, int, str]:
    """Return ``(qam_mode, fec_rate_index, interleaver_mode)`` for a system info index.

    Only whole-codeword-per-frame combinations are supported here because 32QAM
    rate-3 carries 2.5 LDPC codewords per C=3780 frame and needs a multi-frame
    FEC scheduler. Reserved rows and 4QAM-NR are also outside this fixture.
    """

    mapping: dict[int, tuple[QamMode, int, str]] = {
        5: ("4qam", 1, "mode1"),
        7: ("4qam", 2, "mode1"),
        9: ("4qam", 3, "mode1"),
        11: ("16qam", 1, "mode1"),
        13: ("16qam", 2, "mode1"),
        15: ("16qam", 3, "mode1"),
        19: ("64qam", 1, "mode1"),
        21: ("64qam", 2, "mode1"),
        23: ("64qam", 3, "mode1"),
        # even rows are mode2 variants of the odd rows
        6: ("4qam", 1, "mode2"),
        8: ("4qam", 2, "mode2"),
        10: ("4qam", 3, "mode2"),
        12: ("16qam", 1, "mode2"),
        14: ("16qam", 2, "mode2"),
        16: ("16qam", 3, "mode2"),
        20: ("64qam", 1, "mode2"),
        22: ("64qam", 2, "mode2"),
        24: ("64qam", 3, "mode2"),
    }
    try:
        return mapping[int(system_info_index)]
    except KeyError as exc:
        raise ValueError(
            "system_info_index must be one of the supported transmitted values"
        ) from exc


def synthesize_continuous_pre_ldpc_dump(
    transport_payload: bytes,
    *,
    output_path: str | Path,
    system_info_index: int = 22,
    pre_roll_frames: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Write a receiver-shaped pre-LDPC NPZ for a known-good continuous stream.

    This fixture starts at the same seam as ``dtmb.receiver --dump-pre-ldpc``:
    the stored ``data_symbols_before_symbol_deinterleave`` array is the
    continuous C=3780 QAM symbol stream before RX symbol deinterleaving, and
    ``hard_bits``/``llr`` are produced from the post-latency-discard stream.
    It deliberately skips PN, FFT, and equalization so pre-LDPC probes can be
    validated against a standard-derived mode without spending minutes on the
    CI8 receiver fixture. Only the post-discard payload frames are BCH/LDPC
    codewords; startup and drain symbols are random QAM filler that keeps the
    finite interleaver stream from looking zero-filled.
    """

    qam_mode, fec_rate_index, interleaver_mode = _system_info_index_to_parameters(
        system_info_index
    )
    config = ContinuousStreamConfig(
        qam_mode=qam_mode,
        fec_rate_index=fec_rate_index,
        interleaver_mode=interleaver_mode,  # type: ignore[arg-type]
        interleaver_phase=0,
    )
    if len(transport_payload) == 0:
        raise ValueError("transport_payload must contain at least one frame")
    if len(transport_payload) % config.transport_bytes_per_frame:
        raise ValueError(
            "transport_payload must be a whole multiple of "
            f"{config.transport_bytes_per_frame} bytes"
        )
    payload_frames = len(transport_payload) // config.transport_bytes_per_frame
    drain_frames = (
        config.pre_roll_frames
        if pre_roll_frames is None
        else int(pre_roll_frames)
    )
    if drain_frames < 0:
        raise ValueError("pre_roll_frames must be non-negative")

    rng = np.random.default_rng(0 if rng_seed is None else rng_seed)
    payload_bits = np.concatenate(
        [
            encode_frame_bch_ldpc_codewords(
                transport_payload[
                    index * config.transport_bytes_per_frame :
                    (index + 1) * config.transport_bytes_per_frame
                ],
                fec_rate_index=config.fec_rate_index,
            )
            for index in range(payload_frames)
        ]
    ).astype(np.uint8, copy=False)
    drain_bits = rng.integers(
        0,
        2,
        size=drain_frames * config.bits_per_frame,
        dtype=np.uint8,
    )
    tx_symbols = qam_modulate(
        np.concatenate((payload_bits, drain_bits)).astype(np.uint8, copy=False),
        mode=config.qam_mode,
    )
    if config.interleaver_mode is None:
        before_symbols = tx_symbols.astype(np.complex64, copy=False)
    else:
        before_symbols = convolutional_interleave(
            tx_symbols,
            mode=config.interleaver_mode,
            flush=False,
            phase=config.interleaver_phase,
        ).astype(np.complex64, copy=False)
        fill_mask = before_symbols == 0
        fill_count = int(np.count_nonzero(fill_mask))
        if fill_count:
            fill_bits = rng.integers(
                0,
                2,
                size=fill_count * config.bits_per_symbol,
                dtype=np.uint8,
            )
            before_symbols[fill_mask] = qam_modulate(
                fill_bits,
                mode=config.qam_mode,
            )
    if config.interleaver_mode is None:
        deinterleaved = before_symbols
        latency = 0
    else:
        spec = SYMBOL_INTERLEAVERS[config.interleaver_mode]
        latency = int(spec.full_stream_latency_symbols)
        deinterleaved = convolutional_deinterleave(
            before_symbols,
            mode=config.interleaver_mode,
            flush=False,
            phase=config.interleaver_phase,
        )
    discarded = min(latency, int(deinterleaved.size))
    after_symbols = deinterleaved[discarded:].astype(np.complex64, copy=False)
    llr = qam_soft_demodulate(
        after_symbols,
        mode=config.qam_mode,
    ).astype(np.float32, copy=False)
    hard_bits = (llr < 0).astype(np.uint8)
    profile = dtmb_ldpc_profile(config.fec_rate_index)
    bits_per_frame = config.bits_per_frame
    frame_indices, codeword_indices, bit_indices_in_codeword = (
        _pre_ldpc_dump_bit_indices(
            hard_bits.size,
            bits_per_frame=bits_per_frame,
            codeword_bits=profile.codeword_bits,
        )
    )
    total_frames = payload_frames + drain_frames
    data_symbols_per_frame = [DATA_SYMBOLS_PER_FRAME] * total_frames
    dump_path = Path(output_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = dump_path.with_suffix(".json")
    qam_quality_report = {
        "qam_mode": config.qam_mode,
        "normalizer": "standard",
        "before_symbol_deinterleave": qam_symbol_quality(
            before_symbols,
            mode=config.qam_mode,
            normalize=True,
        ),
        "after_symbol_deinterleave": qam_symbol_quality(
            after_symbols,
            mode=config.qam_mode,
            normalize=True,
        ),
    }
    metadata: dict[str, Any] = {
        "capture_path": "synthetic_continuous_pre_ldpc",
        "npz_file": str(dump_path),
        "json_file": str(sidecar_path),
        "hard_bits_dtype": "uint8",
        "hard_bits_encoding": "one_bit_per_byte",
        "llr_dtype": "<f4",
        "llr_count": int(llr.size),
        "hard_bit_count": int(hard_bits.size),
        "bits_per_symbol": int(config.bits_per_symbol),
        "bits_per_frame": int(bits_per_frame),
        "codeword_bits": int(profile.codeword_bits),
        "codewords": int(hard_bits.size // profile.codeword_bits),
        "unused_bits": int(hard_bits.size % profile.codeword_bits),
        "chosen_qam": config.qam_mode,
        "requested_qam": config.qam_mode,
        "chosen_fec_rate": int(config.fec_rate_index),
        "requested_fec_rate": int(config.fec_rate_index),
        "chosen_interleaver_mode": config.interleaver_mode or "none",
        "requested_interleaver_mode": config.interleaver_mode or "none",
        "symbol_deinterleave_phase": int(config.interleaver_phase),
        "symbol_deinterleave_latency_symbols": int(latency),
        "symbol_deinterleave_discarded_symbols": int(discarded),
        "chosen_system_info_vector": int(system_info_index),
        "system_info": {
            "selected": {
                "index": int(system_info_index),
                "qam_mode": config.qam_mode,
                "fec_rate_index": int(config.fec_rate_index),
                "interleaver_mode": config.interleaver_mode,
            }
        },
        "frame_body_mode": "C3780",
        "pn_mode": "none",
        "fft_window_offset": 0,
        "fft_bin_shift": 0,
        "body_window_offset_symbols": 0,
        "data_carrier_order": "logical",
        "carrier_order_version": "synthetic_pre_ldpc_no_frequency_deinterleave",
        "frequency_deinterleaver_version": "none:synthetic_pre_ldpc",
        "equalizer_mode": "none",
        "per_frame_cpe_enabled": False,
        "per_frame_data_evm_rms": [0.0] * total_frames,
        "median_data_evm_rms": 0.0,
        "min_data_evm_rms": 0.0,
        "per_frame_cpe_rad": [0.0] * total_frames,
        "llr_stats": _pre_ldpc_dump_llr_stats(llr),
        "hard_bit_balance_per_plane": _pre_ldpc_dump_hard_bit_balance_per_plane(
            hard_bits,
            bits_per_symbol=config.bits_per_symbol,
        ),
        "qam_symbol_quality": qam_quality_report,
        "data_symbols_before_symbol_deinterleave": int(before_symbols.size),
        "data_symbols_after_symbol_deinterleave": int(after_symbols.size),
        "data_symbols_per_frame": data_symbols_per_frame,
        "frames_used": int(total_frames),
        "payload_frames": int(payload_frames),
        "drain_frames": int(drain_frames),
        "arrays": {
            "hard_bits": "uint8 flat pre-LDPC hard decisions",
            "llr": "float32 flat pre-LDPC LLRs; positive means bit 0",
            "frame_indices": "int32 output-frame index per hard bit",
            "codeword_indices": "int32 LDPC codeword index per hard bit, -1 for tail",
            "bit_indices_in_codeword": "int32 bit position inside LDPC codeword, -1 for tail",
            "data_symbols_before_symbol_deinterleave": "complex64 QAM symbols before convolutional deinterleave",
            "data_symbols_after_symbol_deinterleave": "complex64 QAM symbols after latency discard",
            "data_symbol_frame_indices_before_deinterleave": "int32 source frame index per pre-deinterleave data symbol",
            "frame_start_symbols": "int64 synthetic C=3780 frame start symbol offsets",
        },
    }
    np.savez(
        dump_path,
        hard_bits=hard_bits,
        llr=llr,
        frame_indices=frame_indices,
        codeword_indices=codeword_indices,
        bit_indices_in_codeword=bit_indices_in_codeword,
        data_symbols_before_symbol_deinterleave=before_symbols,
        data_symbols_after_symbol_deinterleave=after_symbols,
        data_symbol_frame_indices_before_deinterleave=_pre_ldpc_dump_symbol_frame_indices(
            data_symbols_per_frame
        ),
        frame_start_symbols=np.arange(total_frames, dtype=np.int64)
        * DATA_SYMBOLS_PER_FRAME,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def synthesize_continuous_c3780_capture(
    transport_payload: bytes,
    *,
    system_info_index: int = 23,
    pre_roll_frames: int | None = None,
    pn_mode: str = "pn945",
    output_path: Any | None = None,
    amplitude: float = 0.75,
    coded_drain_frames: bool = True,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Produce a continuous C=3780 CI8 capture with real coded frames everywhere.

    The capture carries ``len(transport_payload) / transport_bytes_per_frame``
    payload frames followed by ``pre_roll_frames`` drain frames so the
    continuous convolutional interleaver has something real to run through
    while the receiver's deinterleaver latency is flushing. No zero frames
    are inserted anywhere; every frame in the capture looks like a real
    DTMB signal with a fresh PN945 header, a full 3744-carrier frame body,
    and the expected 36 system-information pilots.

    Set ``coded_drain_frames=False`` for fast receiver-seam fixtures that only
    score the post-deinterleaver payload frame. The drain still carries random
    QAM symbols to exercise the continuous interleaver, but those tail frames
    are not BCH/LDPC-valid.

    This is the receiver-side counterpart to the bit-level fixture: running
    ``receive_capture`` on the produced CI8 should recover
    ``transport_payload`` byte-for-byte once the symbol deinterleaver's
    configured pre-roll frames drain.
    """

    from .frequency import frequency_interleave
    from .pn import pn_header_symbols_for_body_power
    from .system_info import SYSTEM_INFO_VECTORS, system_info_symbols

    qam_mode, fec_rate_index, interleaver_mode = _system_info_index_to_parameters(
        system_info_index
    )
    config = ContinuousStreamConfig(
        qam_mode=qam_mode,
        fec_rate_index=fec_rate_index,
        interleaver_mode=interleaver_mode,  # type: ignore[arg-type]
        interleaver_phase=0,
    )
    if len(transport_payload) == 0:
        raise ValueError("transport_payload must contain at least one frame")
    if len(transport_payload) % config.transport_bytes_per_frame:
        raise ValueError(
            "transport_payload must be a whole multiple of "
            f"{config.transport_bytes_per_frame} bytes"
        )
    payload_frames = len(transport_payload) // config.transport_bytes_per_frame
    drain_frames = (
        pre_roll_frames
        if pre_roll_frames is not None
        else config.pre_roll_frames
    )
    if drain_frames < 0:
        raise ValueError("pre_roll_frames must be non-negative")

    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng(0)
    payload_bits = np.concatenate(
        [
            encode_frame_bch_ldpc_codewords(
                transport_payload[
                    index * config.transport_bytes_per_frame :
                    (index + 1) * config.transport_bytes_per_frame
                ],
                fec_rate_index=config.fec_rate_index,
            )
            for index in range(payload_frames)
        ]
    ).astype(np.uint8, copy=False)
    if coded_drain_frames and drain_frames:
        drain_payload = rng.integers(
            0,
            256,
            size=drain_frames * config.transport_bytes_per_frame,
            dtype=np.uint8,
        ).tobytes()
        drain_bits = np.concatenate(
            [
                encode_frame_bch_ldpc_codewords(
                    drain_payload[
                        index * config.transport_bytes_per_frame :
                        (index + 1) * config.transport_bytes_per_frame
                    ],
                    fec_rate_index=config.fec_rate_index,
                )
                for index in range(drain_frames)
            ]
        ).astype(np.uint8, copy=False)
    else:
        drain_bits = rng.integers(
            0,
            2,
            size=drain_frames * config.bits_per_frame,
            dtype=np.uint8,
        )

    # Payload first, drain frames second. The RX deinterleaver latency emits
    # the payload first once (B-1)*M*B symbols have been consumed.
    tx_bits = np.concatenate((payload_bits, drain_bits)).astype(np.uint8, copy=False)
    tx_symbols = qam_modulate(tx_bits, mode=config.qam_mode)
    interleaved_symbols = (
        tx_symbols
        if config.interleaver_mode is None
        else convolutional_interleave(
            tx_symbols,
            mode=config.interleaver_mode,
            flush=False,
            phase=config.interleaver_phase,
        )
    )

    system_info = system_info_symbols(
        SYSTEM_INFO_VECTORS[system_info_index],
        frame_body_mode="C3780",
    )
    frames = payload_frames + drain_frames
    bodies: list[np.ndarray] = []
    for index in range(frames):
        logical = np.empty(3780, dtype=np.complex64)
        logical[:36] = system_info
        logical[36:] = interleaved_symbols[
            index * DATA_SYMBOLS_PER_FRAME :
            (index + 1) * DATA_SYMBOLS_PER_FRAME
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
    if output_path is not None:
        Path(output_path).write_bytes(ci8.tobytes())
    return {
        "config": config.to_dict(),
        "system_info_index": system_info_index,
        "pn_mode": pn_mode,
        "payload_frames": payload_frames,
        "drain_frames": drain_frames,
        "coded_drain_frames": bool(coded_drain_frames),
        "total_frames": frames,
        "symbols_per_frame": 3780 + pn_header.size,
        "pn_header_symbols": int(pn_header.size),
        "header_to_body_power_ratio": float(
            np.mean(np.abs(pn_header) ** 2) / max(body_power, 1e-24)
        ),
        "ci8": ci8,
    }


def _pre_ldpc_dump_bit_indices(
    bit_count: int,
    *,
    bits_per_frame: int,
    codeword_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = np.full(bit_count, -1, dtype=np.int32)
    if bits_per_frame > 0:
        frame_indices = (
            np.arange(bit_count, dtype=np.int64) // int(bits_per_frame)
        ).astype(np.int32)
    codeword_indices = np.full(bit_count, -1, dtype=np.int32)
    bit_indices = np.full(bit_count, -1, dtype=np.int32)
    usable = (bit_count // int(codeword_bits)) * int(codeword_bits)
    if usable:
        idx = np.arange(usable, dtype=np.int64)
        codeword_indices[:usable] = (idx // int(codeword_bits)).astype(np.int32)
        bit_indices[:usable] = (idx % int(codeword_bits)).astype(np.int32)
    return frame_indices, codeword_indices, bit_indices


def _pre_ldpc_dump_symbol_frame_indices(
    data_symbols_per_frame: Sequence[int],
) -> np.ndarray:
    chunks = [
        np.full(int(count), index, dtype=np.int32)
        for index, count in enumerate(data_symbols_per_frame)
        if int(count) > 0
    ]
    if not chunks:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32, copy=False)


def _pre_ldpc_dump_llr_stats(llr: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "positive_fraction": None,
            "negative_fraction": None,
            "zero_fraction": None,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def _pre_ldpc_dump_hard_bit_balance_per_plane(
    hard_bits: np.ndarray,
    *,
    bits_per_symbol: int,
) -> list[float]:
    values = np.asarray(hard_bits, dtype=np.uint8).reshape(-1)
    if bits_per_symbol <= 0:
        return []
    usable = (values.size // int(bits_per_symbol)) * int(bits_per_symbol)
    if usable == 0:
        return []
    grouped = values[:usable].reshape(-1, int(bits_per_symbol))
    return [float(np.mean(grouped[:, index])) for index in range(bits_per_symbol)]
