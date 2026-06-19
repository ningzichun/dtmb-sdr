"""DTMB FEC stage helpers around LDPC, BCH, and descrambling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .bch import BCH_CODE_BITS, bch_decode, bch_encode
from .bits import pack_bits_msb, unpack_bytes_msb
from .ldpc import (
    DtmbLdpcProfile,
    dtmb_ldpc_decode_transmitted_llr,
    dtmb_ldpc_encode_message_bits,
    dtmb_ldpc_profile,
    extract_dtmb_message_bits_from_full_codewords,
)
from .scrambler import scramble_bits


LdpcParityPosition = Literal["front", "back"]


@dataclass(frozen=True)
class FrameFecDecodeResult:
    """Decoded payload bits and FEC diagnostics for one signal frame."""

    transport_bits: np.ndarray
    bch_blocks: int
    bch_corrected_errors: int
    bch_unclean_blocks: int
    ldpc_profile: DtmbLdpcProfile
    ldpc_codewords: int = 0
    ldpc_converged_codewords: int = 0
    ldpc_iterations: tuple[int, ...] = ()
    ldpc_syndrome_weights: tuple[int, ...] = ()

    @property
    def transport_bytes(self) -> bytes:
        return pack_bits_msb(self.transport_bits, truncate=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_bits": int(self.transport_bits.size),
            "transport_bytes": self.transport_bits.size // 8,
            "bch_blocks": self.bch_blocks,
            "bch_corrected_errors": self.bch_corrected_errors,
            "bch_unclean_blocks": self.bch_unclean_blocks,
            "ldpc_profile": self.ldpc_profile.to_dict(),
            "ldpc_codewords": self.ldpc_codewords,
            "ldpc_converged_codewords": self.ldpc_converged_codewords,
            "ldpc_iterations": list(self.ldpc_iterations),
            "ldpc_syndrome_weights": list(self.ldpc_syndrome_weights),
        }


def encode_frame_bch_ldpc_message_bits(
    transport_bytes: bytes,
    *,
    fec_rate_index: int,
) -> np.ndarray:
    """Build LDPC message bits from one frame of transport bytes.

    This helper stops before LDPC parity generation. It is useful for validating
    BCH and descrambler integration while the DTMB LDPC matrix is being wired.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    payload_bits = unpack_bytes_msb(transport_bytes)
    expected_payload_bits = profile.bch_blocks_per_codeword * 752
    if payload_bits.size == 0 or payload_bits.size % expected_payload_bits:
        raise ValueError(
            f"transport_bytes must contain one or more whole "
            f"{expected_payload_bits // 8}-byte LDPC payload groups "
            f"for FEC rate {fec_rate_index}"
        )
    scrambled = scramble_bits(payload_bits)
    blocks = [
        bch_encode(scrambled[start : start + 752])
        for start in range(0, scrambled.size, 752)
    ]
    return np.concatenate(blocks).astype(np.uint8)


def encode_frame_bch_ldpc_codewords(
    transport_bytes: bytes,
    *,
    fec_rate_index: int,
) -> np.ndarray:
    """Encode one DTMB FEC frame through scrambler, BCH, and LDPC."""

    message_bits = encode_frame_bch_ldpc_message_bits(
        transport_bytes,
        fec_rate_index=fec_rate_index,
    )
    return dtmb_ldpc_encode_message_bits(
        message_bits,
        fec_rate_index=fec_rate_index,
        transmitted=True,
    )


def extract_ldpc_message_bits_from_codewords(
    codeword_bits: np.ndarray,
    *,
    fec_rate_index: int,
    parity_position: LdpcParityPosition = "front",
) -> np.ndarray:
    """Extract LDPC message sections from hard codeword bits.

    The DTMB Design Library describes LDPC output as parity bits first. The
    default therefore extracts message bits from the tail of each 7488-bit
    codeword.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(codeword_bits)
    if values.size % profile.codeword_bits:
        raise ValueError("bit count must be a whole number of LDPC codewords")
    rows = values.reshape(-1, profile.codeword_bits)
    if parity_position == "front":
        messages = rows[:, profile.parity_bits :]
    elif parity_position == "back":
        messages = rows[:, : profile.message_bits]
    else:
        raise ValueError("parity_position must be front or back")
    return messages.reshape(-1).astype(np.uint8)


def decode_frame_bch_descramble_from_ldpc_messages(
    ldpc_message_bits: np.ndarray,
    *,
    fec_rate_index: int,
    correct_bch: bool = True,
) -> FrameFecDecodeResult:
    """Decode one frame's LDPC message bits through BCH and descrambling."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(ldpc_message_bits)
    if values.size == 0 or values.size % profile.message_bits:
        raise ValueError("ldpc_message_bits must contain whole frame LDPC messages")
    decoded_blocks = []
    corrected_errors = 0
    unclean = 0
    for start in range(0, values.size, BCH_CODE_BITS):
        block = values[start : start + BCH_CODE_BITS]
        if block.size != BCH_CODE_BITS:
            raise ValueError("LDPC message bits must align to BCH codewords")
        decoded, corrected, clean = bch_decode(block, correct=correct_bch)
        corrected_errors += corrected
        if not clean:
            unclean += 1
        decoded_blocks.append(decoded)
    scrambled_payload = np.concatenate(decoded_blocks).astype(np.uint8)
    transport_bits = scramble_bits(scrambled_payload)
    return FrameFecDecodeResult(
        transport_bits=transport_bits,
        bch_blocks=len(decoded_blocks),
        bch_corrected_errors=corrected_errors,
        bch_unclean_blocks=unclean,
        ldpc_profile=profile,
    )


def decode_frame_bch_descramble_from_ldpc_codewords(
    codeword_bits: np.ndarray,
    *,
    fec_rate_index: int,
    parity_position: LdpcParityPosition = "front",
    correct_bch: bool = True,
) -> FrameFecDecodeResult:
    """Decode a frame when hard LDPC codewords already contain reliable message bits.

    This is not a replacement for LDPC decoding. It is an integration seam for
    synthetic vectors and for validating the BCH/descrambler/TS portion while
    the parity-check matrices are still being transcribed.
    """

    messages = extract_ldpc_message_bits_from_codewords(
        codeword_bits,
        fec_rate_index=fec_rate_index,
        parity_position=parity_position,
    )
    return decode_frame_bch_descramble_from_ldpc_messages(
        messages,
        fec_rate_index=fec_rate_index,
        correct_bch=correct_bch,
    )


def decode_frame_bch_descramble_from_ldpc_llr(
    codeword_llr: np.ndarray,
    *,
    fec_rate_index: int,
    max_ldpc_iterations: int = 50,
    ldpc_attenuation: float = 0.75,
    correct_bch: bool = True,
) -> FrameFecDecodeResult:
    """Decode transmitted LDPC LLRs through LDPC, BCH, and descrambling."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = np.asarray(codeword_llr, dtype=np.float32).reshape(-1)
    if values.size == 0 or values.size % profile.codeword_bits:
        raise ValueError("codeword_llr must contain whole transmitted LDPC codewords")
    decoded_messages = []
    iterations = []
    syndrome_weights = []
    converged = 0
    for row in values.reshape(-1, profile.codeword_bits):
        ldpc = dtmb_ldpc_decode_transmitted_llr(
            row,
            fec_rate_index=fec_rate_index,
            max_iterations=max_ldpc_iterations,
            attenuation=ldpc_attenuation,
        )
        if ldpc.converged:
            converged += 1
        iterations.append(ldpc.iterations)
        syndrome_weights.append(ldpc.syndrome_weight)
        decoded_messages.append(
            extract_dtmb_message_bits_from_full_codewords(
                ldpc.bits,
                fec_rate_index=fec_rate_index,
            )
        )
    message_bits = np.concatenate(decoded_messages).astype(np.uint8)
    bch = decode_frame_bch_descramble_from_ldpc_messages(
        message_bits,
        fec_rate_index=fec_rate_index,
        correct_bch=correct_bch,
    )
    return FrameFecDecodeResult(
        transport_bits=bch.transport_bits,
        bch_blocks=bch.bch_blocks,
        bch_corrected_errors=bch.bch_corrected_errors,
        bch_unclean_blocks=bch.bch_unclean_blocks,
        ldpc_profile=profile,
        ldpc_codewords=len(iterations),
        ldpc_converged_codewords=converged,
        ldpc_iterations=tuple(iterations),
        ldpc_syndrome_weights=tuple(syndrome_weights),
    )


def _as_binary(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return values
