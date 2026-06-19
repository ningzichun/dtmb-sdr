"""DTMB system-information symbol helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SystemInfoQamMode = Literal["4qam", "16qam", "32qam", "64qam"]
SystemInfoInterleaverMode = Literal["mode1", "mode2"]


# GB 20600-2006 Appendix G lists 64 final 32-bit system-information vectors.
# The even rows are the bitwise complements of the preceding odd rows.
SYSTEM_INFO_VECTORS_ODD: dict[int, str] = {
    1: "00011110101011100100100010110011",
    3: "01111000110010000010111011010101",
    5: "01110111110001110010000111011010",
    7: "00100010100100100111010010001111",
    9: "01001011111110110001110111100110",
    11: "00010001101000010100011110111100",
    13: "01111000001101110010111000101010",
    15: "00101101100111010111101110000000",
    17: "01110111001110000010000100100101",
    19: "00100010011011010111010001110000",
    21: "01000100000010110001001000010110",
    23: "00010001010111100100011101000011",
    25: "00101101011000100111101101111111",
    27: "01000100111101000001001011101001",
    29: "01001011000001000001110100011001",
    31: "00011110010100010100100001001100",
    33: "01111000110010001101000100101010",
    35: "00101101100111011000010001111111",
    37: "01001011111110111110001000011001",
    39: "00011110101011101011011101001100",
    41: "01110111110001111101111000100101",
    43: "00100010100100101000101101110000",
    45: "01000100111101001110110100010110",
    47: "00010001101000011011100001000011",
    49: "01111000001101111101000111010101",
    51: "00101101011000101000010010000000",
    53: "01001011000001001110001011100110",
    55: "00011110010100011011011110110011",
    57: "01110111001110001101111011011010",
    59: "00100010011011011000101110001111",
    61: "01000100000010111110110111101001",
    63: "00010001010111101011100010111100",
}

SYSTEM_INFO_VECTORS_EVEN: dict[int, str] = {
    index + 1: "".join("1" if bit == "0" else "0" for bit in bits)
    for index, bits in SYSTEM_INFO_VECTORS_ODD.items()
}

SYSTEM_INFO_VECTORS: dict[int, str] = {
    **SYSTEM_INFO_VECTORS_ODD,
    **SYSTEM_INFO_VECTORS_EVEN,
}
SYSTEM_INFO_VECTORS = dict(sorted(SYSTEM_INFO_VECTORS.items()))

SYSTEM_INFO_MEANINGS: dict[int, str] = {
    1: "first frame indication, odd superframe",
    2: "first frame indication, even superframe",
    3: "4QAM-NR, FEC rate 3, interleave mode 1",
    4: "4QAM-NR, FEC rate 3, interleave mode 2",
    5: "4QAM, FEC rate 1, interleave mode 1",
    6: "4QAM, FEC rate 1, interleave mode 2",
    7: "4QAM, FEC rate 2, interleave mode 1",
    8: "4QAM, FEC rate 2, interleave mode 2",
    9: "4QAM, FEC rate 3, interleave mode 1",
    10: "4QAM, FEC rate 3, interleave mode 2",
    11: "16QAM, FEC rate 1, interleave mode 1",
    12: "16QAM, FEC rate 1, interleave mode 2",
    13: "16QAM, FEC rate 2, interleave mode 1",
    14: "16QAM, FEC rate 2, interleave mode 2",
    15: "16QAM, FEC rate 3, interleave mode 1",
    16: "16QAM, FEC rate 3, interleave mode 2",
    17: "32QAM, FEC rate 3, interleave mode 1",
    18: "32QAM, FEC rate 3, interleave mode 2",
    19: "64QAM, FEC rate 1, interleave mode 1",
    20: "64QAM, FEC rate 1, interleave mode 2",
    21: "64QAM, FEC rate 2, interleave mode 1",
    22: "64QAM, FEC rate 2, interleave mode 2",
    23: "64QAM, FEC rate 3, interleave mode 1",
    24: "64QAM, FEC rate 3, interleave mode 2",
    **{index: "reserved" for index in range(25, 65)},
}


@dataclass(frozen=True)
class DtmbTransmissionParameters:
    """Decoded transmission parameters carried by one system-information vector."""

    index: int
    frame_body_mode: str
    qam_mode: SystemInfoQamMode | None
    qam_label: str
    fec_rate_index: int | None
    interleaver_mode: SystemInfoInterleaverMode | None
    nr_mapping: bool
    supported_by_receiver: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frame_body_mode": self.frame_body_mode,
            "qam_mode": self.qam_mode,
            "qam_label": self.qam_label,
            "fec_rate_index": self.fec_rate_index,
            "interleaver_mode": self.interleaver_mode,
            "nr_mapping": self.nr_mapping,
            "supported_by_receiver": self.supported_by_receiver,
        }


_SystemInfoParameterRow = tuple[
    str,
    SystemInfoQamMode | None,
    int | None,
    SystemInfoInterleaverMode | None,
    bool,
]

_SYSTEM_INFO_PARAMETER_ROWS: dict[int, _SystemInfoParameterRow] = {
    1: ("first-frame-indication", None, None, None, False),
    2: ("first-frame-indication", None, None, None, False),
    3: ("4QAM-NR", "4qam", 3, "mode1", True),
    4: ("4QAM-NR", "4qam", 3, "mode2", True),
    5: ("4QAM", "4qam", 1, "mode1", False),
    6: ("4QAM", "4qam", 1, "mode2", False),
    7: ("4QAM", "4qam", 2, "mode1", False),
    8: ("4QAM", "4qam", 2, "mode2", False),
    9: ("4QAM", "4qam", 3, "mode1", False),
    10: ("4QAM", "4qam", 3, "mode2", False),
    11: ("16QAM", "16qam", 1, "mode1", False),
    12: ("16QAM", "16qam", 1, "mode2", False),
    13: ("16QAM", "16qam", 2, "mode1", False),
    14: ("16QAM", "16qam", 2, "mode2", False),
    15: ("16QAM", "16qam", 3, "mode1", False),
    16: ("16QAM", "16qam", 3, "mode2", False),
    17: ("32QAM", "32qam", 3, "mode1", False),
    18: ("32QAM", "32qam", 3, "mode2", False),
    19: ("64QAM", "64qam", 1, "mode1", False),
    20: ("64QAM", "64qam", 1, "mode2", False),
    21: ("64QAM", "64qam", 2, "mode1", False),
    22: ("64QAM", "64qam", 2, "mode2", False),
    23: ("64QAM", "64qam", 3, "mode1", False),
    24: ("64QAM", "64qam", 3, "mode2", False),
    **{index: ("reserved", None, None, None, False) for index in range(25, 65)},
}


@dataclass(frozen=True)
class SystemInfoMatch:
    """Best match between observed system-information symbols and a standard vector."""

    index: int
    meaning: str
    polarity: int
    frame_body_mode: str
    metric: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "meaning": self.meaning,
            "polarity": self.polarity,
            "frame_body_mode": self.frame_body_mode,
            "metric": self.metric,
            "parameters": transmission_parameters_for_index(
                self.index,
                frame_body_mode=self.frame_body_mode,
            ).to_dict(),
        }


def transmission_parameters_for_index(
    index: int,
    *,
    frame_body_mode: str = "C3780",
) -> DtmbTransmissionParameters:
    """Return structured parameters for a decoded system-information index."""

    try:
        qam_label, qam_mode, fec_rate_index, interleaver_mode, nr_mapping = (
            _SYSTEM_INFO_PARAMETER_ROWS[index]
        )
    except KeyError as exc:
        raise ValueError("unknown system-information index") from exc
    # The 32QAM demapper is implemented, but 32QAM/rate-3 carries 2.5 LDPC
    # codewords per C=3780 frame. Keep auto FEC disabled until the receiver has
    # an explicit multi-frame LDPC scheduler for that mode.
    supported = (
        qam_mode is not None
        and fec_rate_index is not None
        and interleaver_mode is not None
        and not nr_mapping
        and qam_label != "32QAM"
        and frame_body_mode in ("C3780", "C1")
    )
    return DtmbTransmissionParameters(
        index=index,
        frame_body_mode=frame_body_mode,
        qam_mode=qam_mode,
        qam_label=qam_label,
        fec_rate_index=fec_rate_index,
        interleaver_mode=interleaver_mode,
        nr_mapping=nr_mapping,
        supported_by_receiver=supported,
    )


def system_info_symbols(bits32: str, *, frame_body_mode: str = "C3780") -> np.ndarray:
    """Map 4 C-mode bits plus 32 system bits to diagonal 4QAM symbols."""

    if len(bits32) != 32 or any(bit not in "01" for bit in bits32):
        raise ValueError("bits32 must contain 32 binary characters")
    if frame_body_mode == "C1":
        suffix = "0000"
    elif frame_body_mode == "C3780":
        suffix = "1111"
    else:
        raise ValueError("frame_body_mode must be C1 or C3780")
    bits = np.asarray([int(bit) for bit in suffix + bits32], dtype=np.uint8)
    chips = np.where(bits == 0, 1.0, -1.0).astype(np.float32)
    return (chips + 1j * chips).astype(np.complex64)


def classify_system_info(
    symbols: np.ndarray,
    *,
    allow_polarity_inversion: bool = False,
    max_common_phase_radians: float = np.pi / 4,
    frame_body_modes: tuple[str, ...] = ("C3780", "C1"),
) -> list[SystemInfoMatch]:
    """Rank standard system-information vectors against observed symbols.

    ``frame_body_modes`` restricts which reference hypotheses are scored.
    On a receiver path that has already committed to C=3780 frame-body
    demodulation, passing ``("C3780",)`` avoids bogus C=1 top-picks on
    noisy frames where the 4-bit C-mode prefix dominates the correlation.
    """

    if max_common_phase_radians < 0.0 or max_common_phase_radians > np.pi / 2:
        raise ValueError("max_common_phase_radians must be between 0 and pi/2")
    observed = np.asarray(symbols, dtype=np.complex64)
    if observed.size != 36:
        raise ValueError("system information requires 36 symbols")
    observed_power = float(np.sum(np.abs(observed) ** 2))
    if observed_power <= 0:
        return []
    allowed_modes = tuple(frame_body_modes)
    for mode in allowed_modes:
        if mode not in ("C3780", "C1"):
            raise ValueError(f"unknown frame_body_mode {mode}")
    if not allowed_modes:
        raise ValueError("frame_body_modes must contain at least one entry")

    matches: list[SystemInfoMatch] = []
    polarities = (1, -1) if allow_polarity_inversion else (1,)
    for index, bits in SYSTEM_INFO_VECTORS.items():
        for frame_body_mode in allowed_modes:
            reference = system_info_symbols(bits, frame_body_mode=frame_body_mode)
            ref_power = float(np.sum(np.abs(reference) ** 2))
            for polarity in polarities:
                correlation = np.vdot(reference * polarity, observed) / np.sqrt(
                    ref_power * observed_power
                )
                metric = _phase_limited_metric(
                    correlation,
                    max_common_phase_radians=max_common_phase_radians,
                )
                matches.append(
                    SystemInfoMatch(
                        index=index,
                        meaning=SYSTEM_INFO_MEANINGS[index],
                        polarity=polarity,
                        frame_body_mode=frame_body_mode,
                        metric=metric,
                    )
                )
    return sorted(matches, key=lambda match: match.metric, reverse=True)


def _phase_limited_metric(
    correlation: complex,
    *,
    max_common_phase_radians: float,
) -> float:
    if max_common_phase_radians <= 0.0:
        return float(np.real(correlation))
    phase = abs(float(np.angle(correlation)))
    if phase <= max_common_phase_radians:
        return float(abs(correlation))
    return float(np.real(correlation))
