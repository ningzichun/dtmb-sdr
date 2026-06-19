"""DTMB signal-frame payload and FEC scheduling plan helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .frequency import (
    DATA_SYMBOLS_PER_FRAME,
    FRAME_BODY_SYMBOLS,
    SYSTEM_INFO_SYMBOLS,
)
from .ldpc import dtmb_ldpc_profile
from .qam import QAM_DEFINITIONS
from .symbol_interleaver import SYMBOL_INTERLEAVERS
from .system_info import transmission_parameters_for_index


GB20600_FRAME_PLAN_REFERENCES = (
    "GB 20600-2006 4.4.1: input bytes are MSB-first and the scrambler resets "
    "at signal-frame start",
    "GB 20600-2006 4.4.2: BCH(762,752), LDPC(7488,K), five leading parity "
    "bits deleted, BCH codewords enter LDPC in order",
    "GB 20600-2006 4.4.3: the first FEC output bit entering QAM is the symbol "
    "codeword LSB",
    "GB 20600-2006 4.4.4: 3744-symbol basic data blocks are symbol "
    "interleaved; branch 0 is synchronized to the block's first symbol",
    "GB 20600-2006 4.6.4: each frame body carries 36 system-info symbols plus "
    "3744 data symbols",
)


@dataclass(frozen=True)
class DtmbFramePlan:
    """Standard-derived non-NR signal-frame scheduling facts."""

    system_info_index: int
    frame_body_mode: str
    qam_mode: str
    fec_rate_index: int
    interleaver_mode: str
    receiver_supported: bool
    bits_per_symbol: int
    data_symbols_per_frame: int
    system_info_symbols: int
    frame_body_symbols: int
    coded_bits_per_frame: int
    ldpc_codeword_bits: int
    ldpc_full_codeword_bits: int
    ldpc_message_bits: int
    ldpc_parity_bits: int
    bch_blocks_per_codeword: int
    transport_bytes_per_codeword: int
    codewords_per_frame: Fraction
    bch_blocks_per_frame: Fraction
    transport_bits_per_frame: Fraction
    interleaver_branch_count: int
    interleaver_delay_step_symbols: int
    deinterleaver_latency_symbols: int
    deinterleaver_latency_frames: Fraction

    @property
    def whole_codewords_per_frame(self) -> bool:
        return self.codewords_per_frame.denominator == 1

    @property
    def whole_transport_bytes_per_frame(self) -> bool:
        return self.transport_bits_per_frame.denominator == 1 and (
            self.transport_bits_per_frame.numerator % 8 == 0
        )

    @property
    def transport_bytes_per_frame(self) -> Fraction:
        return self.transport_bits_per_frame / 8

    @property
    def codeword_boundary_period_frames(self) -> int:
        """Frames needed for the coded bit stream to return to a codeword boundary."""

        return int(self.codewords_per_frame.denominator)

    @property
    def requires_multiframe_ldpc_scheduler(self) -> bool:
        return not self.whole_codewords_per_frame

    @property
    def receiver_frame_grouping(self) -> str:
        if self.whole_codewords_per_frame:
            return "whole_ldpc_codewords_per_signal_frame"
        return "ldpc_codewords_span_signal_frames"

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_info_index": self.system_info_index,
            "frame_body_mode": self.frame_body_mode,
            "qam_mode": self.qam_mode,
            "fec_rate_index": self.fec_rate_index,
            "interleaver_mode": self.interleaver_mode,
            "receiver_supported": self.receiver_supported,
            "bits_per_symbol": self.bits_per_symbol,
            "data_symbols_per_frame": self.data_symbols_per_frame,
            "system_info_symbols": self.system_info_symbols,
            "frame_body_symbols": self.frame_body_symbols,
            "coded_bits_per_frame": self.coded_bits_per_frame,
            "ldpc_codeword_bits": self.ldpc_codeword_bits,
            "ldpc_full_codeword_bits": self.ldpc_full_codeword_bits,
            "ldpc_message_bits": self.ldpc_message_bits,
            "ldpc_parity_bits": self.ldpc_parity_bits,
            "bch_blocks_per_codeword": self.bch_blocks_per_codeword,
            "transport_bytes_per_codeword": self.transport_bytes_per_codeword,
            "codewords_per_frame": _fraction_to_dict(self.codewords_per_frame),
            "bch_blocks_per_frame": _fraction_to_dict(self.bch_blocks_per_frame),
            "transport_bits_per_frame": _fraction_to_dict(
                self.transport_bits_per_frame
            ),
            "transport_bytes_per_frame": _fraction_to_dict(
                self.transport_bytes_per_frame
            ),
            "whole_codewords_per_frame": self.whole_codewords_per_frame,
            "whole_transport_bytes_per_frame": self.whole_transport_bytes_per_frame,
            "codeword_boundary_period_frames": self.codeword_boundary_period_frames,
            "requires_multiframe_ldpc_scheduler": (
                self.requires_multiframe_ldpc_scheduler
            ),
            "receiver_frame_grouping": self.receiver_frame_grouping,
            "interleaver_branch_count": self.interleaver_branch_count,
            "interleaver_delay_step_symbols": self.interleaver_delay_step_symbols,
            "deinterleaver_latency_symbols": self.deinterleaver_latency_symbols,
            "deinterleaver_latency_frames": _fraction_to_dict(
                self.deinterleaver_latency_frames
            ),
            "scrambler_reset": "per_signal_frame",
            "references": list(GB20600_FRAME_PLAN_REFERENCES),
        }


def frame_plan_for_system_info(
    system_info_index: int,
    *,
    frame_body_mode: str = "C3780",
) -> DtmbFramePlan:
    """Return the standard-derived FEC/QAM/interleaver plan for one mode."""

    parameters = transmission_parameters_for_index(
        int(system_info_index),
        frame_body_mode=frame_body_mode,
    )
    if parameters.nr_mapping:
        raise ValueError("4QAM-NR frame planning is not implemented")
    if (
        parameters.qam_mode is None
        or parameters.fec_rate_index is None
        or parameters.interleaver_mode is None
    ):
        raise ValueError("system-information index does not describe a payload mode")

    profile = dtmb_ldpc_profile(parameters.fec_rate_index)
    qam = QAM_DEFINITIONS[parameters.qam_mode]
    interleaver = SYMBOL_INTERLEAVERS[parameters.interleaver_mode]
    coded_bits_per_frame = DATA_SYMBOLS_PER_FRAME * qam.bits_per_symbol
    codewords_per_frame = Fraction(coded_bits_per_frame, profile.codeword_bits)
    bch_blocks_per_frame = (
        codewords_per_frame * profile.bch_blocks_per_codeword
    )
    transport_bits_per_frame = bch_blocks_per_frame * 752
    latency_symbols = interleaver.full_stream_latency_symbols
    return DtmbFramePlan(
        system_info_index=int(system_info_index),
        frame_body_mode=frame_body_mode,
        qam_mode=parameters.qam_mode,
        fec_rate_index=profile.fec_rate_index,
        interleaver_mode=parameters.interleaver_mode,
        receiver_supported=parameters.supported_by_receiver,
        bits_per_symbol=qam.bits_per_symbol,
        data_symbols_per_frame=DATA_SYMBOLS_PER_FRAME,
        system_info_symbols=SYSTEM_INFO_SYMBOLS,
        frame_body_symbols=FRAME_BODY_SYMBOLS,
        coded_bits_per_frame=coded_bits_per_frame,
        ldpc_codeword_bits=profile.codeword_bits,
        ldpc_full_codeword_bits=profile.full_codeword_bits,
        ldpc_message_bits=profile.message_bits,
        ldpc_parity_bits=profile.parity_bits,
        bch_blocks_per_codeword=profile.bch_blocks_per_codeword,
        transport_bytes_per_codeword=profile.bch_blocks_per_codeword * 94,
        codewords_per_frame=codewords_per_frame,
        bch_blocks_per_frame=bch_blocks_per_frame,
        transport_bits_per_frame=transport_bits_per_frame,
        interleaver_branch_count=interleaver.branch_count,
        interleaver_delay_step_symbols=interleaver.delay_step,
        deinterleaver_latency_symbols=latency_symbols,
        deinterleaver_latency_frames=Fraction(
            latency_symbols,
            DATA_SYMBOLS_PER_FRAME,
        ),
    )


def validate_pre_ldpc_metadata(
    metadata: dict[str, Any],
    plan: DtmbFramePlan,
) -> dict[str, Any]:
    """Compare a receiver pre-LDPC sidecar against a standard frame plan."""

    checks = [
        _check_equal(
            "system_info_index",
            _metadata_int(metadata, "chosen_system_info_vector"),
            plan.system_info_index,
        ),
        _check_equal("qam_mode", metadata.get("chosen_qam"), plan.qam_mode),
        _check_equal(
            "fec_rate_index",
            _metadata_int(metadata, "chosen_fec_rate"),
            plan.fec_rate_index,
        ),
        _check_equal(
            "interleaver_mode",
            metadata.get("chosen_interleaver_mode"),
            plan.interleaver_mode,
        ),
        _check_equal(
            "bits_per_symbol",
            _metadata_int(metadata, "bits_per_symbol"),
            plan.bits_per_symbol,
        ),
        _check_equal(
            "bits_per_frame",
            _metadata_int(metadata, "bits_per_frame"),
            plan.coded_bits_per_frame,
        ),
        _check_equal(
            "codeword_bits",
            _metadata_int(metadata, "codeword_bits"),
            plan.ldpc_codeword_bits,
        ),
        _check_equal(
            "symbol_deinterleave_latency_symbols",
            _metadata_int(metadata, "symbol_deinterleave_latency_symbols"),
            plan.deinterleaver_latency_symbols,
        ),
    ]

    before_symbols = _metadata_int(metadata, "data_symbols_before_symbol_deinterleave")
    after_symbols = _metadata_int(metadata, "data_symbols_after_symbol_deinterleave")
    discarded = _metadata_int(metadata, "symbol_deinterleave_discarded_symbols")
    if (
        before_symbols is not None
        and after_symbols is not None
        and discarded is not None
    ):
        checks.append(
            _check_equal(
                "after_symbols_equal_before_minus_discarded",
                after_symbols,
                max(0, before_symbols - discarded),
            )
        )

    hard_bits = _metadata_int(metadata, "hard_bit_count")
    if after_symbols is not None:
        checks.append(
            _check_equal(
                "hard_bits_from_after_symbols",
                hard_bits,
                after_symbols * plan.bits_per_symbol,
            )
        )
    codewords = _metadata_int(metadata, "codewords")
    expected_codewords = None
    complete_frame_count = None
    if after_symbols is not None and after_symbols % plan.data_symbols_per_frame == 0:
        complete_frame_count = after_symbols // plan.data_symbols_per_frame
        expected_codewords_fraction = complete_frame_count * plan.codewords_per_frame
        if expected_codewords_fraction.denominator == 1:
            expected_codewords = int(expected_codewords_fraction)
            checks.append(
                _check_equal(
                    "codewords_from_complete_after_frames",
                    codewords,
                    expected_codewords,
                )
            )

    return {
        "all_pass": all(check["pass"] for check in checks),
        "checks": checks,
        "derived": {
            "complete_frames_after_deinterleave": complete_frame_count,
            "expected_codewords_after_deinterleave": expected_codewords,
            "expected_codewords_per_frame": _fraction_to_dict(
                plan.codewords_per_frame
            ),
        },
    }


def _fraction_to_dict(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "value": float(value),
        "integer": value.denominator == 1,
    }


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_equal(name: str, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "pass": observed == expected,
        "observed": observed,
        "expected": expected,
    }
