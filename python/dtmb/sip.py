"""Received DTMB second-frame initialization packets (GB/T 28434-2012).

Section 5.3.1.2, Figure 5 fixes the complete TS header to 47 40 15 10.
Tables 1/2 define the modes and packet rates; bytes 20--188 are FF stuffing.
These rules recognize received control packets; they never generate or repair
receiver payload. Reserved SI bits have no defined value and are retained.
"""

from __future__ import annotations

from dataclasses import dataclass


DTMB_SIP_PID = 0x0015


@dataclass(frozen=True)
class DtmbSip:
    system_information: int
    pn_symbols: int
    carriers: int
    mapping: str
    fec_rate: str
    interleaver_length: int
    dual_pilot: bool
    pn_phase_rotates: bool
    maximum_delay_100ns: int
    transmitter_address: int
    independent_delay_100ns: int
    frequency_offset_hz: int
    power_enabled: bool
    power_attenuation_tenths_db: int
    packets_per_second: int


def parse_dtmb_sip_packet(packet: bytes | bytearray | memoryview) -> DtmbSip | None:
    """Recognize only a complete SIP with the defined fields and stuffing."""
    if len(packet) != 188 or packet[:4] != b"\x47\x40\x15\x10":
        return None
    pn_mode = packet[4] >> 6
    mapping = (packet[4] >> 2) & 7
    rate = packet[4] & 3
    if pn_mode == 3 or mapping > 4 or rate == 3:
        return None
    if mapping in (0, 3) and rate != 2:
        return None
    maximum_delay = int.from_bytes(packet[6:9], "big")
    if maximum_delay > 9_999_999 or packet[19:] != b"\xff" * 169:
        return None
    packets_per_frame = (
        2 if mapping == 0 else 10 if mapping == 3 else
        {1: (2, 3, 4), 2: (4, 6, 8), 4: (6, 9, 12)}[mapping][rate]
    )
    power = int.from_bytes(packet[17:19], "big")
    return DtmbSip(
        system_information=int.from_bytes(packet[4:6], "big"),
        pn_symbols=(420, 595, 945)[pn_mode],
        carriers=3780 if packet[4] & 0x20 else 1,
        mapping=("4qam-nr", "4qam", "16qam", "32qam", "64qam")[mapping],
        fec_rate=("0.4", "0.6", "0.8")[rate],
        interleaver_length=720 if packet[5] & 0x80 else 240,
        dual_pilot=bool(packet[5] & 0x40),
        pn_phase_rotates=bool(packet[5] & 0x20),
        maximum_delay_100ns=maximum_delay,
        transmitter_address=int.from_bytes(packet[9:11], "big"),
        independent_delay_100ns=int.from_bytes(packet[11:14], "big"),
        frequency_offset_hz=int.from_bytes(packet[14:17], "big", signed=True),
        power_enabled=bool(power & 0x8000),
        power_attenuation_tenths_db=power & 0x7FFF,
        packets_per_second=packets_per_frame * (1800, 1728, 1600)[pn_mode],
    )
