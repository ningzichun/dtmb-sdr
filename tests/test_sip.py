from __future__ import annotations

import pytest

from dtmb.sip import parse_dtmb_sip_packet


def sip_packet() -> bytes:
    # GB/T 28434-2012 Figure 5/Table 2, with varying transmitter controls.
    return bytes.fromhex("47 40 15 10 b1 a0 4c 4b 40 12 34 00 00 07 ff ff f9 80 0a") + b"\xff" * 169


def test_parse_sip_recovers_defined_fields_without_payload_changes():
    packet = sip_packet()
    info = parse_dtmb_sip_packet(memoryview(packet))
    assert info is not None
    assert (info.pn_symbols, info.carriers, info.mapping, info.fec_rate) == (945, 3780, "64qam", "0.6")
    assert (info.interleaver_length, info.dual_pilot, info.pn_phase_rotates) == (720, False, True)
    assert info.maximum_delay_100ns == 5_000_000
    assert info.transmitter_address == 0x1234
    assert info.independent_delay_100ns == 7
    assert info.frequency_offset_hz == -7
    assert info.power_enabled and info.power_attenuation_tenths_db == 10
    assert info.packets_per_second == 14_400
    assert packet == sip_packet()


@pytest.mark.parametrize(("mode", "packet_rate"), [
    (0x22, 3600), (0x31, 16200), (0x71, 15552), (0xB1, 14400),
    (0xAE, 16000), (0xA9, 9600), (0xA5, 4800), (0xA2, 3200),
])
def test_sip_packet_rates_follow_standard_table_1(mode, packet_rate):
    packet = bytearray(sip_packet())
    packet[4] = mode
    info = parse_dtmb_sip_packet(packet)
    assert info is not None and info.packets_per_second == packet_rate


@pytest.mark.parametrize(("offset", "value"), [
    (0, 0x46), (1, 0), (2, 0x16), (3, 0x11), (4, 0xF1),
    (4, 0xB5), (4, 0xB3), (4, 0xA0), (6, 0xFF), (19, 0), (187, 0),
])
def test_sip_recognition_is_not_a_pid_or_header_only_exemption(offset, value):
    packet = bytearray(sip_packet())
    packet[offset] = value
    assert parse_dtmb_sip_packet(packet) is None


def test_sip_requires_exact_length_and_retains_undefined_reserved_bits():
    assert parse_dtmb_sip_packet(sip_packet()[:-1]) is None
    assert parse_dtmb_sip_packet(sip_packet() + b"\xff") is None
    packet = bytearray(sip_packet())
    packet[5] |= 0x1F
    info = parse_dtmb_sip_packet(packet)
    assert info is not None and info.system_information == 0xB1BF
