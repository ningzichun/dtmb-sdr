import json

from dtmb.ts import (
    TS_NULL_PID,
    TS_SYNC_BYTE,
    analyze_ts_packets,
    extract_ts_packets,
    find_ts_sync,
    main,
    parse_ts_packet_header,
    select_ts_lock,
)


def test_find_ts_sync_detects_188_byte_alignment():
    packet = bytes([TS_SYNC_BYTE]) + bytes(range(1, 188))
    data = b"noise" + packet * 6

    locks = find_ts_sync(data)

    assert locks[0].packet_size == 188
    assert locks[0].offset == 5
    assert locks[0].sync_count == 6
    assert locks[0].sync_ratio == 1.0


def test_extract_ts_packets_uses_detected_lock():
    packet = bytes([TS_SYNC_BYTE]) + bytes([0xAA]) * 187
    data = b"\x00\x00" + packet * 3 + b"tail"
    lock = find_ts_sync(data, packet_sizes=(188,), min_packets=3)[0]

    packets = extract_ts_packets(data, lock)

    assert len(packets) == 3
    assert all(packet[0] == TS_SYNC_BYTE for packet in packets)
    assert all(len(packet) == 188 for packet in packets)


def test_parse_ts_packet_header_decodes_pid_and_flags():
    packet = _ts_packet(pid=0x0123, continuity_counter=5, unit_start=True)

    header = parse_ts_packet_header(packet)

    assert header.pid == 0x0123
    assert header.payload_unit_start_indicator is True
    assert header.adaptation_field_control == 1
    assert header.has_payload is True
    assert header.has_adaptation_field is False
    assert header.continuity_counter == 5


def test_analyze_ts_packets_reports_pid_structure_and_clean_continuity():
    packets = [
        _ts_packet(pid=0x0000, continuity_counter=0, unit_start=True),
        _ts_packet(pid=0x0100, continuity_counter=0, unit_start=True),
        _ts_packet(pid=0x0100, continuity_counter=1),
        _ts_packet(pid=0x0101, continuity_counter=0, unit_start=True),
        _ts_packet(pid=TS_NULL_PID, continuity_counter=9),
    ]

    summary = analyze_ts_packets(packets)
    pids = {item.pid: item for item in summary.top_pids}

    assert summary.packet_count == 5
    assert summary.parsed_packet_count == 5
    assert summary.valid_packet_count == 5
    assert summary.pid_count == 4
    assert summary.null_packet_count == 1
    assert summary.continuity_error_count == 0
    assert summary.invalid_adaptation_control_count == 0
    assert pids[0x0100].packet_count == 2
    assert pids[0x0100].first_continuity_counter == 0
    assert pids[0x0100].last_continuity_counter == 1


def test_select_ts_lock_rejects_sync_only_invalid_packets():
    packet = bytes([TS_SYNC_BYTE]) + bytes(range(1, 188))
    data = packet * 6

    candidate = select_ts_lock(data, min_packets=3)

    assert candidate is None


def test_select_ts_lock_uses_packet_header_validity():
    data = b"noise" + b"".join(
        _ts_packet(pid=0x0100, continuity_counter=index, unit_start=index == 0)
        for index in range(6)
    )

    candidate = select_ts_lock(data, min_packets=3)

    assert candidate is not None
    assert candidate.lock.offset == 5
    assert candidate.lock.sync_ratio == 1.0
    assert candidate.valid_packet_ratio == 1.0


def test_analyze_ts_packets_reports_continuity_counter_jump():
    packets = [
        _ts_packet(pid=0x0100, continuity_counter=0),
        _ts_packet(pid=0x0100, continuity_counter=2),
    ]

    summary = analyze_ts_packets(packets)

    assert summary.continuity_error_count == 1
    assert summary.continuity_errors[0].pid == 0x0100
    assert summary.continuity_errors[0].expected_continuity_counter == 1
    assert summary.continuity_errors[0].actual_continuity_counter == 2


def test_analyze_ts_packets_uses_discontinuity_indicator_as_reset():
    packets = [
        _ts_packet(pid=0x0100, continuity_counter=0),
        _ts_packet(
            pid=0x0100,
            continuity_counter=9,
            adaptation=True,
            discontinuity=True,
        ),
        _ts_packet(pid=0x0100, continuity_counter=10),
    ]

    summary = analyze_ts_packets(packets)

    assert summary.discontinuity_indicator_count == 1
    assert summary.continuity_error_count == 0


def test_analyze_ts_packets_uses_adaptation_only_discontinuity_as_reset():
    packets = [
        _ts_packet(pid=0x0100, continuity_counter=0),
        _ts_packet(
            pid=0x0100,
            continuity_counter=9,
            adaptation=True,
            discontinuity=True,
            adaptation_only=True,
        ),
        _ts_packet(pid=0x0100, continuity_counter=10),
    ]

    summary = analyze_ts_packets(packets)

    assert summary.discontinuity_indicator_count == 1
    assert summary.continuity_error_count == 0


def test_ts_analyze_cli_reports_stream_json(tmp_path, capsys):
    stream_path = tmp_path / "fixture.ts"
    stream_path.write_bytes(
        b"".join(
            _ts_packet(pid=0x0100, continuity_counter=index, unit_start=index == 0)
            for index in range(5)
        )
    )

    result = main([str(stream_path), "--json"])

    diagnostics = json.loads(capsys.readouterr().out)
    assert result == 0
    assert diagnostics["lock"]["packet_size"] == 188
    assert diagnostics["stream"]["valid_packet_count"] == 5
    assert diagnostics["stream"]["continuity_error_count"] == 0


def _ts_packet(
    *,
    pid: int,
    continuity_counter: int,
    unit_start: bool = False,
    adaptation: bool = False,
    discontinuity: bool = False,
    adaptation_only: bool = False,
) -> bytes:
    adaptation_field = b""
    adaptation_control = 1
    if adaptation:
        adaptation_control = 2 if adaptation_only else 3
        adaptation_size = 183 if adaptation_only else 1
        adaptation_field = bytes((adaptation_size, 0x80 if discontinuity else 0x00))
        if adaptation_only:
            adaptation_field += bytes((0xFF,)) * (adaptation_size - 1)
    header = bytes(
        (
            TS_SYNC_BYTE,
            (0x40 if unit_start else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            (adaptation_control << 4) | (continuity_counter & 0x0F),
        )
    )
    payload_len = 188 - len(header) - len(adaptation_field)
    payload = bytes((pid + continuity_counter + index) & 0xFF for index in range(payload_len))
    return header + adaptation_field + payload
