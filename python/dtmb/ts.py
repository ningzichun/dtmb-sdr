"""MPEG transport-stream packet alignment and stream diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


TS_SYNC_BYTE = 0x47
TS_PACKET_SIZE = 188
TS_RS_PACKET_SIZE = 204
TS_NULL_PID = 0x1FFF


@dataclass(frozen=True)
class TsSyncLock:
    """Candidate MPEG-TS packet alignment."""

    packet_size: int
    offset: int
    packet_count: int
    sync_count: int
    sync_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_size": self.packet_size,
            "offset": self.offset,
            "packet_count": self.packet_count,
            "sync_count": self.sync_count,
            "sync_ratio": self.sync_ratio,
        }


@dataclass(frozen=True)
class TsPacketHeader:
    """Parsed MPEG-TS packet header fields."""

    transport_error_indicator: bool
    payload_unit_start_indicator: bool
    transport_priority: bool
    pid: int
    scrambling_control: int
    adaptation_field_control: int
    continuity_counter: int

    @property
    def has_adaptation_field(self) -> bool:
        return self.adaptation_field_control in (2, 3)

    @property
    def has_payload(self) -> bool:
        return self.adaptation_field_control in (1, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_error_indicator": self.transport_error_indicator,
            "payload_unit_start_indicator": self.payload_unit_start_indicator,
            "transport_priority": self.transport_priority,
            "pid": self.pid,
            "pid_hex": f"0x{self.pid:04x}",
            "scrambling_control": self.scrambling_control,
            "adaptation_field_control": self.adaptation_field_control,
            "has_adaptation_field": self.has_adaptation_field,
            "has_payload": self.has_payload,
            "continuity_counter": self.continuity_counter,
        }


@dataclass(frozen=True)
class TsContinuityError:
    """One continuity-counter mismatch in a PID stream."""

    packet_index: int
    pid: int
    expected_continuity_counter: int
    actual_continuity_counter: int
    previous_continuity_counter: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_index": self.packet_index,
            "pid": self.pid,
            "pid_hex": f"0x{self.pid:04x}",
            "expected_continuity_counter": self.expected_continuity_counter,
            "actual_continuity_counter": self.actual_continuity_counter,
            "previous_continuity_counter": self.previous_continuity_counter,
        }


@dataclass(frozen=True)
class TsPidSummary:
    """Per-PID stream statistics."""

    pid: int
    packet_count: int
    payload_packet_count: int
    unit_start_count: int
    transport_error_count: int
    scrambled_packet_count: int
    invalid_adaptation_control_count: int
    invalid_adaptation_field_count: int
    continuity_error_count: int
    continuity_duplicate_count: int
    discontinuity_indicator_count: int
    first_continuity_counter: int | None
    last_continuity_counter: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pid_hex": f"0x{self.pid:04x}",
            "packet_count": self.packet_count,
            "payload_packet_count": self.payload_packet_count,
            "unit_start_count": self.unit_start_count,
            "transport_error_count": self.transport_error_count,
            "scrambled_packet_count": self.scrambled_packet_count,
            "invalid_adaptation_control_count": self.invalid_adaptation_control_count,
            "invalid_adaptation_field_count": self.invalid_adaptation_field_count,
            "continuity_error_count": self.continuity_error_count,
            "continuity_duplicate_count": self.continuity_duplicate_count,
            "discontinuity_indicator_count": self.discontinuity_indicator_count,
            "first_continuity_counter": self.first_continuity_counter,
            "last_continuity_counter": self.last_continuity_counter,
        }


@dataclass(frozen=True)
class TsStreamSummary:
    """MPEG-TS packet stream diagnostics."""

    packet_count: int
    parsed_packet_count: int
    valid_packet_count: int
    short_packet_count: int
    sync_error_count: int
    transport_error_count: int
    scrambled_packet_count: int
    payload_packet_count: int
    unit_start_count: int
    null_packet_count: int
    invalid_adaptation_control_count: int
    invalid_adaptation_field_count: int
    continuity_error_count: int
    continuity_duplicate_count: int
    discontinuity_indicator_count: int
    pid_count: int
    top_pids: tuple[TsPidSummary, ...]
    continuity_errors: tuple[TsContinuityError, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_count": self.packet_count,
            "parsed_packet_count": self.parsed_packet_count,
            "valid_packet_count": self.valid_packet_count,
            "short_packet_count": self.short_packet_count,
            "sync_error_count": self.sync_error_count,
            "transport_error_count": self.transport_error_count,
            "scrambled_packet_count": self.scrambled_packet_count,
            "payload_packet_count": self.payload_packet_count,
            "unit_start_count": self.unit_start_count,
            "null_packet_count": self.null_packet_count,
            "invalid_adaptation_control_count": self.invalid_adaptation_control_count,
            "invalid_adaptation_field_count": self.invalid_adaptation_field_count,
            "continuity_error_count": self.continuity_error_count,
            "continuity_duplicate_count": self.continuity_duplicate_count,
            "discontinuity_indicator_count": self.discontinuity_indicator_count,
            "pid_count": self.pid_count,
            "top_pids": [summary.to_dict() for summary in self.top_pids],
            "continuity_errors": [error.to_dict() for error in self.continuity_errors],
        }


@dataclass(frozen=True)
class TsLockCandidate:
    """A sync-byte lock with parsed stream diagnostics."""

    lock: TsSyncLock
    stream: TsStreamSummary
    valid_packet_ratio: float

    def to_dict(self) -> dict[str, Any]:
        data = self.lock.to_dict()
        data["valid_packet_ratio"] = self.valid_packet_ratio
        data["stream"] = self.stream.to_dict()
        return data


def find_ts_sync(
    data: bytes | bytearray | memoryview,
    *,
    packet_sizes: Iterable[int] = (TS_PACKET_SIZE, TS_RS_PACKET_SIZE),
    min_packets: int = 5,
) -> list[TsSyncLock]:
    """Return possible MPEG-TS sync alignments sorted by sync quality."""

    buffer = memoryview(data)
    if min_packets <= 0:
        raise ValueError("min_packets must be positive")

    locks: list[TsSyncLock] = []
    for packet_size in packet_sizes:
        if packet_size <= 0:
            raise ValueError("packet sizes must be positive")
        if len(buffer) < packet_size * min_packets:
            continue
        for offset in range(packet_size):
            packet_count = 1 + (len(buffer) - offset - 1) // packet_size
            if packet_count < min_packets:
                continue
            sync_count = 0
            for index in range(packet_count):
                if buffer[offset + index * packet_size] == TS_SYNC_BYTE:
                    sync_count += 1
            locks.append(
                TsSyncLock(
                    packet_size=packet_size,
                    offset=offset,
                    packet_count=packet_count,
                    sync_count=sync_count,
                    sync_ratio=sync_count / packet_count,
                )
            )
    return sorted(
        locks,
        key=lambda lock: (lock.sync_ratio, lock.sync_count, -lock.offset),
        reverse=True,
    )


def extract_ts_packets(
    data: bytes | bytearray | memoryview,
    lock: TsSyncLock,
) -> list[bytes]:
    """Extract complete packets using a previously detected sync lock."""

    buffer = memoryview(data)
    packets: list[bytes] = []
    for start in range(lock.offset, len(buffer) - lock.packet_size + 1, lock.packet_size):
        packets.append(bytes(buffer[start : start + lock.packet_size]))
    return packets


def parse_ts_packet_header(packet: bytes | bytearray | memoryview) -> TsPacketHeader:
    """Parse the fixed MPEG-TS header from a 188- or 204-byte packet."""

    buffer = memoryview(packet)
    if len(buffer) < TS_PACKET_SIZE:
        raise ValueError("MPEG-TS packet must contain at least 188 bytes")
    if buffer[0] != TS_SYNC_BYTE:
        raise ValueError("MPEG-TS sync byte missing")
    second = int(buffer[1])
    third = int(buffer[2])
    fourth = int(buffer[3])
    return TsPacketHeader(
        transport_error_indicator=bool(second & 0x80),
        payload_unit_start_indicator=bool(second & 0x40),
        transport_priority=bool(second & 0x20),
        pid=((second & 0x1F) << 8) | third,
        scrambling_control=(fourth >> 6) & 0x03,
        adaptation_field_control=(fourth >> 4) & 0x03,
        continuity_counter=fourth & 0x0F,
    )


def analyze_ts_packets(
    packets: Iterable[bytes | bytearray | memoryview],
    *,
    top_pids: int | None = 16,
    max_continuity_errors: int = 16,
) -> TsStreamSummary:
    """Summarize packet headers, PID structure, and continuity counters.

    Continuity checks are intentionally conservative: they apply to non-null PID
    packets that contain payload bytes. Adaptation-only packets do not advance
    the expected payload continuity counter, and packets with the discontinuity
    indicator reset the continuity check for that PID.
    """

    if top_pids is not None and top_pids < 0:
        raise ValueError("top_pids must be non-negative or None")
    if max_continuity_errors < 0:
        raise ValueError("max_continuity_errors must be non-negative")

    packet_count = 0
    parsed_packet_count = 0
    valid_packet_count = 0
    short_packet_count = 0
    sync_error_count = 0
    transport_error_count = 0
    scrambled_packet_count = 0
    payload_packet_count = 0
    unit_start_count = 0
    null_packet_count = 0
    invalid_adaptation_control_count = 0
    invalid_adaptation_field_count = 0
    continuity_error_count = 0
    continuity_duplicate_count = 0
    discontinuity_indicator_count = 0

    pid_counts: Counter[int] = Counter()
    pid_stats: dict[int, dict[str, Any]] = {}
    last_payload_cc: dict[int, int] = {}
    continuity_errors: list[TsContinuityError] = []

    for packet_index, packet in enumerate(packets):
        packet_count += 1
        buffer = memoryview(packet)
        if len(buffer) < TS_PACKET_SIZE:
            short_packet_count += 1
            continue
        if buffer[0] != TS_SYNC_BYTE:
            sync_error_count += 1
            continue
        header = parse_ts_packet_header(buffer)
        parsed_packet_count += 1
        pid_counts[header.pid] += 1
        stats = _pid_stats(pid_stats, header.pid)
        stats["packet_count"] += 1
        stats["first_continuity_counter"] = (
            header.continuity_counter
            if stats["first_continuity_counter"] is None
            else stats["first_continuity_counter"]
        )
        stats["last_continuity_counter"] = header.continuity_counter
        if header.transport_error_indicator:
            transport_error_count += 1
            stats["transport_error_count"] += 1
        if header.scrambling_control:
            scrambled_packet_count += 1
            stats["scrambled_packet_count"] += 1
        if header.payload_unit_start_indicator:
            unit_start_count += 1
            stats["unit_start_count"] += 1
        if header.pid == TS_NULL_PID:
            null_packet_count += 1

        if header.adaptation_field_control == 0:
            invalid_adaptation_control_count += 1
            stats["invalid_adaptation_control_count"] += 1
            continue
        adaptation_valid = _adaptation_field_valid(buffer, header)
        if not adaptation_valid:
            invalid_adaptation_field_count += 1
            stats["invalid_adaptation_field_count"] += 1
            continue
        valid_packet_count += 1

        discontinuity = _adaptation_discontinuity_indicator(buffer, header)
        if discontinuity:
            discontinuity_indicator_count += 1
            stats["discontinuity_indicator_count"] += 1
            if header.pid != TS_NULL_PID:
                last_payload_cc.pop(header.pid, None)

        if not header.has_payload:
            continue
        payload_packet_count += 1
        stats["payload_packet_count"] += 1
        if header.pid == TS_NULL_PID:
            continue

        previous = last_payload_cc.get(header.pid)
        if discontinuity or previous is None:
            last_payload_cc[header.pid] = header.continuity_counter
            continue
        expected = (previous + 1) & 0x0F
        if header.continuity_counter == previous:
            continuity_duplicate_count += 1
            stats["continuity_duplicate_count"] += 1
            continue
        if header.continuity_counter != expected:
            continuity_error_count += 1
            stats["continuity_error_count"] += 1
            if len(continuity_errors) < max_continuity_errors:
                continuity_errors.append(
                    TsContinuityError(
                        packet_index=packet_index,
                        pid=header.pid,
                        expected_continuity_counter=expected,
                        actual_continuity_counter=header.continuity_counter,
                        previous_continuity_counter=previous,
                    )
                )
        last_payload_cc[header.pid] = header.continuity_counter

    summaries = tuple(
        sorted(
            (
                TsPidSummary(
                    pid=pid,
                    packet_count=int(values["packet_count"]),
                    payload_packet_count=int(values["payload_packet_count"]),
                    unit_start_count=int(values["unit_start_count"]),
                    transport_error_count=int(values["transport_error_count"]),
                    scrambled_packet_count=int(values["scrambled_packet_count"]),
                    invalid_adaptation_control_count=int(
                        values["invalid_adaptation_control_count"]
                    ),
                    invalid_adaptation_field_count=int(
                        values["invalid_adaptation_field_count"]
                    ),
                    continuity_error_count=int(values["continuity_error_count"]),
                    continuity_duplicate_count=int(values["continuity_duplicate_count"]),
                    discontinuity_indicator_count=int(
                        values["discontinuity_indicator_count"]
                    ),
                    first_continuity_counter=values["first_continuity_counter"],
                    last_continuity_counter=values["last_continuity_counter"],
                )
                for pid, values in pid_stats.items()
            ),
            key=lambda item: (-item.packet_count, item.pid),
        )
    )
    if top_pids is not None:
        summaries = summaries[:top_pids]

    return TsStreamSummary(
        packet_count=packet_count,
        parsed_packet_count=parsed_packet_count,
        valid_packet_count=valid_packet_count,
        short_packet_count=short_packet_count,
        sync_error_count=sync_error_count,
        transport_error_count=transport_error_count,
        scrambled_packet_count=scrambled_packet_count,
        payload_packet_count=payload_packet_count,
        unit_start_count=unit_start_count,
        null_packet_count=null_packet_count,
        invalid_adaptation_control_count=invalid_adaptation_control_count,
        invalid_adaptation_field_count=invalid_adaptation_field_count,
        continuity_error_count=continuity_error_count,
        continuity_duplicate_count=continuity_duplicate_count,
        discontinuity_indicator_count=discontinuity_indicator_count,
        pid_count=len(pid_counts),
        top_pids=summaries,
        continuity_errors=tuple(continuity_errors),
    )


def analyze_ts_stream(
    data: bytes | bytearray | memoryview,
    lock: TsSyncLock,
    *,
    top_pids: int | None = 16,
    max_continuity_errors: int = 16,
) -> TsStreamSummary:
    """Extract packets for a sync lock and run stream diagnostics."""

    return analyze_ts_packets(
        extract_ts_packets(data, lock),
        top_pids=top_pids,
        max_continuity_errors=max_continuity_errors,
    )


def analyze_ts_lock_candidates(
    data: bytes | bytearray | memoryview,
    *,
    packet_sizes: Iterable[int] = (TS_PACKET_SIZE, TS_RS_PACKET_SIZE),
    min_packets: int = 5,
    top_pids: int | None = 16,
    max_continuity_errors: int = 16,
) -> list[TsLockCandidate]:
    """Return sync candidates annotated with packet-header diagnostics."""

    candidates = [
        TsLockCandidate(
            lock=lock,
            stream=analyze_ts_stream(
                data,
                lock,
                top_pids=top_pids,
                max_continuity_errors=max_continuity_errors,
            ),
            valid_packet_ratio=0.0,
        )
        for lock in find_ts_sync(
            data,
            packet_sizes=packet_sizes,
            min_packets=min_packets,
        )
    ]
    scored = [
        TsLockCandidate(
            lock=candidate.lock,
            stream=candidate.stream,
            valid_packet_ratio=(
                candidate.stream.valid_packet_count
                / max(1, candidate.stream.packet_count)
            ),
        )
        for candidate in candidates
    ]
    return sorted(scored, key=_lock_candidate_sort_key, reverse=True)


def select_ts_lock(
    data: bytes | bytearray | memoryview,
    *,
    packet_sizes: Iterable[int] = (TS_PACKET_SIZE, TS_RS_PACKET_SIZE),
    min_packets: int = 5,
    min_sync_ratio: float = 0.8,
    min_valid_packet_ratio: float = 0.8,
    top_pids: int | None = 16,
    max_continuity_errors: int = 16,
) -> TsLockCandidate | None:
    """Return the best MPEG-TS lock that also has valid packet headers."""

    if min_sync_ratio < 0.0 or min_sync_ratio > 1.0:
        raise ValueError("min_sync_ratio must be between 0 and 1")
    if min_valid_packet_ratio < 0.0 or min_valid_packet_ratio > 1.0:
        raise ValueError("min_valid_packet_ratio must be between 0 and 1")
    for candidate in analyze_ts_lock_candidates(
        data,
        packet_sizes=packet_sizes,
        min_packets=min_packets,
        top_pids=top_pids,
        max_continuity_errors=max_continuity_errors,
    ):
        if (
            candidate.lock.sync_ratio >= min_sync_ratio
            and candidate.valid_packet_ratio >= min_valid_packet_ratio
        ):
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-ts-analyze",
        description="Analyze MPEG-TS sync, packet headers, PIDs, and continuity counters.",
    )
    parser.add_argument("stream", type=Path, help="Path to a recovered MPEG-TS byte stream")
    parser.add_argument(
        "--packet-size",
        dest="packet_sizes",
        action="append",
        type=_positive_int,
        help="Packet size to try; repeat to try multiple sizes. Defaults to 188 and 204.",
    )
    parser.add_argument("--min-packets", type=_positive_int, default=5)
    parser.add_argument("--min-sync-ratio", type=_ratio, default=0.8)
    parser.add_argument("--min-valid-ratio", type=_ratio, default=0.8)
    parser.add_argument("--top-pids", type=_non_negative_int, default=16)
    parser.add_argument("--max-continuity-errors", type=_non_negative_int, default=16)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = args.stream.read_bytes()
    candidates = analyze_ts_lock_candidates(
        data,
        packet_sizes=args.packet_sizes or (TS_PACKET_SIZE, TS_RS_PACKET_SIZE),
        min_packets=args.min_packets,
        top_pids=args.top_pids,
        max_continuity_errors=args.max_continuity_errors,
    )
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.lock.sync_ratio >= args.min_sync_ratio
            and candidate.valid_packet_ratio >= args.min_valid_ratio
        ),
        None,
    )
    selected_lock = selected.lock if selected is not None else None
    stream = selected.stream if selected is not None else None
    candidate_locks = [
        candidate.to_dict()
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.lock.sync_ratio,
                item.lock.sync_count,
                item.valid_packet_ratio,
                item.stream.valid_packet_count,
                -item.stream.invalid_adaptation_control_count,
                -item.stream.invalid_adaptation_field_count,
                -item.lock.offset,
            ),
            reverse=True,
        )
    ]
    diagnostics = {
        "stream_path": str(args.stream),
        "bytes": len(data),
        "lock": selected_lock.to_dict() if selected_lock else None,
        "candidate_locks": candidate_locks[:8],
        "min_sync_ratio": args.min_sync_ratio,
        "min_valid_ratio": args.min_valid_ratio,
        "stream": stream.to_dict() if stream else None,
    }
    if args.json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
    else:
        print(_format_analysis(diagnostics))
    return 0 if selected_lock is not None else 2


def _format_analysis(diagnostics: dict[str, Any]) -> str:
    lock = diagnostics["lock"]
    stream = diagnostics["stream"]
    if lock is None or stream is None:
        return "MPEG-TS stream: no lock"
    lines = [
        "MPEG-TS stream: lock",
        (
            f"lock: packet_size={lock['packet_size']} offset={lock['offset']} "
            f"sync={lock['sync_count']}/{lock['packet_count']} "
            f"ratio={lock['sync_ratio']:.3f}"
        ),
        (
            f"packets: valid={stream['valid_packet_count']}/"
            f"{stream['packet_count']} pids={stream['pid_count']} "
            f"transport_errors={stream['transport_error_count']} "
            f"continuity_errors={stream['continuity_error_count']}"
        ),
    ]
    for pid in stream["top_pids"]:
        lines.append(
            f"pid {pid['pid_hex']}: packets={pid['packet_count']} "
            f"payload={pid['payload_packet_count']} cc_errors={pid['continuity_error_count']}"
        )
    return "\n".join(lines)


def _lock_candidate_sort_key(candidate: TsLockCandidate) -> tuple[float, ...]:
    stream = candidate.stream
    return (
        candidate.lock.sync_ratio,
        candidate.valid_packet_ratio,
        float(stream.valid_packet_count),
        -float(stream.transport_error_count),
        -float(stream.invalid_adaptation_control_count),
        -float(stream.invalid_adaptation_field_count),
        -float(stream.continuity_error_count),
        float(candidate.lock.sync_count),
        -float(candidate.lock.offset),
    )


def _pid_stats(pid_stats: dict[int, dict[str, Any]], pid: int) -> dict[str, Any]:
    if pid not in pid_stats:
        pid_stats[pid] = {
            "packet_count": 0,
            "payload_packet_count": 0,
            "unit_start_count": 0,
            "transport_error_count": 0,
            "scrambled_packet_count": 0,
            "invalid_adaptation_control_count": 0,
            "invalid_adaptation_field_count": 0,
            "continuity_error_count": 0,
            "continuity_duplicate_count": 0,
            "discontinuity_indicator_count": 0,
            "first_continuity_counter": None,
            "last_continuity_counter": None,
        }
    return pid_stats[pid]


def _adaptation_field_valid(
    packet: memoryview,
    header: TsPacketHeader,
) -> bool:
    if not header.has_adaptation_field:
        return True
    adaptation_length = int(packet[4])
    return 5 + adaptation_length <= TS_PACKET_SIZE


def _adaptation_discontinuity_indicator(
    packet: memoryview,
    header: TsPacketHeader,
) -> bool:
    if not header.has_adaptation_field:
        return False
    adaptation_length = int(packet[4])
    if adaptation_length < 1 or 5 + adaptation_length > TS_PACKET_SIZE:
        return False
    return bool(int(packet[5]) & 0x80)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
