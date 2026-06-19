"""Capture metadata loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class MetadataError(ValueError):
    """Raised when capture metadata is missing or inconsistent."""


@dataclass(frozen=True)
class SampleTimeBounds:
    """UTC Unix-ns bounds for one sample index."""

    sample_index: int
    earliest_unix_ns: int
    latest_unix_ns: int

    @property
    def uncertainty_s(self) -> float:
        return (self.latest_unix_ns - self.earliest_unix_ns) / 1_000_000_000.0


@dataclass(frozen=True)
class CaptureMetadata:
    """Metadata needed to interpret a raw IQ capture."""

    path: Path
    created_utc: str | None
    format: str
    frequency_hz: int | None
    sample_rate_sps: int
    bandwidth_hz: int | None
    duration_s: float | None
    sample_count: int | None
    extra: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CaptureMetadata":
        metadata_path = Path(path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise MetadataError(f"metadata must be a JSON object: {metadata_path}")
        return cls.from_mapping(data, metadata_path)

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any], path: str | Path = "<memory>"
    ) -> "CaptureMetadata":
        metadata_path = Path(path)
        fmt = str(data.get("format", "")).lower()
        if fmt not in {"ci8", "cs8"}:
            raise MetadataError("metadata field 'format' must be 'ci8' or 'cs8'")

        sample_rate = _required_int(data, "sample_rate_sps")
        if sample_rate <= 0:
            raise MetadataError("metadata field 'sample_rate_sps' must be positive")

        duration_s = _optional_float(data, "duration_s")
        if duration_s is not None and duration_s < 0:
            raise MetadataError("metadata field 'duration_s' must be non-negative")

        sample_count = _optional_int(data, "sample_count")
        if sample_count is not None and sample_count < 0:
            raise MetadataError("metadata field 'sample_count' must be non-negative")

        known = {
            "created_utc",
            "format",
            "frequency_hz",
            "sample_rate_sps",
            "bandwidth_hz",
            "duration_s",
            "sample_count",
        }

        return cls(
            path=metadata_path,
            created_utc=_optional_str(data, "created_utc"),
            format="ci8" if fmt == "cs8" else fmt,
            frequency_hz=_optional_int(data, "frequency_hz"),
            sample_rate_sps=sample_rate,
            bandwidth_hz=_optional_int(data, "bandwidth_hz"),
            duration_s=duration_s,
            sample_count=sample_count,
            extra={key: value for key, value in data.items() if key not in known},
        )

    def sample_time_bounds_utc_unix_ns(self, sample_index: int) -> SampleTimeBounds:
        """Return UTC Unix-ns bounds for a zero-based sample index."""

        return sample_time_bounds_utc_unix_ns(self, sample_index)


def find_sidecar(capture_path: str | Path) -> Path | None:
    """Return the preferred metadata sidecar path if it exists."""

    path = Path(capture_path)
    candidates = [
        path.with_name(path.name + ".json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sample_time_bounds_utc_unix_ns(
    metadata: CaptureMetadata | dict[str, Any], sample_index: int
) -> SampleTimeBounds:
    """Return conservative UTC Unix-ns bounds for one zero-based sample index.

    HackRF CI8 files do not carry hardware time-of-day timestamps, so this
    helper propagates the sidecar's explicit timing bounds instead of inventing
    a single absolute timestamp. Those bounds may come from a host-process
    bracket or from an externally timed hardware trigger.
    """

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise MetadataError("sample_index must be an integer")
    if sample_index < 0:
        raise MetadataError("sample_index must be non-negative")

    if isinstance(metadata, CaptureMetadata):
        sample_rate = metadata.sample_rate_sps
        sample_count = metadata.sample_count
        bounds = metadata.extra.get("sample_time_bounds_utc_unix_ns")
    else:
        sample_rate = _required_int(metadata, "sample_rate_sps")
        sample_count = _optional_int(metadata, "sample_count")
        bounds = metadata.get("sample_time_bounds_utc_unix_ns")

    if sample_count is not None and sample_index >= sample_count:
        raise MetadataError("sample_index is outside metadata sample_count")
    if not isinstance(bounds, dict):
        raise MetadataError("metadata field 'sample_time_bounds_utc_unix_ns' is required")

    first_earliest = _required_int(bounds, "first_sample_earliest")
    first_latest = _required_int(bounds, "first_sample_latest")
    if first_latest < first_earliest:
        raise MetadataError("first sample latest bound is earlier than earliest bound")

    offset_ns = int(round(sample_index * 1_000_000_000 / sample_rate))
    return SampleTimeBounds(
        sample_index=sample_index,
        earliest_unix_ns=first_earliest + offset_ns,
        latest_unix_ns=first_latest + offset_ns,
    )


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if value is None:
        raise MetadataError(f"metadata field '{key}' is required")
    return _coerce_int(value, key)


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _coerce_int(value, key)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"metadata field '{key}' must be numeric") from exc


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise MetadataError(f"metadata field '{key}' must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"metadata field '{key}' must be an integer") from exc
    if integer != value and not isinstance(value, str):
        raise MetadataError(f"metadata field '{key}' must be an integer")
    return integer
