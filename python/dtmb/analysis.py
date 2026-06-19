"""Offline capture diagnostics for early DTMB development."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture_metadata import CaptureMetadata, find_sidecar
from .ci8 import inspect_ci8_file, read_ci8


@dataclass(frozen=True)
class SpectrumSummary:
    """Compact spectrum diagnostics."""

    fft_size: int
    peak_frequency_hz: float
    peak_db_over_mean: float
    occupied_bandwidth_99_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fft_size": self.fft_size,
            "peak_frequency_hz": self.peak_frequency_hz,
            "peak_db_over_mean": self.peak_db_over_mean,
            "occupied_bandwidth_99_hz": self.occupied_bandwidth_99_hz,
        }


def analyze_ci8_capture(
    capture_path: str | Path,
    *,
    sample_rate_sps: int | None = None,
    metadata_path: str | Path | None = None,
    max_samples: int = 2_000_000,
    fft_size: int = 16_384,
) -> dict[str, Any]:
    """Analyze a CI8 capture and return JSON-serializable diagnostics."""

    path = Path(capture_path)
    metadata = _load_metadata(path, metadata_path)
    resolved_sample_rate = _resolve_sample_rate(sample_rate_sps, metadata)
    if resolved_sample_rate <= 0:
        raise ValueError("sample rate must be positive")

    info = inspect_ci8_file(path)
    samples = read_ci8(path, max_samples=max_samples)
    if samples.size == 0:
        raise ValueError(f"capture contains no complete CI8 samples: {path}")

    i_values = samples.real
    q_values = samples.imag
    component_power = (i_values * i_values + q_values * q_values) / 2.0
    component_rms = float(np.sqrt(np.mean(component_power)))
    complex_power = float(np.mean(np.abs(samples) ** 2))

    diagnostics: dict[str, Any] = {
        "capture_path": str(path),
        "metadata_path": str(metadata.path) if metadata else None,
        "format": metadata.format if metadata else "ci8",
        "sample_rate_sps": resolved_sample_rate,
        "file": {
            "byte_count": info.byte_count,
            "sample_count": info.sample_count,
            "duration_s": info.sample_count / resolved_sample_rate,
            "has_trailing_byte": info.has_trailing_byte,
        },
        "analyzed": {
            "sample_count": int(samples.size),
            "duration_s": samples.size / resolved_sample_rate,
        },
        "iq": {
            "dc_i": float(np.mean(i_values)),
            "dc_q": float(np.mean(q_values)),
            "rms_i": float(np.sqrt(np.mean(i_values * i_values))),
            "rms_q": float(np.sqrt(np.mean(q_values * q_values))),
            "component_rms": component_rms,
            "component_rms_dbfs": _amplitude_db(component_rms),
            "complex_power": complex_power,
            "complex_power_db": _power_db(complex_power),
            "clip_count_i": int(np.count_nonzero((i_values <= -1.0) | (i_values >= 127.0 / 128.0))),
            "clip_count_q": int(np.count_nonzero((q_values <= -1.0) | (q_values >= 127.0 / 128.0))),
        },
    }

    spectrum = summarize_spectrum(
        samples,
        sample_rate_sps=resolved_sample_rate,
        fft_size=min(fft_size, samples.size),
    )
    if spectrum is not None:
        diagnostics["spectrum"] = spectrum.to_dict()

    if metadata:
        diagnostics["metadata"] = {
            "created_utc": metadata.created_utc,
            "frequency_hz": metadata.frequency_hz,
            "bandwidth_hz": metadata.bandwidth_hz,
            "duration_s": metadata.duration_s,
            "sample_count": metadata.sample_count,
            "extra": metadata.extra,
        }

    return diagnostics


def summarize_spectrum(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    fft_size: int,
) -> SpectrumSummary | None:
    """Return coarse single-segment spectrum diagnostics."""

    if fft_size < 16:
        return None

    segment = samples[:fft_size]
    segment = segment - np.mean(segment)
    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(segment * window))
    power = np.abs(spectrum) ** 2
    mean_power = float(np.mean(power))
    peak_index = int(np.argmax(power))
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_sps))

    cumulative = np.cumsum(power)
    total = float(cumulative[-1])
    if total <= 0.0:
        occupied_bandwidth = 0.0
    else:
        low_index = int(np.searchsorted(cumulative, total * 0.005))
        high_index = int(np.searchsorted(cumulative, total * 0.995))
        occupied_bandwidth = float(frequencies[min(high_index, fft_size - 1)] - frequencies[low_index])

    return SpectrumSummary(
        fft_size=fft_size,
        peak_frequency_hz=float(frequencies[peak_index]),
        peak_db_over_mean=_power_db(float(power[peak_index]) / mean_power) if mean_power > 0 else float("-inf"),
        occupied_bandwidth_99_hz=occupied_bandwidth,
    )


def compute_spectrum_trace(
    samples: np.ndarray,
    *,
    sample_rate_sps: int,
    fft_size: int = 16_384,
    point_count: int = 640,
    segment_count: int = 1,
) -> dict[str, Any]:
    """Return a decimated FFT trace suitable for UI and artifact plots."""

    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    signal = np.asarray(samples, dtype=np.complex64)
    if signal.size < 16:
        return {
            "fft_size": int(signal.size),
            "segment_count": 0,
            "frequency_hz": [],
            "power_db": [],
            "peak_frequency_hz": None,
            "peak_power_db": None,
            "median_power_db": None,
        }

    size = min(int(fft_size), int(signal.size))
    segments = _spectrum_segments(signal, fft_size=size, segment_count=segment_count)
    window = np.hanning(size).astype(np.float32)
    window_power = max(float(np.sum(window * window)), 1e-12)
    accumulated_power = np.zeros(size, dtype=np.float64)
    for segment in segments:
        centered = segment - np.mean(segment, dtype=np.complex128)
        spectrum = np.fft.fftshift(np.fft.fft(centered * window))
        accumulated_power += (np.abs(spectrum) ** 2) / window_power
    power = accumulated_power / max(1, len(segments))
    power_db = 10.0 * np.log10(power + 1e-18)
    frequencies = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / sample_rate_sps))

    peak_index = int(np.argmax(power_db))
    frequencies_out, power_out = _decimate_trace(
        frequencies,
        power_db,
        point_count=max(8, int(point_count)),
    )
    return {
        "fft_size": size,
        "segment_count": int(len(segments)),
        "frequency_hz": [round(float(value), 1) for value in frequencies_out],
        "power_db": [round(float(value), 2) for value in power_out],
        "peak_frequency_hz": float(frequencies[peak_index]),
        "peak_power_db": float(power_db[peak_index]),
        "median_power_db": float(np.median(power_db)),
        "span_hz": float(sample_rate_sps),
    }


def _spectrum_segments(
    signal: np.ndarray,
    *,
    fft_size: int,
    segment_count: int,
) -> list[np.ndarray]:
    if signal.size <= fft_size:
        return [signal[:fft_size]]
    available = signal.size // fft_size
    count = min(int(segment_count), int(available))
    if count <= 1:
        return [signal[:fft_size]]
    max_start = signal.size - fft_size
    starts = np.linspace(0, max_start, count, dtype=np.int64)
    return [signal[int(start) : int(start) + fft_size] for start in starts]


def _decimate_trace(
    frequencies: np.ndarray,
    power_db: np.ndarray,
    *,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frequencies.size <= point_count:
        return frequencies, power_db
    group = max(1, frequencies.size // point_count)
    trimmed = group * point_count
    start = (frequencies.size - trimmed) // 2
    freq_view = frequencies[start : start + trimmed].reshape(point_count, group)
    power_view = power_db[start : start + trimmed].reshape(point_count, group)
    return np.mean(freq_view, axis=1), np.mean(power_view, axis=1)


def _load_metadata(
    capture_path: Path, metadata_path: str | Path | None
) -> CaptureMetadata | None:
    if metadata_path is not None:
        return CaptureMetadata.from_file(metadata_path)
    sidecar = find_sidecar(capture_path)
    if sidecar is None:
        return None
    return CaptureMetadata.from_file(sidecar)


def _resolve_sample_rate(
    sample_rate_sps: int | None, metadata: CaptureMetadata | None
) -> int:
    if sample_rate_sps is not None:
        return int(sample_rate_sps)
    if metadata is not None:
        return metadata.sample_rate_sps
    raise ValueError("sample rate is required when no metadata sidecar is present")


def _amplitude_db(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def _power_db(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(value))
