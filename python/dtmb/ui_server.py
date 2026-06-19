"""Small local web UI for HackRF DTMB capture diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fnmatch
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, unquote, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .analysis import analyze_ci8_capture, compute_spectrum_trace
from .capture import (
    DEFAULT_BANDWIDTH_HZ,
    DEFAULT_SAMPLE_RATE_SPS,
    HackrfCaptureConfig,
    run_capture,
)
from .ci8 import read_ci8
from .conditioning import frequency_shift, remove_dc, resample_to_symbol_rate
from .equalization import equalize_with_system_info_pilots
from .frame_sync import (
    DTMB_SYMBOL_RATE_SPS,
    delay_corrected_phase_offset,
    detect_pn_cyclic_extension_trains,
    detect_pn_phase_family_trains,
    estimate_cfo_from_pn_cyclic_extension,
    score_pn_family_delay_train,
    should_apply_pn_family_delay,
)
from .frames import iter_signal_frames
from .frequency import (
    frame_body_fft,
    frequency_deinterleave,
    frequency_deinterleave_inserted,
    split_system_info_and_data,
)
from .pn import PnMode
from .system_info import classify_system_info


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIR = REPO_ROOT / "apps" / "dtmb-ui" / "static"
DEFAULT_CAPTURE_DIR = REPO_ROOT / "captures" / "ui"
DEFAULT_NATIVE_MONITOR = REPO_ROOT / "build" / "core-cpp" / (
    "dtmb_core_live_monitor.exe" if sys.platform == "win32" else "dtmb_core_live_monitor"
)
DEFAULT_HK_DTMB_CSV = REPO_ROOT / "references" / "HK_DTMB.csv"
DEFAULT_MO_DTMB_CSV = REPO_ROOT / "references" / "MO_DTMB.csv"
DEFAULT_SZ_DTMB_CSV = REPO_ROOT / "references" / "SZ_DTMB.csv"
DEFAULT_NEW_OBSERVATIONS_CSV = REPO_ROOT / "references" / "NEW_OBSERVATIONS.csv"
DTMB_PRESET_SOURCES = (
    ("HK", DEFAULT_HK_DTMB_CSV),
    ("MO", DEFAULT_MO_DTMB_CSV),
    ("SZ", DEFAULT_SZ_DTMB_CSV),
    ("NEW", DEFAULT_NEW_OBSERVATIONS_CSV),
)
DEFAULT_UI_FREQUENCY_HZ = 522_000_000
VISUAL_ARTIFACT_PATTERNS = tuple(
    pattern
    for gate in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e", "gate_f", "echo")
    for pattern in (f"*.{gate}.png", f"*.{gate}.visual.json")
) + ("*.visuals.json",)


@dataclass(frozen=True)
class UiConfig:
    """Operator-facing HackRF and diagnostic settings."""

    frequency_hz: int = DEFAULT_UI_FREQUENCY_HZ
    sample_rate_sps: int = DEFAULT_SAMPLE_RATE_SPS
    bandwidth_hz: int = DEFAULT_BANDWIDTH_HZ
    device_serial: str | None = None
    amp: int = 0
    lna_gain: int = 16
    vga_gain: int = 16
    correct_iq: bool = True
    frequency_shift_hz: float = 0.0
    expected_mode: str = "pn945"
    system_info_index: int = 23
    snapshot_duration_s: float = 0.15
    analysis_max_samples: int = 1_500_000
    pn_max_symbols: int = 250_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UiState:
    """Mutable state shared by HTTP handler threads."""

    def __init__(
        self,
        *,
        config: UiConfig,
        capture_dir: Path,
        executable: str,
        native_monitor: str | Path = DEFAULT_NATIVE_MONITOR,
        command_prefix: Sequence[str] = (),
    ) -> None:
        self.lock = threading.RLock()
        self.live_condition = threading.Condition(self.lock)
        self.live_process_lock = threading.RLock()
        self.config = config
        self.capture_dir = Path(capture_dir)
        self.executable = executable
        self.native_monitor = str(native_monitor)
        self.command_prefix = tuple(command_prefix)
        self.shutting_down = False
        self.capture_in_progress = False
        self.last_snapshot: dict[str, Any] | None = None
        self.live_process: subprocess.Popen[str] | None = None
        self.live_started_utc: str | None = None
        self.live_command: list[str] | None = None
        self.live_latest: dict[str, Any] | None = None
        self.live_events: deque[dict[str, Any]] = deque(maxlen=512)
        self.live_stderr: deque[str] = deque(maxlen=80)
        self.live_sequence = 0


CaptureRunner = Callable[
    [HackrfCaptureConfig], tuple[list[str], Path | None]
]


class UiResourceBusyError(RuntimeError):
    """Raised when another UI operation owns the HackRF."""


def update_ui_config(base: UiConfig, payload: dict[str, Any]) -> UiConfig:
    """Return a validated config updated from a JSON object."""

    if not isinstance(payload, dict):
        raise ValueError("configuration payload must be a JSON object")

    frequency_hz = base.frequency_hz
    if "frequency_hz" in payload:
        frequency_hz = _int_value(payload["frequency_hz"], "frequency_hz")
    elif "frequency_mhz" in payload:
        frequency_hz = int(round(_float_value(payload["frequency_mhz"], "frequency_mhz") * 1_000_000))

    if "sample_rate_msps" in payload:
        sample_rate_sps = int(round(_float_value(payload["sample_rate_msps"], "sample_rate_msps") * 1_000_000))
    else:
        sample_rate_sps = _int_value(payload.get("sample_rate_sps", base.sample_rate_sps), "sample_rate_sps")
    if "bandwidth_mhz" in payload:
        bandwidth_hz = int(round(_float_value(payload["bandwidth_mhz"], "bandwidth_mhz") * 1_000_000))
    else:
        bandwidth_hz = _int_value(payload.get("bandwidth_hz", base.bandwidth_hz), "bandwidth_hz")

    updated = replace(
        base,
        frequency_hz=frequency_hz,
        sample_rate_sps=sample_rate_sps,
        bandwidth_hz=bandwidth_hz,
        device_serial=(
            (
                str(payload["device_serial"]).strip() or None
                if payload["device_serial"] is not None
                else None
            )
            if "device_serial" in payload
            else base.device_serial
        ),
        amp=_amp_value(payload.get("amp", base.amp)),
        lna_gain=_int_value(payload.get("lna_gain", base.lna_gain), "lna_gain"),
        vga_gain=_int_value(payload.get("vga_gain", base.vga_gain), "vga_gain"),
        correct_iq=_bool_value(payload.get("correct_iq", base.correct_iq)),
        frequency_shift_hz=_float_value(
            payload.get("frequency_shift_hz", base.frequency_shift_hz),
            "frequency_shift_hz",
        ),
        expected_mode=str(payload.get("expected_mode", base.expected_mode)).lower(),
        system_info_index=_int_value(
            payload.get("system_info_index", base.system_info_index),
            "system_info_index",
        ),
        snapshot_duration_s=_float_value(
            payload.get("snapshot_duration_s", base.snapshot_duration_s),
            "snapshot_duration_s",
        ),
        analysis_max_samples=_int_value(
            payload.get("analysis_max_samples", base.analysis_max_samples),
            "analysis_max_samples",
        ),
        pn_max_symbols=_int_value(
            payload.get("pn_max_symbols", base.pn_max_symbols),
            "pn_max_symbols",
        ),
    )
    validate_ui_config(updated)
    return updated


def validate_ui_config(config: UiConfig) -> None:
    """Validate HackRF-safe UI settings."""

    if config.frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if config.sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be positive")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if config.device_serial is not None and not config.device_serial.strip():
        raise ValueError("device_serial must be non-empty when provided")
    if config.amp not in {0, 1}:
        raise ValueError("amp must be 0 or 1")
    if config.lna_gain < 0 or config.lna_gain > 40 or config.lna_gain % 8 != 0:
        raise ValueError("lna_gain must be 0..40 dB in 8 dB steps")
    if config.vga_gain < 0 or config.vga_gain > 62 or config.vga_gain % 2 != 0:
        raise ValueError("vga_gain must be 0..62 dB in 2 dB steps")
    if config.expected_mode not in {"auto", "pn420", "pn945"}:
        raise ValueError("expected_mode must be auto, pn420, or pn945")
    if config.system_info_index < 0 or config.system_info_index > 63:
        raise ValueError("system_info_index must be 0..63")
    if config.snapshot_duration_s <= 0 or config.snapshot_duration_s > 5:
        raise ValueError("snapshot_duration_s must be in the range 0..5 seconds")
    if config.analysis_max_samples <= 0:
        raise ValueError("analysis_max_samples must be positive")
    if config.pn_max_symbols <= 0:
        raise ValueError("pn_max_symbols must be positive")


def run_ui_snapshot(
    config: UiConfig,
    *,
    capture_dir: Path,
    executable: str = "hackrf_transfer",
    command_prefix: Sequence[str] = (),
    capture_runner: CaptureRunner = run_capture,
) -> dict[str, Any]:
    """Capture one bounded HackRF snapshot and analyze it for the UI."""

    validate_ui_config(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capture_path = (
        Path(capture_dir)
        / f"snapshot_{config.frequency_hz // 1000}khz_{timestamp}.ci8"
    )
    capture_config = HackrfCaptureConfig.from_duration(
        output_path=capture_path,
        frequency_hz=config.frequency_hz,
        duration_s=config.snapshot_duration_s,
        sample_rate_sps=config.sample_rate_sps,
        bandwidth_hz=config.bandwidth_hz,
        amp=config.amp,
        lna_gain=config.lna_gain,
        vga_gain=config.vga_gain,
        device_serial=config.device_serial,
        executable=executable,
        command_prefix=tuple(command_prefix),
        notes="dtmb-ui snapshot",
    )

    started = time.perf_counter()
    command, metadata_path = capture_runner(capture_config)
    analysis = analyze_ui_capture(capture_path, config=config)
    elapsed_s = time.perf_counter() - started

    return {
        "ok": True,
        "captured_utc": timestamp,
        "elapsed_s": elapsed_s,
        "capture_path": str(capture_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "command": command,
        "config": config.to_dict(),
        "analysis": analysis,
    }


def build_live_monitor_command(
    config: UiConfig,
    *,
    executable: str | Path = DEFAULT_NATIVE_MONITOR,
    fft_size: int = 2048,
    spectrum_bins: int = 192,
    pn_frames: int = 16,
    pn_workers: int = 0,
    report_every_samples: int | None = None,
    max_reports: int = 0,
    max_samples: int = 0,
) -> list[str]:
    """Build the native C++ HackRF live-monitor command."""

    validate_ui_config(config)
    interval = report_every_samples
    if interval is None:
        interval = max(1, config.sample_rate_sps // 5)
    command = [
        str(executable),
        "--hackrf",
        "--sample-rate",
        str(config.sample_rate_sps),
        "--center-frequency",
        str(config.frequency_hz),
        "--bandwidth",
        str(config.bandwidth_hz),
        "--amp",
        str(config.amp),
        "--lna-gain",
        str(config.lna_gain),
        "--vga-gain",
        str(config.vga_gain),
        "--fft-size",
        str(fft_size),
        "--spectrum-bins",
        str(spectrum_bins),
        "--pn-frames",
        str(pn_frames),
        "--pn-workers",
        str(pn_workers),
        "--report-every-samples",
        str(interval),
    ]
    if max_reports:
        command.extend(["--max-reports", str(max_reports)])
    if max_samples:
        command.extend(["--max-samples", str(max_samples)])
    if config.device_serial:
        command.extend(["--serial", config.device_serial])
    return command


def start_live_monitor(state: UiState, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the native C++ HackRF monitor and stream telemetry into state."""

    payload = payload or {}
    with state.live_process_lock:
        with state.live_condition:
            if state.shutting_down:
                raise UiResourceBusyError("UI server is shutting down")
            if state.capture_in_progress:
                raise UiResourceBusyError("snapshot capture in progress")
            if _process_running(state.live_process):
                raise UiResourceBusyError("live monitor already running")
            config = state.config
            command = build_live_monitor_command(
                config,
                executable=state.native_monitor,
                fft_size=_int_value(payload.get("fft_size", 2048), "fft_size"),
                spectrum_bins=_int_value(payload.get("spectrum_bins", 192), "spectrum_bins"),
                pn_frames=_int_value(payload.get("pn_frames", 16), "pn_frames"),
                pn_workers=_int_value(payload.get("pn_workers", 0), "pn_workers"),
                report_every_samples=(
                    _int_value(payload["report_every_samples"], "report_every_samples")
                    if "report_every_samples" in payload
                    else None
                ),
                max_reports=_int_value(payload.get("max_reports", 0), "max_reports"),
                max_samples=_int_value(payload.get("max_samples", 0), "max_samples"),
            )

            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            state.live_process = process
            state.live_started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            state.live_command = command
            state.live_latest = None
            state.live_stderr.clear()
            _publish_live_event_locked(
                state,
                {
                    "schema": "dtmb.live_state.v1",
                    "running": True,
                    "started_utc": state.live_started_utc,
                    "command": command,
                    "config": config.to_dict(),
                },
            )

        if process.stdout is not None:
            threading.Thread(
                target=_read_live_stdout,
                args=(state, process),
                daemon=True,
                name="dtmb-ui-live-stdout",
            ).start()
        if process.stderr is not None:
            threading.Thread(
                target=_read_live_stderr,
                args=(state, process),
                daemon=True,
                name="dtmb-ui-live-stderr",
            ).start()
        threading.Thread(
            target=_wait_live_process,
            args=(state, process),
            daemon=True,
            name="dtmb-ui-live-wait",
        ).start()

    return {
        "ok": True,
        "running": True,
        "started_utc": state.live_started_utc,
        "command": command,
        "config": config.to_dict(),
    }


def stop_live_monitor(state: UiState, *, timeout_s: float = 2.0) -> dict[str, Any]:
    """Stop the native live monitor if it is running."""

    with state.live_process_lock:
        with state.live_condition:
            process = state.live_process
            if not _process_running(process):
                if process is not None:
                    _finish_live_process_locked(state, process, process.returncode)
                return {"ok": True, "running": False, "stopped": False}

        assert process is not None
        killed = _terminate_process(process, timeout_s=timeout_s)
        with state.live_condition:
            _finish_live_process_locked(state, process, process.returncode)
        return {
            "ok": True,
            "running": False,
            "stopped": True,
            "killed": killed,
            "returncode": process.returncode,
        }


def shutdown_ui_state(state: UiState, *, timeout_s: float = 2.0) -> dict[str, Any]:
    """Prevent new HackRF work and stop any live-monitor child."""

    with state.live_condition:
        state.shutting_down = True
    result = stop_live_monitor(state, timeout_s=timeout_s)
    with state.live_condition:
        while state.capture_in_progress:
            state.live_condition.wait()
    return {**result, "snapshot_released": True}


def analyze_ui_capture(capture_path: str | Path, *, config: UiConfig) -> dict[str, Any]:
    """Return capture diagnostics, spectrum trace, PN status, and frame preview."""

    capture_path = Path(capture_path)
    diagnostics = analyze_ci8_capture(
        capture_path,
        sample_rate_sps=config.sample_rate_sps,
        max_samples=config.analysis_max_samples,
        fft_size=16_384,
    )
    samples = read_ci8(capture_path, max_samples=config.analysis_max_samples)
    spectrum = compute_spectrum_trace(
        samples,
        sample_rate_sps=config.sample_rate_sps,
        fft_size=16_384,
        point_count=640,
    )
    pn_status = analyze_pn_status(samples, config=config)
    pn_status["tuning_score"] = score_ui_signal(
        diagnostics,
        pn_status,
        expected_system_info_index=config.system_info_index,
    )
    return {
        "diagnostics": diagnostics,
        "spectrum_trace": spectrum,
        "pn_status": pn_status,
    }


def analyze_pn_status(samples: np.ndarray, *, config: UiConfig) -> dict[str, Any]:
    """Analyze PN acquisition status from raw samples."""

    signal = np.asarray(samples, dtype=np.complex64)
    if config.correct_iq:
        signal = remove_dc(signal)
    if config.frequency_shift_hz:
        signal = frequency_shift(
            signal,
            sample_rate_sps=config.sample_rate_sps,
            shift_hz=config.frequency_shift_hz,
        )

    try:
        if config.sample_rate_sps == DTMB_SYMBOL_RATE_SPS:
            symbols = signal
            up = 1
            down = 1
        else:
            symbols, up, down = resample_to_symbol_rate(
                signal,
                sample_rate_sps=config.sample_rate_sps,
                symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
            )
    except Exception as exc:
        return {
            "status": "resampler-error",
            "error": str(exc),
            "input_sample_count": int(signal.size),
        }

    symbols = symbols[: config.pn_max_symbols]
    modes = _pn_modes(config.expected_mode)
    extension_trains = detect_pn_cyclic_extension_trains(
        symbols,
        modes=modes,
        max_trains_per_mode=3,
    )
    extension_train_dicts: list[dict[str, Any]] = []
    for train in extension_trains:
        train_dict = train.to_dict()
        train_dict["coarse_cfo_hz"] = estimate_cfo_from_pn_cyclic_extension(
            symbols,
            mode=train.mode,
            phase_offset=train.phase_offset,
            symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
        )
        extension_train_dicts.append(train_dict)

    family_trains = detect_pn_phase_family_trains(
        symbols,
        modes=modes,
        candidate_phases_per_mode=4,
        max_trains_per_mode=2,
    )
    family_train_dicts = [train.to_dict() for train in family_trains]
    best_train = extension_trains[0] if extension_trains else None
    best_cfo = (
        extension_train_dicts[0].get("coarse_cfo_hz")
        if extension_train_dicts
        else None
    )
    delay_family = None
    frames = None
    frame_phase_offset = best_train.phase_offset if best_train is not None else None
    frame_phase_source = "cyclic"
    frame_cfo = best_cfo
    if best_train is not None and best_train.max_metric >= 0.35:
        corrected_symbols = symbols
        if best_cfo is not None:
            corrected_symbols = frequency_shift(
                corrected_symbols,
                sample_rate_sps=DTMB_SYMBOL_RATE_SPS,
                shift_hz=-float(best_cfo),
            )
        delay_family_train = score_pn_family_delay_train(
            corrected_symbols,
            mode=best_train.mode,
            phase_offset=best_train.phase_offset,
            max_frames=8,
            max_delay_symbols=256,
        )
        delay_family = delay_family_train.to_dict()
        if should_apply_pn_family_delay(delay_family_train):
            frame_phase_offset = delay_corrected_phase_offset(
                best_train.phase_offset,
                median_delay_symbols=delay_family_train.median_delay_symbols,
                frame_symbols=best_train.frame_symbols,
            )
            frame_phase_source = "pn-family-delay"
            corrected_cfo = estimate_cfo_from_pn_cyclic_extension(
                symbols,
                mode=best_train.mode,
                phase_offset=frame_phase_offset,
                symbol_rate_sps=DTMB_SYMBOL_RATE_SPS,
            )
            if corrected_cfo is not None:
                frame_cfo = corrected_cfo
        frames = summarize_frame_bodies(
            symbols,
            mode=best_train.mode,
            phase_offset=frame_phase_offset,
            coarse_cfo_hz=frame_cfo,
            system_info_index=config.system_info_index,
            max_frames=3,
        )

    return {
        "status": _pn_status_label(
            extension_train_dicts[0] if extension_train_dicts else None,
            delay_family,
        ),
        "correct_iq": config.correct_iq,
        "input_sample_count": int(signal.size),
        "symbol_rate_sps": DTMB_SYMBOL_RATE_SPS,
        "resample_up": up,
        "resample_down": down,
        "analyzed_symbols": int(symbols.size),
        "extension_trains": extension_train_dicts,
        "phase_family_trains": family_train_dicts,
        "delay_family_train": delay_family,
        "frame_phase_offset": frame_phase_offset,
        "frame_phase_source": frame_phase_source if best_train is not None else None,
        "frame_cfo_hz": frame_cfo,
        "frame_preview": frames,
    }


def score_ui_signal(
    diagnostics: dict[str, Any],
    pn_status: dict[str, Any],
    *,
    expected_system_info_index: int = 23,
) -> dict[str, Any]:
    """Return a 0-100 operator score for antenna and gain tuning."""

    iq = diagnostics.get("iq", {}) if isinstance(diagnostics, dict) else {}
    best_train = _first_dict(pn_status.get("extension_trains"))
    delay_family = pn_status.get("delay_family_train") or {}
    frame_preview = pn_status.get("frame_preview") or {}
    frames = frame_preview.get("frames") or []

    family_mean = _finite_float(delay_family.get("mean_metric"), 0.0)
    family_hits = int(delay_family.get("hit_count") or 0)
    family_observed = int(delay_family.get("observed_frames") or 0)
    family_hit_ratio = _ratio(family_hits, family_observed)
    family_component = (
        55.0
        * _normalize(family_mean, low=0.35, high=0.70)
        * _normalize(family_hit_ratio, low=0.25, high=1.0)
    )

    cyclic_mean = _finite_float(best_train.get("mean_metric"), 0.0)
    cyclic_max = _finite_float(best_train.get("max_metric"), 0.0)
    cyclic_hits = int(best_train.get("hit_count") or 0)
    cyclic_observed = int(best_train.get("observed_frames") or 0)
    cyclic_hit_ratio = _ratio(cyclic_hits, cyclic_observed)
    cyclic_component = (
        15.0
        * _normalize(max(cyclic_mean, cyclic_max), low=0.35, high=0.85)
        * _normalize(cyclic_hit_ratio, low=0.20, high=1.0)
    )

    component_rms_dbfs = _finite_float(iq.get("component_rms_dbfs"), -120.0)
    rf_component = 15.0 * _rf_level_score(component_rms_dbfs)

    evm = _finite_float(frame_preview.get("best_data_evm_rms"), float("nan"))
    evm_component = 0.0 if not np.isfinite(evm) else 10.0 * _normalize(1.20 - evm, low=0.0, high=0.70)

    expected_hits = _expected_system_hits(
        frames,
        expected_system_info_index=expected_system_info_index,
    )
    best_system_metric = _best_system_metric(frames)
    system_component = min(
        5.0,
        expected_hits * 2.0 + 3.0 * _normalize(best_system_metric, low=0.20, high=0.45),
    )

    clip_count = int(iq.get("clip_count_i") or 0) + int(iq.get("clip_count_q") or 0)
    clip_penalty = 0.0 if clip_count == 0 else min(40.0, 8.0 + 8.0 * np.log10(clip_count + 1))

    raw_score = (
        family_component
        + cyclic_component
        + rf_component
        + evm_component
        + system_component
        - clip_penalty
    )
    score = _cap_score_for_lock_state(raw_score, str(pn_status.get("status") or ""), clip_count)
    score = int(round(max(0.0, min(100.0, score))))
    return {
        "score": score,
        "label": _score_label(score),
        "components": {
            "pn_family": round(family_component, 1),
            "cyclic": round(cyclic_component, 1),
            "rf_level": round(rf_component, 1),
            "evm": round(evm_component, 1),
            "system": round(system_component, 1),
            "clip_penalty": round(clip_penalty, 1),
        },
        "inputs": {
            "family_mean": family_mean,
            "family_hits": family_hits,
            "family_observed": family_observed,
            "cyclic_metric": max(cyclic_mean, cyclic_max),
            "cyclic_hits": cyclic_hits,
            "cyclic_observed": cyclic_observed,
            "component_rms_dbfs": component_rms_dbfs,
            "clip_count": clip_count,
            "best_data_evm_rms": evm if np.isfinite(evm) else None,
            "expected_system_hits": expected_hits,
            "best_system_metric": best_system_metric,
        },
    }


def summarize_frame_bodies(
    symbols: np.ndarray,
    *,
    mode: PnMode,
    phase_offset: int,
    coarse_cfo_hz: float | None,
    system_info_index: int,
    max_frames: int = 3,
) -> dict[str, Any]:
    """Summarize C=3780 frame bodies for UI lock diagnostics."""

    corrected = np.asarray(symbols, dtype=np.complex64)
    if coarse_cfo_hz is not None:
        corrected = frequency_shift(
            corrected,
            sample_rate_sps=DTMB_SYMBOL_RATE_SPS,
            shift_hz=-float(coarse_cfo_hz),
        )

    reports: list[dict[str, Any]] = []
    try:
        for frame in iter_signal_frames(
            corrected,
            mode=mode,
            phase_offset=phase_offset,
            max_frames=max_frames,
        ):
            spectrum = frame_body_fft(frame.body)
            inserted = frequency_deinterleave_inserted(spectrum)
            deinterleaved = frequency_deinterleave(spectrum)
            system_info, data = split_system_info_and_data(deinterleaved)
            matches = classify_system_info(system_info, frame_body_modes=("C3780",))
            eq = equalize_with_system_info_pilots(
                inserted,
                system_info_index=system_info_index,
                frame_body_mode="C3780",
            )
            reports.append(
                {
                    "start": frame.start,
                    "header_power": _mean_power(frame.header),
                    "body_power": _mean_power(frame.body),
                    "data_power": _mean_power(data),
                    "best_system_info": matches[0].to_dict() if matches else None,
                    "sparse_pilot_equalization": eq.to_dict(),
                }
            )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "frames": reports}

    best_evm = [
        frame["sparse_pilot_equalization"]["data_evm_rms"]
        for frame in reports
        if frame.get("sparse_pilot_equalization")
    ]
    return {
        "ok": True,
        "mode": mode,
        "phase_offset": phase_offset,
        "coarse_cfo_hz": coarse_cfo_hz,
        "frames": reports,
        "best_data_evm_rms": min(best_evm) if best_evm else None,
    }


def list_visual_artifacts(
    capture_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    max_items: int = 200,
) -> list[dict[str, Any]]:
    """Return PNG/JSON gate visual artifacts under approved capture roots."""

    roots = _visual_artifact_roots(capture_dir, repo_root=repo_root)
    rows: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file() or not _is_visual_artifact_file(candidate):
                continue
            try:
                rel = candidate.relative_to(root)
            except ValueError:
                continue
            stat = candidate.stat()
            rows.append(
                {
                    "id": _artifact_id(root_index, rel),
                    "name": candidate.name,
                    "relative_path": rel.as_posix(),
                    "path": str(candidate),
                    "root": str(root),
                    "gate": _artifact_gate(candidate.name),
                    "kind": "image" if candidate.suffix.lower() == ".png" else "json",
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime,
                        timezone.utc,
                    ).isoformat().replace("+00:00", "Z"),
                }
            )
    rows.sort(key=lambda item: str(item["modified_utc"]), reverse=True)
    return rows[:max_items]


def resolve_visual_artifact(
    artifact_id: str,
    capture_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> Path:
    """Resolve a UI artifact id while preventing path traversal."""

    try:
        root_text, rel_text = artifact_id.split(":", 1)
        root_index = int(root_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid artifact id") from exc
    roots = _visual_artifact_roots(capture_dir, repo_root=repo_root)
    if root_index < 0 or root_index >= len(roots):
        raise ValueError("artifact root is out of range")
    rel = Path(rel_text)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("artifact path is not relative")
    root = roots[root_index]
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes capture root") from exc
    if not candidate.is_file() or not _is_visual_artifact_file(candidate):
        raise FileNotFoundError("artifact not found")
    return candidate


def make_handler(
    state: UiState,
    *,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP handler bound to the given UI state."""

    static_root = Path(static_dir).resolve()

    class DtmbUiHandler(BaseHTTPRequestHandler):
        server_version = "dtmb-ui/0.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/status":
                self._send_json(200, _status_payload(state))
                return
            if path == "/api/live/events":
                self._handle_live_events(parsed.query)
                return
            if path == "/api/presets":
                self._send_json(200, {"ok": True, "presets": frequency_presets()})
                return
            if path == "/api/artifacts":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "artifacts": list_visual_artifacts(state.capture_dir),
                    },
                )
                return
            if path == "/api/artifact":
                self._handle_artifact(parsed.query)
                return
            if path.startswith("/api/"):
                self._send_json(404, {"ok": False, "error": "unknown API endpoint"})
                return
            self._serve_static(path, static_root)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            path = urlparse(self.path).path
            if path == "/api/config":
                self._handle_config_update()
                return
            if path == "/api/snapshot":
                self._handle_snapshot()
                return
            if path == "/api/live/start":
                self._handle_live_start()
                return
            if path == "/api/live/stop":
                self._handle_live_stop()
                return
            self._send_json(404, {"ok": False, "error": "unknown API endpoint"})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _handle_config_update(self) -> None:
            try:
                payload = self._read_json()
                with state.lock:
                    state.config = update_ui_config(state.config, payload)
                    config = state.config
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {"ok": True, "config": config.to_dict()})

        def _handle_snapshot(self) -> None:
            try:
                config = _reserve_snapshot_capture(state)
            except UiResourceBusyError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
                return

            result: dict[str, Any] | None = None
            try:
                result = run_ui_snapshot(
                    config,
                    capture_dir=state.capture_dir,
                    executable=state.executable,
                    command_prefix=state.command_prefix,
                )
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "config": config.to_dict()}
                self._send_json(500, result)
                return
            finally:
                _release_snapshot_capture(state, result)

            self._send_json(200, result)

        def _handle_live_start(self) -> None:
            try:
                payload = self._read_json()
                config_payload = payload.get("config")
                if not isinstance(config_payload, dict):
                    config_payload = payload
                with state.lock:
                    state.config = update_ui_config(state.config, config_payload)
                result = start_live_monitor(state, payload)
            except UiResourceBusyError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, result)

        def _handle_live_stop(self) -> None:
            try:
                result = stop_live_monitor(state)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, result)

        def _handle_live_events(self, query: str) -> None:
            try:
                last_id = int((parse_qs(query).get("last") or ["0"])[0])
            except ValueError:
                last_id = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            sequence = last_id
            while True:
                with state.live_condition:
                    state.live_condition.wait_for(
                        lambda: state.live_events
                        and state.live_events[-1]["sequence"] > sequence,
                        timeout=15.0,
                    )
                    events = [
                        event
                        for event in state.live_events
                        if int(event["sequence"]) > sequence
                    ]
                try:
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    for event in events:
                        sequence = int(event["sequence"])
                        data = json.dumps(_json_safe(event["payload"]), allow_nan=False)
                        self.wfile.write(f"id: {sequence}\n".encode("ascii"))
                        self.wfile.write(b"event: telemetry\n")
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return

        def _handle_artifact(self, query: str) -> None:
            artifact_id = (parse_qs(query).get("id") or [""])[0]
            try:
                path = resolve_visual_artifact(artifact_id, state.capture_dir)
            except FileNotFoundError:
                self._send_json(404, {"ok": False, "error": "artifact not found"})
                return
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            content = path.read_bytes()
            mime_type = (
                "application/json; charset=utf-8"
                if path.suffix.lower() == ".json"
                else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64 * 1024:
                raise ValueError("request body is too large")
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("request body must be a JSON object")
            return parsed

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _serve_static(self, request_path: str, root: Path) -> None:
            relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                self.send_error(403)
                return
            if not candidate.is_file():
                self.send_error(404)
                return
            content = candidate.read_bytes()
            mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return DtmbUiHandler


def frequency_presets() -> list[dict[str, Any]]:
    """Return practical center-frequency presets for this project phase."""

    presets: list[dict[str, Any]] = []
    for region, csv_path in DTMB_PRESET_SOURCES:
        presets.extend(load_dtmb_presets(csv_path, region=region))
    return presets if presets else [
        {"label": "HK DTT 482 MHz", "frequency_hz": 482_000_000},
        {"label": "HK DTT 522 MHz", "frequency_hz": 522_000_000},
        {"label": "HK DTT 538 MHz", "frequency_hz": 538_000_000},
        {"label": "HK DTT 586 MHz", "frequency_hz": 586_000_000},
        {"label": "HK DTT 602 MHz", "frequency_hz": 602_000_000},
    ]


def load_hk_dtmb_presets(csv_path: str | Path = DEFAULT_HK_DTMB_CSV) -> list[dict[str, Any]]:
    """Load DTMB multiplex presets from the reference CSV if present."""

    return load_dtmb_presets(csv_path, region="HK")


def load_dtmb_presets(csv_path: str | Path, *, region: str) -> list[dict[str, Any]]:
    """Load multiplex presets from a Chinese DTMB channel-table CSV."""

    path = Path(csv_path)
    if not path.is_file():
        return []

    rows = _read_reference_csv(path)
    if not rows or len(rows) < 2:
        return []
    header = rows[0]
    frequency_index = _first_header_index(header, ("接收频率", "频率"), default=0)
    polarization_index = _first_header_index(header, ("极化",), default=1)
    channel_index = _first_header_index(header, ("频道名称",), default=3)

    muxes: dict[int, dict[str, Any]] = {}
    current_frequency_mhz: float | None = None
    current_polarizations: set[str] = set()
    for raw_row in rows[1:]:
        row = raw_row + [""] * max(0, len(header) - len(raw_row), 12 - len(raw_row))
        frequency_text = row[frequency_index].strip()
        if frequency_text:
            parsed_frequency = _parse_frequency_mhz(frequency_text)
            if parsed_frequency is not None:
                current_frequency_mhz = parsed_frequency
        if current_frequency_mhz is None:
            continue

        polarization = row[polarization_index].strip()
        if polarization:
            current_polarizations = {polarization}
        frequency_hz = int(round(current_frequency_mhz * 1_000_000))
        mux = muxes.setdefault(
            frequency_hz,
            {
                "channels": [],
                "params": set(),
                "polarizations": set(),
            },
        )
        mux["polarizations"].update(current_polarizations)

        channel_name = row[channel_index].strip()
        if channel_name and channel_name not in mux["channels"]:
            mux["channels"].append(channel_name)
        for value in row:
            if _looks_like_dtmb_parameter(value):
                mux["params"].add(value.strip())

    presets: list[dict[str, Any]] = []
    for frequency_hz, data in sorted(muxes.items()):
        channels = "/".join(data["channels"][:4])
        params = " ".join(sorted(data["params"]))
        label_parts = [f"{region} DTMB {frequency_hz // 1_000_000} MHz"]
        if channels:
            label_parts.append(channels)
        if params:
            label_parts.append(params)
        preset: dict[str, Any] = {
            "label": " - ".join(label_parts),
            "frequency_hz": frequency_hz,
            "region": region,
        }
        if channels:
            preset["channels"] = data["channels"]
        if params:
            preset["params"] = sorted(data["params"])
        if data["polarizations"]:
            preset["polarizations"] = sorted(data["polarizations"])
        presets.append(preset)
    return presets


def _read_reference_csv(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.reader(handle))
        except UnicodeDecodeError:
            continue
    return []


def _first_header_index(header: Sequence[str], names: Sequence[str], *, default: int) -> int:
    normalized = [value.strip() for value in header]
    for name in names:
        if name in normalized:
            return normalized.index(name)
    return default


def _parse_frequency_mhz(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _looks_like_dtmb_parameter(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    upper = text.upper()
    return any(
        token in upper
        for token in ("QAM", "PN", "3/5", "2/3", "4/5", "1/2", "0.6")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-ui",
        description="Run a local HackRF DTMB capture and spectrum dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--executable", default="hackrf_transfer")
    parser.add_argument("--native-monitor", type=Path, default=DEFAULT_NATIVE_MONITOR)
    parser.add_argument("--mamba-env", help="Run hackrf_transfer through `mamba run -n ENV`")
    parser.add_argument("--device-serial")
    parser.add_argument("--frequency", type=int, default=DEFAULT_UI_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=0.15)
    parser.add_argument("--lna-gain", type=int, default=16)
    parser.add_argument("--vga-gain", type=int, default=16)
    parser.add_argument("--amp", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command_prefix = ("mamba", "run", "-n", args.mamba_env) if args.mamba_env else ()
    config = UiConfig(
        frequency_hz=args.frequency,
        device_serial=args.device_serial,
        snapshot_duration_s=args.duration,
        amp=args.amp,
        lna_gain=args.lna_gain,
        vga_gain=args.vga_gain,
    )
    validate_ui_config(config)
    state = UiState(
        config=config,
        capture_dir=args.capture_dir,
        executable=args.executable,
        native_monitor=args.native_monitor,
        command_prefix=command_prefix,
    )
    handler = make_handler(state, static_dir=args.static_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"dtmb-ui listening on {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            shutdown_ui_state(state)
        finally:
            server.server_close()
    return 0


def _process_running(process: subprocess.Popen[str] | None) -> bool:
    return process is not None and process.poll() is None


def _terminate_process(process: subprocess.Popen[str], *, timeout_s: float) -> bool:
    killed = False
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            if process.poll() is None:
                raise
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        killed = True
        try:
            process.kill()
        except OSError:
            if process.poll() is None:
                raise
        process.wait(timeout=timeout_s)
    return killed


def _reserve_snapshot_capture(state: UiState) -> UiConfig:
    with state.live_condition:
        if state.shutting_down:
            raise UiResourceBusyError("UI server is shutting down")
        if state.capture_in_progress:
            raise UiResourceBusyError("snapshot capture already in progress")
        if _process_running(state.live_process):
            raise UiResourceBusyError("live monitor already running")
        state.capture_in_progress = True
        return state.config


def _release_snapshot_capture(state: UiState, result: dict[str, Any] | None) -> None:
    with state.live_condition:
        state.capture_in_progress = False
        if result is not None:
            state.last_snapshot = _compact_snapshot(result)
        state.live_condition.notify_all()


def _read_live_stdout(state: UiState, process: subprocess.Popen[str]) -> None:
    stream = process.stdout
    if stream is None:
        return
    try:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                payload = {
                    "schema": "dtmb.live_parse_error.v1",
                    "error": str(exc),
                    "line": text[:2000],
                }
            _publish_live_event(state, payload)
    except Exception as exc:
        _publish_live_event(
            state,
            {"schema": "dtmb.live_reader_error.v1", "stream": "stdout", "error": str(exc)},
        )


def _read_live_stderr(state: UiState, process: subprocess.Popen[str]) -> None:
    stream = process.stderr
    if stream is None:
        return
    try:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            with state.live_condition:
                state.live_stderr.append(text)
            _publish_live_event(
                state,
                {
                    "schema": "dtmb.live_log.v1",
                    "level": "stderr",
                    "message": text,
                },
            )
    except Exception as exc:
        _publish_live_event(
            state,
            {"schema": "dtmb.live_reader_error.v1", "stream": "stderr", "error": str(exc)},
        )


def _wait_live_process(state: UiState, process: subprocess.Popen[str]) -> None:
    returncode = process.wait()
    with state.live_condition:
        _finish_live_process_locked(state, process, returncode)


def _finish_live_process_locked(
    state: UiState,
    process: subprocess.Popen[str],
    returncode: int | None,
) -> bool:
    if state.live_process is not process:
        return False
    state.live_process = None
    _publish_live_event_locked(
        state,
        {
            "schema": "dtmb.live_state.v1",
            "running": False,
            "returncode": returncode,
            "stopped_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )
    return True


def _publish_live_event(state: UiState, payload: dict[str, Any]) -> None:
    with state.live_condition:
        _publish_live_event_locked(state, payload)


def _publish_live_event_locked(state: UiState, payload: dict[str, Any]) -> None:
    state.live_sequence += 1
    state.live_events.append({"sequence": state.live_sequence, "payload": payload})
    if payload.get("schema") == "dtmb.live_monitor.v1":
        state.live_latest = payload
    state.live_condition.notify_all()


def _status_payload(state: UiState) -> dict[str, Any]:
    with state.lock:
        live_running = _process_running(state.live_process)
        return {
            "ok": True,
            "config": state.config.to_dict(),
            "capture_in_progress": state.capture_in_progress,
            "last_snapshot": state.last_snapshot,
            "live": {
                "running": live_running,
                "started_utc": state.live_started_utc if live_running else None,
                "command": state.live_command if live_running else None,
                "latest": state.live_latest,
                "sequence": state.live_sequence,
                "stderr": list(state.live_stderr),
            },
        }


def _visual_artifact_roots(
    capture_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> list[Path]:
    roots: list[Path] = []
    for root in (Path(repo_root) / "captures", Path(capture_dir)):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _is_visual_artifact_file(path: Path) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in VISUAL_ARTIFACT_PATTERNS)


def _artifact_id(root_index: int, relative_path: Path) -> str:
    return f"{root_index}:{relative_path.as_posix()}"


def _artifact_gate(name: str) -> str:
    for gate in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e", "gate_f"):
        if f".{gate}." in name:
            return gate[-1].upper()
    if ".echo." in name:
        return "echo"
    return "pipeline"


def _compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    analysis = snapshot.get("analysis") if isinstance(snapshot, dict) else None
    diagnostics = analysis.get("diagnostics", {}) if isinstance(analysis, dict) else {}
    pn_status = analysis.get("pn_status", {}) if isinstance(analysis, dict) else {}
    spectrum = analysis.get("spectrum_trace", {}) if isinstance(analysis, dict) else {}
    return {
        "ok": snapshot.get("ok", False),
        "captured_utc": snapshot.get("captured_utc"),
        "elapsed_s": snapshot.get("elapsed_s"),
        "capture_path": snapshot.get("capture_path"),
        "iq": diagnostics.get("iq") if isinstance(diagnostics, dict) else None,
        "spectrum": {
            "peak_frequency_hz": spectrum.get("peak_frequency_hz"),
            "peak_power_db": spectrum.get("peak_power_db"),
            "median_power_db": spectrum.get("median_power_db"),
        },
        "pn_status": {
            "status": pn_status.get("status"),
            "tuning_score": pn_status.get("tuning_score"),
            "extension_trains": pn_status.get("extension_trains", [])[:2]
            if isinstance(pn_status, dict)
            else [],
            "delay_family_train": pn_status.get("delay_family_train")
            if isinstance(pn_status, dict)
            else None,
        },
    }


def _pn_modes(expected_mode: str) -> tuple[PnMode, ...]:
    if expected_mode == "auto":
        return ("pn420", "pn945")
    return (expected_mode,)  # type: ignore[return-value]


def _pn_status_label(
    best_train: dict[str, Any] | None,
    delay_family: dict[str, Any] | None = None,
) -> str:
    if not best_train:
        return "no-pn"
    max_metric = float(best_train.get("max_metric") or 0.0)
    hit_count = int(best_train.get("hit_count") or 0)
    if _delay_family_dict_locked(delay_family):
        return "pn-family-lock"
    if max_metric >= 0.70 and hit_count >= 2:
        return "cyclic-only"
    if max_metric >= 0.45:
        return "weak-candidate"
    return "searching"


def _delay_family_dict_locked(delay_family: dict[str, Any] | None) -> bool:
    """Dict equivalent of frame_sync.should_apply_pn_family_delay for UI labels."""

    if not delay_family:
        return False
    observed = int(delay_family.get("observed_frames") or 0)
    if observed < 8:
        return False
    family_hits = int(delay_family.get("hit_count") or 0)
    required_hits = max(2, int(np.ceil(observed * 0.5)))
    if family_hits < required_hits:
        return False
    family_mean = float(delay_family.get("mean_metric") or 0.0)
    if family_mean < 0.45:
        return False
    delay_mad = float(delay_family.get("delay_mad_symbols") or 0.0)
    return delay_mad <= 8.0


def _first_dict(values: Any) -> dict[str, Any]:
    if isinstance(values, list) and values and isinstance(values[0], dict):
        return values[0]
    return {}


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _normalize(value: float, *, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("normalization high must exceed low")
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _rf_level_score(component_rms_dbfs: float) -> float:
    if component_rms_dbfs <= -55.0:
        return 0.0
    if component_rms_dbfs <= -38.0:
        return 0.65 * _normalize(component_rms_dbfs, low=-55.0, high=-38.0)
    if component_rms_dbfs <= -24.0:
        return 0.65 + 0.35 * _normalize(component_rms_dbfs, low=-38.0, high=-24.0)
    if component_rms_dbfs <= -12.0:
        return 1.0 - 0.45 * _normalize(component_rms_dbfs, low=-24.0, high=-12.0)
    return 0.35


def _expected_system_hits(
    frames: Sequence[Any],
    *,
    expected_system_info_index: int,
) -> int:
    hits = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        best = frame.get("best_system_info")
        if not isinstance(best, dict):
            continue
        if int(best.get("index") or -1) == expected_system_info_index:
            hits += 1
    return hits


def _best_system_metric(frames: Sequence[Any]) -> float:
    metrics = []
    for frame in frames:
        if isinstance(frame, dict) and isinstance(frame.get("best_system_info"), dict):
            metrics.append(_finite_float(frame["best_system_info"].get("metric"), 0.0))
    return max(metrics) if metrics else 0.0


def _cap_score_for_lock_state(score: float, status: str, clip_count: int) -> float:
    if status == "pn-family-lock":
        capped = score
    elif status == "cyclic-only":
        capped = min(score, 70.0)
    elif status == "weak-candidate":
        capped = min(score, 55.0)
    elif status in {"no-pn", "searching"}:
        capped = min(score, 35.0)
    else:
        capped = min(score, 25.0)
    if clip_count > 0:
        capped = min(capped, 75.0)
    return capped


def _score_label(score: int) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "usable"
    if score >= 30:
        return "weak"
    return "bad"


def _mean_power(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values) ** 2))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _int_value(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return parsed


def _float_value(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _amp_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return _int_value(value, "amp")


if __name__ == "__main__":
    raise SystemExit(main())
