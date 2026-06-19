"""PN945/C=3780 DTMB simulation harness.

The simulator generates a known-good continuous DTMB stream, applies a
deterministic RF impairment model, quantizes to CI8, and optionally runs the
normal receiver over the produced capture. It is a validation harness, not a
second receiver.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import analyze_ci8_capture, compute_spectrum_trace
from .continuous_stream import (
    DATA_SYMBOLS_PER_FRAME,
    ContinuousStreamConfig,
    _system_info_index_to_parameters,
    encode_continuous_symbol_stream,
)
from .frequency import FRAME_BODY_SYMBOLS, frequency_interleave
from .pn import pn_header_symbols_for_body_power
from .receiver import receive_capture
from .system_info import SYSTEM_INFO_VECTORS, system_info_symbols
from .ts import TS_PACKET_SIZE, TS_SYNC_BYTE, analyze_ts_packets


SAMPLE_RATE_SPS = 7_560_000
PN_MODE = "pn945"
FRAME_SYMBOLS = 4725
SIM_SCHEMA = "dtmb.simulate.pn945_c3780.v1"
PRESETS = (
    "clean",
    "awgn",
    "echo-in-guard",
    "echo-out-of-guard",
    "cfo-cpe",
    "sample-drift",
    "urban-stress",
)


@dataclass(frozen=True)
class EchoTap:
    """One synthetic multipath echo."""

    delay_samples: float
    attenuation_db: float
    phase_rad: float = 0.0

    @property
    def complex_gain(self) -> complex:
        gain = 10.0 ** (float(self.attenuation_db) / 20.0)
        return gain * np.exp(1j * float(self.phase_rad))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelImpairments:
    """RF impairment parameters applied before CI8 quantization."""

    awgn_cn_db: float | None = None
    cfo_hz: float = 0.0
    cpe_initial_rad: float = 0.0
    cpe_step_std_rad: float = 0.0
    sample_rate_offset_ppm: float = 0.0
    timing_offset_samples: float = 0.0
    echoes: tuple[EchoTap, ...] = ()
    dc_i: float = 0.0
    dc_q: float = 0.0
    gain: float = 1.0
    clip: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["echoes"] = [tap.to_dict() for tap in self.echoes]
        return data


@dataclass(frozen=True)
class SimulationPreset:
    """Named simulator preset and receiver defaults."""

    name: str
    impairments: ChannelImpairments
    receiver_equalizer: str = "none"
    correct_per_frame_cpe: bool = False
    expected_recovery: str = "pass"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-simulate",
        description="Generate PN945/C=3780 synthetic captures and run receiver diagnostics.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="clean",
        help="Named impairment preset.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--payload-ts", type=Path)
    parser.add_argument("--payload-frames", type=_positive_int, default=1)
    parser.add_argument("--system-info-index", type=int, default=23)
    parser.add_argument("--rng-seed", type=int, default=20260523)
    parser.add_argument("--amplitude", type=float, default=0.75)
    parser.add_argument("--run-receiver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receiver-equalizer", default=None)
    parser.add_argument("--receiver-pn-channel-taps", type=_positive_int)
    parser.add_argument(
        "--receiver-pn-estimator",
        choices=("compact", "wideband"),
        default="compact",
    )
    parser.add_argument("--receiver-pn-mmse", type=float)
    parser.add_argument("--receiver-csi-demap", action="store_true")
    parser.add_argument("--correct-per-frame-cpe", action="store_true")
    parser.add_argument("--awgn-cn-db", type=float)
    parser.add_argument("--cfo-hz", type=float)
    parser.add_argument("--cpe-initial-rad", type=float)
    parser.add_argument("--cpe-step-std-rad", type=float)
    parser.add_argument("--sample-rate-offset-ppm", type=float)
    parser.add_argument("--timing-offset-samples", type=float)
    parser.add_argument(
        "--echo",
        action="append",
        default=[],
        help="Echo as DELAY_SAMPLES:ATTENUATION_DB[:PHASE_RAD]. Repeatable.",
    )
    parser.add_argument("--dc-i", type=float)
    parser.add_argument("--dc-q", type=float)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--no-clip", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = preset_config(args.preset)
    impairments = _apply_cli_overrides(preset.impairments, args)
    receiver_equalizer = args.receiver_equalizer or preset.receiver_equalizer
    correct_cpe = bool(args.correct_per_frame_cpe or preset.correct_per_frame_cpe)
    report = run_simulation(
        output_prefix=args.output_prefix,
        payload_ts_path=args.payload_ts,
        payload_frames=args.payload_frames,
        system_info_index=args.system_info_index,
        impairments=impairments,
        rng_seed=args.rng_seed,
        amplitude=args.amplitude,
        run_receiver=args.run_receiver,
        visuals=args.visuals,
        receiver_equalizer=receiver_equalizer,
        receiver_pn_channel_taps=args.receiver_pn_channel_taps,
        receiver_pn_estimator=args.receiver_pn_estimator,
        receiver_pn_equalizer_noise_variance=args.receiver_pn_mmse,
        receiver_csi_demap=args.receiver_csi_demap,
        correct_per_frame_cpe=correct_cpe,
        preset_name=args.preset,
        expected_recovery=preset.expected_recovery,
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report["verdict"]["ok"] else 2


def run_simulation(
    *,
    output_prefix: str | Path,
    payload_ts_path: str | Path | None = None,
    payload_frames: int = 1,
    system_info_index: int = 23,
    impairments: ChannelImpairments | None = None,
    rng_seed: int = 20260523,
    amplitude: float = 0.75,
    run_receiver: bool = True,
    visuals: bool = True,
    receiver_equalizer: str = "none",
    receiver_pn_channel_taps: int | None = None,
    receiver_pn_estimator: str = "compact",
    receiver_pn_equalizer_noise_variance: float | None = None,
    receiver_csi_demap: bool = False,
    correct_per_frame_cpe: bool = False,
    preset_name: str = "custom",
    expected_recovery: str = "pass",
) -> dict[str, Any]:
    """Generate a simulation artifact bundle and return its report."""

    if payload_frames <= 0:
        raise ValueError("payload_frames must be positive")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    impairments = impairments or ChannelImpairments()
    qam_mode, fec_rate_index, interleaver_mode = _system_info_index_to_parameters(
        system_info_index
    )
    config = ContinuousStreamConfig(
        qam_mode=qam_mode,
        fec_rate_index=fec_rate_index,
        interleaver_mode=interleaver_mode,  # type: ignore[arg-type]
        interleaver_phase=0,
    )
    payload = _load_or_generate_payload(
        payload_ts_path,
        payload_frames=payload_frames,
        frame_bytes=config.transport_bytes_per_frame,
    )
    clean = synthesize_pn945_c3780_complex(
        payload,
        config=config,
        system_info_index=system_info_index,
        amplitude=amplitude,
        rng_seed=rng_seed,
    )
    impaired, channel_report = apply_channel_impairments(
        clean["samples"],
        impairments=impairments,
        sample_rate_sps=SAMPLE_RATE_SPS,
        frame_symbols=FRAME_SYMBOLS,
        seed=rng_seed + 1,
    )
    ci8_path = prefix.with_suffix(".ci8")
    _write_ci8(ci8_path, impaired)
    ci8_sidecar = _write_ci8_sidecar(
        ci8_path,
        sample_count=int(impaired.size),
        simulation_preset=preset_name,
        system_info_index=system_info_index,
        impairments=impairments,
    )
    capture_analysis = analyze_ci8_capture(
        ci8_path,
        sample_rate_sps=SAMPLE_RATE_SPS,
        metadata_path=ci8_sidecar,
        max_samples=min(int(impaired.size), 2_000_000),
    )

    recovered_ts_path = prefix.with_suffix(".recovered.ts")
    receiver_json_path = prefix.with_suffix(".receiver.json")
    pre_ldpc_path = prefix.with_suffix(".pre_ldpc.npz")
    receiver_report: dict[str, Any] | None = None
    receiver_error: str | None = None
    if run_receiver:
        try:
            receiver_report = receive_capture(
                ci8_path,
                sample_rate_sps=SAMPLE_RATE_SPS,
                symbol_rate_sps=SAMPLE_RATE_SPS,
                max_samples=int(impaired.size),
                mode=PN_MODE,
                phase_offset=0,
                max_frames=int(clean["total_frames"]),
                qam_mode=qam_mode,
                equalizer=receiver_equalizer,
                pn_channel_taps=receiver_pn_channel_taps,
                pn_estimator=receiver_pn_estimator,
                pn_equalizer_noise_variance=receiver_pn_equalizer_noise_variance,
                csi_demap=receiver_csi_demap,
                symbol_deinterleave=interleaver_mode,
                timing_search=False,
                fec_mode="ldpc",
                fec_rate_index=fec_rate_index,
                system_info_index=system_info_index,
                correct_per_frame_cpe=correct_per_frame_cpe,
                min_ts_packets=max(1, len(payload) // TS_PACKET_SIZE),
                min_ts_sync_ratio=0.0,
                min_ts_valid_ratio=0.0,
                output_path=recovered_ts_path,
                pre_ldpc_dump_path=pre_ldpc_path,
            )
            _write_json(receiver_json_path, receiver_report)
        except Exception as exc:  # pragma: no cover - artifact boundary
            receiver_error = f"{type(exc).__name__}: {exc}"
            _write_json(receiver_json_path, {"error": receiver_error})

    ts_report = _recovered_ts_report(recovered_ts_path, expected_payload=payload)
    sim_json_path = prefix.with_suffix(".sim.json")
    visual_path = prefix.with_suffix(".sim.png")
    report: dict[str, Any] = {
        "schema": SIM_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preset": preset_name,
        "tx": {
            "system_info_index": int(system_info_index),
            "pn_mode": PN_MODE,
            "frame_body_mode": "C3780",
            "qam_mode": qam_mode,
            "fec_rate_index": int(fec_rate_index),
            "interleaver_mode": interleaver_mode,
            "payload_frames": int(payload_frames),
            "drain_frames": int(config.pre_roll_frames),
            "total_frames": int(clean["total_frames"]),
            "transport_bytes": int(len(payload)),
            "header_to_body_power_ratio": float(
                clean["header_to_body_power_ratio"]
            ),
        },
        "channel": channel_report,
        "capture": {
            "file": capture_analysis.get("file"),
            "iq": capture_analysis.get("iq"),
            "spectrum": capture_analysis.get("spectrum"),
        },
        "receiver": _compact_receiver_report(receiver_report, receiver_error),
        "ts": ts_report,
        "artifacts": {
            "ci8": str(ci8_path),
            "ci8_json": str(ci8_sidecar),
            "sim_json": str(sim_json_path),
            "receiver_json": str(receiver_json_path),
            "recovered_ts": str(recovered_ts_path),
            "pre_ldpc_npz": str(pre_ldpc_path) if pre_ldpc_path.exists() else None,
            "pre_ldpc_json": str(pre_ldpc_path.with_suffix(".json"))
            if pre_ldpc_path.with_suffix(".json").exists()
            else None,
            "visual": str(visual_path) if visuals else None,
        },
    }
    report["verdict"] = _simulation_verdict(report, expected_recovery=expected_recovery)
    if visuals:
        try:
            report["artifacts"]["visual_status"] = write_simulation_visual(
                visual_path,
                capture_path=ci8_path,
                clean_samples=clean["samples"],
                impaired_samples=impaired,
                system_info_index=system_info_index,
                channel_report=channel_report,
                receiver_report=receiver_report,
                ts_report=ts_report,
            )
        except Exception as exc:  # pragma: no cover - artifact boundary
            report["artifacts"]["visual_status"] = f"error: {type(exc).__name__}: {exc}"
    _write_json(sim_json_path, report)
    return report


def preset_config(name: str) -> SimulationPreset:
    """Return a named simulator preset."""

    if name == "clean":
        return SimulationPreset(name, ChannelImpairments(), expected_recovery="pass")
    if name == "awgn":
        return SimulationPreset(name, ChannelImpairments(awgn_cn_db=28.0))
    if name == "echo-in-guard":
        # In-guard multipath needs PN-header channel estimation: the sparse/DD
        # equalizers undersample the channel ripple (a 160-sample echo puts a
        # ~24-carrier notch pattern across 3780 carriers, far finer than the 36
        # system-info pilots can track) and lock to a low-EVM but bit-wrong
        # solution. The pn equalizer's PN945 CIR + TDS-OFDM circular restore
        # recovers the transport exactly.
        return SimulationPreset(
            name,
            ChannelImpairments(echoes=(EchoTap(160.0, -14.0, 0.35),)),
            receiver_equalizer="pn",
        )
    if name == "echo-out-of-guard":
        return SimulationPreset(
            name,
            ChannelImpairments(
                echoes=(EchoTap(1200.0, -8.0, 0.2),),
                awgn_cn_db=32.0,
            ),
            receiver_equalizer="dd",
            expected_recovery="may_fail",
        )
    if name == "cfo-cpe":
        return SimulationPreset(
            name,
            ChannelImpairments(cfo_hz=120.0, cpe_initial_rad=0.15, cpe_step_std_rad=0.018),
            correct_per_frame_cpe=True,
        )
    if name == "sample-drift":
        return SimulationPreset(
            name,
            ChannelImpairments(sample_rate_offset_ppm=18.0, awgn_cn_db=35.0),
            expected_recovery="may_fail",
        )
    if name == "urban-stress":
        return SimulationPreset(
            name,
            ChannelImpairments(
                awgn_cn_db=24.0,
                cfo_hz=250.0,
                cpe_initial_rad=0.25,
                cpe_step_std_rad=0.035,
                sample_rate_offset_ppm=8.0,
                timing_offset_samples=0.35,
                echoes=(
                    EchoTap(96.0, -9.0, 0.0),
                    EchoTap(620.0, -15.0, 1.7),
                ),
                dc_i=0.01,
                dc_q=-0.008,
                gain=1.05,
            ),
            receiver_equalizer="dd",
            correct_per_frame_cpe=True,
            expected_recovery="may_fail",
        )
    raise ValueError(f"unknown preset: {name}")


def synthesize_pn945_c3780_complex(
    transport_payload: bytes,
    *,
    config: ContinuousStreamConfig,
    system_info_index: int,
    amplitude: float,
    rng_seed: int,
) -> dict[str, Any]:
    """Return complex baseband samples before RF impairments and quantization."""

    if len(transport_payload) == 0:
        raise ValueError("transport_payload must not be empty")
    if len(transport_payload) % config.transport_bytes_per_frame:
        raise ValueError("transport_payload must contain whole C=3780 frames")
    payload_frames = len(transport_payload) // config.transport_bytes_per_frame
    rng = np.random.default_rng(rng_seed)
    drain_payload = rng.integers(
        0,
        256,
        size=config.pre_roll_frames * config.transport_bytes_per_frame,
        dtype=np.uint8,
    ).tobytes()
    joined = transport_payload + drain_payload
    total_frames = payload_frames + config.pre_roll_frames
    frame_payloads = [
        joined[
            index * config.transport_bytes_per_frame :
            (index + 1) * config.transport_bytes_per_frame
        ]
        for index in range(total_frames)
    ]
    interleaved = encode_continuous_symbol_stream(frame_payloads, config=config)
    system_info = system_info_symbols(
        SYSTEM_INFO_VECTORS[system_info_index],
        frame_body_mode="C3780",
    )
    bodies: list[np.ndarray] = []
    for frame_index in range(total_frames):
        logical = np.empty(FRAME_BODY_SYMBOLS, dtype=np.complex64)
        logical[:36] = system_info
        start = frame_index * DATA_SYMBOLS_PER_FRAME
        stop = start + DATA_SYMBOLS_PER_FRAME
        logical[36:] = interleaved[start:stop]
        bodies.append(np.fft.ifft(frequency_interleave(logical)).astype(np.complex64))
    body_power = float(np.mean([np.mean(np.abs(body) ** 2) for body in bodies]))
    pn_header = pn_header_symbols_for_body_power(PN_MODE, body_power=body_power)
    frames = [np.concatenate((pn_header, body)) for body in bodies]
    samples = np.concatenate(frames).astype(np.complex64, copy=False)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.0:
        samples = (samples / peak * float(amplitude)).astype(np.complex64)
    return {
        "samples": samples,
        "payload_frames": int(payload_frames),
        "drain_frames": int(config.pre_roll_frames),
        "total_frames": int(total_frames),
        "symbols_per_frame": int(FRAME_SYMBOLS),
        "header_to_body_power_ratio": float(
            np.mean(np.abs(pn_header) ** 2) / max(body_power, 1e-24)
        ),
    }


def apply_channel_impairments(
    samples: np.ndarray,
    *,
    impairments: ChannelImpairments,
    sample_rate_sps: int = SAMPLE_RATE_SPS,
    frame_symbols: int = FRAME_SYMBOLS,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic RF impairments and return samples plus truth report."""

    rng = np.random.default_rng(seed)
    output = np.asarray(samples, dtype=np.complex64).reshape(-1).copy()
    input_power = _mean_power(output)
    if impairments.timing_offset_samples:
        output = fractional_delay(output, float(impairments.timing_offset_samples))
    if impairments.echoes:
        echoed = output.astype(np.complex64, copy=True)
        for tap in impairments.echoes:
            echoed = echoed + tap.complex_gain * fractional_delay(
                output,
                float(tap.delay_samples),
            )
        output = echoed.astype(np.complex64)
    if impairments.cfo_hz:
        n = np.arange(output.size, dtype=np.float64)
        output = (
            output
            * np.exp(2j * np.pi * float(impairments.cfo_hz) * n / sample_rate_sps)
        ).astype(np.complex64)

    cpe_phases = np.empty(0, dtype=np.float32)
    if impairments.cpe_initial_rad or impairments.cpe_step_std_rad:
        frame_count = int(np.ceil(output.size / frame_symbols))
        steps = rng.normal(
            0.0,
            float(impairments.cpe_step_std_rad),
            size=max(0, frame_count - 1),
        )
        cpe_phases = np.empty(frame_count, dtype=np.float32)
        cpe_phases[0] = float(impairments.cpe_initial_rad)
        if frame_count > 1:
            cpe_phases[1:] = cpe_phases[0] + np.cumsum(steps).astype(np.float32)
        phase_per_sample = np.repeat(cpe_phases, frame_symbols)[: output.size]
        output = (output * np.exp(1j * phase_per_sample)).astype(np.complex64)

    if impairments.sample_rate_offset_ppm:
        output = resample_by_ppm(output, float(impairments.sample_rate_offset_ppm))
    if impairments.awgn_cn_db is not None:
        signal_power = _mean_power(output)
        noise_power = signal_power / (10.0 ** (float(impairments.awgn_cn_db) / 10.0))
        noise = (
            rng.normal(0.0, np.sqrt(noise_power / 2.0), size=output.size)
            + 1j * rng.normal(0.0, np.sqrt(noise_power / 2.0), size=output.size)
        ).astype(np.complex64)
        output = (output + noise).astype(np.complex64)
    if impairments.gain != 1.0:
        output = (output * float(impairments.gain)).astype(np.complex64)
    if impairments.dc_i or impairments.dc_q:
        output = (
            output + np.complex64(float(impairments.dc_i) + 1j * float(impairments.dc_q))
        ).astype(np.complex64)
    if impairments.clip:
        output = np.clip(output.real, -1.0, 1.0) + 1j * np.clip(
            output.imag,
            -1.0,
            1.0,
        )
        output = output.astype(np.complex64)

    report = {
        "sample_rate_sps": int(sample_rate_sps),
        "input_samples": int(np.asarray(samples).size),
        "output_samples": int(output.size),
        "impairments": impairments.to_dict(),
        "input_power": input_power,
        "output_power": _mean_power(output),
        "output_peak": float(np.max(np.abs(output))) if output.size else 0.0,
        "cpe_phase_rad": [float(value) for value in cpe_phases[:128]],
        "cpe_frame_count": int(cpe_phases.size),
        "guard_interval_symbols": 945,
        "echoes_outside_guard_count": sum(
            1 for tap in impairments.echoes if abs(float(tap.delay_samples)) > 945.0
        ),
    }
    return output.astype(np.complex64, copy=False), report


def fractional_delay(samples: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay complex samples by a fractional number of samples using interpolation."""

    values = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if values.size == 0 or delay_samples == 0.0:
        return values.copy()
    positions = np.arange(values.size, dtype=np.float64) - float(delay_samples)
    x = np.arange(values.size, dtype=np.float64)
    real = np.interp(positions, x, values.real, left=0.0, right=0.0)
    imag = np.interp(positions, x, values.imag, left=0.0, right=0.0)
    return (real + 1j * imag).astype(np.complex64)


def resample_by_ppm(samples: np.ndarray, ppm: float) -> np.ndarray:
    """Apply a simple sample-rate offset by linear interpolation."""

    values = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if values.size == 0 or ppm == 0.0:
        return values.copy()
    ratio = 1.0 + float(ppm) * 1e-6
    output_size = max(1, int(round(values.size / ratio)))
    positions = np.arange(output_size, dtype=np.float64) * ratio
    x = np.arange(values.size, dtype=np.float64)
    real = np.interp(positions, x, values.real, left=0.0, right=0.0)
    imag = np.interp(positions, x, values.imag, left=0.0, right=0.0)
    return (real + 1j * imag).astype(np.complex64)


def write_simulation_visual(
    path: str | Path,
    *,
    capture_path: str | Path,
    clean_samples: np.ndarray,
    impaired_samples: np.ndarray,
    system_info_index: int,
    channel_report: dict[str, Any],
    receiver_report: dict[str, Any] | None,
    ts_report: dict[str, Any],
) -> str:
    """Write a best-effort multi-panel simulation PNG."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable"

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = compute_spectrum_trace(
        impaired_samples,
        sample_rate_sps=SAMPLE_RATE_SPS,
        fft_size=min(16_384, max(16, int(impaired_samples.size))),
        point_count=512,
        segment_count=8,
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.reshape(-1)
    _plot_spectrum(axes[0], trace)
    _plot_constellation(axes[1], receiver_report)
    _plot_echoes(axes[2], channel_report)
    _plot_cpe(axes[3], channel_report)
    _plot_ldpc(axes[4], receiver_report)
    _plot_ts(axes[5], ts_report)
    fig.suptitle(
        f"DTMB simulation PN945/C=3780 system-info {system_info_index}: "
        f"{Path(capture_path).name}"
    )
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return "rendered"


def _plot_spectrum(axis: Any, trace: dict[str, Any]) -> None:
    freqs = np.asarray(trace.get("frequency_hz") or [], dtype=np.float64) / 1_000_000.0
    power = np.asarray(trace.get("power_db") or [], dtype=np.float64)
    if freqs.size and power.size:
        axis.plot(freqs, power, color="#087f8c", linewidth=1.2)
    axis.set_title("FFT spectrum")
    axis.set_xlabel("MHz offset")
    axis.set_ylabel("dB")
    axis.grid(True, alpha=0.25)


def _plot_constellation(axis: Any, receiver_report: dict[str, Any] | None) -> None:
    quality = (receiver_report or {}).get("qam_symbol_quality") or {}
    after = quality.get("after_symbol_deinterleave") or {}
    text = "\n".join(
        (
            "Constellation quality",
            f"EVM: {_fmt(after.get('grid_evm_rms'))}",
            f"inner frac: {_fmt((after.get('axis_level_occupancy') or {}).get('i', {}).get('inner_fraction'))}",
        )
    )
    axis.text(0.05, 0.8, text, va="top", transform=axis.transAxes)
    axis.set_axis_off()


def _plot_echoes(axis: Any, channel_report: dict[str, Any]) -> None:
    echoes = channel_report.get("impairments", {}).get("echoes") or []
    if echoes:
        delays = [0.0] + [float(tap["delay_samples"]) for tap in echoes]
        gains = [0.0] + [float(tap["attenuation_db"]) for tap in echoes]
        axis.stem(delays, gains)
        axis.axvline(945, color="#e76f51", linestyle="--", linewidth=1.0)
    axis.set_title("Echo profile")
    axis.set_xlabel("delay symbols")
    axis.set_ylabel("dBc")
    axis.grid(True, alpha=0.25)


def _plot_cpe(axis: Any, channel_report: dict[str, Any]) -> None:
    phases = np.asarray(channel_report.get("cpe_phase_rad") or [], dtype=np.float64)
    if phases.size:
        axis.plot(np.arange(phases.size), phases, color="#264653")
    axis.set_title("CPE trajectory")
    axis.set_xlabel("frame")
    axis.set_ylabel("rad")
    axis.grid(True, alpha=0.25)


def _plot_ldpc(axis: Any, receiver_report: dict[str, Any] | None) -> None:
    parity = ((receiver_report or {}).get("fec") or {}).get("ldpc_parity_check") or {}
    counts = parity.get("mismatch_counts") or parity.get("mismatches") or []
    if counts:
        axis.hist(counts, bins=min(20, max(1, len(counts))), color="#2a9d8f")
    axis.set_title(f"LDPC mismatch mean {_fmt(parity.get('mean_mismatch_ratio'))}")
    axis.set_xlabel("mismatches/codeword")
    axis.set_ylabel("count")
    axis.grid(True, alpha=0.25)


def _plot_ts(axis: Any, ts_report: dict[str, Any]) -> None:
    lines = [
        "TS status",
        f"byte exact: {ts_report.get('byte_exact_match')}",
        f"recovered bytes: {ts_report.get('recovered_bytes')}",
        f"packets: {(ts_report.get('stream') or {}).get('packet_count')}",
        f"continuity errors: {(ts_report.get('stream') or {}).get('continuity_error_count')}",
    ]
    axis.text(0.05, 0.8, "\n".join(lines), va="top", transform=axis.transAxes)
    axis.set_axis_off()


def _load_or_generate_payload(
    payload_ts_path: str | Path | None,
    *,
    payload_frames: int,
    frame_bytes: int,
) -> bytes:
    if payload_ts_path is not None:
        payload = Path(payload_ts_path).read_bytes()
        if not payload:
            raise ValueError("payload TS is empty")
        if len(payload) % frame_bytes:
            raise ValueError(
                f"payload TS length must be a multiple of {frame_bytes} bytes"
            )
        return payload
    packet_count = payload_frames * frame_bytes // TS_PACKET_SIZE
    if packet_count * TS_PACKET_SIZE != payload_frames * frame_bytes:
        raise ValueError("frame payload is not an integer number of TS packets")
    return b"".join(
        _ts_packet(pid=0x0100, continuity_counter=index & 0x0F, unit_start=index == 0)
        for index in range(packet_count)
    )


def _ts_packet(*, pid: int, continuity_counter: int, unit_start: bool) -> bytes:
    header = bytes(
        (
            TS_SYNC_BYTE,
            (0x40 if unit_start else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0x10 | (continuity_counter & 0x0F),
        )
    )
    payload = bytes((pid + continuity_counter + index) & 0xFF for index in range(184))
    return header + payload


def _write_ci8(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, dtype=np.complex64)
    ci8 = np.empty(values.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(values.real * 127.0), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(values.imag * 127.0), -128, 127).astype(np.int8)
    path.write_bytes(ci8.tobytes())


def _write_ci8_sidecar(
    path: Path,
    *,
    sample_count: int,
    simulation_preset: str,
    system_info_index: int,
    impairments: ChannelImpairments,
) -> Path:
    sidecar = path.with_suffix(path.suffix + ".json")
    _write_json(
        sidecar,
        {
            "format": "ci8",
            "sample_rate_sps": SAMPLE_RATE_SPS,
            "bandwidth_hz": SAMPLE_RATE_SPS,
            "frequency_hz": 0,
            "byte_count": sample_count * 2,
            "sample_count": sample_count,
            "duration_s": sample_count / SAMPLE_RATE_SPS,
            "antenna": "synthetic-simulator",
            "location": "simulated",
            "simulation_preset": simulation_preset,
            "system_info_index": int(system_info_index),
            "impairments": impairments.to_dict(),
        },
    )
    return sidecar


def _recovered_ts_report(path: Path, *, expected_payload: bytes) -> dict[str, Any]:
    data = path.read_bytes() if path.exists() else b""
    packet_count = len(data) // TS_PACKET_SIZE
    packets = [
        data[index * TS_PACKET_SIZE : (index + 1) * TS_PACKET_SIZE]
        for index in range(packet_count)
    ]
    stream = analyze_ts_packets(packets).to_dict() if packets else None
    return {
        "recovered_bytes": int(len(data)),
        "expected_bytes": int(len(expected_payload)),
        "byte_exact_match": bool(data == expected_payload),
        "first_mismatch_byte": _first_mismatch(data, expected_payload),
        "stream": stream,
    }


def _compact_receiver_report(
    receiver_report: dict[str, Any] | None,
    receiver_error: str | None,
) -> dict[str, Any]:
    if receiver_report is None:
        return {"ran": False, "error": receiver_error}
    fec = receiver_report.get("fec") or {}
    parity = fec.get("ldpc_parity_check") or {}
    frame_reports = fec.get("frame_reports") or []
    sysinfo = receiver_report.get("system_info") or {}
    return {
        "ran": True,
        "error": receiver_error,
        "acquisition": receiver_report.get("acquisition"),
        "system_info_selected": sysinfo.get("selected"),
        "ldpc_mean_mismatch_ratio": parity.get("mean_mismatch_ratio"),
        "ldpc_zero_mismatch_codewords": parity.get("zero_mismatch_codewords"),
        "ldpc_codewords": int(
            sum(int(row.get("ldpc_codewords", 0) or 0) for row in frame_reports)
        ),
        "ldpc_converged_codewords": int(
            sum(
                int(row.get("ldpc_converged_codewords", 0) or 0)
                for row in frame_reports
            )
        ),
        "bch_unclean_blocks": int(
            sum(int(row.get("bch_unclean_blocks", 0) or 0) for row in frame_reports)
        ),
        "ts_lock": (receiver_report.get("ts") or {}).get("lock"),
        "symbol_deinterleave_discarded_symbols": receiver_report.get(
            "symbol_deinterleave_discarded_symbols"
        ),
    }


def _simulation_verdict(report: dict[str, Any], *, expected_recovery: str) -> dict[str, Any]:
    ts_ok = bool(report["ts"].get("byte_exact_match"))
    receiver = report.get("receiver") or {}
    sysinfo = receiver.get("system_info_selected") or {}
    system_info_ok = sysinfo.get("index") == report["tx"]["system_info_index"]
    ldpc_codewords = int(receiver.get("ldpc_codewords", 0) or 0)
    ldpc_ok = (
        ldpc_codewords > 0
        and int(receiver.get("ldpc_converged_codewords", 0) or 0) == ldpc_codewords
        and int(receiver.get("bch_unclean_blocks", 0) or 0) == 0
    )
    expected_pass = expected_recovery == "pass"
    ok = ts_ok and system_info_ok and ldpc_ok if expected_pass else True
    return {
        "ok": bool(ok),
        "expected_recovery": expected_recovery,
        "byte_exact_ts": ts_ok,
        "system_info_match": bool(system_info_ok),
        "ldpc_clean": bool(ldpc_ok),
    }


def _apply_cli_overrides(
    base: ChannelImpairments,
    args: argparse.Namespace,
) -> ChannelImpairments:
    echoes = tuple(_parse_echo(value) for value in args.echo) if args.echo else base.echoes
    return ChannelImpairments(
        awgn_cn_db=base.awgn_cn_db if args.awgn_cn_db is None else args.awgn_cn_db,
        cfo_hz=base.cfo_hz if args.cfo_hz is None else args.cfo_hz,
        cpe_initial_rad=base.cpe_initial_rad
        if args.cpe_initial_rad is None
        else args.cpe_initial_rad,
        cpe_step_std_rad=base.cpe_step_std_rad
        if args.cpe_step_std_rad is None
        else args.cpe_step_std_rad,
        sample_rate_offset_ppm=base.sample_rate_offset_ppm
        if args.sample_rate_offset_ppm is None
        else args.sample_rate_offset_ppm,
        timing_offset_samples=base.timing_offset_samples
        if args.timing_offset_samples is None
        else args.timing_offset_samples,
        echoes=echoes,
        dc_i=base.dc_i if args.dc_i is None else args.dc_i,
        dc_q=base.dc_q if args.dc_q is None else args.dc_q,
        gain=base.gain if args.gain is None else args.gain,
        clip=base.clip and not args.no_clip,
    )


def _parse_echo(value: str) -> EchoTap:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("echo must be DELAY:ATTENUATION[:PHASE]")
    delay = float(parts[0])
    attenuation = float(parts[1])
    phase = float(parts[2]) if len(parts) == 3 else 0.0
    return EchoTap(delay, attenuation, phase)


def _mean_power(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.complex64).reshape(-1)
    return float(np.mean(np.abs(array) ** 2)) if array.size else 0.0


def _first_mismatch(a: bytes, b: bytes) -> int | None:
    compared = min(len(a), len(b))
    for index in range(compared):
        if a[index] != b[index]:
            return index
    if len(a) != len(b):
        return compared
    return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.4g}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
