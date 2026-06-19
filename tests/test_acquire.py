import numpy as np

from dtmb.acquire import refine_acquisition, sweep_acquisition
from dtmb.pn import pn_bipolar, pn_cyclic_family_bipolar


def test_refine_acquisition_reports_delay_tolerant_family_score(tmp_path):
    family = pn_cyclic_family_bipolar("pn945")
    reference = family[17].astype(np.complex64)
    header = reference.copy()
    header[30:] += 0.4 * reference[:-30]
    rng = np.random.default_rng(123)
    body = (
        rng.normal(scale=0.05, size=3780)
        + 1j * rng.normal(scale=0.05, size=3780)
    ).astype(np.complex64)
    signal = np.tile(np.concatenate((header, body)), 4)
    path = tmp_path / "synthetic.ci8"
    _write_ci8(path, signal * 0.5)

    diagnostics = refine_acquisition(
        path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=signal.size,
        mode="pn945",
        family_frames=3,
        max_delay_symbols=64,
    )

    selected = diagnostics["selected"]
    delay_family = selected["delay_family_train"]
    assert selected["mode"] == "pn945"
    assert selected["phase_offset"] == 0
    assert delay_family["hit_count"] == 3
    assert delay_family["mean_metric"] > 0.85
    assert delay_family["delay_mad_symbols"] <= 1.0
    assert delay_family["dominant_pn_phase_count"] == 3


def test_sweep_acquisition_ranks_delay_family_results(tmp_path):
    family = pn_cyclic_family_bipolar("pn945")
    reference = family[5].astype(np.complex64)
    rng = np.random.default_rng(456)
    body = (
        rng.normal(scale=0.05, size=3780)
        + 1j * rng.normal(scale=0.05, size=3780)
    ).astype(np.complex64)
    signal = np.tile(np.concatenate((reference, body)), 3)
    path = tmp_path / "synthetic.ci8"
    _write_ci8(path, signal * 0.5)

    diagnostics = sweep_acquisition(
        path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=signal.size - 1,
        frequency_shifts_hz=(0.0,),
        input_skip_samples=(1, 0),
        mode="pn945",
        family_frames=2,
        max_delay_symbols=32,
    )

    assert diagnostics["selected"]["input_skip_samples"] == 0
    family_score = diagnostics["selected"]["selected"]["delay_family_train"]
    assert family_score["mean_metric"] > 0.9


def test_refine_acquisition_supports_pn595_direct_train(tmp_path):
    reference = pn_bipolar("pn595").astype(np.complex64)
    rng = np.random.default_rng(789)
    body = (
        rng.normal(scale=0.05, size=3780)
        + 1j * rng.normal(scale=0.05, size=3780)
    ).astype(np.complex64)
    leading = np.zeros(17, dtype=np.complex64)
    signal = np.concatenate((leading, np.tile(np.concatenate((reference, body)), 4)))
    path = tmp_path / "synthetic_pn595.ci8"
    _write_ci8(path, signal * 0.5)

    diagnostics = refine_acquisition(
        path,
        sample_rate_sps=7_560_000,
        symbol_rate_sps=7_560_000,
        max_samples=signal.size,
        mode="pn595",
        family_frames=3,
        max_delay_symbols=64,
    )

    selected = diagnostics["selected"]
    assert selected["mode"] == "pn595"
    assert selected["phase_offset"] == 17
    assert selected["acquisition_kind"] == "direct-pn-correlation"
    assert selected["hit_count"] == 4
    assert selected["mean_metric"] > 0.99


def _write_ci8(path, samples):
    interleaved = np.empty(samples.size * 2, dtype=np.int8)
    interleaved[0::2] = np.clip(np.round(samples.real * 127), -128, 127).astype(np.int8)
    interleaved[1::2] = np.clip(np.round(samples.imag * 127), -128, 127).astype(np.int8)
    path.write_bytes(interleaved.tobytes())
