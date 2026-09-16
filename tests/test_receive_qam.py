import subprocess

import numpy as np
import pytest

from dtmb.ts import analyze_ts_packets
from synthetic_support import DATA, ci8_frames, modulate, symbol_stream


@pytest.mark.parametrize("rate", [1, 2, 3])
@pytest.mark.parametrize("mode,phase", [("mode1", 17), ("mode2", 9)])
@pytest.mark.parametrize("noise", [0.0, 0.65])
def test_16qam_continuous_fec_exact_ts(native, rate, mode, phase, noise):
    stream, bits, expected = symbol_stream(rate, 16, mode, phase)
    if noise:
        rng = np.random.default_rng(734)
        stream += (noise * (rng.normal(size=stream.size) + 1j * rng.normal(size=stream.size))).astype(np.complex64)
    result = subprocess.run(native("deinterleave_qam", "--qam", "16qam", "--mode", mode,
                                  "--phase", phase, "--chunk-symbols", "4093",
                                  "--workers", "3", "--noise-variance", "0.845", "-", "-"),
                            input=stream.tobytes(), capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    llr = np.frombuffer(result.stdout, dtype=np.float32)
    assert llr.size == bits.size == 4 * 14976
    if noise:
        assert np.count_nonzero((llr < 0) != bits) > 0
    else:
        np.testing.assert_array_equal(llr < 0, bits)
    fec = subprocess.run(native("ldpc_bch_decode", "--fec-rate", rate,
                                "--alist", DATA / f"dtmb_ldpc_rate{rate}.alist",
                                "--codewords-per-frame", 2, "--workers", 3,
                                "--clean-frames-only", "--fail-on-unclean-frame", "--require-output", "-", "-"),
                         input=result.stdout, capture_output=True, timeout=90)
    assert fec.returncode == 0, fec.stderr.decode(errors="replace")
    assert fec.stdout == expected and len(expected) > 0
    assert b"clean_frames=4\n" in fec.stderr.replace(b"\r\n", b"\n")
    assert b"converged_codewords=8" in fec.stderr
    report = analyze_ts_packets(fec.stdout[n:n+188] for n in range(0, len(fec.stdout), 188)).to_dict()
    assert report["continuity_error_count"] == 0
    assert report["transport_error_count"] == 0


def test_16qam_all_bit_planes_and_legacy_executable(native):
    bits = np.array([[(label >> bit) & 1 for bit in range(4)] for label in range(16)], dtype=np.uint8).ravel()
    symbols = modulate(bits, 16)
    for exe in ("qam_softdemap", "deinterleave_qam64"):
        args = ["--qam", "16qam", "--chunk-symbols", "3"]
        if exe == "deinterleave_qam64":
            # Branch zero has no deinterleaver delay when phase=51.
            args += ["--mode", "mode1", "--keep-latency", "--phase", "51"]
            input_symbols = np.repeat(symbols, 52)
        else:
            input_symbols = symbols
        result = subprocess.run(native(exe, *args, "-", "-"), input=input_symbols.tobytes(), capture_output=True, check=True)
        llr = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 4)
        if exe == "deinterleave_qam64":
            llr = llr[::52]
        np.testing.assert_array_equal(llr.ravel() < 0, bits)


def test_16qam_weights_confidence_and_llr_scale_follow_symbols(native, tmp_path):
    phase = 17
    stream, bits, _ = symbol_stream(2, 16, "mode1", phase)
    args = ("deinterleave_qam", "--qam", "16qam", "--mode", "mode1",
            "--phase", phase, "--chunk-symbols", "65536")
    plain = subprocess.run(native(*args, "-", "-"), input=stream.tobytes(), capture_output=True, check=True)
    csi = np.random.default_rng(302).uniform(0.1, 1.5, stream.size).astype(np.float32)
    path = tmp_path / "weights.f32"
    csi.tofile(path)
    weighted = subprocess.run(native(*args, "--csi-weights", path,
                                      "--source-frame-confidence", "inverse-mse",
                                      "--source-frame-llr-scale", "17:0:100000:0.5", "-", "-"),
                              input=stream.tobytes(), capture_output=True, check=True)
    n = np.arange(bits.size // 4)
    branches = (n + phase) % 52
    source = n + 52 * 51 * 240 - (51 - branches) * 240 * 52
    weight = csi[source] * 4 * np.where(branches == 17, 0.5, 1.0)
    expected = np.frombuffer(plain.stdout, dtype=np.float32).reshape(-1, 4) * weight[:, None]
    np.testing.assert_allclose(np.frombuffer(weighted.stdout, dtype=np.float32).reshape(-1, 4), expected, rtol=1e-6)


def test_empty_output_is_failure(native):
    result = subprocess.run(native("ldpc_bch_decode", "--fec-rate", 2,
                                   "--alist", DATA / "dtmb_ldpc_rate2.alist",
                                   "--codewords-per-frame", 2, "--require-output", "-", "-"),
                            input=b"", capture_output=True)
    assert result.returncode != 0 and result.stdout == b""
    assert b"no transport bytes recovered" in result.stderr


@pytest.mark.parametrize("profile", [11, 14, 15])
def test_16qam_ci8_frontend_uses_selected_constellation(native, profile, tmp_path):
    rate = (profile - 11) // 2 + 1
    mode = "mode1" if profile % 2 else "mode2"
    symbols, bits, expected = symbol_stream(rate, 16, mode)
    diagnostics = tmp_path / "residuals.csv"
    result = subprocess.run(native("c3780_extract", "--qam", "16qam", "--system-info-index", profile,
                                   "--normalization", "qam", "--frame-residual-diagnostics", diagnostics,
                                   "--workers", 3, "-", "-"),
                            input=ci8_frames(symbols, profile, 16), capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"qam=16qam" in result.stderr
    assert "slot2" not in diagnostics.read_text().splitlines()[0]
    demap = subprocess.run(native("deinterleave_qam", "--qam", "16qam", "--mode", mode, "-", "-"),
                           input=result.stdout, capture_output=True, check=True, timeout=90)
    np.testing.assert_array_equal(np.frombuffer(demap.stdout, dtype=np.float32) < 0, bits)
    fec = subprocess.run(native("ldpc_bch_decode", "--fec-rate", rate,
                                "--alist", DATA / f"dtmb_ldpc_rate{rate}.alist",
                                "--codewords-per-frame", 2, "--clean-frames-only",
                                "--fail-on-unclean-frame", "--require-output", "-", "-"),
                         input=demap.stdout, capture_output=True, check=True, timeout=90)
    assert fec.stdout == expected


@pytest.mark.parametrize("flag", ["--qam64-integer-timing-correction", "--data-dd-refine",
                                   "--timing-trajectory-local-search"])
def test_16qam_rejects_qam64_calibration(native, flag):
    result = subprocess.run(native("c3780_extract", "--qam", "16qam", "--system-info-index", 11,
                                   "--normalization", "qam", flag, "-", "-"), input=b"", capture_output=True)
    assert result.returncode != 0 and b"not supported for 16QAM" in result.stderr


def test_explicit_16qam_selects_corresponding_default_profile(native):
    bits = np.random.default_rng(113).integers(0, 2, 3744 * 4, dtype=np.uint8)
    result = subprocess.run(native("c3780_extract", "--qam", "16qam", "--normalization", "qam", "-", "-"),
                            input=ci8_frames(modulate(bits, 16), 15, 16), capture_output=True, check=True)
    assert b"system_info_index=15" in result.stderr
    demap = subprocess.run(native("qam_softdemap", "--qam", "16qam", "-", "-"),
                           input=result.stdout, capture_output=True, check=True)
    np.testing.assert_array_equal(np.frombuffer(demap.stdout, dtype=np.float32) < 0, bits)
