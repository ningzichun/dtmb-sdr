import subprocess
import sys

import numpy as np
import pytest

from dtmb.ts import analyze_ts_packets
from synthetic_support import PN, ci8_frames, modulate, symbol_stream


@pytest.mark.parametrize("pn_mode", PN)
@pytest.mark.parametrize("qam,profile", [(16, 11), (64, 22)])
def test_selected_pn_ci8_to_c3780(native, pn_mode, qam, profile):
    width = 4 if qam == 16 else 6
    bits = np.random.default_rng(455).integers(0, 2, 4 * 3744 * width, dtype=np.uint8)
    symbols = modulate(bits, qam)
    result = subprocess.run(native("c3780_extract", "--pn-mode", pn_mode,
                                   "--system-info-index", profile, "--normalization", "qam",
                                   "--auto-sync", "--sync-frames", 4, "--workers", 3, "-", "-"),
                            input=ci8_frames(symbols, profile, qam, pn_mode=pn_mode,
                                             scheduled=True, leading=37, cfo=1200),
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert f"pn_mode={pn_mode}".encode() in result.stderr
    assert f"pn_frame_symbols={PN[pn_mode][0] + 3780}".encode() in result.stderr
    demap = subprocess.run(native("qam_softdemap", "--qam", f"{qam}qam", "-", "-"),
                           input=result.stdout, capture_output=True, check=True)
    np.testing.assert_array_equal(np.frombuffer(demap.stdout, dtype=np.float32) < 0, bits)


@pytest.mark.parametrize("pn_mode", PN)
@pytest.mark.parametrize("profile", range(11, 17))
def test_pn_16qam_profiles_exact_ts(native_bin, pn_mode, profile, tmp_path):
    rate = (profile - 11) // 2 + 1
    interleaver = "mode1" if profile % 2 else "mode2"
    symbols, _, expected = symbol_stream(rate, 16, interleaver)
    # A complete following frame supplies the receiver's next-header context.
    symbols = np.concatenate((symbols, symbols[-3744:]))
    samples = ci8_frames(symbols, profile, 16, pn_mode=pn_mode, scheduled=True, noise=0.001)
    command = [sys.executable, "-m", "dtmb.decode", "--input-rate", "7560000",
               "--system-info-index", str(profile), "--pn-mode", pn_mode,
               "--bin-dir", str(native_bin), "--workers", "3", "--pipeline-buffer-mib", "0"]
    if profile % 2:
        path = tmp_path / "synthetic.ci8"
        path.write_bytes(samples)
        command += ["--input", str(path)]
        samples = None
    result = subprocess.run(command, input=samples, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == expected and len(result.stdout) > 0
    assert b"clean_frames=4\n" in result.stderr.replace(b"\r\n", b"\n")
    assert b"converged_codewords=8" in result.stderr
    report = analyze_ts_packets(result.stdout[n:n + 188] for n in range(0, len(result.stdout), 188)).to_dict()
    assert report["continuity_error_count"] == 0 and report["transport_error_count"] == 0


@pytest.mark.parametrize("pn_mode", PN)
@pytest.mark.parametrize("qam,profile", [(16, 15), (64, 22)])
def test_pn_multipath_cfo_exact_ts(native_bin, pn_mode, qam, profile):
    rate = 3 if qam == 16 else 2
    interleaver = "mode1" if profile % 2 else "mode2"
    symbols, _, expected = symbol_stream(rate, qam, interleaver)
    symbols = np.concatenate((symbols, symbols[-3744:]))
    channel = np.zeros(14, dtype=np.complex64)
    channel[0], channel[3], channel[13] = 1, 0.12 - 0.08j, 0.04 + 0.07j
    samples = ci8_frames(symbols, profile, qam, pn_mode=pn_mode, scheduled=True,
                         leading=37, cfo=275, noise=0.0005, channel=channel)
    result = subprocess.run([sys.executable, "-m", "dtmb.decode", "--input-rate", "7560000",
                             "--system-info-index", str(profile), "--pn-mode", pn_mode,
                             "--bin-dir", str(native_bin), "--workers", "3", "--pipeline-buffer-mib", "0"],
                            input=samples, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == expected
    assert b"clean_frames=4\n" in result.stderr.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("argument,value", [("--pn-wideband-header-observation", "core-postfix"),
                                           ("--pn-wideband-header-observation", "core-cyclic-safe"),
                                           ("--pn-schedule-tracking", None)])
def test_pn595_rejects_cyclic_assumptions(native, argument, value):
    args = [argument] + ([value] if value else [])
    result = subprocess.run(native("c3780_extract", "--pn-mode", "pn595", *args, "-", "-"),
                            input=b"", capture_output=True)
    assert result.returncode != 0 and b"PN595" in result.stderr
    assert result.stdout == b""


@pytest.mark.parametrize("pn_mode", PN)
def test_pn_timing_trajectory_preserves_symbols_across_sample_displacement(native, pn_mode):
    frames = 280
    bits = np.random.default_rng(257).integers(0, 2, (frames + 1) * 3744 * 4, dtype=np.uint8)
    samples = ci8_frames(modulate(bits, 16), 11, 16, pn_mode=pn_mode, scheduled=True, leading=37)
    slip_byte = (37 + 243 * (PN[pn_mode][0] + 3780)) * 2
    samples = samples[:slip_byte] + b"\0" * 4 + samples[slip_byte:]
    args = ["--pn-mode", pn_mode, "--system-info-index", 11, "--normalization", "qam",
            "--auto-sync", "--sync-frames", 240, "--equalizer", "pn", "--pn-estimator", "wideband",
            "--pn-wideband-block-frames", 2, "--pn-wideband-scale-estimator", "masked-frame-taps",
            "--timing-search-radius", 3, "--timing-trajectory-interval-frames", 20,
            "--timing-trajectory-max-innovation-samples", 3, "--workers", 3]
    if pn_mode != "pn595":
        args += ["--pn-schedule-tracking", "--pn-current-header-tracking"]
    result = subprocess.run(native("c3780_extract", *args, "-", "-"),
                            input=samples, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    demap = subprocess.run(native("qam_softdemap", "--qam", "16qam", "-", "-"),
                           input=result.stdout, capture_output=True, check=True)
    np.testing.assert_array_equal(np.frombuffer(demap.stdout, dtype=np.float32) < 0, bits[:frames * 3744 * 4])
    if pn_mode != "pn595":
        assert b"pn_schedule_locked=true" in result.stderr
