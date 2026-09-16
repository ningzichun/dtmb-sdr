"""Hardware-free 32QAM checks with an independent standard H-matrix encoder."""

from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dtmb.ts import analyze_ts_packets
from synthetic_support import DATA, PN, ci8_frames, modulate, symbol_stream


@lru_cache(maxsize=4)
def _stream(mode, phase=0):
    return symbol_stream(3, 32, mode, phase)


def _fec(native, *args):
    return native("ldpc_bch_decode", "--qam", "32qam", "--fec-rate", 3,
                  "--alist", DATA / "dtmb_ldpc_rate3.alist", "--codewords-per-frame", 5,
                  "--clean-frames-only", "--require-output", "--workers", 3,
                  "--decode-batch-frames", 1, *args, "-", "-")


@pytest.mark.parametrize("mode,phase", [("mode1", 17), ("mode2", 9)])
@pytest.mark.parametrize("start_frame", [0, 1])
@pytest.mark.parametrize("noise", [0, 0.65])
def test_32qam_continuous_cross_frame_fec(native, mode, phase, start_frame, noise):
    stream, bits, expected = _stream(mode, phase)
    stream = stream[start_frame * 3744:].copy()
    if noise:
        rng = np.random.default_rng(932)
        stream += (noise * (rng.normal(size=stream.size) + 1j * rng.normal(size=stream.size))).astype(np.complex64)
    demap = subprocess.run(native("deinterleave_qam", "--qam", "32qam", "--mode", mode,
                                  "--phase", phase, "--chunk-symbols", 4093, "--workers", 3, "-", "-"),
                           input=stream.tobytes(), capture_output=True, timeout=90)
    assert demap.returncode == 0, demap.stderr.decode(errors="replace")
    llr = np.frombuffer(demap.stdout, dtype=np.float32)
    wanted_bits = bits[start_frame * 18720:]
    assert llr.size == wanted_bits.size
    if noise:
        assert np.count_nonzero((llr < 0) != wanted_bits) > 0
    else:
        np.testing.assert_array_equal(llr < 0, wanted_bits)
    fec = subprocess.run(_fec(native, "--fail-on-unclean-frame"), input=demap.stdout,
                         capture_output=True, timeout=90)
    assert fec.returncode == 0, fec.stderr.decode(errors="replace")
    assert fec.stdout == expected[2 * start_frame * 1880:]
    assert f"packing_discarded_signal_frames={start_frame}".encode() in fec.stderr
    report = analyze_ts_packets(fec.stdout[n:n + 188] for n in range(0, len(fec.stdout), 188))
    assert report.sync_error_count == report.continuity_error_count == report.transport_error_count == 0


@pytest.mark.parametrize("pn_mode", ["pn420", "pn595", "pn945"])
@pytest.mark.parametrize("profile", [17, 18])
@pytest.mark.parametrize("start_frame", [0, 1])
def test_32qam_cli_ci8_to_exact_ts(native_bin, tmp_path, pn_mode, profile, start_frame):
    mode = "mode1" if profile == 17 else "mode2"
    symbols, _, expected = _stream(mode)
    # One following source frame supplies wideband PN boundary context.
    guard = modulate(np.random.default_rng(139).integers(0, 2, 3744 * 5, dtype=np.uint8), 32)
    capture = ci8_frames(np.r_[symbols, guard], profile, 32, pn_mode=pn_mode,
                         scheduled=True, leading=37, cfo=600)
    if start_frame:
        capture = capture[2 * (37 + PN[pn_mode][0] + 3780):]
    input_path = tmp_path / "32qam.ci8"
    input_path.write_bytes(capture)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "python")
    result = subprocess.run([
        sys.executable, "-m", "dtmb.decode", "--bin-dir", str(native_bin),
        "--input", str(input_path) if profile == 17 else "-", "--output", "-",
        "--input-rate", "7560000", "--pn-mode", pn_mode, "--system-info-index", str(profile),
        "--workers", "3", "--max-iterations", "50", "--pipeline-buffer-mib", "1",
    ], input=capture if profile == 18 else None, capture_output=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == expected[2 * start_frame * 1880:]
    assert b"scrambler_reset_bits=15040" in result.stderr
    assert f"packing_discarded_signal_frames={start_frame}".encode() in result.stderr
    report = analyze_ts_packets(result.stdout[n:n + 188] for n in range(0, len(result.stdout), 188))
    assert report.sync_error_count == report.continuity_error_count == report.transport_error_count == 0


def test_32qam_strict_failure_keeps_only_the_clean_group(native):
    _, bits, expected = _stream("mode1")
    llr = (1 - 2 * bits.astype(np.float32)) * 8
    llr[7 * 7488:8 * 7488] = np.random.default_rng(65).normal(size=7488)
    result = subprocess.run(_fec(native, "--fail-on-unclean-frame"), input=llr.tobytes(),
                            capture_output=True, timeout=90)
    assert result.returncode != 0
    assert result.stdout == expected[:3760]


@pytest.mark.parametrize("case", ["odd-tail", "ambiguous", "nonfinite"])
def test_32qam_rejects_incomplete_or_ambiguous_stream(native, case):
    _, bits, expected = _stream("mode1")
    llr = (1 - 2 * bits.astype(np.float32)) * 8
    if case == "odd-tail":
        llr = llr[:-18720]
    elif case == "ambiguous":
        llr[:] = 0
    else:
        llr[0] = np.nan
    result = subprocess.run(_fec(native, "--decode-batch-frames", 256), input=llr.tobytes(), capture_output=True, timeout=90)
    assert result.returncode != 0
    if case != "odd-tail":
        assert result.stdout == b""
    else:
        assert result.stdout == expected[:3760]
