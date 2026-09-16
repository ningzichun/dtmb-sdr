from pathlib import Path

import pytest

from dtmb import decode


@pytest.mark.parametrize("profile", range(11, 25))
def test_receive_profiles_build_all_stage_parameters(monkeypatch, profile, tmp_path):
    monkeypatch.setattr(decode, "_exe", lambda name, bin_dir: tmp_path / name)
    args = decode.build_parser().parse_args(["--input-rate", "7560000",
                                           "--system-info-index", str(profile),
                                           "--pipeline-buffer-mib", "0"])
    frontend, demap, fec = decode.build_commands(args)
    base = 11 if profile < 17 else 19
    qam = "16qam" if profile < 17 else "32qam" if profile < 19 else "64qam"
    assert frontend[frontend.index("--qam") + 1] == qam
    assert demap[demap.index("--qam") + 1] == qam
    assert demap[demap.index("--mode") + 1] == ("mode1" if profile % 2 else "mode2")
    assert fec[fec.index("--fec-rate") + 1] == ("3" if qam == "32qam" else str(1 + (profile - base) // 2))
    assert fec[fec.index("--codewords-per-frame") + 1] == {"16qam": "2", "32qam": "5", "64qam": "3"}[qam]
    if qam == "32qam":
        assert fec[fec.index("--qam") + 1] == "32qam"
        assert fec[fec.index("--qam32-frame-phase") + 1] == "auto"
    assert "--require-output" in fec and "--fail-on-unclean-frame" in fec
    assert "--qam64-integer-timing-correction" not in frontend


@pytest.mark.parametrize("pn_mode", ["pn420", "pn595", "pn945"])
@pytest.mark.parametrize("profile", range(11, 17))
def test_pn_profile_command_construction(monkeypatch, tmp_path, pn_mode, profile):
    monkeypatch.setattr(decode, "_exe", lambda name, bin_dir: tmp_path / name)
    args = decode.build_parser().parse_args(["--input-rate", "20000000", "--pn-mode", pn_mode,
                                           "--system-info-index", str(profile), "--pipeline-buffer-mib", "0"])
    _, frontend, demap, fec = decode.build_commands(args)
    assert frontend[frontend.index("--pn-mode") + 1] == pn_mode
    assert frontend[frontend.index("--auto-phase-adjustment") + 1] == "0"
    assert frontend[frontend.index("--pn-wideband-header-observation") + 1] == (
        "direct" if pn_mode == "pn595" else "core-postfix")
    assert "--pn-schedule-tracking" not in frontend
    assert demap[demap.index("--qam") + 1] == "16qam"
    assert fec[fec.index("--codewords-per-frame") + 1] == "2"


def test_file_pipeline_is_vendor_neutral(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    suffix = ".exe" if decode.os.name == "nt" else ""
    for name in (
        "dtmb_core_ci8_resample",
        "dtmb_core_c3780_extract",
        "dtmb_core_deinterleave_qam64",
        "dtmb_core_ldpc_bch_decode",
        "dtmb_core_pipe_buffer",
    ):
        (bin_dir / f"{name}{suffix}").write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "dtmb_ldpc_rate2.alist").write_text("fixture", encoding="utf-8")
    parser = decode.build_parser()
    args = parser.parse_args(
        [
            "--input",
            "capture.ci8",
            "--input-rate",
            "16000000",
            "--output",
            "-",
            "--error-policy",
            "continue",
            "--acceleration",
            "cpu",
            "--bin-dir",
            str(bin_dir),
            "--data-dir",
            str(data),
        ]
    )
    commands = decode.build_commands(args)
    assert len(commands) == 7
    assert commands[0][-2:] == ["capture.ci8", "-"]
    assert Path(commands[1][0]).stem == "dtmb_core_pipe_buffer"
    assert commands[-1][-1] == "-"
    assert commands[-1][commands[-1].index("--ldpc-accel") + 1] == "cpu"
    assert commands[-1][commands[-1].index("--decode-batch-frames") + 1] == "256"
    assert "--insert-discontinuity-packets" in commands[-1]


def test_live_receiver_controls_remain_explicit(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    suffix = ".exe" if decode.os.name == "nt" else ""
    for name in (
        "dtmb_core_ci8_resample",
        "dtmb_core_c3780_extract",
        "dtmb_core_deinterleave_qam64",
        "dtmb_core_ldpc_bch_decode",
        "dtmb_core_pipe_buffer",
    ):
        (bin_dir / f"{name}{suffix}").write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "dtmb_ldpc_rate2.alist").write_text("fixture", encoding="utf-8")
    args = decode.build_parser().parse_args([
        "--input-rate", "16000000", "--bin-dir", str(bin_dir), "--data-dir", str(data),
        "--pn-schedule-tracking", "--source-frame-confidence", "inverse-mse",
        "--resample-headroom-db", "6",
    ])
    commands = decode.build_commands(args)
    frontend = next(command for command in commands if Path(command[0]).stem == "dtmb_core_c3780_extract")
    demap = next(command for command in commands if Path(command[0]).stem == "dtmb_core_deinterleave_qam64")
    resample = commands[0]
    assert "--pn-schedule-tracking" in frontend
    assert "--pn-current-header-tracking" in frontend
    assert demap[demap.index("--source-frame-confidence") + 1] == "inverse-mse"
    assert resample[resample.index("--output-scale") + 1] == "0.0747125816"
    assert frontend[frontend.index("--pn-mmse") + 1] == "0.00100475457"


def test_packaged_binary_directory_is_preferred(
    monkeypatch,
    tmp_path: Path,
) -> None:
    suffix = ".exe" if decode.os.name == "nt" else ""
    package_bin = tmp_path / "bin"
    package_bin.mkdir()
    executable = package_bin / f"dtmb_core_ci8_resample{suffix}"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(decode, "__file__", str(tmp_path / "decode.py"))
    assert decode._exe("dtmb_core_ci8_resample", None) == executable.resolve()


def test_explicit_binary_directory_overrides_package(
    monkeypatch,
    tmp_path: Path,
) -> None:
    suffix = ".exe" if decode.os.name == "nt" else ""
    package_bin = tmp_path / "package" / "bin"
    package_bin.mkdir(parents=True)
    (package_bin / f"dtmb_core_ldpc_bch_decode{suffix}").write_bytes(b"cpu")
    external_bin = tmp_path / "cuda" / "Release"
    external_bin.mkdir(parents=True)
    cuda = external_bin / f"dtmb_core_ldpc_bch_decode{suffix}"
    cuda.write_bytes(b"cuda")
    monkeypatch.setattr(decode, "__file__", str(package_bin.parent / "decode.py"))
    assert decode._exe("dtmb_core_ldpc_bch_decode", external_bin.parent) == cuda.resolve()


def test_default_data_directory_is_package_relative(monkeypatch, tmp_path: Path) -> None:
    package_dir = tmp_path / "installed" / "dtmb"
    monkeypatch.setattr(decode, "__file__", str(package_dir / "decode.py"))
    args = decode.build_parser().parse_args(["--input-rate", "7560000"])
    assert args.data_dir == package_dir / "data"
