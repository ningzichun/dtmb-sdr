from pathlib import Path

from dtmb import decode


def test_file_pipeline_is_vendor_neutral(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    suffix = ".exe" if decode.os.name == "nt" else ""
    for name in (
        "dtmb_core_ci8_resample",
        "dtmb_core_c3780_extract",
        "dtmb_core_deinterleave_qam64",
        "dtmb_core_ldpc_bch_decode",
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
    assert len(commands) == 4
    assert commands[0][-2:] == ["capture.ci8", "-"]
    assert commands[-1][-1] == "-"
    assert commands[-1][commands[-1].index("--ldpc-accel") + 1] == "cpu"
    assert commands[-1][commands[-1].index("--decode-batch-frames") + 1] == "256"
    assert "--insert-discontinuity-packets" in commands[-1]
