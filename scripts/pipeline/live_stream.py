"""Stage: capture source -> DTMB receiver -> ffmpeg transcode, all via stdio.

This is the "live video stream" entry point. It wires three subprocesses
into one streaming chain:

    source | dtmb.receiver - --output - | ffmpeg -f mpegts -i - ... -f <fmt> <sink>

No intermediate files. The source is one of:

  - ``--source synthetic``  : ``dtmb.tx_c3780 input.ts --output -`` emits a
    continuous DTMB CI8 baseband. Good for end-to-end verification without
    a HackRF attached.
  - ``--source hackrf``     : ``hackrf_transfer -r -`` streams live IQ.
  - ``--source ci8-file``   : ``cat file.ci8`` replays a recorded capture.

The sink is one of:

  - ``--sink mp4``   : fragmented MP4 written to ``--output``.
  - ``--sink udp``   : MPEG-TS over UDP to ``--udp-url`` (e.g. ``udp://127.0.0.1:5555``).
    Tune VLC / mpv / ffplay to that URL to watch.
  - ``--sink ffplay``: pipe to ``ffplay -`` (no X on Windows headless; requires GUI).
  - ``--sink stdout``: raw MPEG-TS to stdout (for ``... | ffplay -`` composition).

Gating:

  The full streaming decode only produces a playable output on captures
  whose DTMB signal is standard-compliant enough to clear the offline
  chain. On real RF today the LDPC plateau blocks useful video (see
  ``docs/trial-log.md``). The script still runs the full pipe so users
  can probe a live mux and see the exact stage that fails.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from _common import ROOT, child_env


DTMB_SYMBOL_RATE_SPS = 7_560_000
DEFAULT_HACKRF_SAMPLE_RATE = 20_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtmb-live-stream",
        description=(
            "Stream HackRF / file / synthetic IQ through the DTMB receiver "
            "into ffmpeg, with no intermediate files."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("hackrf", "synthetic", "ci8-file"),
        required=True,
    )
    parser.add_argument(
        "--sink",
        choices=("mp4", "udp", "ffplay", "stdout"),
        default="mp4",
    )

    rx_group = parser.add_argument_group("receiver")
    rx_group.add_argument("--sample-rate", type=int, default=DEFAULT_HACKRF_SAMPLE_RATE)
    rx_group.add_argument(
        "--qam",
        default="64qam",
        choices=("4qam", "16qam", "32qam", "64qam"),
    )
    rx_group.add_argument(
        "--symbol-deinterleave",
        default="mode1",
        choices=("none", "mode1", "mode2"),
    )
    rx_group.add_argument(
        "--equalizer",
        default="dd",
        choices=(
            "none",
            "sparse",
            "pn",
            "pn-sparse",
            "dd",
            "pn-dd",
            "dd-data",
            "pn-dd-data",
            "dd-raw",
            "pn-dd-raw",
        ),
    )
    rx_group.add_argument("--fec-rate", type=int, default=3, choices=(1, 2, 3))
    rx_group.add_argument("--system-info-index", type=int, default=23)
    rx_group.add_argument("--frames", type=int, default=600)
    rx_group.add_argument(
        "--correct-per-frame-cpe",
        action="store_true",
        default=True,
    )
    rx_group.add_argument(
        "--no-correct-per-frame-cpe",
        dest="correct_per_frame_cpe",
        action="store_false",
    )
    rx_group.add_argument("--max-samples", type=int, default=40_000_000)
    rx_group.add_argument(
        "--synthetic-loopback",
        action="store_true",
        help=(
            "When --source=synthetic or --source=ci8-file, pin "
            "phase_offset=0, disable equalizer + timing search. Matches "
            "scripts/pipeline/receive.py's synthetic-loopback mode."
        ),
    )

    hackrf = parser.add_argument_group("hackrf source")
    hackrf.add_argument("--frequency", type=int)
    hackrf.add_argument("--hackrf-bin", default="hackrf_transfer")
    hackrf.add_argument("--hackrf-sample-rate", type=int, default=20_000_000)
    hackrf.add_argument(
        "--bandwidth",
        type=int,
        default=8_000_000,
        help="Centered DTMB filter width; 8 MHz limits adjacent-channel ingress.",
    )
    hackrf.add_argument("--amp", type=int, default=0)
    hackrf.add_argument("--lna-gain", type=int, default=16)
    hackrf.add_argument("--vga-gain", type=int, default=16)
    hackrf.add_argument("--hackrf-samples", type=int, default=0,
                        help="If >0, stop hackrf after this many samples (bounded stream).")

    synthetic = parser.add_argument_group("synthetic source")
    synthetic.add_argument("--input-ts", type=Path)
    synthetic.add_argument("--pad-null-packets", action="store_true", default=True)
    synthetic.add_argument("--no-pad-null-packets", dest="pad_null_packets", action="store_false")

    ci8 = parser.add_argument_group("ci8-file source")
    ci8.add_argument("--ci8-input", type=Path)

    sink_group = parser.add_argument_group("ffmpeg sink")
    sink_group.add_argument("--ffmpeg-bin", default="ffmpeg")
    sink_group.add_argument("--ffplay-bin", default="ffplay")
    sink_group.add_argument(
        "--output",
        type=Path,
        help="Output media file when --sink=mp4 (required).",
    )
    sink_group.add_argument(
        "--udp-url",
        default="udp://127.0.0.1:5555?pkt_size=1316",
        help="Destination URL when --sink=udp.",
    )
    sink_group.add_argument(
        "--video-codec",
        default="mpeg4",
        help="ffmpeg video codec (mpeg4 / libx264 / copy).",
    )
    sink_group.add_argument(
        "--audio-codec",
        default="aac",
        help="ffmpeg audio codec (aac / copy / none).",
    )
    sink_group.add_argument(
        "--video-bitrate",
        default="",
        help="Override video bitrate for ffmpeg (default: q:v 5 for mpeg4).",
    )
    return parser


def _build_source_cmd(args: argparse.Namespace) -> list[str]:
    """Return the argv for the upstream IQ producer."""

    if args.source == "hackrf":
        if args.frequency is None:
            raise SystemExit("--source=hackrf requires --frequency")
        cmd = [
            args.hackrf_bin,
            "-r",
            "-",
            "-f",
            str(args.frequency),
            "-s",
            str(args.hackrf_sample_rate),
            "-b",
            str(args.bandwidth),
            "-a",
            str(args.amp),
            "-l",
            str(args.lna_gain),
            "-g",
            str(args.vga_gain),
        ]
        if args.hackrf_samples > 0:
            cmd.extend(["-n", str(args.hackrf_samples)])
        return cmd
    if args.source == "synthetic":
        if not args.input_ts:
            raise SystemExit("--source=synthetic requires --input-ts")
        cmd = [
            sys.executable,
            "-m",
            "dtmb.tx_c3780",
            str(args.input_ts),
            "--output",
            "-",
            "--system-info-index",
            str(args.system_info_index),
        ]
        if args.pad_null_packets:
            cmd.append("--pad-null-packets")
        return cmd
    if args.source == "ci8-file":
        if not args.ci8_input:
            raise SystemExit("--source=ci8-file requires --ci8-input")
        if sys.platform.startswith("win"):
            return ["cmd", "/c", "type", str(args.ci8_input)]
        return ["cat", str(args.ci8_input)]
    raise SystemExit(f"unknown source {args.source}")


def _build_receiver_cmd(args: argparse.Namespace) -> list[str]:
    """Return the argv for the DTMB receiver (reads stdin, writes TS to stdout)."""

    # When streaming from tx_c3780 or a file we know the sample rate is
    # the DTMB symbol rate, so we can skip the polyphase resample and all
    # the acquisition overhead.
    if args.source == "hackrf":
        sample_rate = args.hackrf_sample_rate
    elif args.source == "synthetic":
        sample_rate = DTMB_SYMBOL_RATE_SPS
    else:
        sample_rate = args.sample_rate
    cmd = [
        sys.executable,
        "-m",
        "dtmb.receiver",
        "-",
        "--sample-rate",
        str(sample_rate),
        "--max-samples",
        str(args.max_samples),
        "--mode",
        "pn945",
        "--qam",
        args.qam,
        "--equalizer",
        "none" if args.synthetic_loopback else args.equalizer,
        "--symbol-deinterleave",
        args.symbol_deinterleave,
        "--fec-mode",
        "ldpc",
        "--fec-rate",
        str(args.fec_rate),
        "--system-info-index",
        str(args.system_info_index),
        "--frames",
        str(args.frames),
        "--min-ts-packets",
        "1",
        "--min-ts-sync-ratio",
        "0.0",
        "--min-ts-valid-ratio",
        "0.0",
        "--output",
        "-",
        "--quiet",
    ]
    if args.synthetic_loopback:
        cmd.extend(["--phase-offset", "0", "--no-timing-search"])
    elif args.correct_per_frame_cpe:
        cmd.append("--correct-per-frame-cpe")
    return cmd


def _build_ffmpeg_cmd(args: argparse.Namespace) -> list[str]:
    """Return the argv for the ffmpeg sink (reads TS from stdin)."""

    if args.sink == "ffplay":
        # ffplay doesn't transcode; it decodes and displays directly.
        return [args.ffplay_bin, "-hide_banner", "-loglevel", "warning", "-"]

    base = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "mpegts",
        "-i",
        "-",
    ]
    # Video codec options.
    if args.video_codec == "copy":
        base.extend(["-c:v", "copy"])
    else:
        base.extend(["-c:v", args.video_codec])
        if args.video_bitrate:
            base.extend(["-b:v", args.video_bitrate])
        elif args.video_codec == "mpeg4":
            base.extend(["-q:v", "5"])
    # Audio codec options.
    if args.audio_codec == "none":
        base.append("-an")
    elif args.audio_codec == "copy":
        base.extend(["-c:a", "copy"])
    else:
        base.extend(["-c:a", args.audio_codec, "-b:a", "64k"])

    if args.sink == "mp4":
        if not args.output:
            raise SystemExit("--sink=mp4 requires --output")
        # frag_keyframe/empty_moov make the MP4 readable even if we stop
        # mid-stream (e.g. live preview).
        base.extend([
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "-y",
            str(args.output),
        ])
    elif args.sink == "udp":
        base.extend(["-f", "mpegts", args.udp_url])
    elif args.sink == "stdout":
        base.extend(["-f", "mpegts", "-"])
    else:
        raise SystemExit(f"unknown sink {args.sink}")
    return base


def _run_streaming_pipeline(
    source_cmd: list[str],
    receiver_cmd: list[str],
    ffmpeg_cmd: list[str],
) -> int:
    """Spawn the three subprocesses with stdio pipes and wait for the graph.

    Returns the ffmpeg exit code (the terminal stage) as the pipeline result.
    Upstream failures propagate via broken-pipe closure; we report them as
    warnings but treat the ffmpeg exit code as authoritative because that is
    the stage that owns the sink artifact.
    """

    env = child_env()
    print(f"[live] source : {_pretty(source_cmd)}", file=sys.stderr)
    print(f"[live] receiver: {_pretty(receiver_cmd)}", file=sys.stderr)
    print(f"[live] sink    : {_pretty(ffmpeg_cmd)}", file=sys.stderr)

    source = subprocess.Popen(
        source_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
        env=env,
    )
    assert source.stdout is not None
    receiver = subprocess.Popen(
        receiver_cmd,
        stdin=source.stdout,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(ROOT),
        env=env,
    )
    # Close our handle so SIGPIPE propagates correctly when ffmpeg exits.
    source.stdout.close()
    assert receiver.stdout is not None
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=receiver.stdout,
        stdout=sys.stdout.buffer,
        stderr=sys.stderr,
        cwd=str(ROOT),
        env=env,
    )
    receiver.stdout.close()

    try:
        ffmpeg_rc = ffmpeg.wait()
    except KeyboardInterrupt:
        print("[live] interrupted, stopping pipeline", file=sys.stderr)
        for proc in (ffmpeg, receiver, source):
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
        ffmpeg_rc = ffmpeg.wait()

    # Drain upstream processes with a short timeout so stale hackrf_transfer
    # or tx_c3780 processes don't outlive the pipeline.
    for proc, name in ((receiver, "receiver"), (source, "source")):
        try:
            rc = proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                rc = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
        if rc not in (0, None) and rc != -15:
            print(f"[live] {name} exited with rc={rc}", file=sys.stderr)

    return ffmpeg_rc


def _pretty(cmd: Sequence[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(cmd)
    return " ".join(shlex.quote(part) for part in cmd)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_cmd = _build_source_cmd(args)
    receiver_cmd = _build_receiver_cmd(args)
    ffmpeg_cmd = _build_ffmpeg_cmd(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rc = _run_streaming_pipeline(source_cmd, receiver_cmd, ffmpeg_cmd)
    elapsed = time.monotonic() - start
    print(f"[live] pipeline exit={rc} elapsed={elapsed:.2f}s", file=sys.stderr)
    if args.sink == "mp4" and args.output:
        if args.output.exists():
            print(
                f"[live] wrote {args.output} ({args.output.stat().st_size} bytes)",
                file=sys.stderr,
            )
        else:
            print(f"[live] MP4 not produced at {args.output}", file=sys.stderr)
            rc = rc or 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
