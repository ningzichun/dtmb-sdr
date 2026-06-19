"""Stage 4: ffmpeg verify and transcode recovered MPEG-TS to MP4.

Inputs:
    <capture>.recovered.ts

Outputs:
    <capture>.probe.json   (ffprobe structured output, or an empty-status
                            JSON if the TS carries no decodable streams)
    <capture>.mp4          (only written when ffprobe finds at least one
                            valid elementary stream and ffmpeg succeeds)

An empty / garbage TS is not an error for this stage: we record the
probe verdict and exit 0 so `make` continues. The `--require-video`
flag promotes "no usable streams" into a hard failure; use it on
loopback targets where ffmpeg output is a required gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from _common import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline-transcode")
    parser.add_argument("--input-ts", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument(
        "--require-video",
        action="store_true",
        help="Exit non-zero if ffprobe finds no streams or ffmpeg fails.",
    )
    return parser


def _ffprobe(input_ts: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_ts),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        return {
            "ok": False,
            "reason": "ffprobe_error",
            "stderr": str(exc),
            "streams": [],
        }
    if result.returncode != 0 or not result.stdout.strip():
        return {"ok": False, "stderr": result.stderr.strip(), "streams": []}
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "stderr": f"json decode: {exc}", "streams": []}
    parsed["ok"] = bool(parsed.get("streams"))
    return parsed


def _remove_stale_output(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.probe_json.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_output(args.output_mp4)
    if not args.input_ts.exists() or args.input_ts.stat().st_size == 0:
        write_json(
            args.probe_json,
            {
                "ok": False,
                "reason": "recovered_ts_missing_or_empty",
                "input": str(args.input_ts),
            },
        )
        return 1 if args.require_video else 0

    probe = _ffprobe(args.input_ts, args.ffprobe_bin)
    probe["input"] = str(args.input_ts)
    write_json(args.probe_json, probe)
    if not probe["ok"]:
        print(
            f"[transcode] ffprobe rejected {args.input_ts}: "
            f"{probe.get('stderr', '')}",
            file=sys.stderr,
        )
        return 1 if args.require_video else 0

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(args.input_ts),
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(args.output_mp4),
    ]
    print("[transcode]", " ".join(cmd), file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        rc = int(result.returncode)
        probe["ffmpeg"] = {
            "ok": rc == 0,
            "returncode": rc,
            "stderr": result.stderr.strip(),
        }
    except OSError as exc:
        rc = 127
        probe["ffmpeg"] = {
            "ok": False,
            "returncode": rc,
            "stderr": str(exc),
        }
    write_json(args.probe_json, probe)
    if rc != 0 and args.require_video:
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
