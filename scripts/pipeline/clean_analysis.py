"""Safely remove disposable capture-analysis artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clean-capture-analysis")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("captures") / "analysis",
        help="Analysis directory to clean. Must resolve under captures/analysis.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis_dir = _resolve_under_root(args.analysis_dir)
    captures_root = _resolve_under_root(Path("captures"))
    try:
        relative = analysis_dir.relative_to(captures_root)
    except ValueError as exc:
        raise SystemExit(f"refusing to clean outside captures/: {analysis_dir}") from exc
    if not relative.parts or relative.parts[0] == "raw":
        raise SystemExit(f"refusing to clean raw capture storage: {analysis_dir}")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for child in analysis_dir.iterdir():
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"[clean-analysis] cleaned {analysis_dir}")
    return 0


def _resolve_under_root(path: Path) -> Path:
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"refusing path outside repository: {resolved}") from exc
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
