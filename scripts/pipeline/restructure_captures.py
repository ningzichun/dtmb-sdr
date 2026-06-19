"""Move existing capture files into captures/raw and captures/analysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
RAW_SUFFIXES = (".ci8", ".ci8.json")
RAW_INPUT_NAMES = {"demo_input.ts", "hackrf_sweep.csv"}
SKIP_DIRS = {"raw", "analysis"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="restructure-captures")
    parser.add_argument("--captures-dir", type=Path, default=Path("captures"))
    parser.add_argument("--execute", action="store_true", help="Apply moves; otherwise dry-run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    captures_dir = _resolve_under_root(args.captures_dir)
    raw_dir = captures_dir / "raw"
    analysis_dir = captures_dir / "analysis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    moves = _plan_moves(captures_dir, raw_dir, analysis_dir)
    for src, dst in moves:
        if dst.exists():
            raise SystemExit(f"destination already exists: {dst}")
        if args.execute:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if args.execute:
        _remove_empty_dirs(captures_dir)
    print(
        f"[restructure-captures] {'moved' if args.execute else 'would move'} "
        f"{len(moves)} files"
    )
    return 0


def _plan_moves(captures_dir: Path, raw_dir: Path, analysis_dir: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for path in sorted(captures_dir.rglob("*")):
        rel = path.relative_to(captures_dir)
        if not rel.parts:
            continue
        if rel.parts[0] in SKIP_DIRS:
            continue
        if path.is_dir():
            continue
        if rel == Path("README.md"):
            continue
        destination_root = raw_dir if _is_raw_input(path) else analysis_dir
        moves.append((path, destination_root / rel))
    return moves


def _is_raw_input(path: Path) -> bool:
    name = path.name
    return name.endswith(RAW_SUFFIXES) or name in RAW_INPUT_NAMES


def _remove_empty_dirs(captures_dir: Path) -> None:
    for path in sorted(captures_dir.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if not path.is_dir():
            continue
        rel = path.relative_to(captures_dir)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def _resolve_under_root(path: Path) -> Path:
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"refusing path outside repository: {resolved}") from exc
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
