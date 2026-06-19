"""Export DTMB LDPC parity-check matrices in the MacKay .alist format.

External LDPC decoders and simulation frameworks (AFF3CT, the Python
`ldpc` package, Radford Neal's LDPC tools, liquid-dsp's LDPC benchmarks)
consume the `.alist` representation of a Tanner graph. Exporting our
Appendix-B parity-check shifts in this format gives us a reproducible
way to feed the exact GB 20600-2006 matrices to an independent decoder
and score the same LLR dump the dtmb receiver produces. That decouples
"is our parity-check matrix transcription correct" from "is our decoder
implementation correct" without requiring an entire reference DTMB
receiver to exist.

The emitted format follows MacKay's specification (column and row
definitions, one-based variable indexing, zero-padded to the max column
weight / max row weight so downstream tools can parse the blocks).

Output files are intentionally placed under ``python/dtmb/data`` so the
exported artifacts sit next to the JSON definitions they are derived
from, and third-party tools can read them without any Python runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np

from dtmb.ldpc import (  # noqa: E402
    _dtmb_ldpc_graph_cached,
    dtmb_ldpc_profile,
)


def _graph_adjacency(graph) -> tuple[list[list[int]], list[list[int]]]:
    """Return per-variable and per-check neighbor lists as 1-based indices."""

    variable_neighbors: list[list[int]] = [[] for _ in range(graph.variable_count)]
    check_neighbors: list[list[int]] = [[] for _ in range(graph.check_count)]
    for check_index, edges in enumerate(graph.check_edges):
        for edge in edges:
            variable = int(graph.edge_variables[int(edge)])
            check_neighbors[check_index].append(variable + 1)
            variable_neighbors[variable].append(check_index + 1)
    for entries in variable_neighbors:
        entries.sort()
    for entries in check_neighbors:
        entries.sort()
    return variable_neighbors, check_neighbors


def format_alist(graph) -> str:
    """Produce a MacKay .alist representation for a Tanner graph."""

    variable_neighbors, check_neighbors = _graph_adjacency(graph)
    column_weights = [len(entries) for entries in variable_neighbors]
    row_weights = [len(entries) for entries in check_neighbors]
    max_column_weight = max(column_weights)
    max_row_weight = max(row_weights)

    lines: list[str] = []
    lines.append(f"{graph.variable_count} {graph.check_count}")
    lines.append(f"{max_column_weight} {max_row_weight}")
    lines.append(" ".join(str(value) for value in column_weights))
    lines.append(" ".join(str(value) for value in row_weights))
    # Zero-padded column neighbor lists (one per column).
    for entries in variable_neighbors:
        padded = entries + [0] * (max_column_weight - len(entries))
        lines.append(" ".join(str(value) for value in padded))
    # Zero-padded row neighbor lists (one per row).
    for entries in check_neighbors:
        padded = entries + [0] * (max_row_weight - len(entries))
        lines.append(" ".join(str(value) for value in padded))
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_ldpc_alist",
        description=(
            "Emit MacKay .alist files for the DTMB QC-LDPC mother codes "
            "so third-party decoders can verify our parity-check layout."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python/dtmb/data"),
        help="Directory to write dtmb_ldpc_rate{1,2,3}.alist into",
    )
    parser.add_argument(
        "--fec-rate",
        type=int,
        choices=(1, 2, 3),
        action="append",
        help="One or more FEC rate indices to export (default: all three)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rates = args.fec_rate or (1, 2, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        profile = dtmb_ldpc_profile(rate)
        graph = _dtmb_ldpc_graph_cached(rate)
        text = format_alist(graph)
        filename = args.output_dir / f"dtmb_ldpc_rate{rate}.alist"
        filename.write_text(text, encoding="ascii", newline="\n")
        print(
            f"wrote {filename} ({graph.variable_count}x{graph.check_count}, "
            f"{graph.edge_count} edges, message {profile.message_bits} bits)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
