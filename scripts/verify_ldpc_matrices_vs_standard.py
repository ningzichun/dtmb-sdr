#!/usr/bin/env python
"""Verify stored LDPC generator/parity matrices against GB 20600-2006 text.

The self-consistent synthetic loopback proves our generator (Appendix A) and
parity-check (Appendix B) matrices are *mutually* consistent, but NOT that they
match the normative GB 20600-2006 tables. If both were transcribed from a wrong
source, the synthetic round-trip still passes byte-exact while every real
broadcast codeword fails LDPC parity at the 0.50 random plateau -- exactly the
fingerprint observed on the HK captures.

This script parses Appendix A (``G[i][j] : <32 hex>``) and Appendix B
(``A[r][c] = shift``) directly from ``references/GB_20600-2006.txt`` and
compares them entry-for-entry against ``python/dtmb/data/dtmb_ldpc_generator.json``
and ``python/dtmb/data/dtmb_ldpc_parity_check.json``.

Each appendix block is preceded by an ``LDPC(7493, NNNN)`` header that selects
the FEC rate (3048->rate1, 4572->rate2, 6096->rate3).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STD = next((ROOT / "references").glob("GB_20600*2006.txt"))
GEN_JSON = ROOT / "python" / "dtmb" / "data" / "dtmb_ldpc_generator.json"
PAR_JSON = ROOT / "python" / "dtmb" / "data" / "dtmb_ldpc_parity_check.json"

# LDPC(7493, message_bits) -> fec rate index
MSG_TO_RATE = {3048: 1, 4572: 2, 6096: 3}
# (k rows, c columns) of the generator circulant grid per rate
GEN_SHAPE = {1: (24, 35), 2: (36, 23), 3: (48, 11)}
# parity check: c rows, (c+k) columns
PAR_SHAPE = {1: (35, 59), 2: (23, 59), 3: (11, 59)}

G_RE = re.compile(r"G\[\s*(\d+)\]\[\s*(\d+)\]\s*:\s*([0-9A-Fa-f]{32})")
A_RE = re.compile(r"A\[\s*(\d+)\]\[\s*(\d+)\]\s*=\s*(\d+)")
HDR_RE = re.compile(r"LDPC\(\s*7493\s*,\s*(\d+)\s*\)")


def _rate_segments(text: str) -> list[tuple[int, int, int]]:
    """Return (rate_index, start_pos, end_pos) for each LDPC(7493,N) header."""
    headers = [(m.start(), int(m.group(1))) for m in HDR_RE.finditer(text)]
    segs = []
    for idx, (pos, msg) in enumerate(headers):
        if msg not in MSG_TO_RATE:
            continue
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
        segs.append((MSG_TO_RATE[msg], pos, end))
    return segs


def parse_generator(text: str) -> dict[int, dict[tuple[int, int], str]]:
    """Parse Appendix A generator hex per rate.

    The generator section appears before the parity 'LDPC(7493,N)' header
    introduces the parity block; but both G[][] and A[][] entries can share
    the same 'LDPC(7493,N)' segment. We collect G entries per segment.
    """
    out: dict[int, dict[tuple[int, int], str]] = {1: {}, 2: {}, 3: {}}
    for rate, start, end in _rate_segments(text):
        for m in G_RE.finditer(text, start, end):
            i, j, hexval = int(m.group(1)), int(m.group(2)), m.group(3).upper()
            out[rate][(i, j)] = hexval
    return out


def parse_parity(text: str) -> dict[int, dict[tuple[int, int], int]]:
    out: dict[int, dict[tuple[int, int], int]] = {1: {}, 2: {}, 3: {}}
    for rate, start, end in _rate_segments(text):
        for m in A_RE.finditer(text, start, end):
            r, c, shift = int(m.group(1)), int(m.group(2)), int(m.group(3))
            out[rate][(r, c)] = shift
    return out


def main() -> int:
    text = STD.read_text(encoding="utf-8", errors="replace")
    gen_std = parse_generator(text)
    par_std = parse_parity(text)
    gen_stored = json.loads(GEN_JSON.read_text(encoding="utf-8"))
    par_stored = json.loads(PAR_JSON.read_text(encoding="utf-8"))

    print("=" * 70)
    print("GENERATOR (Appendix A) — stored JSON vs GB 20600-2006 text")
    print("=" * 70)
    gen_ok = True
    for rate in (1, 2, 3):
        rows = gen_stored[str(rate)]
        k, c = GEN_SHAPE[rate]
        std = gen_std[rate]
        n_std = len(std)
        mism = []
        checked = 0
        for i in range(k):
            for j in range(c):
                if (i, j) in std:
                    checked += 1
                    stored_hex = rows[i][j].upper()
                    if stored_hex != std[(i, j)]:
                        mism.append((i, j, stored_hex, std[(i, j)]))
        print(
            f"rate{rate}: grid {k}x{c}={k*c}, parsed_from_std={n_std}, "
            f"compared={checked}, mismatches={len(mism)}"
        )
        for i, j, a, b in mism[:5]:
            print(f"   G[{i}][{j}] stored={a} std={b}")
        if mism or checked == 0:
            gen_ok = False

    print()
    print("=" * 70)
    print("PARITY CHECK (Appendix B) — stored JSON vs GB 20600-2006 text")
    print("=" * 70)
    par_ok = True
    for rate in (1, 2, 3):
        rows = par_stored[str(rate)]  # list of rows; each row = list of [col, shift]
        stored_map = {}
        for r, row in enumerate(rows):
            for col, shift in row:
                stored_map[(r, int(col))] = int(shift)
        std = par_std[rate]
        # Compare in both directions.
        only_std = [(rc, v) for rc, v in std.items() if rc not in stored_map]
        only_stored = [(rc, v) for rc, v in stored_map.items() if rc not in std]
        valdiff = [
            (rc, stored_map[rc], std[rc])
            for rc in std
            if rc in stored_map and stored_map[rc] != std[rc]
        ]
        print(
            f"rate{rate}: std_entries={len(std)}, stored_entries={len(stored_map)}, "
            f"value_mismatches={len(valdiff)}, only_in_std={len(only_std)}, "
            f"only_in_stored={len(only_stored)}"
        )
        for rc, a, b in valdiff[:5]:
            print(f"   A[{rc[0]}][{rc[1]}] stored={a} std={b}")
        for rc, v in only_std[:5]:
            print(f"   MISSING from stored: A[{rc[0]}][{rc[1]}] = {v}")
        for rc, v in only_stored[:5]:
            print(f"   EXTRA in stored: A[{rc[0]}][{rc[1]}] = {v}")
        if valdiff or only_std or only_stored or len(std) == 0:
            par_ok = False

    print()
    print("=" * 70)
    print(f"GENERATOR match: {'OK' if gen_ok else 'MISMATCH'}")
    print(f"PARITY match:    {'OK' if par_ok else 'MISMATCH'}")
    print("=" * 70)
    return 0 if (gen_ok and par_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
