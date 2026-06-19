"""Quick-read scan_*.acquire.json files and print best PN945 metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def best_metric(data: dict, mode: str, group: str) -> float:
    best = 0.0
    trains = (data.get("pn_search") or {}).get(group) or []
    for t in trains:
        if t.get("mode") == mode:
            m = float(t.get("max_metric") or 0.0)
            if m > best:
                best = m
    return best


def main() -> int:
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("captures/scan_20260510")
    rows = []
    for p in sorted(scan_dir.glob("scan_*.acquire.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        row = {
            "file": p.name,
            "pn945_cyc_ext": round(best_metric(data, "pn945", "cyclic_extension_trains"), 3),
            "pn945_frame": round(best_metric(data, "pn945", "frame_trains"), 3),
            "pn945_family": round(best_metric(data, "pn945", "phase_family_trains"), 3),
            "pn420_cyc_ext": round(best_metric(data, "pn420", "cyclic_extension_trains"), 3),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["pn945_cyc_ext"], reverse=True)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
