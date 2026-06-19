"""Print a one-line TS-lock + LDPC summary from a receiver diagnostics JSON."""

import json
import sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
d = json.load(open(path, encoding="utf-8"))
ts = d.get("ts", {})
lock = ts.get("lock")
pc = d.get("fec", {}).get("ldpc_parity_check", {})
mismatch = pc.get("mean_mismatch_ratio")
zero_cw = pc.get("zero_mismatch_codewords")
cw = pc.get("codewords")
lock_str = f"YES {ts.get('packet_count')}pkt sync={lock.get('sync_ratio')}" if lock else "NONE"
print(
    f"{label}: ts_lock={lock_str}  "
    f"ldpc_mean_mismatch={round(mismatch, 4) if mismatch is not None else None}  "
    f"zero_cw={zero_cw}/{cw}"
)
