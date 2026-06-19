"""Check recovered TS bytes against the transmitted payload (byte-exact)."""
import json
import sys

import numpy as np

receive_json = sys.argv[1]
recovered_ts = sys.argv[2]
payload = sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else receive_json

d = json.load(open(receive_json, encoding="utf-8"))
pc = d.get("fec", {}).get("ldpc_parity_check", {})
mismatch = pc.get("mean_mismatch_ratio")
zero_cw = pc.get("zero_mismatch_codewords")
cw = pc.get("codewords")

try:
    rec = open(recovered_ts, "rb").read()
except FileNotFoundError:
    rec = b""
pay = open(payload, "rb").read()

ber = None
found = -1
if rec:
    found = rec.find(pay[:188])
    if found >= 0:
        n = min(len(pay), len(rec) - found)
        a = np.frombuffer(rec[found:found + n], dtype=np.uint8)
        b = np.frombuffer(pay[:n], dtype=np.uint8)
        ber = float(np.mean(a != b))

print(
    f"{label}: ldpc_mismatch={round(mismatch, 4) if mismatch is not None else None} "
    f"zero_cw={zero_cw}/{cw} recovered_bytes={len(rec)} payload_found_at={found} "
    f"byte_error_rate={ber}"
)
