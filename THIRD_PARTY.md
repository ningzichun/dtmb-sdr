# Third-Party References

## Standard

| Name | Purpose | Reuse |
|------|---------|-------|
| GB 20600-2006 | Normative DTMB standard: PN sequences, constellation mappings, LDPC/BCH parameters, system information tables | Receiver constants and LDPC interoperability matrices |

All DTMB-specific algorithms are reimplemented from the standard, not copied from other projects.

## Derived Artifacts

Files generated from GB 20600-2006 (not third-party code):

| Artifact | Format | Purpose |
|----------|--------|---------|
| `python/dtmb/data/dtmb_ldpc_rate{1,2,3}.alist` | MacKay .alist | LDPC parity-check matrices for interoperability with external decoders (AFF3CT, Radford Neal's tools, etc.) |
