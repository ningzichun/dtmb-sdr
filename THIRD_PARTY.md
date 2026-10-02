# Third-Party References

## Local ZIP extraction

The static web build bundles [fflate](https://github.com/101arrowz/fflate) under
MIT for local ZIP extraction. Its license is distributed as
`vendor/fflate-LICENSE.txt`. Dependencies are pinned in `web/package-lock.json`.

## CUDA runtime

The optional `cuda` extra depends on NVIDIA's separately distributed
`nvidia-cuda-runtime-cu12` package and its NVIDIA license. Its runtime libraries
are not bundled in the DTMB wheel. CUDA-enabled builds use NVIDIA's CUDA 12
Toolkit; the DTMB LDPC implementation is project source compiled into an
optional shared backend.

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
| `core/cpp/tests/nr_appendix_c.txt` | Hex input/output pairs | Complete GB 20600-2006 Appendix C NR mapping table; independent test oracle for the equation-based receiver. The fixture header records the source-text SHA-256. |
