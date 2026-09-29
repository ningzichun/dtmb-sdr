# Third-Party References

## Browser playback

The static web build bundles [mpegts.js 1.8.0](https://github.com/xqq/mpegts.js)
under Apache-2.0 for optional local MPEG-TS preview. Its license is distributed
as `vendor/mpegts-LICENSE.txt`. Dependencies are pinned in `web/package-lock.json`.
It handles media playback, independently of the DTMB receiver.

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
