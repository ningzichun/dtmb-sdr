# dtmb-sdr

A vendor-neutral DTMB receiver with a browser application. It accepts interleaved
CU8, CI8/CS8, CI16/SC16 or CF32 samples from a file or standard input and emits MPEG transport
stream packets to a file or standard output.

**[Open DTMB Lab in your browser](https://ningzichun.github.io/dtmb-sdr/)** to
inspect a capture with a spectrum and WebGL waterfall, decode it locally,
preview supported video codecs, and export a diagnostic report before opening
an issue. The browser uses the same C++ receiver compiled to WebAssembly.
See [browser build and format documentation](web/README.md).

The new formats, normal 4QAM and browser app are available from the current
source and Pages deployment. The existing PyPI 0.4.0 release predates these
additions; use a source build for the updated native CLI.

```text
CI8 file/stdin
  -> sample-rate conversion
  -> PN420/PN595/PN945 synchronization and C=3780 equalization
  -> 4QAM/16QAM/32QAM/64QAM deinterleaving and soft demapping
  -> LDPC, BCH, and descrambling
  -> MPEG-TS file/stdout
```

Hardware acquisition is outside this repository. Applications producing raw
interleaved IQ bytes at a known sample rate can feed the receiver.

## Build

Requirements are CMake 3.20+, a C++20 compiler, and Python 3.10+.

For a released platform wheel, installation is simply:

```bash
pip install dtmb-sdr
```

The wheel contains the native receiver executables and required LDPC data; no
separate source checkout or CMake build is required.

Version 0.4.0 adds the 16QAM, 32QAM, PN420 and PN595 receive paths. These paths
have synthetic regression coverage; their over-the-air validation remains pending.

To build from a source checkout:

```bash
python -m pip install -e ".[dev]"
cmake -S core/cpp -B build/core-cpp -DDTMB_CORE_BUILD_TESTS=ON
cmake --build build/core-cpp --config Release
ctest --test-dir build/core-cpp -C Release --output-on-failure
pytest
```

Build a distributable wheel with:

```bash
python -m build --wheel
```

Prebuilt wheels use the portable CPU decoder. CUDA requires a source build with
a supported NVIDIA toolkit.

## Decode a CI8 file

Use `--input-format cu8|ci8|cs8|ci16|sc16|cf32` for the actual export format.
CI16/SC16 and CF32 use little-endian components and preserve precision through
the native frontend. The default remains CI8 for compatibility.
Rates such as 10, 11.52, 12.5 and 15.12 MS/s are accepted; there is no 16 MS/s
minimum. Rate conversion does not restore bandwidth missing from a capture.

```bash
dtmb-decode \
  --input capture.ci8 \
  --input-rate 16000000 \
  --output recovered.ts \
  --acceleration cpu \
  --error-policy continue
```

The default profile is QAM64, FEC rate 0.6, interleaver mode 2
(`--system-info-index 22`). Select the modulation, FEC rate and interleaver
with an explicit system-information profile:

| FEC rate | Interleaver | 16QAM profile | 32QAM profile | 64QAM profile |
| --- | --- | --- | --- | --- |
| 0.4 (rate 1) | mode1 | 11 | — | 19 |
| 0.4 (rate 1) | mode2 | 12 | — | 20 |
| 0.6 (rate 2) | mode1 | 13 | — | 21 |
| 0.6 (rate 2) | mode2 | 14 | — | 22 |
| 0.8 (rate 3) | mode1 | 15 | 17 | 23 |
| 0.8 (rate 3) | mode2 | 16 | 18 | 24 |

Normal 4QAM profiles 5/6, 7/8 and 9/10 select FEC 0.4, 0.6 and 0.8 respectively
(odd = mode1, even = mode2), using one LDPC codeword per signal frame. All six
profiles have exact synthetic transport tests with PN420, PN595 and PN945.
Profiles 5/6, 7/8 and 9/10 cover the normal 4QAM mapping.

The native CLI also supports **4QAM-NR profiles 3/4** (mode1/mode2), both with
FEC rate 3, nominally 0.8. NR means Nordstrom-Robinson: an 8-to-16-bit code
before ordinary 4QAM mapping. The receiver performs joint soft NR decoding
before bit deinterleaving. One LDPC codeword spans two signal frames, with
descrambler resets every 3,008 payload bits (376 transport bytes per frame).
Use `--system-info-index 3` or `4`; native stages accept `--qam 4qam-nr`.

`--nr-frame-phase auto` requires two LDPC/BCH-clean words at exactly one phase
in a five-signal-frame prefix, then replays the retained input. A one-frame
startup discard is reported. `0` selects a known boundary; `1` skips one frame.
FEC frame counters and batch sizes count pairs of signal frames. The bit-interleaver
latency is 170/510 signal frames. CSI and source confidence weight observations
before NR decoding.

NR is covered by all 256 normative Appendix C entries, independent likelihood
calculations and exact synthetic CI8-to-TS tests for both profiles and every PN
mode. NR OTA reception, CUDA parity and playable-media validation remain pending.

For a PN595/4QAM rate-3 mode2 SC16 capture:

```bash
dtmb-decode --input capture.sc16 --input-format sc16 --input-rate 15120000 \
  --pn-mode pn595 --system-info-index 10 --output recovered.ts
```

Each 16QAM, 32QAM, and 64QAM symbol carries 4, 5, or 6 soft bits. The native
frontend and demappers accept `--qam 16qam|32qam|64qam`, and
`--normalization qam` uses the selected alphabet. The existing 64QAM API and
`dtmb_core_deinterleave_qam64` executable remain compatible; the generic
executable is `dtmb_core_deinterleave_qam`.

Synthetic regressions cover all six 16QAM profiles, continuous interleaving,
noisy natural LDPC/BCH recovery and exact transport bytes. Over-the-air validation is a separate step.

32QAM profiles 17/18 group two signal frames into five rate-3 LDPC codewords.

`--qam32-frame-phase auto` detects packing alignment from two naturally
LDPC/BCH-clean codewords at exactly one of two possible phases in a bounded
prefix. The prefix is replayed and any one-frame startup discard is reported.
Use `0` for a known group boundary or `1` to skip the first signal frame. FEC frame counters
and `--decode-batch-frames` count two-signal-frame groups for 32QAM; diagnostics
also report signal-frame counts and the grouping factor. All five codewords
must pass the clean gates before the group is emitted. Synthetic tests cover
both packing phases, both interleavers, noise and exact CI8-to-TS recovery.

Select the known frame header explicitly with `--pn-mode pn420|pn595|pn945`.
The default remains `pn945`; PN mode is independent of the system-information
profile. Select the PN mode explicitly; automatic detection is a possible future addition.

| Mode | Header symbols | Frame symbols | Frames per superframe | Header/body power |
| --- | ---: | ---: | ---: | ---: |
| PN420 | 420 | 4200 | 225 | 2 |
| PN595 | 595 | 4375 | 216 | 1 |
| PN945 | 945 | 4725 | 200 | 2 |

For example, a known PN595/16QAM rate-2, mode2 signal uses:

```bash
dtmb-decode --input input.ci8 --input-rate 7560000 --output recovered.ts \
  --pn-mode pn595 --system-info-index 14 --acceleration cpu
```

Use the actual input sample rate. Synthetic CI8 tests cover all three PN modes,
all six 16QAM profiles, multipath, CFO, sample displacement and exact TS recovery.
PN420/PN595 and 32QAM over-the-air validation is still pending. PN595/32QAM
is supported by the same native stages and has synthetic coverage. For a
known rate-3, mode2 transmission:

```bash
dtmb-decode --input input.ci8 --input-rate 7560000 --output recovered.ts \
  --pn-mode pn595 --system-info-index 18 --acceleration cpu
```

Strict mode is the default. `--error-policy continue` omits unclean frames and
adds MPEG-TS discontinuity indications before later clean output.
The runtime fails when no transport bytes are recovered. Validate nonempty
output with `dtmb-ts-analyze`; video claims additionally require an actual
codec decode, not only stream identification.

## Optional CUDA LDPC acceleration

CPU decoding is the portable baseline. Build the optional CUDA LDPC backend
when a supported NVIDIA toolchain is available:

```bash
cmake -S core/cpp -B build/core-cpp-cuda \
  -DDTMB_CORE_ENABLE_CUDA_LDPC=ON \
  -DDTMB_CORE_BUILD_TESTS=ON
cmake --build build/core-cpp-cuda --config Release
```

Then select it explicitly:

```bash
dtmb-decode \
  --bin-dir build/core-cpp-cuda \
  --input capture.ci8 \
  --input-rate 16000000 \
  --output recovered.ts \
  --acceleration cuda
```

The CUDA backend batches LDPC codewords and reports host-to-device, kernel,
device-to-host, and total timing metrics. CPU remains available as the
correctness and portability reference.

## Decode a pipe

Use `-` for stdin or stdout:

```bash
ci8-producing-command |
  dtmb-decode --input - --input-rate 16000000 --output recovered.ts
```

The source command is deliberately unspecified: the receiver depends only on
the byte-stream contract, not an SDR vendor or device API.

## Play while decoding

With ffplay:

```bash
ci8-producing-command |
  dtmb-decode --input - --input-rate 16000000 --output - --error-policy continue |
  ffplay -fflags nobuffer -flags low_delay -f mpegts -
```

With VLC:

```bash
ci8-producing-command |
  dtmb-decode --input - --input-rate 16000000 --output - \
    --error-policy continue |
  vlc - --demux=ts
```

The producer owns the SDR device, tuning and capture policy. `dtmb-decode`
only consumes CI8 bytes, so the same receiver command works with a
hardware adapter, a vendor utility or another SDR framework on any platform.
Keep diagnostics on stderr; stdout is binary MPEG-TS.

Long-running receiver controls are explicit and off by default. A deployment
that has independently validated them for its RF environment can enable:

```bash
ci8-producing-command |
  dtmb-decode --input - --input-rate 16000000 --output - \
    --pn-schedule-tracking \
    --source-frame-confidence inverse-mse \
    --resample-headroom-db 6 \
    --error-policy continue |
  vlc - --demux=ts
```

The receiver inserts bounded 64 MiB queues after bulk stages by default so
capture, frontend, demapping and FEC can progress concurrently. Use
`--pipeline-buffer-mib 0` only for controlled diagnostics. These options are
not universal RF defaults and do not replace transport or codec validation.

PN schedule options apply to PN420 and PN945. PN595 has a fixed header and uses
direct correlation for timing tracking. In the native frontend, `--pn-estimator wideband`
selects the appropriate cyclic or direct channel estimator for the chosen mode.
For PN595, use `--pn-wideband-header-observation direct` (or `core`). Its default
linear fit spans 149 signed delays; `--pn-wideband-max-span-symbols` controls
that span, up to 298 taps. The existing PN945 API and command defaults remain
available.

Playback quality depends on RF quality and FEC cleanliness. A player detecting
a service is not evidence that the complete stream is error-free.

## Inspect MPEG-TS output

```bash
dtmb-ts-analyze recovered.ts
ffprobe -v error -show_programs -show_streams recovered.ts
```

## Scope

Included:

- portable C++20 receive stages;
- CU8, CI8/CS8, CI16/SC16 and CF32 file/stdin integration;
- a static browser receiver, spectrum/waterfall and diagnostic reports;
- MPEG-TS file and stdout output;
- required LDPC matrices;
- core and pipeline-construction tests.

Not included:

- SDR drivers or hardware-control code;
- capture recipes, device identifiers, frequencies, gains, or locations;
- hardware scanning, sweeps, and research tooling;
- real broadcast captures or derived analysis artifacts.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
