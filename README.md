# dtmb-sdr

A minimal, vendor-neutral DTMB receiver. It accepts interleaved signed 8-bit
complex samples (CI8) from a file or standard input and emits MPEG transport
stream packets to a file or standard output.

```text
CI8 file/stdin
  -> sample-rate conversion
  -> PN420/PN595/PN945 synchronization and C=3780 equalization
  -> 16QAM/32QAM/64QAM deinterleaving and soft demapping
  -> LDPC, BCH, and descrambling
  -> MPEG-TS file/stdout
```

Hardware acquisition is intentionally outside this repository. Any application
that can produce CI8 bytes at a known sample rate can feed the receiver.

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

Each 16QAM, 32QAM, and 64QAM symbol carries 4, 5, or 6 soft bits. The native
frontend and demappers accept `--qam 16qam|32qam|64qam`, and
`--normalization qam` uses the selected alphabet. The existing 64QAM API and
`dtmb_core_deinterleave_qam64` executable remain compatible; the generic
executable is `dtmb_core_deinterleave_qam`.

Synthetic regressions cover all six 16QAM profiles, continuous interleaving,
noisy natural LDPC/BCH recovery and exact transport bytes. They do not establish
over-the-air validation.

32QAM profiles 17/18 group two signal frames into five rate-3 LDPC codewords.

`--qam32-frame-phase auto` detects packing alignment from two naturally
LDPC/BCH-clean codewords at exactly one of two possible phases in a bounded
prefix. The prefix is replayed and any one-frame startup discard is reported.
Use `0` for a known group boundary or `1` to skip the first signal frame. An
ambiguous prefix or incomplete terminal group is an error. FEC frame counters
and `--decode-batch-frames` count two-signal-frame groups for 32QAM; diagnostics
also report signal-frame counts and the grouping factor. All five codewords
must pass the clean gates before the group is emitted. Synthetic tests cover
both packing phases, both interleavers, noise and exact CI8-to-TS recovery.

Select the known frame header explicitly with `--pn-mode pn420|pn595|pn945`.
The default remains `pn945`; PN mode is independent of the system-information
profile. Automatic PN-mode selection is not implemented.

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
direct correlation for timing tracking; cyclic schedule and cyclic header
averaging options are rejected. In the native frontend, `--pn-estimator wideband`
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
- CI8 file and stdin integration;
- MPEG-TS file and stdout output;
- required LDPC matrices;
- core and pipeline-construction tests.

Not included:

- SDR drivers or hardware-control code;
- capture recipes, device identifiers, frequencies, gains, or locations;
- signal-generation, scanning, UI, visualization, sweep, or research tooling;
- real broadcast captures or derived analysis artifacts.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
