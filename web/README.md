# DTMB Lab browser receiver

DTMB Lab lets you inspect a raw I/Q recording, identify DTMB transmission
settings and recover an MPEG transport stream in your browser.

**[Open DTMB Lab](https://ningzichun.github.io/dtmb-sdr/)**

Your recording stays on your device. The static application runs the shared
C++ receiver in WebAssembly workers and saves results locally.

## Get started

1. Open a local I/Q file or ZIP archive, or choose a sample from the catalog.
2. Select the recording's **Sample format**. Confirm the encoding used by the
   recording application rather than relying on the filename.
3. Enter the **Sample rate** in samples per second, such as `12500000` for
   12.5 MS/s. If the rate is unknown, leave the field empty and click **Guess**.
4. Choose a start time and a short duration for the first attempt. A duration
   of `0` selects the rest of the file.
5. Leave **PN mode** and **Transmission profile** on **Auto**, then click
   **Inspect capture**. The spectrum and waterfall appear first, followed by
   a scan of PN420, PN595 and PN945. Clear **Scan PN modes after inspection**
   if you only want the spectrum and capture checks.
6. Review the detected settings, then click **Decode to MPEG-TS**. Inspect
   **Transport stream info**, download the `.ts` file or save a report.

Use **Stop receiver** to cancel an active scan or decode. You can select a
specific PN mode or profile at any time to override Auto.

## Input settings

All formats contain interleaved **I then Q** samples, with no container header.

| Format | I/Q component encoding | Bytes per complex sample |
| --- | --- | ---: |
| CU8 | Unsigned 8-bit, centered at 128 | 2 |
| CI8 / CS8 | Signed 8-bit | 2 |
| CI16 / SC16 / CS16 | Signed 16-bit, little-endian | 4 |
| CF32 | IEEE 754 float32, little-endian, nominal range ±1 | 8 |

The sample-rate field starts empty. Use the known recording rate whenever
available. **Guess** checks these common rates (MS/s) against all three PN modes:

7.56, 7.68, 8, 8.192, 9.6, 10, 10.24, 11.52, 12, 12.288, 12.5, 15.12,
15.36, 16, 16.384, 19.2, 20, 24, 25, 30.72, 32, 40, 50 and 61.44.

Repeated header matches with an unambiguous result
supply an editable rate and start inspection. Ambiguous, incomplete or unmatched
scans leave the field empty.

A guessed rate is a candidate from this list, not recording metadata or an
exact clock measurement. It does not establish system-information lock or
successful decoding. Rate conversion also cannot restore bandwidth missing
from the recording.

**Center frequency** changes the plot labels only; it does not tune a receiver.

## Understand the results

RF energy in the spectrum and recognized DTMB frame structure are different
observations. The receiver reports four stages separately:

| Stage | What the result establishes |
| --- | --- |
| PN acquisition | Repeated PN headers match a DTMB frame structure. |
| System-information lock | The receiver identifies the profile, modulation, FEC rate, interleaver and frame body. |
| FEC recovery | LDPC/BCH checks pass for the recovered data groups. |
| TS validation | The recovered bytes pass packet alignment, transport-error flag and PCR syntax checks. |

For example, **PN595 acquired; system information not locked** means that the
receiver recognized the header but has not confirmed the transmission profile.
It is neither a complete decode nor a failed PN acquisition.

### PN and profile selection

PN **Auto** resolves a mode from a complete three-mode scan: either the only
acquired mode, or the only mode with system-information lock when several modes
acquire. Ambiguous results leave Auto unresolved. Profile **Auto** waits for
system-information lock.

The panel shows the current Auto choices alongside the observations. Selecting
a mode or profile manually changes the receiver configuration; it does not
turn that choice into a detection result. Modes that have not been tested
remain **Not tested**.

Click a PN entry to view its measurements:

- Acquisition state and header hits, such as `16/16`.
- Correlation metric and threshold, such as `0.543 / 0.350`. This is **not SNR**.
- Estimated frequency offset, with acquisition timing in the expandable details.
- Header and frame dimensions, frame period and superframe size.
- Input format, sample rate and the analyzed time window.

**Test configured PN** checks one configured mode; **Scan PN modes** checks
all three. If no PN is acquired, check the input format, sample rate, PN mode
or signal quality. That result alone does not show that the signal is too weak.

The browser selects the frame body automatically and displays
**C=1 · single-carrier** or **C=3780 · multicarrier** after system-information
lock. C=1 reception covers PN595 with normal 4QAM profiles 5–10. See the
[profile table](../README.md#pn-mode-profile-and-frame-body) for all profiles.

### Progress and probe limits

The progress bar measures **input read**, not decode success. It is shown only
while work is active and is replaced by a final status when the operation
finishes, fails, times out or is cancelled. An early stop retains the actual
input-read percentage rather than showing 100%.

Probes are bounded so you can inspect results before committing to a full decode:

| Action | Input per probe | Time limit |
| --- | --- | --- |
| Guess sample rate | Up to 0.04 seconds or 16 MiB, whichever is smaller | 3 seconds per probe; 45 seconds overall |
| Test or scan PN modes | Up to 0.2 seconds or 16 MiB, whichever is smaller | 30 seconds per mode |

PN probes examine the beginning of the selected interval, using up to 16
headers for acquisition and 32 frames for system-information checks. Scanning
adds receiver work beyond spectrum inspection; its cost depends on the device
and input settings. The plots stay visible as results arrive.

## Transport stream and reports

**Transport stream info** reads CRC-valid PAT/PMT tables and shows programs,
PMT/PCR PIDs, declared tracks, stream types, languages and observed packet
counts. Missing or damaged tables leave the corresponding information
unavailable. These are received stream declarations; use an external player
with suitable codec support to play the downloaded TS.

Only groups that pass the native LDPC/BCH checks are emitted. Failed groups
are skipped with discontinuity indications. A recovered TS can still fail
packet checks; the file and its program information remain available.
For continuity checks, use `dtmb-ts-analyze` from the
[command-line package](../README.md). Audio/video integrity requires actual
codec decoding.

**Save report** records input settings, analyzed windows, level/DC/clipping
measurements, PN observations, receiver logs, core build identity and transport
checks. Its SHA-256 identifies the first 1 MiB of the input, or the whole input
if smaller; it is a prefix hash for larger files.

## Working with large recordings

Inspection samples up to 128 windows across the selected interval. The spectrum
and capture checks describe those windows rather than every sample in the file.

Decoding reads the recording in bounded chunks and runs on CPU workers.
Performance depends on your device and the recording. Start with a short
interval, then increase it after reviewing the results. TS downloads are capped
at 256 MiB; select a shorter interval if the cap is reached. Each WebAssembly
worker has a 1 GiB memory limit.

If the app reports an incompatible receiver build, refresh the page. For a
local installation, rebuild and serve the complete output directory. Script
versions and WebAssembly hashes are checked to keep receiver assets consistent.

## Build and run locally

Install and activate Emscripten **4.0.15**, and install CMake 3.20+, Python
3.10+ and Node.js 22+. From the repository root:

```sh
npm ci --prefix web
python web/build.py
python -m http.server 8765 --bind 127.0.0.1 --directory dist/web
```

Open **http://127.0.0.1:8765/**. Serve the app over localhost or HTTPS so the
workers can load normally.

If Emscripten is not active in your shell, point the build script at the SDK:

```sh
python web/build.py --emsdk /path/to/emsdk
```

Run the JavaScript unit tests with:

```sh
npm test --prefix web
```

The browser build compiles the receiver stages from `core/cpp`, using one
worker per stage and bounded byte streams between them. The waterfall is
rendered with WebGL. The [native build instructions](../README.md#build-from-source)
cover the shared receiver tests.

## Deploy to GitHub Pages

Enable **GitHub Actions** as the Pages source in the repository settings.
The [Pages workflow](../.github/workflows/pages.yml) builds and tests the app,
then publishes `dist/web`. Pull requests build and test without deploying.

Asset paths are relative, so the same output works at the site root or under
a project path such as `/dtmb-sdr/`.
