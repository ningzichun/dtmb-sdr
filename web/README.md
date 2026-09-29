# DTMB Lab browser receiver

Open **https://ningzichun.github.io/dtmb-sdr/**. Choose a local raw IQ file, its
actual sample format and sample rate, inspect the spectrum and waterfall, then
select the PN mode and transmission profile. Decode a short interval first.
The resulting MPEG-TS can be downloaded or previewed when its codecs are
supported by the browser. H.264/AAC support depends on the browser; AVS and
other broadcast codecs play through a suitable external player.

No recording is uploaded. The application is a static site: all receiver
stages execute in WebAssembly workers on the user's device. Saving a report
and opening an issue are explicit user actions.

## Shared receiver

`web/CMakeLists.txt` compiles the existing `core/cpp` resampler, C3780 frontend,
symbol deinterleaver/demapper and LDPC/BCH tools. `web_stdio.hpp` adapts their
binary stdin/stdout streams to asynchronous browser messages. Each stage has
a dedicated worker; demand/ack messages bound each connection to one buffer.
No server, SharedArrayBuffer, cross-origin isolation, Python runtime, WebGPU,
or browser-specific copy of the receiver algorithms is required.

WebGL renders the waterfall. The diagnostic FFT and power spectrum are separate
from the receiver. A good-looking spectrum is a first check; the receiver makes the final call.
Reports include sample windows, level, DC, clipping, a SHA-256 of the first
1 MiB (explicitly **not** a whole-file hash), selected settings, core build
identity, native receiver logs and the observed transport checks.

## Formats and rates

All formats are interleaved **I then Q**, with no WAV or other container header.

| Format | Component | Bytes per complex sample |
| --- | --- | ---: |
| CU8 | unsigned 8-bit, zero at 128 | 2 |
| CI8 / CS8 | signed two's-complement 8-bit | 2 |
| CI16 / SC16 / CS16 | signed 16-bit, little endian | 4 |
| CF32 | IEEE float32, little endian, nominal full scale ±1 | 8 |

Native `--input-format` accepts the same names. CI16 is converted to float
without losing its low bits and remains float through the frontend. The browser
uses float conditioning for every format. Native legacy CI8 pipelines retain
their existing resampler/scaling contract; `--output-format cf32` on the native
conditioner selects the same float path as the browser.

The bounded polyphase SRRC converter accepts positive integer sample rates,
including 10, 11.52, 12.5, 15.12, 16 and 20 MS/s, and relatively prime rates.
Its phase table does not grow with the greatest-common-divisor ratio. Rates
are limited to unsigned 32-bit values; extreme ratios exceeding the 8192-tap
half-width are rejected with an explicit error. At 7.56 MS/s it passes already
conditioned symbol-rate input through. No rate converter restores missing RF
bandwidth: arbitrary-rate acceptance does not mean every rate captures an
entire DTMB channel.

The browser exposes 4QAM-NR profiles 3–4, normal 4QAM profiles 5–10,
16QAM 11–16, 32QAM 17–18 and 64QAM 19–24, with PN420, PN595 or PN945.
PN mode/profile selection is explicit in the browser.

## Build and test

Install/activate Emscripten **4.0.15**, CMake, Python and Node.js 22+:

```sh
npm ci --prefix web
python web/build.py
# Or: python web/build.py --emsdk /path/to/emsdk
python -m http.server 8765 --bind 127.0.0.1 --directory dist/web
```

Open http://127.0.0.1:8765. Serve over localhost or HTTPS, rather than file URLs.

```sh
python -m pip install numpy
python web/tests/make_fixture.py
npm test --prefix web
npx --prefix web playwright install chromium
npm --prefix web run test:browser
```

Browser tests inspect a real WebGL canvas, run the complete WASM pipeline,
compare recovered transport bytes against an independent synthetic transmitter,
and exercise report downloads, cancellation, malformed float input and mobile
layout. Native format/rate/4QAM tests live in `tests/test_iq_formats.py`.

GitHub Actions builds, tests and publishes `dist/web` using the Pages workflow.
Enable Pages with **GitHub Actions** as its source. All asset paths are relative
so project sites such as `/dtmb-sdr/` work. The private project's app delegates
to this same build instead of maintaining a second web receiver.

## Limits and interpretation

Inspection samples up to 128 windows across the selected interval; it covers these sampled windows rather than the entire recording. The file extension is a hint, not proof of the byte format.
Browser decoding runs one CPU worker per stage and is not claimed to sustain
live RF rates. Cancel terminates the workers immediately. Captures stream from
disk in bounded reads; downloadable TS is capped at 256 MiB, with an explicit
request to select a shorter interval if reached. Each WASM stage has a 1 GiB
memory ceiling. The app has no SDR hardware driver or tuning interface.

Only naturally clean LDPC/BCH groups are emitted. Unclean groups are omitted
with native discontinuity indications. The browser checks TS packet alignment
and transport error flags; receiver logs preserve FEC failures. For continuity and codec-correctness claims, use `dtmb-ts-analyze` and actual codec decoding. The supplied issue recordings are pending OTA validation of the new modes.
