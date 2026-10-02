# dtmb-sdr

dtmb-sdr receives DTMB digital television from raw I/Q samples and writes an
MPEG transport stream (MPEG-TS). Use the command-line receiver with a file or
stream, or open DTMB Lab to inspect and decode a recording in your browser.

**[Open DTMB Lab](https://ningzichun.github.io/dtmb-sdr/)** — view the spectrum
and waterfall, identify PN headers and transmission settings, inspect recovered
programs and tracks, and save the transport stream or a diagnostic report.
Recordings are processed on your device. See the [browser guide](web/README.md)
for instructions.

The command-line and browser applications share the same C++ receiver.
An SDR utility or another application can supply the raw I/Q samples; device
setup and recording remain with that application.

## Installation

Python 3.10 or later is required. Install a released package with:

```sh
python -m pip install dtmb-sdr
```

Platform wheels include the native receiver executables and LDPC data, so a
compiler is not needed when installing a wheel.

## Decode a recording

This example uses signed 8-bit I/Q samples at 16 MS/s, PN945 and profile 22
(64QAM, FEC rate 0.6, interleaver mode 2):

```sh
dtmb-decode --input capture.ci8 --input-rate 16000000 --pn-mode pn945 --system-info-index 22 --output recovered.ts
dtmb-ts-analyze recovered.ts
```

Set the format, sample rate, PN mode and profile to match your recording.
The CLI defaults to CI8, PN945, profile 22, a C=3780 frame body and CPU decoding.
The sample rate must be supplied in samples per second.

By default, `--error-policy fail` stops decoding on an unclean FEC frame.
Use `--error-policy continue` to skip failed groups and mark discontinuities
before subsequent clean output. Transport checks are a separate step:
`dtmb-ts-analyze` checks the recovered stream, while assessing audio or video
integrity requires decoding it with a suitable codec.

### Input formats and sample rates

Input is raw interleaved **I then Q**, with no container header.

| Format | I/Q component encoding | Bytes per complex sample |
| --- | --- | ---: |
| CU8 | Unsigned 8-bit, centered at 128 | 2 |
| CI8 / CS8 | Signed 8-bit | 2 |
| CI16 / SC16 / CS16 | Signed 16-bit, little-endian | 4 |
| CF32 | IEEE 754 float32, little-endian, nominal range ±1 | 8 |

Select the encoding with `--input-format`. A filename extension is a hint;
confirm the format used by the recording application.

The receiver converts the input to its 7.56 MS/s processing rate. Common input
rates include 10, 11.52, 12.5, 15.12, 16 and 20 MS/s. Enter the actual recording
rate, for example `--input-rate 12500000` for 12.5 MS/s. Rate conversion cannot
restore channel bandwidth that was missing from the recording.

### PN mode, profile and frame body

Select the PN header with `--pn-mode pn420`, `pn595` or `pn945`. PN mode
and transmission profile are separate settings. If they are unknown, the
[browser receiver](web/README.md#understand-the-results) can scan PN modes and
identify the profile from system information.

A profile combines modulation, FEC code rate and interleaver mode:

| Modulation | FEC rate | Mode 1 profile | Mode 2 profile |
| --- | ---: | ---: | ---: |
| 4QAM-NR | 0.8 | 3 | 4 |
| 4QAM | 0.4 | 5 | 6 |
| 4QAM | 0.6 | 7 | 8 |
| 4QAM | 0.8 | 9 | 10 |
| 16QAM | 0.4 | 11 | 12 |
| 16QAM | 0.6 | 13 | 14 |
| 16QAM | 0.8 | 15 | 16 |
| 32QAM | 0.8 | 17 | 18 |
| 64QAM | 0.4 | 19 | 20 |
| 64QAM | 0.6 | 21 | 22 |
| 64QAM | 0.8 | 23 | 24 |

4QAM-NR adds Nordstrom–Robinson coding to the 4QAM path. Choose a profile with
`--system-info-index`; the CLI uses the configured value rather than
automatically selecting a detected profile.

C=3780 multicarrier reception supports all three PN modes and profiles 3–24.
For PN595 with normal 4QAM (profiles 5–10), the receiver also supports C=1
single-carrier reception. Use `--frame-body-mode auto` to check the frame body,
or `--frame-body-mode c1` when C=1 is known. The CLI defaults to `c3780);
the browser uses Auto.

For example, decode a PN595, profile 10 CF32 recording with automatic
frame-body selection:

```sh
dtmb-decode --input capture.cf32 --input-format cf32 --input-rate 12500000 --pn-mode pn595 --system-info-index 10 --frame-body-mode auto --output recovered.ts
```

Run `dtmb-decode --help` for the full set of receiver options. Use additional
tracking, confidence and calibration controls after validating them for your
input.

## Optional CUDA acceleration

CUDA accelerates the LDPC stage. CUDA-enabled Windows and Linux x86-64 wheels
contain both the CPU decoder and an optional CUDA 12 backend. The `cuda`
extra installs the CUDA runtime separately; CPU decoding does not load it.
macOS wheels use the CPU decoder.

When using a CUDA-enabled wheel, install the runtime and select CUDA explicitly:

```sh
python -m pip install "dtmb-sdr[cuda]"
dtmb-decode --input capture.ci8 --input-rate 16000000 --output recovered.ts --acceleration cuda
```

A compatible NVIDIA GPU and driver are required. Wheel users do not need the
CUDA Toolkit. The launcher locates the installed runtime and reports missing
runtime or device errors.

Source builds use the CPU backend by default. To build with CUDA, install a
CUDA 12 Toolkit and run this command from the repository root:

```sh
python -m pip install ".[cuda]" --config-settings "cmake.define.DTMB_CORE_ENABLE_CUDA_LDPC=ON"
```

The build setting enables the native backend; selecting the extra alone only
installs the runtime.

## Stream input and playback

Use `-` for standard input or standard output. Replace `your-iq-source`
with a command that produces raw CI8 samples at the stated rate:

```sh
your-iq-source | dtmb-decode --input - --input-rate 16000000 --output recovered.ts
```

To send the recovered stream directly to an external player such as ffplay:

```sh
your-iq-source | dtmb-decode --input - --input-rate 16000000 --output - --error-policy continue | ffplay -fflags nobuffer -flags low_delay -f mpegts -
```

Use the PN mode and profile options when the transmission differs from the
defaults. Diagnostics go to standard error; standard output carries binary TS
data. The source application controls the SDR device, tuning and recording.

To list programs and streams in a saved TS:

```sh
ffprobe -v error -show_programs -show_streams recovered.ts
```

Stream identification is useful metadata, not a substitute for transport
checks or an actual audio/video decode.

## Build from source

Install CMake 3.20 or later, a C++20 compiler and Python 3.10 or later.
From the repository root, install the package with:

```sh
python -m pip install .
```

For an editable installation with development and test dependencies:

```sh
python -m pip install -e ".[dev]"
```

Build and run the native tests, then the Python tests:

```sh
cmake -S core/cpp -B build/core-cpp -DDTMB_CORE_BUILD_TESTS=ON
cmake --build build/core-cpp --config Release
ctest --test-dir build/core-cpp -C Release --output-on-failure
python -m pytest
```

Use `--bin-dir build/core-cpp` to run the CLI with the executables from that
build directory instead of the packaged copies.

Build a distributable wheel with:

```sh
python -m build --wheel
```

For browser build and deployment instructions, see the
[DTMB Lab guide](web/README.md#build-and-run-locally).

## Citation

If you use dtmb-sdr in research, please cite the software and include the
version or commit used.

```bibtex
@misc{dtmb_sdr,
  author       = {{dtmb-sdr contributors}},
  title        = {{dtmb-sdr}},
  howpublished = {\url{https://github.com/ningzichun/dtmb-sdr}},
  note         = {Open-source software}
}
```

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
