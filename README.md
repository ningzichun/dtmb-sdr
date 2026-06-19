# dtmb-sdr

A software-defined DTMB (Digital Terrestrial Multimedia Broadcasting, GB 20600-2006) receiver. Takes IQ samples from a capture file or any supported SDR and decodes them into MPEG-TS video, audio, and subtitle streams.

This project was developed with **AI-assisted hardware-specific debugging and DSP implementation** -- using large-language-model agents for iterative signal-chain diagnosis, LDPC/FEC fixture generation, carrier-mapping audits, and real-RF experiment coordination across the full PHY stack.

## What It Does

```text
IQ input (file or SDR)
 -> PN945 frame sync + coarse/fine CFO correction
 -> C=3780 OFDM FFT + frequency deinterleave + MMSE channel equalization
 -> Mode1/Mode2 convolutional symbol deinterleave
 -> 64-QAM soft demap (float32 LLR)
 -> LDPC (normalized min-sum) + BCH(762,752) + descrambler
 -> MPEG-TS output
```

Recovers 1920x1080 H.264 video, AC-3 audio, and DVB subtitles from real over-the-air UHF DTMB broadcasts.

## Input Sources

| Source | Format | Notes |
|--------|--------|-------|
| **Capture file** | CI8 (interleaved int8 I/Q) | Primary offline workflow; any SDR can produce these |
| **SDR (live)** | Via libhackrf, SoapySDR, or GNU Radio | Real-time capture and decode pipeline |
| **Synthetic** | Generated test fixtures | For development and validation without hardware |

The receiver is input-agnostic. Any source that provides complex baseband samples at the correct rate works. The SDR adapter layer is separate from the core demodulator.

## Proof of Operation

The following ffprobe output was produced from a **live SDR capture** at 602 MHz (16 Msps, external 10 MHz reference clock). The receiver decoded the RF into valid MPEG-TS containing multiple TV programs:

```
ffprobe output from live 602 MHz capture (40s, real RF):

  Program 1:
    Stream #0: h264  video  1920x1080  (25 frames decoded)
    Stream #1: ac3   audio             (23 frames decoded)
    Stream #2: ac3   audio             (20 frames decoded)
    Stream #3: ac3   audio             (22 frames decoded)
    Stream #4: dvb_subtitle
    Stream #5: dvb_subtitle
    Stream #6: dvb_subtitle

  Program 2:
    Stream #0: h264  video  1920x1080  (28 frames decoded)
    Stream #1: ac3   audio             (12 frames decoded)
    Stream #2: ac3   audio             (18 frames decoded)
    Stream #3: ac3   audio             (25 frames decoded)
    Stream #4: dvb_subtitle
    Stream #5: dvb_subtitle
    Stream #6: dvb_subtitle

  Program 3:
    Stream #0: h264  video  1920x1080  (23 frames decoded)
    Stream #1: ac3   audio             (23 frames decoded)
    Stream #2: ac3   audio             (20 frames decoded)
    Stream #3: ac3   audio             (22 frames decoded)
    Stream #4: dvb_subtitle

  Container: MPEG-TS, 19.07s, 7.21 MB, ~3.0 Mbps
  Format:    epg data + 3x h264 1080p + 9x ac3 audio + 8x dvb_subtitle
```

Key metrics from the best validated run:
- **LDPC convergence:** 95.47% of selected codewords
- **TS syntax:** 48,366 valid packets, zero sync errors, zero TEI
- **Capture:** zero dropped bytes, external 10 MHz clock reference

## Architecture

```text
core/cpp/          Portable C++20 demodulator library (no OS-specific APIs)
  src/               Core DSP: PN sync, C3780 FFT, channel estimation, FEC
  tools/             Pipe-shaped CLI executables (stdin/stdout streaming)
  include/dtmb/      Public headers

python/dtmb/       Python reference implementation and offline tools
  data/              LDPC parity-check matrices (.alist), generator data

adapters/          SDR integration layer (pluggable backends)
scripts/           Pipeline orchestration and synthetic test generation
tests/             Core algorithm tests (PN, QAM, LDPC, BCH, TS, interleaver)
```

The demodulator core has no SDR dependencies. SDR-specific code lives entirely in `adapters/` and the acquisition tools, keeping the signal processing portable and testable without hardware.

## Prerequisites

- **C++20 compiler** (GCC 12+, Clang 15+, or MSVC 2022+)
- **CMake 3.20+**
- **Python 3.10+** with numpy, scipy
- **ffmpeg/ffprobe** (for TS inspection and video extraction)
- **SDR library** (optional, for live capture -- e.g. libhackrf, SoapySDR)
- Optional: external 10 MHz reference clock for frequency accuracy

## Build

```bash
# C++ core library and tools
cmake -B build/core-cpp core/cpp
cmake --build build/core-cpp -j$(nproc)

# Python package
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dsp]"

# Run tests
ctest --test-dir build/core-cpp --output-on-failure
pytest tests/
```

Or use the Makefile shortcut:

```bash
make -f pipeline.mk native-core PYTHON=python
```

## Usage

### Decode a Capture File

The primary workflow takes a CI8 IQ capture and produces MPEG-TS:

```bash
build/core-cpp/dtmb_core_c3780_extract \
    --auto-sync --equalizer pn --pn-estimator wideband \
    --pn-wideband-block-frames 2 --pn-mmse 0.004 \
    --remove-dc --normalization qam64 --system-info-index 22 \
    capture.ci8 - \
  | build/core-cpp/dtmb_core_deinterleave_qam64 \
    --mode mode2 --phase 0 --workers 8 - - \
  | build/core-cpp/dtmb_core_ldpc_bch_decode \
    --fec-rate 2 \
    --alist python/dtmb/data/dtmb_ldpc_rate2.alist \
    --codewords-per-frame 3 --workers 3 \
    --hard-h-gate-threshold 0.43 \
    --clean-frames-only --mark-discontinuities \
    --insert-discontinuity-packets - - \
  > output.ts

# Inspect the result
ffprobe -show_streams -show_programs output.ts
```

Any tool that records complex int8 I/Q at the appropriate sample rate can produce the input file. The pipe-shaped tools read from stdin and write to stdout, so they compose naturally.

### Live SDR Capture + Decode

For live RF, pipe the SDR output directly into the decoder:

```bash
# Example with the included acquisition tool:
build/core-cpp/dtmb_core_hackrf_acquire \
    --frequency 602000000 --sample-rate 16000000 \
    --bandwidth 12000000 --amp 0 --lna-gain 24 --vga-gain 14 \
    --duration 20 --output - \
  | build/core-cpp/dtmb_core_c3780_extract \
    --auto-sync --equalizer pn --pn-estimator wideband \
    --pn-wideband-block-frames 2 --pn-mmse 0.004 \
    --remove-dc --normalization qam64 --system-info-index 22 - - \
  | build/core-cpp/dtmb_core_deinterleave_qam64 \
    --mode mode2 --phase 0 --workers 8 - - \
  | build/core-cpp/dtmb_core_ldpc_bch_decode \
    --fec-rate 2 \
    --alist python/dtmb/data/dtmb_ldpc_rate2.alist \
    --codewords-per-frame 3 --workers 3 \
    --hard-h-gate-threshold 0.43 \
    --clean-frames-only --mark-discontinuities \
    --insert-discontinuity-packets - - \
  > live.ts
```

Other SDR tools (e.g. `rx_sdr`, `hackrf_transfer`, GNU Radio) can substitute the first stage as long as they output CI8 to stdout.

### Python Offline Tools

```bash
# Detect DTMB frames in an IQ recording
python -m dtmb.detect capture.ci8 --sample-rate 16000000 --pn-search

# Run the Python reference receiver
python -m dtmb.receiver capture.ci8 --sample-rate 16000000 --mode pn945

# Analyze TS packet structure
python -m dtmb.ts output.ts

# Generate synthetic test fixture (no hardware needed)
python scripts/generate_synthetic_ts.py
```

### Make Pipeline (Full Automated Run)

```bash
# Synthetic loopback test (no hardware needed)
make -f pipeline.mk synthetic

# Live capture + full decode pipeline
make -f pipeline.mk native-live-stream \
    FREQUENCY=602000000 \
    NATIVE_LIVE_SAMPLE_RATE=16000000 \
    NATIVE_LIVE_BANDWIDTH=12000000 \
    AMP=0 LNA_GAIN=24 VGA_GAIN=14 \
    NATIVE_LIVE_DURATION=20 \
    NATIVE_SYSTEM_INFO_INDEX=22 \
    NATIVE_LIVE_OUTPUT=output.ts
```

## Supported DTMB Modes

| Parameter | Supported Values |
|-----------|-----------------|
| PN header | PN420, PN595, PN945 |
| FFT size | C=3780 (8 MHz channel) |
| QAM | 4-QAM, 16-QAM, 64-QAM |
| FEC rate | 0.4 (rate 1), 0.6 (rate 2), 0.8 (rate 3) |
| Interleaver | Mode 1, Mode 2 |
| LDPC | Normalized min-sum, configurable iterations |
| BCH | BCH(762,752) |

## Current Status

The receiver decodes real over-the-air DTMB broadcasts into valid MPEG-TS with H.264 video, AC-3 audio, and DVB subtitles. LDPC convergence exceeds 95% on selected codewords under good RF conditions. The output contains clean TS islands with complete video keyframes; continuous gap-free playback across an entire capture is still in progress.

Known limitations:
- Decode throughput is below real-time (roughly 10x slower than wall-clock)
- Contiguous H.264 PES coverage has gaps under weak RF conditions
- Single-frequency operation (no automatic channel scanning yet)

## Standard Reference

This implementation follows **GB 20600-2006** (Framing Structure, Channel Coding and Modulation for Digital Television Terrestrial Broadcasting System). LDPC parity-check matrices, PN sequences, system information tables, and constellation mappings are extracted directly from the standard.

## AI-Assisted Development

This project was developed using AI coding agents for hardware-specific debugging and DSP implementation:

- **Signal chain diagnosis:** Iterative debugging of carrier mapping, deinterleaver chronology, QAM bit-plane ordering, and LDPC/FEC framing against real RF captures
- **Fixture-driven correctness audits:** Generating tagged synthetic frames to verify every stage of the receive chain (FFT bin to logical carrier to data-only index to FEC codeword)
- **Real-RF experiment coordination:** Systematic parameter sweeps with gated promotion criteria, trial logging, and artifact provenance tracking
- **Standard extraction:** Translating GB 20600-2006 tables into code-testable fixtures and parity-check matrices

The development workflow demonstrates that AI agents can be effective collaborators on low-level hardware/DSP projects where correctness depends on exact bit-level layout, physical-layer timing, and standards compliance.

## License

MIT. See [LICENSE](LICENSE).
