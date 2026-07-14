# dtmb-sdr

A minimal, vendor-neutral DTMB receiver. It accepts interleaved signed 8-bit
complex samples (CI8) from a file or standard input and emits MPEG transport
stream packets to a file or standard output.

```text
CI8 file/stdin
  -> sample-rate conversion
  -> PN945 synchronization and C=3780 equalization
  -> QAM64 deinterleaving and soft demapping
  -> LDPC, BCH, and descrambling
  -> MPEG-TS file/stdout
```

Hardware acquisition is intentionally outside this repository. Any application
that can produce CI8 bytes at a known sample rate can feed the receiver.

## Build

Requirements are CMake 3.20+, a C++20 compiler, and Python 3.10+.

```bash
python -m pip install -e ".[dev]"
cmake -S core/cpp -B build/core-cpp -DDTMB_CORE_BUILD_TESTS=ON
cmake --build build/core-cpp --config Release
ctest --test-dir build/core-cpp -C Release --output-on-failure
pytest
```

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
(`--system-info-index 22`). Profiles 19 through 24 select the supported QAM64
FEC/interleaver combinations.

Strict mode is the default. `--error-policy continue` omits unclean frames and
adds MPEG-TS discontinuity indications before later clean output.

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

### Parallelism and observed acceleration

The receiver uses independent concurrency at several stages:

- multithreaded rational resampling and PN acquisition;
- worker-parallel C=3780 frame equalization;
- chunked QAM64 deinterleaving and soft demapping;
- batched LDPC decoding on either CPU workers or CUDA kernels.

One observed live-pipeline comparison used 12 CPU workers and 256-frame FEC
batches. The CPU path processed 198,660 codewords in 30.95 seconds of measured
FEC batch time (about 6,419 codewords/s). The CUDA path processed 267,486
codewords in 17.22 seconds of measured FEC batch time (about 15,534
codewords/s), while 10.41 seconds were attributed to CUDA transfer plus kernel
execution. After normalizing by processed codewords, the observed FEC-stage
throughput improvement was approximately 2.4x.

These runs did not use an identical retained LLR stream, so the figure is an
operational observation rather than a controlled cross-machine benchmark.
Performance depends on the proportion of early-rejected codewords, iteration
count, batch size, CPU, GPU, and memory-transfer overhead. Reproducible release
claims should use the same retained LLR input and compare byte-identical output.

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
  dtmb-decode --input - --input-rate 16000000 --output - --error-policy continue |
  vlc - --demux=ts
```

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
