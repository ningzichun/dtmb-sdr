"""Deterministic standard-based transmit fixtures; no received samples or payload repair."""
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "python" / "dtmb" / "data"
DATA_SYMBOLS = 3744

# GB 20600-2006 Figure 4, with the printed labels read b0 first.
QAM32_FIGURE4 = (
    ("10111", -4.5, 7.5), ("10011", -1.5, 7.5), ("11011", 1.5, 7.5), ("11111", 4.5, 7.5),
    ("10010", -7.5, 4.5), ("00111", -4.5, 4.5), ("00011", -1.5, 4.5), ("01011", 1.5, 4.5),
    ("01111", 4.5, 4.5), ("11010", 7.5, 4.5), ("10110", -7.5, 1.5), ("00110", -4.5, 1.5),
    ("00010", -1.5, 1.5), ("01010", 1.5, 1.5), ("01110", 4.5, 1.5), ("11110", 7.5, 1.5),
    ("10100", -7.5, -1.5), ("00100", -4.5, -1.5), ("00000", -1.5, -1.5), ("01000", 1.5, -1.5),
    ("01100", 4.5, -1.5), ("11100", 7.5, -1.5), ("10000", -7.5, -4.5), ("00101", -4.5, -4.5),
    ("00001", -1.5, -4.5), ("01001", 1.5, -4.5), ("01101", 4.5, -4.5), ("11000", 7.5, -4.5),
    ("10101", -4.5, -7.5), ("10001", -1.5, -7.5), ("11001", 1.5, -7.5), ("11101", 4.5, -7.5),
)


def transport_frames(rate: int, qam: int, count: int = 4) -> list[bytes]:
    if qam == 32:
        assert rate == 3 and count % 2 == 0
    packets_per_frame = 10 if qam == 32 else (2 if qam == 16 else 3) * (rate + 1)
    rng = np.random.default_rng(823 + rate)
    packets = [bytes((0x47, 0x01, 0x00, 0x10 | (n % 16)))
               + rng.bytes(184) for n in range(packets_per_frame * count)]
    return [b"".join(packets[n:n + packets_per_frame])
            for n in range(0, len(packets), packets_per_frame)]


@lru_cache(maxsize=3)
def parity_basis(rate: int) -> tuple[tuple[int, ...], int]:
    """Systematic GF(2) encoder derived from the standard mother-code H matrix.

    Python integers represent rows. Eliminate only the parity columns; retain
    the message coefficients. This test encoder is independent of the native
    iterative receiver and verifies every generated mother-code syndrome.
    """
    lines = (DATA / f"dtmb_ldpc_rate{rate}.alist").read_text().splitlines()
    variables, checks = map(int, lines[0].split())
    rows = [sum(1 << (int(v) - 1) for v in line.split() if int(v))
            for line in lines[4 + variables:4 + variables + checks]]
    pivots = [0] * checks
    mask = (1 << checks) - 1
    for row in rows:
        while row & mask:
            pivot = ((row & mask) & -(row & mask)).bit_length() - 1
            if pivots[pivot]:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
        else:
            raise AssertionError("parity matrix is not full rank")
    assert all(pivots) and len(rows) == checks
    return tuple(pivots), checks


def encode_frame(payload: bytes, rate: int, *, scrambler_frame_bytes: int | None = None) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    state = [int(b) for b in "100101010000000"]
    scrambled = bits.copy()
    for n in range(bits.size):
        if scrambler_frame_bytes and n % (scrambler_frame_bytes * 8) == 0:
            state = [int(b) for b in "100101010000000"]
        feedback = state[13] ^ state[14]
        scrambled[n] ^= feedback
        state = [feedback] + state[:-1]
    bch_blocks = []
    for block in scrambled.reshape(-1, 752):
        word = np.concatenate((block, np.zeros(10, dtype=np.uint8)))
        work = word.copy()
        for n in range(752):
            if work[n]:
                # x^10 + x^3 + 1, MSB first.
                work[n] ^= 1; work[n + 7] ^= 1; work[n + 10] ^= 1
        word[-10:] = work[-10:]
        bch_blocks.append(word)
    message_bits = np.concatenate(bch_blocks)
    basis, parity_count = parity_basis(rate)
    encoded = []
    for message in message_bits.reshape(-1, (rate + 1) * 1524):
        word = sum(int(bit) << (parity_count + n) for n, bit in enumerate(message))
        for pivot in range(parity_count - 1, -1, -1):
            if (basis[pivot] & word).bit_count() & 1:
                word |= 1 << pivot
        assert all((row & word).bit_count() % 2 == 0 for row in basis)
        # The first five mother-code parity bits are punctured, never transmitted.
        encoded.append(np.array([(word >> n) & 1 for n in range(5, 7493)], dtype=np.uint8))
    return np.concatenate(encoded)


def modulate(bits: np.ndarray, qam: int) -> np.ndarray:
    if qam == 32:
        points = np.empty(32, dtype=np.complex64)
        for label, real, imag in QAM32_FIGURE4:
            points[sum(int(bit) << n for n, bit in enumerate(label))] = real + 1j * imag
        return points[np.sum(bits.reshape(-1, 5) * (1 << np.arange(5)), axis=1)]
    axis_bits = 2 if qam == 16 else 3
    levels = np.array([-6, -2, 2, 6] if qam == 16 else [-7, -5, -3, -1, 1, 3, 5, 7])
    labels = np.arange(len(levels)) ^ (np.arange(len(levels)) >> 1)
    lookup = np.empty_like(levels)
    lookup[labels] = levels
    words = bits.reshape(-1, 2, axis_bits)
    indices = np.sum(words * (1 << np.arange(axis_bits)), axis=2)
    return (lookup[indices[:, 0]] + 1j * lookup[indices[:, 1]]).astype(np.complex64)


def interleave(symbols: np.ndarray, mode: str, phase: int = 0) -> np.ndarray:
    step = 240 if mode == "mode1" else 720
    # Valid random constellation symbols supply unknown prehistory, not zero padding.
    rng = np.random.default_rng(452)
    out = rng.choice(symbols, size=symbols.size).astype(np.complex64)
    for branch in range(52):
        first = (branch - phase) % 52
        source = symbols[first::52]
        target = out[first::52]
        delay = branch * step
        if delay == 0:
            target[:] = source
        elif source.size > delay:
            target[delay:] = source[:-delay]
    return out


def symbol_stream(rate: int, qam: int, mode: str, phase: int = 0, count: int = 4):
    frames = transport_frames(rate, qam, count)
    bits = (encode_frame(b"".join(frames), rate, scrambler_frame_bytes=1880) if qam == 32
            else np.concatenate([encode_frame(frame, rate) for frame in frames]))
    payload = modulate(bits, qam)
    latency = 52 * 51 * (240 if mode == "mode1" else 720)
    rng = np.random.default_rng(78)
    tail = modulate(rng.integers(0, 2, latency * {16: 4, 32: 5, 64: 6}[qam], dtype=np.uint8), qam)
    return interleave(np.concatenate((payload, tail)), mode, phase), bits, b"".join(frames)


PN = {
    "pn420": (420, 255, 82, 83, 225, 2, "10110000", (0, 2, 3, 7)),
    "pn595": (595, 595, 0, 0, 216, 1, "0000000001", (0, 7)),
    "pn945": (945, 511, 217, 217, 200, 2, "111110111", (0, 1, 2, 7)),
}


def pn_phase(mode: str, frame: int) -> int:
    if mode == "pn595":
        return 0
    _, core, _, _, period, *_ = PN[mode]
    index = frame % period
    if index > period // 2:
        index = 2 * (period // 2) - index
    return ((index + 1) // 2 if index % 2 else -index // 2) % core


@lru_cache(maxsize=1024)
def pn_header(mode: str, phase: int = 0) -> np.ndarray:
    length, core_length, prefix, suffix, _, _, seed, taps = PN[mode]
    bits = [int(x) for x in seed]
    while len(bits) < length:
        n = len(bits) - len(seed)
        bit = 0
        for tap in taps:
            bit ^= bits[n + tap]
        bits.append(bit)
    if prefix:
        core = np.roll(np.array(bits[prefix:prefix + core_length]), -phase)
        bits = np.concatenate((core[-prefix:], core, core[:suffix]))
    else:
        assert phase == 0
    return ((1 - 2 * np.asarray(bits)) * (1 + 1j)).astype(np.complex64)


def pn945_header(phase: int = 0) -> np.ndarray:
    return pn_header("pn945", phase)


def ci8_frames(symbols: np.ndarray, profile: int, qam: int, *, pn_mode: str = "pn945",
               scheduled: bool = False, leading: int = 0, cfo: float = 0.0,
               noise: float = 0.0, channel: np.ndarray | None = None) -> bytes:
    info_vectors = {
        11: "00010001101000010100011110111100",
        13: "01111000001101110010111000101010",
        15: "00101101100111010111101110000000",
        17: "01110111001110000010000100100101",
        19: "00100010011011010111010001110000",
        21: "01000100000010110001001000010110",
        23: "00010001010111100100011101000011",
    }
    info = np.array([int(x) for x in info_vectors[profile if profile % 2 else profile - 1]])
    if profile % 2 == 0:
        info = 1 - info
    pilots = (1 - 2 * np.r_[np.ones(4, dtype=int), info]) * (1 + 1j)
    pilot_positions = np.array([0, 140, 279, 419])
    pilot_positions = (pilot_positions[None, :] + 420 * np.arange(9)[:, None]).ravel()
    data_positions = np.setdiff1d(np.arange(3780), pilot_positions)
    mapping = np.arange(3780).reshape(7, 5, 2, 2, 3, 3, 3).transpose(6, 5, 4, 3, 2, 1, 0).ravel()
    logical = np.empty((symbols.size // 3744, 3780), dtype=np.complex64)
    logical[:, pilot_positions] = pilots
    logical[:, data_positions] = symbols.reshape(-1, 3744)
    physical = np.empty_like(logical)
    physical[:, mapping] = logical
    bodies = np.fft.ifft(physical, axis=1)
    body_power = ({16: 40, 32: 45, 64: 42}[qam] * 3744 + 2 * 36) / 3780**2
    headers = np.stack([pn_header(pn_mode, pn_phase(pn_mode, frame) if scheduled else 0)
                        for frame in range(len(bodies))])
    headers *= np.sqrt(body_power * PN[pn_mode][5] / 2)
    frames = np.concatenate((headers, bodies), axis=1)
    signal = np.concatenate((np.zeros(leading, dtype=np.complex64), frames.ravel()))
    if channel is not None:
        signal = np.convolve(signal, channel)[:signal.size]
    if cfo:
        signal = signal * np.exp(2j * np.pi * cfo * np.arange(signal.size) / 7_560_000)
    if noise:
        rng = np.random.default_rng(374)
        signal += noise * (rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size))
    # Common complex gain, quantization and a finite headroom margin.
    signal = signal * (220 * np.exp(0.05j))
    components = np.column_stack((signal.real, signal.imag)).ravel()
    assert np.max(np.abs(components)) < 127
    return np.rint(components).astype(np.int8).tobytes()
