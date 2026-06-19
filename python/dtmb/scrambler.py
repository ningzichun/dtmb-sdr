"""DTMB transport-stream bit scrambler."""

from __future__ import annotations

import numpy as np


SCRAMBLER_INITIAL_STATE = "100101010000000"


def scrambling_sequence(
    length: int,
    *,
    initial_state: str = SCRAMBLER_INITIAL_STATE,
) -> np.ndarray:
    """Generate the DTMB data scrambling PN bits.

    GB 20600-2006 gives `G(x)=1+x^14+x^15` and initial state
    `100101010000000`. Figure 2 numbers the registers D1..D15 from left to
    right: the output is the D14 XOR D15 feedback bit, which shifts into D1
    while D1 shifts toward D2. The same sequence is used for scrambling and
    descrambling because the operation is XOR.
    """

    if length < 0:
        raise ValueError("length must be non-negative")
    if len(initial_state) != 15 or any(bit not in "01" for bit in initial_state):
        raise ValueError("initial_state must contain 15 binary digits")
    state = [int(bit) for bit in initial_state]
    output: list[int] = []
    for _ in range(length):
        next_bit = state[13] ^ state[14]
        output.append(next_bit)
        state = [next_bit] + state[:-1]
    return np.asarray(output, dtype=np.uint8)


def scramble_bits(
    bits: np.ndarray,
    *,
    initial_state: str = SCRAMBLER_INITIAL_STATE,
) -> np.ndarray:
    """Scramble or descramble bits with the DTMB data PN sequence."""

    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return (values ^ scrambling_sequence(values.size, initial_state=initial_state)).astype(
        np.uint8
    )
