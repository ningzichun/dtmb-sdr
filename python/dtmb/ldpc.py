"""Generic LDPC helpers plus DTMB LDPC profile metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DtmbLdpcProfile:
    """DTMB LDPC profile metadata from GB 20600-2006."""

    fec_rate_index: int
    nominal_rate: float
    k: int
    c: int
    b: int
    message_bits: int
    erased_parity_bits: int = 5
    codeword_bits: int = 7488

    @property
    def full_parity_bits(self) -> int:
        return self.c * self.b

    @property
    def full_codeword_bits(self) -> int:
        return self.message_bits + self.full_parity_bits

    @property
    def parity_bits(self) -> int:
        return self.full_parity_bits - self.erased_parity_bits

    @property
    def bch_blocks_per_codeword(self) -> int:
        return self.message_bits // 762

    def to_dict(self) -> dict[str, Any]:
        return {
            "fec_rate_index": self.fec_rate_index,
            "nominal_rate": self.nominal_rate,
            "k": self.k,
            "c": self.c,
            "b": self.b,
            "message_bits": self.message_bits,
            "full_codeword_bits": self.full_codeword_bits,
            "codeword_bits": self.codeword_bits,
            "full_parity_bits": self.full_parity_bits,
            "parity_bits": self.parity_bits,
            "erased_parity_bits": self.erased_parity_bits,
            "bch_blocks_per_codeword": self.bch_blocks_per_codeword,
        }


DTMB_LDPC_PROFILES: dict[int, DtmbLdpcProfile] = {
    1: DtmbLdpcProfile(
        fec_rate_index=1,
        nominal_rate=0.4,
        k=24,
        c=35,
        b=127,
        message_bits=3048,
    ),
    2: DtmbLdpcProfile(
        fec_rate_index=2,
        nominal_rate=0.6,
        k=36,
        c=23,
        b=127,
        message_bits=4572,
    ),
    3: DtmbLdpcProfile(
        fec_rate_index=3,
        nominal_rate=0.8,
        k=48,
        c=11,
        b=127,
        message_bits=6096,
    ),
}


@dataclass(frozen=True)
class LdpcDecodeResult:
    """Result from a generic LDPC hard decision decoder."""

    bits: np.ndarray
    iterations: int
    converged: bool
    syndrome_weight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "syndrome_weight": self.syndrome_weight,
            "bits": int(self.bits.size),
        }


@dataclass(frozen=True)
class LdpcSparseGraph:
    """Sparse LDPC Tanner graph in edge-index form."""

    check_edges: tuple[np.ndarray, ...]
    variable_edges: tuple[np.ndarray, ...]
    edge_variables: np.ndarray
    check_count: int
    variable_count: int

    @property
    def edge_count(self) -> int:
        return int(self.edge_variables.size)


def dtmb_ldpc_profile(fec_rate_index: int) -> DtmbLdpcProfile:
    """Return one DTMB LDPC profile by system-information FEC rate index."""

    try:
        return DTMB_LDPC_PROFILES[int(fec_rate_index)]
    except KeyError as exc:
        raise ValueError("fec_rate_index must be 1, 2, or 3") from exc


def dtmb_ldpc_generator_hex(fec_rate_index: int) -> tuple[tuple[str, ...], ...]:
    """Return DTMB LDPC quasi-cyclic generator blocks as standard hex strings."""

    rate_index = dtmb_ldpc_profile(fec_rate_index).fec_rate_index
    raw = _load_generator_hex()[str(rate_index)]
    return tuple(tuple(row) for row in raw)


def dtmb_ldpc_generator_polynomials(fec_rate_index: int) -> np.ndarray:
    """Return generator block polynomials with shape `(k, c, b)`.

    Each 32-hex-digit standard entry stores one 127-bit circulant polynomial in
    the low 127 bits of a 128-bit field.
    """

    return _generator_polynomials_cached(fec_rate_index).copy()


def dtmb_ldpc_parity_check_shifts(
    fec_rate_index: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return standard QC-LDPC parity-check shifts as `(column, shift)` rows."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    raw = _load_parity_check_shifts()[str(profile.fec_rate_index)]
    rows = tuple(tuple((int(col), int(shift)) for col, shift in row) for row in raw)
    if len(rows) != profile.c:
        raise ValueError("LDPC parity-check row count does not match profile")
    column_count = profile.c + profile.k
    for row in rows:
        for col, shift in row:
            if col < 0 or col >= column_count or shift < 0 or shift >= profile.b:
                raise ValueError("LDPC parity-check entry is out of range")
    return rows


def dtmb_ldpc_encode_message_bits(
    message_bits: np.ndarray,
    *,
    fec_rate_index: int,
    transmitted: bool = True,
) -> np.ndarray:
    """Encode BCH output bits with the DTMB LDPC generator matrix.

    The default return value is the transmitted 7488-bit DTMB layout with the
    five erased parity bits removed. Set `transmitted=False` to keep the full
    7493-bit mother-code layout.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(message_bits)
    if values.size == 0 or values.size % profile.message_bits:
        raise ValueError("message_bits must contain whole LDPC messages")
    parity_matrix = _parity_generator_matrix_cached(fec_rate_index)
    messages = values.reshape(-1, profile.message_bits)
    # bits.size * full_parity_bits ops through numpy matmul is ~10x faster than
    # the per-bit Python loop that lived here previously, which matters when
    # the continuous-stream fixture pushes 500+ codewords per rate.
    parity = (
        messages.astype(np.uint32) @ parity_matrix.astype(np.uint32)
    ) & 1
    full = np.concatenate(
        (parity.astype(np.uint8), messages),
        axis=1,
    ).reshape(-1)
    if not transmitted:
        return full
    return collapse_dtmb_full_codeword_bits(full, fec_rate_index=fec_rate_index)


def dtmb_ldpc_parity_mismatch_counts(
    codeword_bits: np.ndarray,
    *,
    fec_rate_index: int,
) -> np.ndarray:
    """Return parity mismatch counts for transmitted DTMB LDPC codewords.

    This is a hard-decision consistency check, not an LDPC decoder. For a valid
    transmitted 7488-bit codeword the count is zero. For random hard decisions
    it should be near half of the transmitted parity bits.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(codeword_bits)
    if values.size % profile.codeword_bits:
        raise ValueError("bit count must contain whole transmitted LDPC codewords")
    counts = []
    for row in values.reshape(-1, profile.codeword_bits):
        message = row[profile.parity_bits :]
        expected = dtmb_ldpc_encode_message_bits(
            message,
            fec_rate_index=fec_rate_index,
            transmitted=True,
        )
        counts.append(int(np.count_nonzero(row[: profile.parity_bits] != expected[: profile.parity_bits])))
    return np.asarray(counts, dtype=np.int32)


def expand_dtmb_transmitted_llr(
    llr: np.ndarray,
    *,
    fec_rate_index: int,
    erased_llr: float = 0.0,
) -> np.ndarray:
    """Insert DTMB's five erased LDPC parity-bit LLRs before decoding.

    DTMB transmits `[p(5), ..., p(c*b-1), m(0), ..., m(k*b-1)]`.
    The mother code is `[p(0), ..., p(c*b-1), m(0), ..., m(k*b-1)]`.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    if values.size % profile.codeword_bits:
        raise ValueError("LLR count must contain whole transmitted LDPC codewords")
    rows = values.reshape(-1, profile.codeword_bits)
    expanded = np.empty((rows.shape[0], profile.full_codeword_bits), dtype=np.float32)
    expanded[:, : profile.erased_parity_bits] = np.float32(erased_llr)
    expanded[:, profile.erased_parity_bits : profile.full_parity_bits] = rows[
        :, : profile.parity_bits
    ]
    expanded[:, profile.full_parity_bits :] = rows[:, profile.parity_bits :]
    return expanded.reshape(-1)


def collapse_dtmb_full_codeword_bits(
    bits: np.ndarray,
    *,
    fec_rate_index: int,
) -> np.ndarray:
    """Drop the five erased parity positions from full DTMB LDPC codewords."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(bits)
    if values.size % profile.full_codeword_bits:
        raise ValueError("bit count must contain whole full LDPC codewords")
    rows = values.reshape(-1, profile.full_codeword_bits)
    collapsed = np.concatenate(
        (
            rows[:, profile.erased_parity_bits : profile.full_parity_bits],
            rows[:, profile.full_parity_bits :],
        ),
        axis=1,
    )
    return collapsed.reshape(-1).astype(np.uint8)


def extract_dtmb_message_bits_from_full_codewords(
    bits: np.ndarray,
    *,
    fec_rate_index: int,
) -> np.ndarray:
    """Return BCH-protected message bits from full DTMB LDPC codewords."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(bits)
    if values.size % profile.full_codeword_bits:
        raise ValueError("bit count must contain whole full LDPC codewords")
    rows = values.reshape(-1, profile.full_codeword_bits)
    return rows[:, profile.full_parity_bits :].reshape(-1).astype(np.uint8)


def dtmb_ldpc_syndrome_weight(bits: np.ndarray, *, fec_rate_index: int) -> int:
    """Return the QC-LDPC syndrome weight for full 7493-bit DTMB codewords."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(bits)
    if values.size != profile.full_codeword_bits:
        raise ValueError("bits must contain one full DTMB LDPC mother codeword")
    return ldpc_sparse_syndrome_weight(
        values,
        _dtmb_ldpc_graph_cached(profile.fec_rate_index),
    )


def dtmb_ldpc_appendix_b_syndrome_counts(
    codeword_bits: np.ndarray,
    *,
    fec_rate_index: int,
) -> tuple[np.ndarray, int]:
    """Syndrome weights over Appendix-B rows that miss the 5 erased parity bits.

    The DTMB transmitted codeword is the 7493-bit mother codeword with its
    first five parity bits deleted. For a generator-independent consistency
    check we reconstruct a best-effort full codeword by setting those five
    erased bits to zero, compute ``H @ c`` with the Appendix-B parity-check
    matrix, and then only count the rows whose edges all miss the erased
    positions. Those rows' syndromes are determined entirely by the received
    bits, so they can never be reconciled by toggling the erased bits.

    For a valid transmitted codeword this weight is zero. For random bits it
    averages half of the row count. The diagnostic lets the receiver tell a
    generator-convention fault (generator parity mismatch is ~0.5 yet this
    syndrome is zero) apart from an upstream bit-order fault (both are ~0.5).
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = _as_binary(codeword_bits)
    if values.size == 0 or values.size % profile.codeword_bits:
        raise ValueError("bit count must contain whole transmitted LDPC codewords")
    codewords = values.reshape(-1, profile.codeword_bits)
    erased_padding = np.zeros(
        (codewords.shape[0], profile.erased_parity_bits),
        dtype=np.uint8,
    )
    full = np.concatenate((erased_padding, codewords), axis=1)
    clean_variables = _dtmb_ldpc_clean_check_variables(profile.fec_rate_index)
    weights = np.zeros(codewords.shape[0], dtype=np.int64)
    for variables in clean_variables:
        weights += np.bitwise_xor.reduce(full[:, variables], axis=1, initial=0)
    return weights.astype(np.int64, copy=False), int(len(clean_variables))


def dtmb_ldpc_decode_transmitted_llr(
    llr: np.ndarray,
    *,
    fec_rate_index: int,
    max_iterations: int = 50,
    attenuation: float = 0.75,
    erased_llr: float = 0.0,
) -> LdpcDecodeResult:
    """Decode one transmitted 7488-bit DTMB LDPC codeword.

    The returned bit vector uses the full 7493-bit mother-code layout, including
    the five erased parity bits recovered by the decoder when convergence is
    possible.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    if values.size != profile.codeword_bits:
        raise ValueError("LLR input must contain one transmitted DTMB LDPC codeword")
    expanded = expand_dtmb_transmitted_llr(
        values,
        fec_rate_index=profile.fec_rate_index,
        erased_llr=erased_llr,
    )
    return ldpc_decode_min_sum_sparse(
        expanded,
        _dtmb_ldpc_graph_cached(profile.fec_rate_index),
        max_iterations=max_iterations,
        attenuation=attenuation,
    )


def ldpc_decode_min_sum(
    llr: np.ndarray,
    parity_check: np.ndarray,
    *,
    max_iterations: int = 50,
    attenuation: float = 0.75,
) -> LdpcDecodeResult:
    """Decode LDPC bits with normalized min-sum belief propagation."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if attenuation <= 0:
        raise ValueError("attenuation must be positive")
    h = np.asarray(parity_check, dtype=np.uint8)
    if h.ndim != 2:
        raise ValueError("parity_check must be a two-dimensional matrix")
    if np.any((h != 0) & (h != 1)):
        raise ValueError("parity_check must be binary")
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    check_count, variable_count = h.shape
    if values.size != variable_count:
        raise ValueError("LLR count must match parity-check column count")

    check_neighbors = [np.flatnonzero(h[row]) for row in range(check_count)]
    variable_neighbors = [np.flatnonzero(h[:, col]) for col in range(variable_count)]
    check_to_var = np.zeros((check_count, variable_count), dtype=np.float32)
    var_to_check = np.zeros((check_count, variable_count), dtype=np.float32)
    for check_index, variables in enumerate(check_neighbors):
        var_to_check[check_index, variables] = values[variables]

    hard = (values < 0).astype(np.uint8)
    syndrome_weight = syndrome(h, hard)
    if syndrome_weight == 0:
        return LdpcDecodeResult(bits=hard, iterations=0, converged=True, syndrome_weight=0)

    for iteration in range(1, max_iterations + 1):
        for check_index, variables in enumerate(check_neighbors):
            if variables.size == 0:
                continue
            messages = var_to_check[check_index, variables]
            signs = np.where(messages < 0, -1.0, 1.0)
            abs_messages = np.abs(messages)
            sign_product = float(np.prod(signs))
            if variables.size == 1:
                check_to_var[check_index, variables[0]] = 0.0
                continue
            order = np.argsort(abs_messages)
            min1 = float(abs_messages[order[0]])
            min2 = float(abs_messages[order[1]])
            for local_index, variable in enumerate(variables):
                magnitude = min2 if local_index == order[0] else min1
                sign = sign_product * signs[local_index]
                check_to_var[check_index, variable] = attenuation * sign * magnitude

        posterior = values.copy()
        for variable, checks in enumerate(variable_neighbors):
            posterior[variable] += float(np.sum(check_to_var[checks, variable]))
            for check in checks:
                var_to_check[check, variable] = posterior[variable] - check_to_var[
                    check, variable
                ]

        hard = (posterior < 0).astype(np.uint8)
        syndrome_weight = syndrome(h, hard)
        if syndrome_weight == 0:
            return LdpcDecodeResult(
                bits=hard,
                iterations=iteration,
                converged=True,
                syndrome_weight=0,
            )

    return LdpcDecodeResult(
        bits=hard,
        iterations=max_iterations,
        converged=False,
        syndrome_weight=syndrome_weight,
    )


def ldpc_decode_min_sum_sparse(
    llr: np.ndarray,
    graph: LdpcSparseGraph,
    *,
    max_iterations: int = 50,
    attenuation: float = 0.75,
) -> LdpcDecodeResult:
    """Decode LDPC bits with normalized min-sum over a sparse Tanner graph."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if attenuation <= 0:
        raise ValueError("attenuation must be positive")
    values = np.asarray(llr, dtype=np.float32).reshape(-1)
    if values.size != graph.variable_count:
        raise ValueError("LLR count must match graph variable count")

    check_to_var = np.zeros(graph.edge_count, dtype=np.float32)
    var_to_check = values[graph.edge_variables].astype(np.float32, copy=True)
    hard = (values < 0).astype(np.uint8)
    syndrome_weight = ldpc_sparse_syndrome_weight(hard, graph)
    if syndrome_weight == 0:
        return LdpcDecodeResult(bits=hard, iterations=0, converged=True, syndrome_weight=0)

    for iteration in range(1, max_iterations + 1):
        for edges in graph.check_edges:
            messages = var_to_check[edges]
            signs = np.where(messages < 0, -1.0, 1.0).astype(np.float32)
            abs_messages = np.abs(messages)
            sign_product = float(np.prod(signs))
            if edges.size == 1:
                check_to_var[edges[0]] = 0.0
                continue
            order = np.argsort(abs_messages)
            min1 = float(abs_messages[order[0]])
            min2 = float(abs_messages[order[1]])
            magnitudes = np.full(edges.size, min1, dtype=np.float32)
            magnitudes[order[0]] = min2
            check_to_var[edges] = attenuation * sign_product * signs * magnitudes

        posterior = values.copy()
        np.add.at(posterior, graph.edge_variables, check_to_var)
        var_to_check = posterior[graph.edge_variables] - check_to_var

        hard = (posterior < 0).astype(np.uint8)
        syndrome_weight = ldpc_sparse_syndrome_weight(hard, graph)
        if syndrome_weight == 0:
            return LdpcDecodeResult(
                bits=hard,
                iterations=iteration,
                converged=True,
                syndrome_weight=0,
            )

    return LdpcDecodeResult(
        bits=hard,
        iterations=max_iterations,
        converged=False,
        syndrome_weight=syndrome_weight,
    )


def ldpc_sparse_syndrome_weight(bits: np.ndarray, graph: LdpcSparseGraph) -> int:
    """Return the syndrome Hamming weight for a sparse Tanner graph."""

    values = _as_binary(bits)
    if values.size != graph.variable_count:
        raise ValueError("bit count must match graph variable count")
    weight = 0
    for edges in graph.check_edges:
        variables = graph.edge_variables[edges]
        weight += int(np.bitwise_xor.reduce(values[variables], initial=0))
    return weight


def llr_from_hard_bits(bits: np.ndarray, *, magnitude: float = 8.0) -> np.ndarray:
    """Convert hard bits to signed LLR values for decoder tests."""

    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return np.where(values == 0, magnitude, -magnitude).astype(np.float32)


def syndrome(parity_check: np.ndarray, bits: np.ndarray) -> int:
    """Return the parity-check syndrome Hamming weight."""

    h = np.asarray(parity_check, dtype=np.uint8)
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    residual = (h @ values) & 1
    return int(np.count_nonzero(residual))


@lru_cache(maxsize=1)
def _load_generator_hex() -> dict[str, list[list[str]]]:
    path = resources.files("dtmb").joinpath("data/dtmb_ldpc_generator.json")
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_parity_check_shifts() -> dict[str, list[list[list[int]]]]:
    path = resources.files("dtmb").joinpath("data/dtmb_ldpc_parity_check.json")
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


@lru_cache(maxsize=3)
def _generator_polynomials_cached(fec_rate_index: int) -> np.ndarray:
    profile = dtmb_ldpc_profile(fec_rate_index)
    matrix = dtmb_ldpc_generator_hex(fec_rate_index)
    if len(matrix) != profile.k or any(len(row) != profile.c for row in matrix):
        raise ValueError("LDPC generator matrix dimensions do not match profile")
    polynomials = np.empty((profile.k, profile.c, profile.b), dtype=np.uint8)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            polynomials[row_index, col_index] = _hex_to_127_bits(value)
    return polynomials


@lru_cache(maxsize=3)
def _parity_generator_matrix_cached(fec_rate_index: int) -> np.ndarray:
    """Build the dense `(message_bits, full_parity_bits)` generator matrix.

    Row `(mb * b + j)` holds the bit pattern that message-bit `j` of message
    block `mb` contributes to every parity block when XORed through the DTMB
    QC-LDPC encoder. This mirrors the ``np.roll(polynomials[mb, pb], j)``
    accumulation in the original per-bit encoder while letting numpy matmul
    compute every codeword at once.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    polynomials = _generator_polynomials_cached(fec_rate_index)
    b = profile.b
    matrix = np.empty(
        (profile.k * b, profile.c * b),
        dtype=np.uint8,
    )
    indices = np.arange(b)
    for mb in range(profile.k):
        for pb in range(profile.c):
            poly = polynomials[mb, pb]
            # circulant[j, k] = poly[(k - j) mod b] matches np.roll(poly, j)[k]
            shift = (indices[None, :] - indices[:, None]) % b
            matrix[
                mb * b : (mb + 1) * b,
                pb * b : (pb + 1) * b,
            ] = poly[shift]
    return matrix


@lru_cache(maxsize=3)
def _dtmb_ldpc_graph_cached(fec_rate_index: int) -> LdpcSparseGraph:
    profile = dtmb_ldpc_profile(fec_rate_index)
    rows = dtmb_ldpc_parity_check_shifts(profile.fec_rate_index)
    check_count = profile.c * profile.b
    variable_count = (profile.c + profile.k) * profile.b
    check_edges: list[list[int]] = [[] for _ in range(check_count)]
    variable_edges: list[list[int]] = [[] for _ in range(variable_count)]
    edge_variables: list[int] = []
    for row_block, row in enumerate(rows):
        for column_block, shift in row:
            for row_position in range(profile.b):
                check_index = row_block * profile.b + row_position
                variable_index = (
                    column_block * profile.b
                    + ((row_position + shift) % profile.b)
                )
                edge_index = len(edge_variables)
                edge_variables.append(variable_index)
                check_edges[check_index].append(edge_index)
                variable_edges[variable_index].append(edge_index)
    return LdpcSparseGraph(
        check_edges=tuple(
            np.asarray(edges, dtype=np.int32) for edges in check_edges
        ),
        variable_edges=tuple(
            np.asarray(edges, dtype=np.int32) for edges in variable_edges
        ),
        edge_variables=np.asarray(edge_variables, dtype=np.int32),
        check_count=check_count,
        variable_count=variable_count,
    )


@lru_cache(maxsize=3)
def _dtmb_ldpc_clean_check_rows(fec_rate_index: int) -> np.ndarray:
    """Indices of Appendix-B check rows that do not touch the erased parity bits.

    DTMB transmits the 7493-bit mother codeword with its first
    ``erased_parity_bits`` variables (mother-code indices 0..4) removed. Any
    parity-check row that touches one of those variables cannot be evaluated
    from the receiver's bit stream alone; filtering them out leaves a subset
    of the parity matrix whose syndrome is fully determined by the received
    bits and so serves as a generator-independent consistency check.
    """

    profile = dtmb_ldpc_profile(fec_rate_index)
    graph = _dtmb_ldpc_graph_cached(profile.fec_rate_index)
    erased = set(range(profile.erased_parity_bits))
    clean: list[int] = []
    for check_index, edges in enumerate(graph.check_edges):
        if edges.size == 0:
            continue
        variables = graph.edge_variables[edges]
        if not any(int(v) in erased for v in variables):
            clean.append(check_index)
    return np.asarray(clean, dtype=np.int32)


@lru_cache(maxsize=3)
def _dtmb_ldpc_clean_check_variables(fec_rate_index: int) -> tuple[np.ndarray, ...]:
    """Variable indices for clean Appendix-B rows, cached for syndrome sweeps."""

    profile = dtmb_ldpc_profile(fec_rate_index)
    graph = _dtmb_ldpc_graph_cached(profile.fec_rate_index)
    return tuple(
        graph.edge_variables[graph.check_edges[int(clean_row)]]
        for clean_row in _dtmb_ldpc_clean_check_rows(profile.fec_rate_index)
    )


def _encode_one_ldpc_message(
    message: np.ndarray,
    *,
    profile: DtmbLdpcProfile,
    polynomials: np.ndarray,
) -> np.ndarray:
    message_blocks = message.reshape(profile.k, profile.b)
    parity_blocks = np.zeros((profile.c, profile.b), dtype=np.uint8)
    for message_block_index in range(profile.k):
        one_positions = np.flatnonzero(message_blocks[message_block_index])
        if one_positions.size == 0:
            continue
        for parity_block_index in range(profile.c):
            polynomial = polynomials[message_block_index, parity_block_index]
            parity = parity_blocks[parity_block_index]
            for bit_position in one_positions:
                parity ^= np.roll(polynomial, int(bit_position))
    return np.concatenate((parity_blocks.reshape(-1), message)).astype(np.uint8)


def _hex_to_127_bits(value: str) -> np.ndarray:
    if len(value) != 32:
        raise ValueError("LDPC generator entries must be 32 hex digits")
    integer = int(value, 16)
    return np.asarray([(integer >> shift) & 1 for shift in range(127)], dtype=np.uint8)


def _as_binary(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must be binary")
    return values
