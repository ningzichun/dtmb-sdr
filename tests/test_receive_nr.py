"""GB 20600 NR table and joint-likelihood regressions.

The transmitter uses Appendix C, independently of the receiver's equations.
No received capture, payload repair or receiver-generated reference is used.
"""

from functools import lru_cache
from pathlib import Path
import subprocess

import numpy as np
import pytest

from synthetic_support import modulate

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def appendix_c():
    rows = np.array([line.split() for line in (ROOT / "core/cpp/tests/nr_appendix_c.txt").read_text().splitlines()
                     if line and not line.startswith("#")])
    assert [int(x, 16) for x in rows[:, 0]] == list(range(256))
    words = np.array([int(x, 16) for x in rows[:, 1]], dtype=np.uint16)
    return ((words[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.uint8)


def nr_modulate(bits):
    indices = np.packbits(bits.reshape(-1, 8), axis=1).ravel()
    return modulate(appendix_c()[indices].ravel(), 4)


def run(command, data=None, cwd=None):
    result = subprocess.run(command, input=data, capture_output=True, timeout=120, cwd=cwd)
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    return result


@pytest.mark.parametrize("method", ["max-log", "log-sum-exp"])
@pytest.mark.parametrize("chunk", [8, 1872, 3744])
def test_nr_soft_decode_matches_all_appendix_c_joint_likelihoods(native, method, chunk):
    reference = modulate(appendix_c().ravel(), 4).reshape(256, 8)
    flat = reference.reshape(-1)
    result = run(native("qam_softdemap", "--qam", "4qam-nr",
                        "--soft-demod-method", method,
                        "--chunk-symbols", str(chunk), "-", "-"),
                 flat.astype(np.complex64).tobytes())
    llr = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 8)
    np.testing.assert_array_equal(np.packbits((llr < 0).astype(np.uint8), axis=1).ravel(),
                                  np.arange(256, dtype=np.uint8))
