import numpy as np

from dtmb.ci8 import inspect_ci8_file, read_ci8


def test_read_ci8_converts_signed_interleaved_iq(tmp_path):
    capture = tmp_path / "sample.ci8"
    capture.write_bytes(bytes([0, 0, 127, 129]))

    samples = read_ci8(capture)

    np.testing.assert_allclose(samples, np.array([0 + 0j, 127 / 128 - 127j / 128], dtype=np.complex64))


def test_read_ci8_ignores_trailing_byte(tmp_path):
    capture = tmp_path / "odd.ci8"
    capture.write_bytes(bytes([1, 2, 3]))

    info = inspect_ci8_file(capture)
    samples = read_ci8(capture)

    assert info.byte_count == 3
    assert info.sample_count == 1
    assert info.has_trailing_byte is True
    assert samples.size == 1


def test_read_ci8_seeks_to_sample_offset(tmp_path):
    capture = tmp_path / "offset.ci8"
    capture.write_bytes(bytes([1, 2, 3, 4, 5, 6, 7, 8]))

    samples = read_ci8(capture, max_samples=2, skip_samples=1)

    np.testing.assert_allclose(
        samples,
        np.array([3 / 128 + 4j / 128, 5 / 128 + 6j / 128], dtype=np.complex64),
    )
