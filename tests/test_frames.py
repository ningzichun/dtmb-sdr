import numpy as np
import pytest

from dtmb.frames import iter_signal_frames


def test_iter_signal_frames_splits_header_and_body():
    signal = np.arange(10_000, dtype=np.float32).astype(np.complex64)

    frames = list(iter_signal_frames(signal, mode="pn420", phase_offset=10, max_frames=2))

    assert len(frames) == 2
    assert frames[0].start == 10
    assert frames[0].header.size == 420
    assert frames[0].body.size == 3780
    assert frames[1].start == 4210


def test_iter_signal_frames_rejects_negative_phase_offset():
    with pytest.raises(ValueError, match="phase_offset"):
        list(iter_signal_frames(np.zeros(100, dtype=np.complex64), mode="pn945", phase_offset=-1))
