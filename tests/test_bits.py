import numpy as np

from dtmb.bits import pack_bits_msb, unpack_bytes_msb


def test_pack_unpack_bits_msb_round_trips_bytes():
    data = bytes([0x47, 0x80, 0x01, 0xFE])

    bits = unpack_bytes_msb(data)

    assert bits.tolist()[:8] == [0, 1, 0, 0, 0, 1, 1, 1]
    assert pack_bits_msb(bits) == data


def test_pack_bits_rejects_incomplete_byte_by_default():
    bits = np.asarray([1, 0, 1], dtype=np.uint8)

    try:
        pack_bits_msb(bits)
    except ValueError as exc:
        assert "multiple of 8" in str(exc)
    else:
        raise AssertionError("expected incomplete byte rejection")
