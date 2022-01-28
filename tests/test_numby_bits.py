import numpy as np
from grl.numby.bits import *


def test_pack_unpack():
    # test shared routine on all 8-bit numbers
    for number in np.arange(2**8):
        assert pack(unpack(np.array([number]), 8), 8) == number
    # test shared routine on all 16-bit numbers
    for number in np.arange(2**16):
        assert pack(unpack(np.array([number]), 16), 16) == number


def test_pack_unpack32():
    # test on a bunch of random numbers
    x = np.random.randint(2**32-1, size=2**15, dtype=np.uint32)
    assert np.all(pack32(unpack32(x)) == x)
    # test the edge cases
    x = np.array([0, 2**32-1], dtype=np.uint32)
    assert np.all(pack32(unpack32(x)) == x)


def test_pack_unpack64():
    # test on a bunch of random numbers
    x = np.random.randint(2**64-1, size=2**15, dtype=np.uint64)
    assert np.all(pack64(unpack64(x)) == x)
    # test the edge cases
    x = np.array([0, 2**64-1], dtype=np.uint64)
    assert np.all(pack64(unpack64(x)) == x)
