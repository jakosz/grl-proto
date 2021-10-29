import json

import numpy as np


class JsonNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64) or isinstance(obj, np.uint32) or isinstance(obj, np.uint64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def random_hex(n=16):
    return np.random.randint(0, 2**63, 8).tobytes().hex()[:n]


NumpyJson = JsonNumpy  # I don't want to have to remember this
