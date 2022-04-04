from grl import numby


def get(activation):
    if activation is None:
        return numby.identity
    else:
        return getattr(numby, activation)
