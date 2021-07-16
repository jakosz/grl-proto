import logging
import sys


def get_stdout_logger(name, level=logging.DEBUG):
    log = logging.getLogger(name)
    log.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    f = logging.Formatter('%(asctime)s %(name)s [%(levelname)s] %(message)s')
    h.setFormatter(f)
    log.addHandler(h)
    return log
