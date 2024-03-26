import time
from functools import wraps
from typing import Callable

import numpy as np
from loguru import logger


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f'{func.__name__} Time cost: {end - start}s')
        return result

    return wrapper


def triangle_wave(_x: float, _c: float, _c0: float, _hc: float) -> float:
    _x = _x - int(_x)
    if _x >= _c:
        r = 0.0
    elif _x < _c0:
        r = _x / _c0 * _hc
    else:
        r = (_c - _x) / (_c - _c0) * _hc

    return r


@timer
def type1(_x: np.ndarray) -> np.ndarray:
    y1 = np.array([triangle_wave(t, 0.6, 0.4, 1.0) for t in _x])
    return y1


@timer
def type2(_x: np.ndarray) -> np.ndarray:
    triangle_ufunc1 = np.frompyfunc(triangle_wave, 4, 1)
    y2 = triangle_ufunc1(_x, 0.6, 0.4, 1.0)

    return y2


@timer
def type3(_x: np.ndarray) -> np.ndarray:
    triangle_ufunc2 = np.frompyfunc(lambda s: triangle_wave(s, 0.6, 0.4, 1.0), 1, 1)
    y3 = triangle_ufunc2(_x)

    return y3


@timer
def type4(_x: np.ndarray) -> np.ndarray:
    triangle_ufunc3 = np.vectorize(triangle_wave, otypes=[np.float64])
    y4 = triangle_ufunc3(x, 0.6, 0.4, 1.0)

    return y4


if __name__ == '__main__':
    x = np.linspace(0, 2, 1000)

    type1(x)
    type2(x)
    type3(x)
    type4(x)
