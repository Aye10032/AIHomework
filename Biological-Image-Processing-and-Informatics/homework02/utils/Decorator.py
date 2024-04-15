import time
from functools import wraps
from typing import Callable

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
