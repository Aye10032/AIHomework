import numpy as np
from scipy.optimize import fsolve

d = 140
L = 156


def func(x: np.ndarray) -> np.ndarray:
    r, a = x.tolist()

    return np.array([
        np.cos(a) - 1 + d ** 2 / (2 * r ** 2),
        L - a * r
    ])


def jacobian(x: np.ndarray) -> np.ndarray:
    r, a, = x.tolist()
    return np.array([
        [-np.sin(a), -d ** 2 * r ** (-3)],
        [-a, -r]
    ])


result = fsolve(func, np.array([1, 1]), fprime=jacobian)

print(result)
print(func(result))
