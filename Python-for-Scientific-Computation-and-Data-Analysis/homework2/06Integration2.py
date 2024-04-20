import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import fsolve


def int_n(n: float) -> float:
    i_n = dblquad(
        lambda x, t: (np.exp(-x * t)) / t ** n,
        np.inf, 0,
        np.inf, 1
    )

    return i_n[0]


def func(inputs: np.ndarray) -> np.ndarray:
    n, = inputs.tolist()
    return np.array([int_n(n) - 1 / n])


result = fsolve(
    func,
    np.array([1])
)

print(result)
