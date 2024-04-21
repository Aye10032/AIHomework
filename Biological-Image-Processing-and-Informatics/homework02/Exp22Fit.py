from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import j1
from loguru import logger


@dataclass
class Variable:
    LAMBDA: float
    NA: float
    FIT_SIGMA: float = 0

    def get_sigma(self, ndigits=2):
        return round((0.61 * self.LAMBDA / self.NA) / 3, ndigits)


def psf(r: np.ndarray, light_wave_l: float, na: float) -> np.ndarray:
    k = 2 * np.pi / light_wave_l
    rho = k * na * r

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (2 * j1(rho) / rho) ** 2
        z[rho == 0] = 1

    return z


def gauss(_x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-_x ** 2 / (2 * sigma ** 2))


variables = [
    Variable(0.48, 0.5),
    Variable(0.52, 0.5),
    Variable(0.68, 0.5),
    Variable(0.52, 1.0),
    Variable(0.52, 1.4),
    Variable(0.68, 1.5),
]

x = np.linspace(-1.8, 1.8, 200)
for vari in variables:
    min_loss = np.inf
    for si in range(1, 600):
        _sigma = 0.005 * si
        y1 = psf(x, vari.LAMBDA, vari.NA)
        y2 = gauss(x, _sigma)
        loss = abs(np.sum(y1 - y2))
        if loss < min_loss:
            min_loss = loss
            vari.FIT_SIGMA = _sigma

    print(vari.FIT_SIGMA, vari.get_sigma(3))
