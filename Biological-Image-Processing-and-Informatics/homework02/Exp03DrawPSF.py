import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j1


def psf(r, light_wave_l, na):
    k = 2 * np.pi / light_wave_l
    rho = k * na * r

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (2 * j1(rho) / rho) ** 2
        z[rho == 0] = 1

    return z


LAMBDA = 0.48
NA = 0.5

x = np.linspace(-2, 2, 200)
y = psf(x, LAMBDA, NA)
r0 = 0.61 * LAMBDA / NA

plt.plot(x, y)
plt.axvline(x=r0, linestyle='--', color='r')
plt.show()
