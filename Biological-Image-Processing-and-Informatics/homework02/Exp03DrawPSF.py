import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import j1


def psf(r: np.ndarray, light_wave_l, na) -> np.ndarray:
    k = 2 * np.pi / light_wave_l
    rho = k * na * r

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (2 * j1(rho) / rho) ** 2
        z[rho == 0] = 1

    return z


def gauss(_x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2))


LAMBDA = 0.48
NA = 0.5

x = np.linspace(-2, 2, 200)
y = psf(x, LAMBDA, NA)
r0 = 0.61 * LAMBDA / NA
y2 = gauss(x, r0 / 3)

sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.lineplot(x=x, y=y, ax=ax)
sns.lineplot(x=x, y=y2, ax=ax)
ax.axvline(x=r0, color='red', linestyle='--')

plt.show()
