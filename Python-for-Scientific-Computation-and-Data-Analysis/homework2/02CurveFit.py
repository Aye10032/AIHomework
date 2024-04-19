import numpy as np
from scipy.optimize import curve_fit

a_real = 1
b_real = 5
c_real = 2


def gauss(x: np.ndarray, a: float, b: float, c: float):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


x_data = np.linspace(0, 10, 200)
y = gauss(x_data, a_real, b_real, c_real)
y_noise = y + 0.2 * np.random.normal(size=x_data.size)

popt, pcov = curve_fit(gauss, x_data, y_noise, p0=[1, 1, 1])

print(popt)
