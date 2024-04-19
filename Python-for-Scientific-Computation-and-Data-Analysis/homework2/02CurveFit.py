import numpy as np
from matplotlib import pyplot as plt
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

a_fit, b_fit, c_fit = popt
y_fit = gauss(x_data, a_fit, b_fit, c_fit)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y, label='Real Data', linestyle='--')
plt.scatter(x_data, y_noise, color='red', label='Noisy Data', alpha=0.5)
plt.plot(x_data, y_fit, label='Fit Result', linestyle='-', linewidth=2)
plt.legend()
plt.show()
