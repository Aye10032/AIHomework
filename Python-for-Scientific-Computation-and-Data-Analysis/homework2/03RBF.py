import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def func(_x, _y):
    return (_x + _y) * np.exp(-5.0 * (_x ** 2 + _y ** 2))


x = np.array([-1, 0, 2.0, 1.0])
y = np.array([1.0, 0.3, -0.5, 0.8])
z = func(x, y)

xi, yi = np.meshgrid(np.linspace(-3, 4, 100), np.linspace(-3, 4, 100))

methods = ['multiquadric', 'gaussian', 'linear']

fig = plt.figure(figsize=(18, 6))

for i, method in enumerate(methods):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    rbf = Rbf(x, y, z, function=method)
    zi = rbf(xi, yi)

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)

    ax.scatter(x, y, z, color='red', s=50)

    ax.set_title(f'Method: {method}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
