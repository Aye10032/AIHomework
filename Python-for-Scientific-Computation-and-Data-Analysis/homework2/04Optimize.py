import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_bfgs, fminbound, brute


def func(x: np.ndarray) -> np.ndarray:
    return x ** 2 + 10 * np.sin(x)


result1 = fmin_bfgs(func, np.array([0]))
print(result1)

result2 = fminbound(func, -10, 10)
print(result2)

result3 = brute(func, (slice(-10, 10, 0.1),))
print(result3)

x = np.linspace(-10, 10, 400)
y = func(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f(x) = x^2 + 10*sin(x)$')
plt.scatter(result1, func(result1[0]), color='r', label='fmin_bfgs')
plt.scatter(result2, func(result2), color='g', label='fminbound')
plt.scatter(result3, func(result3[0]), color='b', label='brute')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
