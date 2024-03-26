import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


def z(_x):
    return (_x - 1) * 5


def g(_x):
    return np.sin(z(_x) ** 2) + np.sin(z(_x)) ** 2


x = Chebyshev.basis(100).roots()
xd = np.linspace(-1, 1, 1000)

c1 = Chebyshev.fit(x, g(x), 10, domain=[-1, 1])
c2 = Polynomial.fit(x, g(x), 10, domain=[-1, 1])

print('插值多项式的最大误差：')
print(f'契比雪夫节点：{abs(c1(xd) - g(xd)).max()}')
print(f'多项式节点：{abs(c2(xd) - g(xd)).max()}')
