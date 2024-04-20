import numpy as np
from scipy.integrate import quad, dblquad

a = quad(
    lambda x: np.cos(np.exp(x)) ** 2,
    3, 0
)
print(a[0])

b = dblquad(
    lambda x, y: 16 * x * y,
    0.5, 0,
    lambda y: np.sqrt(1 - 4 * y ** 2), 0
)
print(b[0])
