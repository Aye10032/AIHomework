import numpy as np

a = np.array([1, 2, 3, 4, 5])

b_length = a.shape[0] * 3 - 2
b = np.zeros(b_length, dtype=a.dtype)
b[::3] = a

print(b)
