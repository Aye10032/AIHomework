import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, coo_matrix

data = np.array([
    [3, 0, 8, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])
a = dok_matrix(data)
b = lil_matrix(data)
c = coo_matrix(data)

print(a)
print('---------')
print(b)
print('---------')
print(c)
