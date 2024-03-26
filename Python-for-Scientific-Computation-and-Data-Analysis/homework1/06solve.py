import numpy as np

# 解方程：3x + 6y -5z = 12；x-3y+2z = -2；5x -y +4z = 10
a = np.array([[3, 6, -5], [1, -3, 2], [5, -1, 4]])
b = np.array([12, -2, 10])
x = np.linalg.solve(a, b)
print(x)
