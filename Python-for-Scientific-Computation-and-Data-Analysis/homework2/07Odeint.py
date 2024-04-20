import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

M = 1.0
b = 0.2
k = 0.5
F = 1.0


def spring_damper_system(u, t, M, b, k, F):
    u1, u2 = u
    du1_dt = u2
    du2_dt = (F - b * u2 - k * u1) / M
    return [du1_dt, du2_dt]


init_status = [-1.0, 0.0]
t = np.arange(0, 50, 0.02)

solution = odeint(spring_damper_system, init_status, t, args=(M, b, k, F))

x = solution[:, 0]

plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.grid(True)
plt.show()
