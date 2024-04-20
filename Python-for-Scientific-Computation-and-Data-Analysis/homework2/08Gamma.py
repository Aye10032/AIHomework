import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

data = np.random.gamma(1, 1, 1000)
plt.hist(data, bins=50, density=True, alpha=0.5, color='b')

x = np.linspace(0, max(data), 1000)
pdf = gamma.pdf(x, 1, scale=1)
plt.plot(x, pdf, 'r', lw=1.5)

plt.show()
