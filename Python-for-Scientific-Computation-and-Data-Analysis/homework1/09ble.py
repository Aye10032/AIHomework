import numpy as np
from matplotlib.pyplot import plot, show

cash = np.zeros(10000)
cash[0] = 1000
outcome = np.random.binomial(5, 0.5, size=len(cash))

for i in range(1, len(cash)):
    if outcome[i] < 3:
        cash[i] = cash[i - 1] - 8
    elif outcome[i] < 6:
        cash[i] = cash[i - 1] + 8
    else:
        raise AssertionError("Unexpected outcome " + outcome)
print(outcome.min(), outcome.max())

plot(np.arange(len(cash)), cash)
show()
