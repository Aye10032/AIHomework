import numpy as np


def normalize(z: np.ndarray) -> np.ndarray:
    return (z - np.min(z)) / (np.max(z) - np.min(z))


if __name__ == '__main__':
    np.random.seed(0)
    Z = np.random.random((5, 5))
    print(Z)
    print(normalize(Z))
