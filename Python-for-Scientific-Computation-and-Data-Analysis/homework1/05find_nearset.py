import numpy as np


def find_nearest(z: np.ndarray, x: float) -> int:
    return np.abs(z - x).argmin()


if __name__ == '__main__':
    Z = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    print(find_nearest(Z, 5.1))
