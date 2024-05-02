import os
from dataclasses import dataclass

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import j1


@dataclass
class Variable:
    LAMBDA: float
    NA: float


def psf(r: np.ndarray, light_wave_l: float, na: float) -> np.ndarray:
    """
    计算点扩散函数（PSF）

    :param r: 距离中心的距离。
    :param light_wave_l: 波长
    :param na: 数值孔径
    :return: 对应输入距离r的光强
    """
    k = 2 * np.pi / light_wave_l
    rho = k * na * r

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (2 * j1(rho) / rho) ** 2
        z[rho == 0] = 1

    return z


def main() -> None:
    os.makedirs('image/exp2', exist_ok=True)

    configs = [
        Variable(0.48, 0.5),
        Variable(0.52, 0.5),
        Variable(0.68, 0.5),
        Variable(0.52, 1.0),
        Variable(0.52, 1.4),
        Variable(0.52, 1.5),
    ]

    x = np.linspace(-1.8, 1.8, 200)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8*2, 4 * 3))
    for index, config in enumerate(configs):
        ax = axes.flat[index]
        sns.lineplot(x=x, y=psf(x, config.LAMBDA, config.NA), ax=ax)
        ax.set_xlabel(fr'$\lambda$: {config.LAMBDA}, NA: {config.NA}')

    plt.tight_layout()
    plt.savefig('image/exp2/psf.png')
    plt.show()


if __name__ == '__main__':
    main()
