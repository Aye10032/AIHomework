import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Variable:
    LAMBDA: float
    NA: float
    PIX_SIZE: float
    SIGMA: float = field(init=False)

    def __post_init__(self):
        self.SIGMA = ((0.61 * self.LAMBDA / self.NA) / 3) / self.PIX_SIZE


def oversample(input_mat: np.ndarray, factor: int) -> np.ndarray:
    """
    对输入的图像进行上采样

    :param input_mat: 输入的原图片
    :param factor: 缩放放因子，指定矩阵在行和列方向上重复的次数
    :return: 返回放大后的图片
    """

    output_y = np.repeat(input_mat, factor, axis=0)
    output = np.repeat(output_y, factor, axis=1)
    return output


def gauss_filter(sigma: float, kernel_size: int) -> np.ndarray:
    """
    用于拟合的高斯核

    :param sigma: 高斯函数的标准差
    :param kernel_size: 高斯核的大小
    :return: 高斯函数拟合的Airy disk
    """

    offset = kernel_size // 2

    x = y = np.arange(kernel_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    gaussian_mask = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x_mesh - offset) ** 2 + (y_mesh - offset) ** 2) / (2 * sigma ** 2))
    gaussian_mask /= gaussian_mask.max()

    return gaussian_mask


def loss_func(y: np.ndarray, fit: np.ndarray) -> float:
    return abs(np.sum(y - fit)) / 225


def fit_max_points(input_mat: np.ndarray, max_points: np.ndarray, sigma: float) -> np.ndarray:
    """
    根据输入的矩阵和最大点集，利用高斯核拟合Airy disk实现精确定位

    :param input_mat: 上采样后的图片
    :param max_points: 初始的最大点集，每个点为二维坐标的数组
    :param sigma: 高斯滤核的标准差
    :return: 精确定位后的最大点集
    """

    input_mat = input_mat.T
    mask = gauss_filter(sigma, 15)
    for index, start_point in enumerate(max_points):
        min_loss = np.inf
        center = [0, 0]
        for i in range(5):
            for j in range(5):
                x, y = start_point + np.array([i, j])
                roi = input_mat[x - 7:x + 8, y - 7:y + 8]
                gauss = mask * float(input_mat[x, y])
                loss = loss_func(roi, gauss)

                if loss < min_loss:
                    min_loss = loss
                    center = [x, y]

        max_points[index] = np.array(center)

    return max_points


def main() -> None:
    os.makedirs('image/BIP_Project02_image_sequence/sub_pix', exist_ok=True)

    variable = Variable(515, 1.4, 13)

    for i in range(1, 6):
        file_name = f'001_a5_002_t00{i}'

        src = cv2.imread(f'image/BIP_Project02_image_sequence/gauss/{file_name}.tif', cv2.IMREAD_UNCHANGED)
        max_points = np.load(f'image/BIP_Project02_image_sequence/points/{file_name}.npy')

        src_oversample = oversample(src, 5)
        max_points = max_points * 5

        sub_pix_max = fit_max_points(src_oversample, max_points, variable.SIGMA)

        point_mat = cv2.cvtColor(src_oversample, cv2.COLOR_GRAY2RGB)
        for point in sub_pix_max:
            cv2.circle(point_mat, point, 3, (0, 255, 0), -1)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/sub_pix/{file_name}.png', point_mat.astype('uint8'))


if __name__ == '__main__':
    main()
