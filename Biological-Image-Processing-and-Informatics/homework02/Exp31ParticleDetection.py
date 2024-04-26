from typing import Tuple, List, Dict, Any

import cv2
import numpy as np

from loguru import logger
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

from Exp11GaussianKernel import conv2d, PaddingMode
from utils.Decorator import timer


def get_background_noise(input_mat: np.ndarray) -> Tuple[float, float]:
    """
    计算输入图片中选定区域的背景噪声的平均值和标准差。

    :param input_mat: 输入的numpy数组格式的图片
    :return: 返回一个元组，包含背景噪声的平均值和标准差。
    """

    init_8 = input_mat.astype('uint8')
    bbox = cv2.selectROI('SELECT BACKGROUND', init_8, printNotice=False)
    img_out = init_8[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    mean = np.mean(img_out)
    standard_deviation = np.std(img_out)

    logger.info(f'mean: {mean}, standard deviation: {standard_deviation}')
    return mean, standard_deviation


@timer
def gauss_filter(input_mat: np.ndarray, wavelength: float, NA: float) -> np.ndarray:
    """
    使用高斯滤波器对输入的图像进行滤波处理。

    :param input_mat: 输入的图像，将对此矩阵进行高斯滤波。
    :param wavelength: 光波波长，用于计算高斯滤波器的方差。
    :param NA: 数值孔径，用于计算高斯滤波器的方差。
    :return: 返回经过高斯滤波处理后的图像。
    """

    kernel_size = 3
    sigma = (0.61 * wavelength / NA) / 3
    offset = kernel_size // 2
    logger.info(f'sigma={sigma}')

    x = y = np.arange(kernel_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    gaussian_mask = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x_mesh - offset) ** 2 + (y_mesh - offset) ** 2) / (2 * sigma ** 2))
    gaussian_mask /= gaussian_mask.sum()

    output_mat = conv2d(input_mat, gaussian_mask, PaddingMode.SAME)
    return output_mat


@timer
def local_min_max(input_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    寻找输入矩阵中的局部最小值和局部最大值。

    :param input_mat: 输入的二维numpy数组，寻找其中的局部最小和局部最大值。
    :return: 返回一个元组，包含两个元素。第一个元素是标记局部最大值的二维数组（255表示局部最大值），第二个元素是标记局部最小值的二维数组（255表示局部最小值）。
    """

    local_max = np.zeros(input_mat.shape, dtype=np.uint16)
    local_min = np.zeros(input_mat.shape, dtype=np.uint16)
    kernel_size = 3

    for i in range(kernel_size, input_mat.shape[0] - kernel_size):
        for j in range(kernel_size, input_mat.shape[1] - kernel_size):
            roi = input_mat[i - kernel_size:i + kernel_size + 1, j - kernel_size:j + kernel_size + 1]
            local_min[i, j] = (input_mat[i, j] == roi.min()) * 255
            local_max[i, j] = (input_mat[i, j] == roi.max()) * 255

    return local_max, local_min


@timer
def establish_connections(local_max: np.ndarray, local_min: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_positions = np.argwhere(local_max == 255)
    min_positions = np.argwhere(local_min == 255)

    tri = Delaunay(min_positions, incremental=False)

    triangle_in = tri.simplices[tri.find_simplex(max_positions)]

    result = []
    for i, pos in enumerate(max_positions):
        result.append({'max': pos, 'min': min_positions[triangle_in[i]]})

    return max_positions, min_positions[triangle_in]


def main() -> None:
    init_src = cv2.imread('image/BIP_Project02_image_sequence/001_a5_002_t001.tif', cv2.IMREAD_UNCHANGED)

    # mean, std = get_background_noise(init_src)

    src_gauss = gauss_filter(init_src, 5.15, 1.4)
    cv2.imwrite('image/BIP_Project02_image_sequence/gauss/001_a5_002_t001.tif', src_gauss.astype('uint8'))

    local_maxima, local_minima = local_min_max(src_gauss)
    cv2.imwrite('image/BIP_Project02_image_sequence/minmax/001_a5_002_t001_max.tif', local_maxima.astype('uint8'))
    cv2.imwrite('image/BIP_Project02_image_sequence/minmax/001_a5_002_t001_min.tif', local_minima.astype('uint8'))

    tra_list: tuple[np.ndarray, np.ndarray] = establish_connections(local_maxima, local_minima)
    logger.info(f'find {tra_list[0].shape[0]} local maxima')

    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
