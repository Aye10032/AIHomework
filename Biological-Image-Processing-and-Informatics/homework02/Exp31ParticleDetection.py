import os
from typing import Tuple

import cv2
import numpy as np

from loguru import logger
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
    """
    建立最大值点和最小值点之间的连接。

    :param local_max: 包含局部最大值点的numpy数组，其中最大值点被标记为255。
    :param local_min: 包含局部最小值点的numpy数组，同样，最小值点被标记为255。
    :return: 一个元组，包含两个numpy数组。第一个是最大值点的位置数组，第二个是与每个最大值点相关联的最小值点的位置数组。
    """

    max_positions = np.argwhere(local_max == 255)
    min_positions = np.argwhere(local_min == 255)

    tri = Delaunay(min_positions, incremental=False)

    triangle_in = tri.simplices[tri.find_simplex(max_positions)]

    return max_positions, min_positions[triangle_in]


@timer
def statistical_selection(
        point_tuple: tuple[np.ndarray, np.ndarray],
        input_mat: np.ndarray,
        q: float,
        sigma: float
):
    i_net = (input_mat[point_tuple[0][..., 0], point_tuple[0][..., 1]] -
             input_mat[point_tuple[1][..., 0], point_tuple[1][..., 1]].sum(axis=1) / 3)

    output = point_tuple[0][i_net > q * sigma]
    return output


def main() -> None:
    os.makedirs('image/BIP_Project02_image_sequence/gauss', exist_ok=True)
    os.makedirs('image/BIP_Project02_image_sequence/minmax', exist_ok=True)
    os.makedirs('image/BIP_Project02_image_sequence/points', exist_ok=True)

    init_src = cv2.imread('image/BIP_Project02_image_sequence/001_a5_002_t001.tif', cv2.IMREAD_UNCHANGED)
    mean, std = get_background_noise(init_src)
    quantile = 2.5

    for i in range(1, 6):
        file_name = f'001_a5_002_t00{i}'
        src = cv2.imread(f'image/BIP_Project02_image_sequence/{file_name}.tif', cv2.IMREAD_UNCHANGED)

        src_gauss = gauss_filter(src, 5.15, 1.4)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/gauss/{file_name}.tif', src_gauss.astype('uint8'))

        local_maxima, local_minima = local_min_max(src_gauss)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_max.tif', local_maxima.astype('uint8'))
        cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_min.tif', local_minima.astype('uint8'))

        tra_list: tuple[np.ndarray, np.ndarray] = establish_connections(local_maxima, local_minima)
        logger.info(f'find {tra_list[0].shape[0]} local maxima')

        true_max: np.ndarray = statistical_selection(tra_list, src_gauss, quantile, std)
        logger.info(f'{true_max.shape[0]} points find')

        point_mat = cv2.cvtColor(init_src, cv2.COLOR_GRAY2RGB)
        for point in true_max:
            cv2.circle(point_mat, (point[1], point[0]), 3, (0, 255, 0), -1)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/points/{file_name}.tif', point_mat.astype('uint8'))


if __name__ == '__main__':
    main()
