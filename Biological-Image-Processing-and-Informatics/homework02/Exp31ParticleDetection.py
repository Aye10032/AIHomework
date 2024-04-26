from typing import Tuple

import cv2
import numpy as np

from loguru import logger
from Exp11GaussianKernel import conv2d, PaddingMode


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


def main() -> None:
    init_src = cv2.imread('image/BIP_Project02_image_sequence/001_a5_002_t001.tif', cv2.IMREAD_UNCHANGED)

    mean, std = get_background_noise(init_src)

    src_gauss = gauss_filter(init_src, 5.15, 1.4)
    cv2.imwrite('image/BIP_Project02_image_sequence/gauss/001_a5_002_t001.tif', src_gauss.astype('uint8'))

    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
