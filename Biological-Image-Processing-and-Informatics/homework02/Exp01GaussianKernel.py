import math
import os
from enum import IntEnum
from loguru import logger
from utils.Decorator import timer

import cv2
import numpy as np


class PaddingMode(IntEnum):
    VALID = 0
    SAME = 1
    FULL = 2


def conv2d(src: np.ndarray, mask: np.ndarray, pad_mode: int = PaddingMode.SAME) -> np.ndarray:
    """
    计算二维卷积
    :param src: 输入的图像
    :param mask: 卷积核
    :param pad_mode: 填充模式
    :return: 卷积后的图像
    """
    _h, _w = src.shape
    offset = mask.shape[0] - 1

    match pad_mode:
        case PaddingMode.VALID:
            logger.info('padding mode: valid')
            output_img = np.zeros((_h - offset, _w - offset))

        case PaddingMode.SAME:
            logger.info('padding mode: same')
            output_img = np.zeros(src.shape)
            src = np.pad(src, (int(offset / 2), int(offset / 2)), mode="mean")

        case PaddingMode.FULL:
            logger.info('padding mode: full')
            output_img = np.zeros((_h + offset * 2, _w + offset * 2))
            src = np.pad(src, (offset, offset), mode="mean")

        case _:
            logger.error('invalid padding mode')
            exit()

    for x in range(output_img.shape[0]):
        for y in range(output_img.shape[1]):
            src_cut = src[x:x + offset + 1, y:y + offset + 1]
            output_img[x, y] = np.sum(src_cut * mask)

    return output_img


@timer
def gaussian_filter(src: np.ndarray, sigma: float) -> np.ndarray:
    """
    对图片使用高斯核进行滤波

    :param src: 原始输入图片
    :param sigma: 高斯核的方差
    :return: 经过高斯核滤波后的图片
    """
    kernel_size = 1 + math.ceil(sigma * 3) * 2
    logger.info(f'sigma={sigma}, kernel size={kernel_size}')
    offset = kernel_size // 2

    x = y = np.arange(kernel_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    gaussian_mask = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x_mesh - offset) ** 2 + (y_mesh - offset) ** 2) / (2 * sigma ** 2))
    gaussian_mask /= gaussian_mask.sum()

    output_mat = conv2d(src, gaussian_mask, PaddingMode.SAME)
    return output_mat


def main() -> None:
    os.makedirs('image/exp1', exist_ok=True)

    img = cv2.imread('axon01.tif', cv2.IMREAD_UNCHANGED)
    src_8 = img.astype('uint8')
    cv2.imshow('origin', src_8)

    sigmas = [1, 2, 5, 7]

    for _sigma in sigmas:
        dst = gaussian_filter(img, _sigma)

        img_out = dst.astype('uint8')
        cv2.imshow(f'gauss_sigma_{_sigma}', img_out)
        cv2.imwrite(f'image/exp1/gauss_sigma_{_sigma}.tif', img_out)
        cv2.imwrite(f'image/exp1/gauss_sigma_{_sigma}_uint8.tif', dst)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
