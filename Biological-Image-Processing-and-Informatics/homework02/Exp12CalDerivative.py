import os

import cv2
import numpy as np

from loguru import logger

from utils.Decorator import timer
from Exp11GaussianKernel import gaussian_filter


@timer
def get_v_derivative(src: np.ndarray) -> np.ndarray:
    """
    计算输入图像在垂直方向上的导数。

    :param src: 输入的二维numpy数组，代表待处理的图像。
    :return: 经过垂直方向导数计算后的图像，为一个二维numpy数组。

    此函数首先对输入图像进行边缘填充，然后计算图像在垂直方向上的导数，
    并将结果归一化到0-255的范围内，最后返回处理后的图像。
    """

    src = np.pad(src, ((0, 0), (1, 1)), "edge")

    output_img = (src[:, 2:] - src[:, :-2]) / 2
    output_img = (output_img / output_img.max()) * 255

    return output_img.astype('uint8')



@timer
def get_h_derivative(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, ((1, 1), (0, 0)), "edge")

    output_img = (src[2:, :] - src[:-2, :]) / 2
    output_img = (output_img / output_img.max()) * 255

    return output_img.astype('uint8')


def main() -> None:
    os.makedirs('image/exp1/derivative', exist_ok=True)

    image1 = cv2.imread('image/axon02.tif', cv2.IMREAD_GRAYSCALE).astype('uint16')
    image2 = cv2.imread('image/cell_nucleus.tif', cv2.IMREAD_UNCHANGED).astype('uint16')

    sigmas = [1, 2, 5]

    for _sigma in sigmas:
        image1_gauss = gaussian_filter(image1, _sigma)
        image1_dv: np.ndarray = get_v_derivative(image1_gauss)
        image1_dh: np.ndarray = get_h_derivative(image1_gauss)

        cv2.imwrite(f'image/exp1/derivative/axon02_gauss_sigma_{_sigma}_dv.png', image1_dv)
        cv2.imwrite(f'image/exp1/derivative/axon02_gauss_sigma_{_sigma}_dh.png', image1_dh)

        image2_gauss = gaussian_filter(image2, _sigma)
        image2_dv = get_v_derivative(image2_gauss)
        image2_dh = get_h_derivative(image2_gauss)

        cv2.imwrite(f'image/exp1/derivative/cell_nucleus_gauss_sigma_{_sigma}_dv.png', image2_dv)
        cv2.imwrite(f'image/exp1/derivative/cell_nucleus_gauss_sigma_{_sigma}_dh.png', image2_dh)


if __name__ == '__main__':
    main()
