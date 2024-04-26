import os

import cv2
import numpy as np

from loguru import logger

from utils.Decorator import timer
from Exp11GaussianKernel import gaussian_filter


@timer
def get_v_derivative(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, ((0, 0), (1, 1)), "edge")

    output_img = (src[:, 2:] - src[:, :-2]) / 2
    output_img = (output_img / output_img.max()) * 255
    logger.info(f'image type: {output_img.dtype}')

    return output_img.astype('uint8')


@timer
def get_h_derivative(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, ((1, 1), (0, 0)), "edge")

    output_img = (src[2:, :] - src[:-2, :]) / 2
    output_img = (output_img / output_img.max()) * 255
    logger.info(f'image type: {output_img.dtype}')

    return output_img.astype('uint8')


def main() -> None:
    os.makedirs('image/exp2', exist_ok=True)

    image1 = cv2.imread('image/axon02.tif', cv2.IMREAD_GRAYSCALE).astype('uint16')
    image2 = cv2.imread('image/cell_nucleus.tif', cv2.IMREAD_UNCHANGED).astype('uint16')

    sigmas = [1, 2, 5]

    for _sigma in sigmas:
        image1_gauss = gaussian_filter(image1, _sigma)
        image1_dv: np.ndarray = get_v_derivative(image1_gauss)
        image1_dh: np.ndarray = get_h_derivative(image1_gauss)

        cv2.imwrite(f'image/exp2/axon02_gauss_sigma_{_sigma}_dv.tif', image1_dv)
        cv2.imwrite(f'image/exp2/axon02_gauss_sigma_{_sigma}_dh.tif', image1_dh)

        image2_gauss = gaussian_filter(image2, _sigma)
        image2_dv = get_v_derivative(image2_gauss)
        image2_dh = get_h_derivative(image2_gauss)

        cv2.imwrite(f'image/exp2/cell_nucleus_gauss_sigma_{_sigma}_dv.tif', image2_dv)
        cv2.imwrite(f'image/exp2/cell_nucleus_gauss_sigma_{_sigma}_dh.tif', image2_dh)


if __name__ == '__main__':
    main()
