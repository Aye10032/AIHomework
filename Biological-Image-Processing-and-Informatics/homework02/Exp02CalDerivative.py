import os

import cv2
import numpy as np

from utils.Decorator import timer

from Exp01GaussianKernel import gaussian_filter


@timer
def get_v_derivative(src: np.ndarray) -> np.ndarray:
    output_img = np.zeros(src.shape, dtype=np.float16)
    src = np.pad(src, (1, 1), "edge")
    for x in range(output_img.shape[0]):
        for y in range(output_img.shape[1]):
            output_img[x, y] = (src[x + 1, y + 2] - src[x + 1, y]) / 2

    return output_img


@timer
def get_h_derivative(src: np.ndarray) -> np.ndarray:
    output_img = np.zeros(src.shape, dtype=np.float16)
    src = np.pad(src, (1, 1), "edge")
    for x in range(output_img.shape[0]):
        for y in range(output_img.shape[1]):
            output_img[x, y] = (src[x + 2, y + 1] - src[x, y + 1]) / 2

    return output_img


def main() -> None:
    os.makedirs('image/exp2', exist_ok=True)

    image1 = cv2.imread('axon02.tif', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('cell_nucleus.tif', cv2.IMREAD_UNCHANGED)

    sigmas = [1, 2, 5]

    for _sigma in sigmas:
        image1_gauss: np.ndarray = gaussian_filter(image1, _sigma)
        image1_dv = get_v_derivative(image1_gauss)
        image1_dh = get_h_derivative(image1_gauss)

        # cv2.imshow(f'gauss_sigma_{_sigma}_v', image1_dv)
        # cv2.imshow(f'gauss_sigma_{_sigma}_h', image1_dh)
        cv2.imwrite(f'image/exp2/axon02_gauss_sigma_{_sigma}_dv.png', image1_dv)
        cv2.imwrite(f'image/exp2/axon02_gauss_sigma_{_sigma}_dh.png', image1_dh)

        image2_gauss = gaussian_filter(image2, _sigma)
        image2_dv = get_v_derivative(image2_gauss)
        image2_dh = get_h_derivative(image2_gauss)

        cv2.imwrite(f'image/exp2/cell_nucleus_gauss_sigma_{_sigma}_dv.tif', image2_dv)
        cv2.imwrite(f'image/exp2/cell_nucleus_gauss_sigma_{_sigma}_dh.tif', image2_dh)


if __name__ == '__main__':
    # main()
    image1 = cv2.imread('axon02.tif', cv2.IMREAD_GRAYSCALE)
    image1_gauss: np.ndarray = gaussian_filter(image1, 2)
    image1_dv = get_v_derivative(image1_gauss)
