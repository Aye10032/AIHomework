import os
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

from loguru import logger
from scipy.spatial import Delaunay

from Exp11GaussianKernel import conv2d, PaddingMode
from utils.Decorator import timer


@dataclass
class Variable:
    LAMBDA: float
    NA: float
    PIX_SIZE: float
    SIGMA: float = field(init=False)

    def __post_init__(self):
        self.SIGMA = ((0.61 * self.LAMBDA / self.NA) / 3) / self.PIX_SIZE


def get_background_noise(input_mat: np.ndarray) -> Tuple[float, float]:
    """
    计算输入图片中选定区域的背景噪声的平均值和标准差。

    :param input_mat: 输入的16位深度原图
    :return: 返回一个元组，包含背景噪声的平均值和标准差。
    """

    init_8 = input_mat.astype('uint8')
    bbox = cv2.selectROI('SELECT BACKGROUND', init_8, printNotice=False)
    img_out = input_mat[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    mean = np.mean(img_out)
    standard_deviation = np.std(img_out)

    logger.info(f'mean: {mean}, standard deviation: {standard_deviation}')
    return mean, standard_deviation


@timer
def gauss_filter(input_mat: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = 3
    offset = kernel_size // 2

    x = y = np.arange(kernel_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    gaussian_mask = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x_mesh - offset) ** 2 + (y_mesh - offset) ** 2) / (2 * sigma ** 2))
    gaussian_mask /= gaussian_mask.sum()

    output_mat = conv2d(input_mat, gaussian_mask, PaddingMode.SAME)
    return output_mat


@timer
def local_min_max(
        input_mat: np.ndarray,
        kernel_size: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    寻找输入矩阵中的局部最小值和局部最大值

    :param input_mat: 输入的二维numpy数组，寻找其中的局部最小和局部最大值
    :param kernel_size: 局部极值检测的范围，默认为3
    :return: 返回一个元组，包含两个元素。第一个元素是标记局部最大值的二维数组（255表示局部最大值），第二个元素是标记局部最小值的二维数组（255表示局部最小值）
    """

    local_max = np.zeros(input_mat.shape, dtype=np.uint8)
    local_min = np.zeros(input_mat.shape, dtype=np.uint8)

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

    max_positions = np.argwhere(local_max == 255)[:, ::-1]
    min_positions = np.argwhere(local_min == 255)[:, ::-1]

    tri = Delaunay(min_positions, incremental=False)

    triangle_in = tri.simplices[tri.find_simplex(max_positions)]

    return max_positions, min_positions[triangle_in]


@timer
def statistical_selection(
        point_tuple: tuple[np.ndarray, np.ndarray],
        input_mat: np.ndarray,
        q: float,
        sigma: float
) -> np.ndarray:
    """
    执行假设检验。

    :param point_tuple: 包含局部极值的元组。
    :param input_mat: 输入图像。
    :param q: 统计量的阈值。
    :param sigma: 标准差。
    :return: 满足条件的点的集合。
    """

    i_net = (input_mat[point_tuple[0][..., 1], point_tuple[0][..., 0]] -
             input_mat[point_tuple[1][..., 1], point_tuple[1][..., 0]].sum(axis=1) / 3)

    output = point_tuple[0][i_net > q * sigma]
    return output


def draw_triangle(
        input_mat: np.ndarray,
        max_positions: np.ndarray,
        min_positions: np.ndarray
) -> None:
    """

    :param input_mat: 输入图片
    :param max_positions: 局部极大值点
    :param min_positions: 局部极小值点
    :return:
    """
    os.makedirs('image/BIP_Project02_image_sequence/triangle', exist_ok=True)

    img_big = zoom_in(input_mat, 5).astype('uint8')
    img_big = cv2.cvtColor(img_big, cv2.COLOR_GRAY2RGB)

    max_positions = max_positions * 5 + 2
    min_positions = min_positions * 5 + 2

    for index, points in enumerate(min_positions):
        points: np.ndarray
        pos_list = np.int32(points.reshape((-1, 1, 2)))
        cv2.polylines(img_big, [pos_list], True, (0, 255, 0), 1)

        cv2.circle(img_big, max_positions[index], 2, (0, 0, 255), -1)

    cv2.imwrite('image/BIP_Project02_image_sequence/triangle/triangle.png', img_big)


def zoom_in(input_mat: np.ndarray, factor: int) -> np.ndarray:
    """
    对输入的图像进行放大处理。

    :param input_mat: 输入的原图片
    :param factor: 缩放放因子，指定矩阵在行和列方向上重复的次数
    :return: 返回放大后的图片
    """

    output_y = np.repeat(input_mat, factor, axis=0)
    output = np.repeat(output_y, factor, axis=1)
    return output


def main() -> None:
    os.makedirs('image/BIP_Project02_image_sequence/gauss', exist_ok=True)
    os.makedirs('image/BIP_Project02_image_sequence/minmax', exist_ok=True)
    os.makedirs('image/BIP_Project02_image_sequence/points', exist_ok=True)

    init_src = cv2.imread('image/BIP_Project02_image_sequence/001_a5_002_t001.tif', cv2.IMREAD_UNCHANGED)
    mean, std = get_background_noise(init_src)
    quantile = 3.5

    variable = Variable(515, 1.4, 65)

    for i in range(1, 6):
        file_name = f'001_a5_002_t00{i}'
        logger.info(f'loading {file_name}.tif...')
        src = cv2.imread(f'image/BIP_Project02_image_sequence/{file_name}.tif', cv2.IMREAD_UNCHANGED)

        src_gauss = gauss_filter(src, variable.SIGMA)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/gauss/{file_name}.png', src_gauss.astype('uint8'))

        local_maxima, local_minima = local_min_max(src_gauss)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_max.png', local_maxima.astype('uint8'))
        cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_min.png', local_minima.astype('uint8'))
        logger.info(f'find {local_maxima.shape[0]} local maxima, {local_minima.shape[0]} local minima')
        if i == 1:
            logger.info('compare')
            local_maxima_1, local_minima_1 = local_min_max(src_gauss, 5)
            cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_max_5X5.png', local_maxima_1.astype('uint8'))
            cv2.imwrite(f'image/BIP_Project02_image_sequence/minmax/{file_name}_min_5X5.png', local_minima_1.astype('uint8'))
            logger.info(f'find {local_maxima_1.shape[0]} local maxima, {local_minima_1.shape[0]} local minima')

        tra_list: tuple[np.ndarray, np.ndarray] = establish_connections(local_maxima, local_minima)
        if i == 1:
            draw_triangle(src_gauss, tra_list[0], tra_list[1])

        true_max: np.ndarray = statistical_selection(tra_list, src_gauss, quantile, std)
        logger.info(f'{true_max.shape[0]} points find')

        point_mat = cv2.cvtColor(init_src, cv2.COLOR_GRAY2RGB)
        for point in true_max:
            cv2.circle(point_mat, point, 3, (0, 255, 0), -1)
        cv2.imwrite(f'image/BIP_Project02_image_sequence/points/{file_name}.png', point_mat.astype('uint8'))
        np.save(f'image/BIP_Project02_image_sequence/points/{file_name}.npy', true_max)


if __name__ == '__main__':
    main()
