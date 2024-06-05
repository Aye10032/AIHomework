import os
import cv2
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                ├── ...
                ...
        """
        self.img_ids = img_ids  # 用于记录数据集中的图像id
        self.img_dir = img_dir  # 用于存储图像文件的目录路径
        self.mask_dir = mask_dir  # 用于存储标签文件的目录路径
        self.img_ext = img_ext  # 用于指定图像文件的扩展名
        self.mask_ext = mask_ext  # 用于指定标签文件的扩展名
        self.transform = transform  # 用于对图像和掩膜进行增强处理。默认为None，表示不进行数据增强。

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 构建训练的batch，这是一张图片的处理
        img_id = self.img_ids[idx]
        # 第一步，读取图像
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext), cv2.IMREAD_GRAYSCALE)
        # 第二步，读标签
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255  # 二值化
        # 第三步数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉
            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        # 第四步归一化
        img = img.astype('float32') / 255.
        mask = mask.astype('float32') / 255.

        return img[np.newaxis, :], mask[np.newaxis, :], {'img_id': img_id}
