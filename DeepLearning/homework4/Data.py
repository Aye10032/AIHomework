import os.path
from enum import IntEnum

import numpy as np
from torch.utils.data import Dataset
from loguru import logger


class DataType(IntEnum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class TransData(Dataset):
    def __init__(self, data_path: str, data_type: DataType):
        match data_type:
            case DataType.TRAIN:
                self.src_file = os.path.join(data_path, 'train.zh')
                self.target_file = os.path.join(data_path, 'train.en')
            case DataType.VALID:
                self.src_file = os.path.join(data_path, 'dev.zh')
                self.target_file = os.path.join(data_path, 'dev.en')
            case DataType.TEST:
                self.src_file = os.path.join(data_path, 'test.zh')
                self.target_file = os.path.join(data_path, 'test.en')
            case _:
                raise RuntimeError('无效的参数！')

        self.src_vocab = os.path.join(data_path, 'vocab.zh')
        self.target_vocab = os.path.join(data_path, 'vocab.en')

        self.__load_data()

    def __getitem__(self, item):
        raise NotImplementedError

    def __load_data(self):
        logger.info('loading dataset...')
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_lines = [line.strip().split(' ') for line in f]

        with open(self.target_file, 'r', encoding='utf-8') as f:
            target_lines = [line.strip().split(' ') for line in f]

        self.max_len = 0
        for line in src_lines:
            self.max_len = max(len(line), self.max_len)

        for line in target_lines:
            self.max_len = max(len(line), self.max_len)

        return src_lines, target_lines


def main() -> None:
    dataset = TransData('data', DataType.TRAIN)


if __name__ == '__main__':
    main()
