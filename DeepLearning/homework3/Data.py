from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TangData(Dataset):
    def __init__(self, max_length: int = 50):
        datas = np.load('data/tang.npz', allow_pickle=True)
        data = datas['data']
        self.ix2word = datas['ix2word'].item()
        self.word2ix = datas['word2ix'].item()
        self.max_length = max_length

        self.inputs, self.targets = self.resize_data(data)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def resize_data(self, data: np.ndarray) -> Tuple[Tensor, Tensor]:
        data = data.reshape(-1)

        tensor_list = torch.tensor(data[data != 8292])
        remainder = (self.max_length - len(tensor_list) % self.max_length) + 1

        padding_vector = torch.full((remainder,), 8292)
        result = torch.cat([tensor_list, padding_vector])

        inputs = result[:-1].view(-1, self.max_length)
        targets = result[1:].view(-1, self.max_length)

        return inputs, targets


def main() -> None:
    dataset = TangData()
    print(dataset.ix2word[8292])


if __name__ == '__main__':
    main()
