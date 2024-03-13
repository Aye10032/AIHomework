import io
import pickle
import tarfile
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 20种氨基酸序列对应的的字典表
code_dict = {
    'A': 0,
    'F': 1,
    'C': 2,
    'D': 3,
    'N': 4,
    'E': 5,
    'Q': 6,
    'G': 7,
    'H': 8,
    'L': 9,
    'I': 10,
    'K': 11,
    'M': 12,
    'P': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}

type_dict = {'C': 0, 'E': 1, 'H': 2}


def load_data() -> List[Dict]:
    datas = []
    with tarfile.open('../dataset/assignment1_data.tar', 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.pkl'):
                if file := tar.extractfile(member):
                    byte_stream = io.BytesIO(file.read())
                    data = pickle.load(byte_stream)

                    datas.append(data)

    return datas


class MyDataset(Dataset):
    def __init__(self, device):
        self.data: list[dict] = load_data()
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq_str = self.data[item].get('seq')
        seq_list = [code_dict.get(letter) for letter in seq_str]
        seq = torch.zeros((250, 20), dtype=torch.float, device=self.device)
        seq[range(len(seq_str)), seq_list] = 1

        ssp_str = self.data[item].get('ssp')
        ssp_list = [type_dict.get(letter) for letter in ssp_str]
        ssp = torch.zeros((250, 3), dtype=torch.float, device=self.device)
        ssp[range(len(ssp_str)), ssp_list] = 1

        return seq, ssp


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(DEVICE)
    data_load = DataLoader(dataset, batch_size=64)

    for train_features, train_labels in data_load:
        print(train_features.shape)
