import io
import pickle
import tarfile
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

type_dict = {'C': 1, 'E': 2, 'H': 3}


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
        seq_list = [ord(letter) - ord('A') + 1 for letter in seq_str]
        seq_origin = torch.tensor(seq_list, dtype=torch.float, device=self.device)
        seq = F.pad(seq_origin, (0, 250 - seq_origin.size(0)), mode='constant', value=0)

        ssp_str = self.data[item].get('ssp')
        ssp_list = [type_dict.get(letter) for letter in ssp_str]
        ssp_origin = torch.tensor(ssp_list, dtype=torch.float, device=self.device)
        ssp = F.pad(ssp_origin, (0, 250 - ssp_origin.size(0)), mode='constant', value=0)

        return seq, ssp


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(DEVICE)
    data_load = DataLoader(dataset, batch_size=64)

    for train_features, train_labels in data_load:
        print(train_features.shape)
