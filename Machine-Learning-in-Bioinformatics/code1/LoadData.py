import io
import pickle
import tarfile
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


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


def collate_fn(batch):
    seq_batch, ssp_batch = zip(*batch)
    seq_padded = pad_sequence(seq_batch, batch_first=True, padding_value=0)
    ssp_padded = pad_sequence(ssp_batch, batch_first=True, padding_value=0)

    return seq_padded, ssp_padded


class MyDataset(Dataset):
    def __init__(self):
        self.data: list[dict] = load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq_str = self.data[item].get('seq')
        seq_list = [ord(letter) - ord('A') + 1 for letter in seq_str]
        seq_origin = torch.tensor(seq_list, dtype=torch.float16)
        seq = F.pad(seq_origin, (0, 250 - seq_origin.size(0)), mode='constant', value=0)

        ssp_str = self.data[item].get('ssp')
        ssp_list = [ord(letter) - ord('A') + 1 for letter in ssp_str]
        ssp_origin = torch.tensor(ssp_list, dtype=torch.float16)
        ssp = F.pad(ssp_origin, (0, 250 - ssp_origin.size(0)), mode='constant', value=0)

        return seq, ssp


if __name__ == '__main__':
    dataset = MyDataset()
    data_load = DataLoader(dataset, batch_size=64)

    for train_features, train_labels in data_load:
        print(train_features.shape)
