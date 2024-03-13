import io
import pickle
import tarfile
from typing import Dict, List

import torch
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
        seq = torch.tensor(seq_list, dtype=torch.float16)

        ssp_str = self.data[item].get('ssp')
        ssp_list = [ord(letter) - ord('A') + 1 for letter in ssp_str]
        ssp = torch.tensor(ssp_list, dtype=torch.float16)

        return seq, ssp


if __name__ == '__main__':
    dataset = MyDataset()
    data_load = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    train_features, train_labels = next(iter(data_load))
