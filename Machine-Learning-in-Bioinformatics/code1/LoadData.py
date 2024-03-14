import io
import pickle
import tarfile
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 20种氨基酸序列对应的的字典表
code_dict = {
    'A': 1,
    'F': 2,
    'C': 3,
    'D': 4,
    'N': 5,
    'E': 6,
    'Q': 7,
    'G': 8,
    'H': 9,
    'L': 10,
    'I': 11,
    'K': 12,
    'M': 13,
    'P': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20
}

# 三种蛋白质二级结构
type_dict = {'C': 1, 'E': 2, 'H': 3}


def load_data() -> List[Dict]:
    """
    :return: datas中的每一条序列长度不一定相同
    """
    datas = []
    with tarfile.open('../dataset/assignment1_data.tar', 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.pkl'):
                if file := tar.extractfile(member):
                    byte_stream = io.BytesIO(file.read())
                    data = pickle.load(byte_stream)

                    """
                    每一条记录中包含一个键值对，分别是seq和ssp，两者长度相同
                    seq代表氨基酸序列
                    ssp代表对应的蛋白质二级结构
                    如：{'seq':'MHPLSIEGAWSQEPVIHSDHRGR','ssp':'CEECCCCCEEEECCCEEEECCEE'}
                    """
                    datas.append(data)

    return datas


class MyDataset(Dataset):
    def __init__(self, device, padding: bool = False):
        self.data: list[dict] = load_data()
        self.device = device
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq_str = self.data[item].get('seq')
        seq_origin = torch.tensor([code_dict.get(letter) for letter in seq_str],
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=True)

        ssp_str = self.data[item].get('ssp')
        ssp_origin = torch.tensor([type_dict.get(letter) for letter in ssp_str],
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=True)

        if self.padding:
            seq = F.pad(seq_origin, (0, 250 - seq_origin.size(0)), mode='constant', value=0)
            ssp = F.pad(ssp_origin, (0, 250 - ssp_origin.size(0)), mode='constant', value=0)
            return seq, ssp
        else:
            return seq_origin, ssp_origin


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(DEVICE)
    data_load = DataLoader(dataset, batch_size=64)

    for train_features, train_labels in data_load:
        print(train_features.shape, train_labels.shape)
