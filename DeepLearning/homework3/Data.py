import numpy as np
from torch.utils.data import Dataset


class TangData(Dataset):
    def __init__(self):
        datas = np.load('data/tang.npz', allow_pickle=True)
        self.data = datas['data']
        self.ix2word = datas['ix2word'].item()
        self.word2ix = datas['word2ix'].item()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]
