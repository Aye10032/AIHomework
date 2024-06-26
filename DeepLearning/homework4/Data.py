import os.path
from enum import IntEnum

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
from tqdm import tqdm

from Config import PAD_IDX, EOB_IDX, SOB_IDX


class DataType(IntEnum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class Tokenizer:
    def __init__(
            self,
            i2w: dict[int, str],
            w2i: dict[str, int],
    ):
        self.i2w = i2w
        self.w2i = w2i

    def tokenize(self, sentence: list[str]) -> Tensor:
        return torch.LongTensor([
            self.w2i[word]
            if word in self.w2i else
            self.w2i['<UNK>']
            for word in sentence
        ])

    def detokenize(self, tensor: Tensor) -> str:
        output_list = [
            self.i2w[idx.item()]
            for idx in tensor
            if idx not in [PAD_IDX, EOB_IDX, SOB_IDX]
        ]

        return ' '.join(output_list)


class TransData(Dataset):
    def __init__(self, data_path: str, data_type: DataType, show_log: bool):
        match data_type:
            case DataType.TRAIN:
                self.src_file = os.path.join(data_path, 'train.zh')
                self.tgt_file = os.path.join(data_path, 'train.en')
            case DataType.VALID:
                self.src_file = os.path.join(data_path, 'dev.zh')
                self.tgt_file = os.path.join(data_path, 'dev.en')
            case DataType.TEST:
                self.src_file = os.path.join(data_path, 'test.zh')
                self.tgt_file = os.path.join(data_path, 'test.en')
            case _:
                raise RuntimeError('无效的参数！')

        self.src_vocab = os.path.join(data_path, 'vocab.zh')
        self.tgt_vocab = os.path.join(data_path, 'vocab.en')

        self.__build_tokenizer(show_log)
        self.__load_data(show_log)

    def __getitem__(self, item):
        return self.pad_src[item], self.pad_tgt[item]

    def __len__(self):
        return self.pad_src.shape[0]

    def __build_tokenizer(self, show_log: bool):
        if show_log:
            logger.info('building tokenizer...')
        with open(self.src_vocab, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f]
        self.src_id2word = {index: v for index, v in enumerate(src_lines)}
        self.src_word2id = {v: index for index, v in enumerate(src_lines)}

        with open(self.tgt_vocab, 'r', encoding='utf-8') as f:
            tgt_lines = [line.strip() for line in f]
        self.tgt_id2word = {index: v for index, v in enumerate(tgt_lines)}
        self.tgt_word2id = {v: index for index, v in enumerate(tgt_lines)}

        self.src_tokenizer = Tokenizer(self.src_id2word, self.src_word2id)
        self.tgt_tokenizer = Tokenizer(self.tgt_id2word, self.tgt_word2id)

    def __load_data(self, show_log: bool):
        if show_log:
            logger.info('loading dataset...')
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_lines = [
                ['<SOB>'] + line.strip().split(' ') + ['<EOB>']
                for line in f
            ]

        with open(self.tgt_file, 'r', encoding='utf-8') as f:
            tgt_lines = [
                ['<SOB>'] + line.strip().split(' ') + ['<EOB>']
                for line in f
            ]

        if show_log:
            logger.info('tokenize sentences...')
        self.src_ids = []
        for line in tqdm(src_lines, disable=not show_log):
            self.src_ids.append(self.src_tokenizer.tokenize(line))

        self.tgt_ids = []
        for line in tqdm(tgt_lines, disable=not show_log):
            self.tgt_ids.append(self.tgt_tokenizer.tokenize(line))

        self.pad_src = pad_sequence(self.src_ids, True, PAD_IDX)
        self.pad_tgt = pad_sequence(self.tgt_ids, True, PAD_IDX)


def collate_fn(batch: list[tuple[Tensor, Tensor]]):
    src_ids, tgt_ids = [], []

    for src, tgt in batch:
        src_ids.append(src)
        tgt_ids.append(tgt)

    pad_src = pad_sequence(src_ids, True, PAD_IDX)
    pad_tgt = pad_sequence(tgt_ids, True, PAD_IDX)

    return pad_src, pad_tgt


def main() -> None:
    dataset = TransData('data', DataType.TRAIN, True)
    logger.info('build dataloader')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)
    logger.info('done')
    data0 = next(iter(dataloader))
    print(dataset.src_tokenizer.detokenize(data0[0][0]))
    print(dataset.tgt_tokenizer.detokenize(data0[0][1]))


if __name__ == '__main__':
    main()
