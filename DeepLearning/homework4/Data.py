import os.path
from enum import IntEnum

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
from tqdm import tqdm


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

        self.__build_tokenizer()
        self.__load_data()

    def __getitem__(self, item):
        return self.src_tensor[item], self.target_tensor[item]

    def __len__(self):
        return self.src_tensor.shape[0]

    def __build_tokenizer(self):
        logger.info('building tokenizer...')
        with open(self.src_vocab, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f]
        self.src_id2word = {index: v for index, v in enumerate(src_lines)}
        self.src_word2id = {v: index for index, v in enumerate(src_lines)}

        with open(self.target_vocab, 'r', encoding='utf-8') as f:
            target_lines = [line.strip() for line in f]
        self.target_id2word = {index: v for index, v in enumerate(target_lines)}
        self.target_word2id = {v: index for index, v in enumerate(target_lines)}

    def __load_data(self):
        logger.info('loading dataset...')
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_lines = [['<SOB>'] + line.strip().split(' ') + ['<EOB>'] for line in f]

        with open(self.target_file, 'r', encoding='utf-8') as f:
            target_lines = [['<SOB>'] + line.strip().split(' ') + ['<EOB>'] for line in f]

        logger.info('tokenize src...')
        src_ids = []
        for line in tqdm(src_lines):
            src_ids.append(torch.LongTensor([
                self.src_word2id[word]
                if word in self.src_word2id else
                self.src_word2id['<UNK>']
                for word in line
            ]))

        target_ids = []
        for line in tqdm(target_lines):
            target_ids.append(torch.LongTensor([
                self.target_word2id[word]
                if word in self.target_word2id else
                self.target_word2id['<UNK>']
                for word in line
            ]))

        logger.info('padding dataset...')
        self.src_tensor = pad_sequence(src_ids, True, self.src_word2id['<PAD>'])
        self.target_tensor = pad_sequence(target_ids, True, self.target_word2id['<PAD>'])


def main() -> None:
    dataset = TransData('data', DataType.TRAIN)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    print(next(iter(dataloader)))


if __name__ == '__main__':
    main()
