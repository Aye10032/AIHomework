from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from Data import TransData, DataType
from Model import Encoder, Decoder
from Config import *


def collate_fn(batch: list[Tensor]):
    src_batch = []
    target_batch = []

    for src, target in batch:
        src_batch.append(src)
        target_batch.append(target)

    pad_src = pad_sequence(src_batch, True, PAD_IDX)
    pad_target = pad_sequence(target_batch, True, PAD_IDX)

    return pad_src, pad_target


def main() -> None:
    dataset = TransData('data', DataType.TRAIN)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_data, test_target = next(iter(dataloader))
    vocab = len(dataset.src_word2id)

    encoder = Encoder(vocab, 512, 128, 3, 256, 2)
    decoder = Decoder(len(dataset.target_word2id), 512, 128, 3, 256, 2)
    print(test_data.shape, test_target.shape)
    output = encoder.forward(test_data)
    # print(output.shape)
    decoder_out = decoder.forward(test_target, test_data, output)
    print(decoder_out.shape)


if __name__ == '__main__':
    main()
