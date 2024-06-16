from torch.utils.data import DataLoader

from Data import TransData, DataType
from Model import Encoder


def main() -> None:
    dataset = TransData('data', DataType.TRAIN)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    test_data, test_target = next(iter(dataloader))
    vocab = len(dataset.src_word2id)

    net = Encoder(vocab, 512, 128, 3, 256, 2)
    output = net.forward(test_data)
    print(output.shape)


if __name__ == '__main__':
    main()
