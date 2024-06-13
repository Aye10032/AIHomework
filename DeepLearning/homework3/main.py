import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchnet import meter

from DeepLearning.homework3.Data import TangData


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embeddings(x)
        # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = x.size()
        if hidden is None:
            h_0 = x.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output)
        output = output.reshape(batch_size * seq_len, -1)

        return output, hidden


def train(model, dataloader, optimizer, criterion, loss_meter):
    model.train()
    # TODO


def main() -> None:
    dataset = TangData()
    ix2word = dataset.ix2word
    word2ix = dataset.word2ix

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    with torch.no_grad():
        data0 = next(iter(dataloader))
        print(data0.shape)

    model = PoetryModel(len(word2ix), 1024, 512).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()


if __name__ == '__main__':
    main()
