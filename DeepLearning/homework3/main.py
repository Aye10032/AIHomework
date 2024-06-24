import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchnet import meter
from tqdm import tqdm

from Data import TangData


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(PoetryModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, vocab_size)
        )

    def forward(self, x, hidden=None):
        embeds = self.embeddings(x)
        # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = x.size()
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output)
        output = output.reshape(batch_size * seq_len, -1)

        return output, hidden


def train(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.CrossEntropyLoss,
        loss_meter: meter.AverageValueMeter,
        epoch
):
    model.train()
    for i, (inputs, targets) in tqdm(enumerate(dataloader), desc=f'Epoc {epoch}', total=len(dataloader)):
        optimizer.zero_grad()
        inputs: Tensor = inputs.cuda()
        targets: Tensor = targets.cuda().view(-1)

        outputs, hidden = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())


def main() -> None:
    dataset = TangData()
    word2ix = dataset.word2ix

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    emb_dim = 128
    hidden_size = 1024
    num_layers = 3

    writer = SummaryWriter(f'runs/emb{emb_dim}_hidden{hidden_size}_layer{num_layers}')

    model = PoetryModel(len(word2ix), emb_dim, hidden_size, num_layers).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    with torch.no_grad():
        data0 = next(iter(dataloader))
        writer.add_graph(model, input_to_model=data0[0].cuda())

    max_loss = 100
    for epoch in range(30):
        train(model, dataloader, optimizer, criterion, loss_meter, epoch)
        writer.add_scalar('loss', loss_meter.mean, epoch)
        if max_loss > loss_meter.mean:
            max_loss = loss_meter.mean
            torch.save(model.state_dict(), 'model/model.pth')
        loss_meter.reset()


if __name__ == '__main__':
    main()
