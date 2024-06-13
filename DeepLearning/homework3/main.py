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


def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    inputs = torch.tensor([word2ix['<START>']], device='cuda').view(1, 1).long()

    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(50):
            output, hidden = model(inputs, hidden)
            # 如果在给定的句首中，input 为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                inputs = inputs.data.new([word2ix[w]]).view(1, 1)
            # 否则将 output 作为下一个 input 进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                inputs = inputs.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break

    return ''.join(results)


def main() -> None:
    dataset = TangData()
    ix2word = dataset.ix2word
    word2ix = dataset.word2ix

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    writer = SummaryWriter(f'runs/emb{1024}_hidden{512}')

    model = PoetryModel(len(word2ix), 1024, 512).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    with torch.no_grad():
        data0 = next(iter(dataloader))
        writer.add_graph(model, input_to_model=data0[0].cuda())

    max_loss = 100
    for epoch in range(20):
        train(model, dataloader, optimizer, criterion, loss_meter, epoch)
        writer.add_scalar('loss', loss_meter.mean, epoch)
        if max_loss > loss_meter.mean:
            max_loss = loss_meter.mean
            torch.save(model.state_dict(), 'model/model.pth')
        loss_meter.reset()

    model_output = generate(model, '举头', ix2word, word2ix)
    print(model_output)


if __name__ == '__main__':
    main()
