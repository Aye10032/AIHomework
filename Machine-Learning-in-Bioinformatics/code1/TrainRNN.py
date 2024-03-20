import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LoadData import OneHotDataset, MyDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.word_embeddings(x)
        rnn_out, _ = self.rnn(embeds)
        out = self.fc(rnn_out)
        out = self.softmax(out)
        return out


INPUT_SIZE = 20
EMBEDDING_DIM = 64
HIDDEN_SIZE = 250
OUTPUT_SIZE = 3
LEARNING_RATE = 0.01
EPOCH = 5

writer = SummaryWriter(log_dir=f'runs/RNN_embedding{EMBEDDING_DIM}_hidden{HIDDEN_SIZE}_LR{LEARNING_RATE}')

dataset = MyDataset(DEVICE)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

model = RNN(EMBEDDING_DIM, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)

with torch.no_grad():
    print(model)
    writer.add_graph(model, input_to_model=dataset.__getitem__(0)[0].squeeze())

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

length = len(train_loader)
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        model.zero_grad()
        sequences, labels = sequences.squeeze(), labels.squeeze()

        outputs = model(sequences)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_total += labels.shape[0]
        predicted = torch.argmax(outputs.data, dim=1)
        train_correct += torch.sum(predicted == labels).item()
        train_loss += loss.item()

    writer.add_scalar('Train/Acc', train_correct / train_total, epoch)
    writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
    writer.flush()

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    for test_sequences, test_labels in test_loader:
        test_sequences, test_labels = test_sequences.squeeze(), test_labels.squeeze()
        outputs = model(test_sequences)

        loss = loss_function(outputs, test_labels)

        test_total += test_labels.shape[0]
        predicted = torch.argmax(outputs.data, dim=1)
        test_correct += torch.sum(predicted == test_labels).item()
        test_loss += loss.item()

    writer.add_scalar('Test/Acc', test_correct / test_total, epoch)
    writer.add_scalar('Test/Loss', test_loss / len(test_loader), epoch)
    writer.flush()

    if epoch % 9 == 0 and epoch != 0:
        logger.info(f'Accuracy: {100 * test_correct / test_total}%, Loss: {test_loss / len(test_loader)}')

writer.close()
