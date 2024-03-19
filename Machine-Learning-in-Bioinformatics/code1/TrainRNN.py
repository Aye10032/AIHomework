import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LoadData import OneHotDataset, MyDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = torch.zeros(1, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(1, self.hidden_size).to(DEVICE)

        return h0, c0

    def forward(self, x):
        embeds = self.word_embeddings(x)
        # print(embeds.shape)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(lstm_out.shape)
        out = self.fc(lstm_out) # .view(x.shape[0], -1)
        out = F.log_softmax(out, dim=1)

        return out


INPUT_SIZE = 20
EMBEDDING_DIM = 32
HIDDEN_SIZE = 15
NUM_LAYERS = 3
OUTPUT_SIZE = 3
LEARNING_RATE = 0.1
EPOCH = 10

dataset = MyDataset(DEVICE)
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

model = RNN(EMBEDDING_DIM, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)

with torch.no_grad():
    print(model)
    # writer.add_graph(model, input_to_model=dataset.__getitem__(0)[0])

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

length = len(train_loader)
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        model.zero_grad()
        model.hidden = model.init_hidden()
        sequences, labels = sequences.squeeze(), labels.squeeze()

        outputs = model(sequences)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    writer.add_scalar('Loss', train_loss / len(train_loader), epoch)

    model.eval()
    correct = 0
    total = 0
    for test_sequences, test_labels in test_loader:
        test_sequences, test_labels = test_sequences.squeeze(), test_labels.squeeze()
        # print(test_sequences.shape)
        outputs = model(test_sequences)
        # print(outputs.shape)
        predicted = torch.argmax(outputs.data, dim=1)
        total += test_labels.shape[0]
        # print(predicted.shape, test_labels.shape)
        correct += torch.sum(predicted == test_labels).item()
        # print(correct, total)

    writer.add_scalar('Accuracy', correct / total, epoch)
    writer.flush()

    if epoch % 10 == 0:
        logger.info(f'Accuracy: {100 * correct / total}%, Loss: {train_loss / len(train_loader)}')

writer.close()
