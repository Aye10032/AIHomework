import torch
from loguru import logger
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LoadData import OneHotDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)

        return out


INPUT_SIZE = 20
HIDDEN_SIZE = 15
NUM_LAYERS = 2
OUTPUT_SIZE = 3
LEARNING_RATE = 0.02
EPOCH = 10

dataset = OneHotDataset(DEVICE)
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(DEVICE)
writer.add_graph(model, input_to_model=torch.rand(1, 250, 20).to(DEVICE))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

length = len(train_loader)
for epoch in range(EPOCH):
    train_loss = 0
    for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        outputs = model(sequences)
        loss = criterion(outputs.view(-1, OUTPUT_SIZE), labels.view(-1).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    writer.add_scalar('Loss', train_loss / len(train_loader), epoch)

    correct = 0
    total = 0
    for sequences, labels in test_loader:
        outputs = model(sequences)
        predicted = torch.argmax(outputs.data, dim=2)
        total += labels.shape[0]
        correct += torch.sum(predicted == labels).item()

    writer.add_scalar('Accuracy', correct / total, epoch)
    writer.flush()
    logger.info(f'Accuracy: {correct / total}%, Loss: {train_loss / len(train_loader)}')

writer.close()
