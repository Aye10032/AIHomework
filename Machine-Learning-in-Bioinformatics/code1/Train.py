import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LoadData import OneHotDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

dataset = OneHotDataset(DEVICE, True)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Conv1d(250, 250, 3, padding='same'),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)

        return x


model = MLP().to(DEVICE)

writer = SummaryWriter(log_dir=f'runs/CNN')
with torch.no_grad():
    print(model)
    writer.add_graph(model, input_to_model=dataset.__getitem__(0)[0].squeeze())

opt = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# length = len(train_loader)
# for epoch in range(20):  # number of epochs
#     for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
#         sequences, labels = sequences.squeeze(), labels.squeeze()
#
#         opt.zero_grad()
#         output = model(sequences)
#         # print(output.shape, labels.shape)
#         loss = loss_function(output, labels.long())
#         loss.backward()
#         opt.step()
#
#     print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

length = len(train_loader)
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        opt.zero_grad()
        sequences, labels = sequences.squeeze(), labels.squeeze()

        output = model(sequences)

        loss = loss_function(output, labels.long())
        loss.backward()
        opt.step()

        train_total += labels.shape[0]
        predicted = torch.argmax(output.data, dim=1)
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

        loss = loss_function(outputs, test_labels.long())

        test_total += test_labels.shape[0]
        predicted = torch.argmax(outputs.data, dim=1)
        test_correct += torch.sum(predicted == test_labels).item()
        test_loss += loss.item()

    writer.add_scalar('Test/Acc', test_correct / test_total, epoch)
    writer.add_scalar('Test/Loss', test_loss / len(test_loader), epoch)
    writer.flush()

writer.close()
