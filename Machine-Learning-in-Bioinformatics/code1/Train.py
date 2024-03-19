import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LoadData import MyDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

dataset = MyDataset(DEVICE)
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(250, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, 250)
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.softmax(x)

        return x


model = MyModel().to(DEVICE)
print(model)

opt = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.rand(1, 250).to(DEVICE))


def convert_sequence(tensor: torch.tensor):
    mask_0 = tensor < 0.5
    tensor = torch.where(mask_0, torch.tensor(0.0, device=DEVICE), tensor)

    mask_1 = (tensor >= 0.5) & (tensor < 1.5)
    tensor = torch.where(mask_1, torch.tensor(1.0, device=DEVICE), tensor)

    mask_2 = (tensor >= 1.5) & (tensor < 2.5)
    tensor = torch.where(mask_2, torch.tensor(2.0, device=DEVICE), tensor)

    mask_3 = tensor >= 2.5
    tensor = torch.where(mask_3, torch.tensor(3.0, device=DEVICE), tensor)

    return tensor


length = len(train_loader) / 32
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i, (sequences, labels) in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        opt.zero_grad()
        output = model(sequences)

        loss = loss_func(output, labels)
        loss.backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            output = model(inputs)
            correct += torch.sum(convert_sequence(output) == targets).item()

    accuracy = correct / len(test_loader)

    print(f'Accuracy: {accuracy}')
