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

k_fold = KFold(n_splits=3, shuffle=True)
opt = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

writer = SummaryWriter()


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


for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):
    logger.info(f'fold {fold}')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=test_subsampler)

    length = len(train_loader) / 32
    epochs = 10
    for epoch in range(epochs):
        model.train()

        for batch_index, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            output = model(data)

            loss = loss_func(output, target)
            loss.backward()
            opt.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                output = model(inputs)
                test_loss += loss_func(output, targets).item()

                correct += torch.sum(convert_sequence(output) == targets).item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        print(f'Test Loss: {test_loss} , Test Accuracy: {accuracy}')
