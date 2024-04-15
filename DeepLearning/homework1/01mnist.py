import os

import torch.nn
from torch.nn import Module, Sequential, Conv2d, Flatten, GELU, MaxPool2d, Linear, Softmax, ReLU, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

BATCH_SIZE = 50
LR = 0.001
EPOCH = 15
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = MNIST(root='./data/', train=True, transform=ToTensor())
test_data = MNIST(root='./data/', train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = Sequential(
            Conv2d(1, 3, 3, 1, 1),
            GELU(),
            Conv2d(3, 16, 3, 1, 1),
            ReLU(),
            MaxPool2d(2, 2)
        )

        self.layer2 = Sequential(
            Conv2d(16, 32, 3, 1, 1),
            GELU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.flatten = Flatten()

        self.mlp = Sequential(
            Linear(32 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10),
        )

        self.softmax = Softmax(dim=0)

    def forward(self, x):
        # print(x.shape)
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.flatten(output)
        # print(x.shape)
        output = self.mlp(output)
        # output = self.softmax(x)

        return output


model = Net().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss()

writer = SummaryWriter(log_dir=f'runs/mnist_gelu_LR{LR}')

with torch.no_grad():
    print(model)
    writer.add_graph(model, input_to_model=train_data.__getitem__(0)[0].unsqueeze(0).to(DEVICE))

length = train_data.data.shape[0] / BATCH_SIZE
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for step, epoch_data in tqdm(enumerate(train_loader), total=length, desc=f'epoch{epoch}'):
        datas = epoch_data[0].to(DEVICE)
        labels = epoch_data[1].to(DEVICE)

        optimizer.zero_grad()

        train_output = model(datas)
        loss = loss_func(train_output, labels)
        loss.backward()
        optimizer.step()

        predict_label = torch.max(train_output, 1)[1]
        # print(torch.sum(predict_label == labels).item())
        train_correct += torch.sum(predict_label == labels).item()
        train_total += labels.shape[0]
        train_loss += loss.item()

    writer.add_scalar('Train/Acc', train_correct / train_total, epoch)
    writer.add_scalar('Train/Loss', train_loss / train_total, epoch)
    writer.flush()

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), epoch)
    writer.flush()

    if epoch >= 6:
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/mnist_epoch{epoch}.pth')

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    for step, epoch_data in enumerate(test_loader):
        datas = epoch_data[0].to(DEVICE)
        labels = epoch_data[1].to(DEVICE)

        test_output = model(datas)
        loss = loss_func(test_output, labels)

        test_label = torch.max(test_output, 1)[1]
        test_correct += torch.sum(test_label == labels).item()
        test_total += labels.shape[0]
        test_loss += loss.item()

    writer.add_scalar('Test/Acc', test_correct / test_total, epoch)
    writer.add_scalar('Test/Loss', test_loss / len(test_loader), epoch)
    writer.flush()

writer.close()
