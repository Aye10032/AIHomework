import os

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    Compose,
    Resize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize
)
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

from VitModel import ViT, DEVICE, ChannelSelection


def sparse_selection(net: nn.Module):
    s = 1e-4
    for m in net.modules():
        if isinstance(m, ChannelSelection):
            m.indexes.grad.data.add_(s * torch.sign(m.indexes.data))


def train(
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler.CosineAnnealingLR,
        epoch: int,
        train_loader: DataLoader,
        writer: SummaryWriter
):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    output_embed = torch.empty((0, 10))
    target_embeds = torch.empty(0)

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in loop:
        inputs: Tensor
        targets: Tensor

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sparse_selection(net)
        optimizer.step()
        schedule.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total)

        if batch_idx <= 3:
            output_embed = torch.cat((output_embed, outputs.clone().cpu()), 0)
            target_embeds = torch.cat((target_embeds, targets.data.clone().cpu()), 0)

    writer.add_scalar('lr', schedule.get_last_lr()[0], epoch)

    if epoch % 9 == 0:
        writer.add_embedding(
            output_embed,
            metadata=target_embeds,
            global_step=epoch + 1,
            tag='cifar10'
        )

    writer.add_scalar('Train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Train/acc', 100. * correct / total, epoch)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), epoch)

    writer.flush()


def test(net: nn.Module, epoch: int, test_loader: DataLoader, writer: SummaryWriter):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test {epoch}')
        for batch_idx, (inputs, targets) in loop:
            inputs: Tensor
            targets: Tensor

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=100. * correct / total)

        writer.add_scalar('Test/loss', test_loss / (batch_idx + 1), epoch)
        writer.add_scalar('Test/acc', 100. * correct / total, epoch)
        writer.flush()


def main() -> None:
    lr = 3 * 1e-3
    max_epoch = 100

    trans_train = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trans_valid = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = CIFAR10(root='./cifar10', train=True, download=True, transform=trans_train)
    test_set = CIFAR10(root='./cifar10', train=False, download=False, transform=trans_valid)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    net = ViT(
        image_size=256,
        patch_size=32,
        num_classes=10,
        dim=512,
        depth=3,
        heads=4,
        mlp_dim=128,
        dropout=.1,
        emb_dropout=.1
    ).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch * len(train_loader), eta_min=1e-5)
    writer = SummaryWriter(log_dir=f'runs/cif10_lr{lr}_ep{max_epoch}')

    with torch.no_grad():
        writer.add_graph(net, input_to_model=train_set.__getitem__(0)[0].unsqueeze(0).to(DEVICE))

    for i in range(max_epoch):
        train(net, optimizer, scheduler, i, train_loader, writer)
        test(net, i, test_loader, writer)


if __name__ == '__main__':
    main()
