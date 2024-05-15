import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel
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
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from tqdm import tqdm

from VitModel import ViT, ChannelSelection

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast


def sparse_selection(net: nn.Module):
    s = 1e-4
    for m in net.modules():
        if isinstance(m, ChannelSelection):
            m.indexes.grad.data.add_(s * torch.sign(m.indexes.data))


def train(
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler.CosineAnnealingLR,
        device: list[torch.device],
        epoch: int,
        train_loader: DataLoader,
        writer: SummaryWriter
):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # output_embed = torch.empty((0, 10))
    # target_embeds = torch.empty(0)

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in loop:
        inputs: Tensor
        targets: Tensor

        inputs, targets = inputs.to(device[0]), targets.to(device[-1])
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

        # if batch_idx <= 3:
        #     output_embed = torch.cat((output_embed, outputs.clone().cpu()), 0)
        #     target_embeds = torch.cat((target_embeds, targets.data.clone().cpu()), 0)

    writer.add_scalar('lr', schedule.get_last_lr()[0], epoch)

    # if epoch % 9 == 0:
    #     writer.add_embedding(
    #         output_embed,
    #         metadata=target_embeds,
    #         global_step=epoch + 1,
    #         tag='image_net'
    #     )

    writer.add_scalar('Train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Train/acc', 100. * correct / total, epoch)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), epoch)

    writer.flush()


def test(net: nn.Module, device: list[torch.device], epoch: int, test_loader: DataLoader, writer: SummaryWriter):
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

            inputs, targets = inputs.to(device[0]), targets.to(device[0])
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


def load_data() -> Tuple[DataLoader, DataLoader]:
    trans_train = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        # RandomRotation(45),
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

    logger.info('start loading ImageNet train dataset')
    train_set = ImageNet(root='./ImageNet', split='train', transform=trans_train)
    logger.info('start loading ImageNet valid dataset')
    test_set = ImageNet(root='./ImageNet', split='val', transform=trans_valid)
    logger.info('done')

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, test_loader


def main() -> None:
    device0 = torch.device('cuda', 0)
    device1 = torch.device('cuda', 1)
    lr = 3 * 1e-3
    max_epoch = 300

    train_loader, test_loader = load_data()

    model = ViT(
        image_size=224,
        patch_size=(16, 16),
        num_classes=1000,
        dim=256,
        layers=12,
        heads=6,
        hidden_size=384,
        mlp_size=1536,
        dropout=.1,
        emb_dropout=.1,
        device=[device0, device1]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch * len(train_loader), eta_min=1e-5)
    writer = SummaryWriter(log_dir=f'runs/imagenet_base_ep{max_epoch}')

    with torch.no_grad():
        tensor = torch.rand(1, 3, 224, 224).to(device0)
        writer.add_graph(model, input_to_model=tensor)

    for i in range(max_epoch):
        train(model, optimizer, scheduler, [device0, device1], i, train_loader, writer)

        if i % 5 == 0:
            test(model, [device0, device1], i, test_loader, writer)

        if i % 50 == 0 and i != 0:
            torch.save(model.module.state_dict(), f'models/imagenet_base_ep{i}.pth')


if __name__ == '__main__':
    main()
