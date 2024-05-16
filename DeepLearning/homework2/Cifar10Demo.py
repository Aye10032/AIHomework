import argparse
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    Compose,
    Resize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ToTensor,
    Normalize
)
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from VitModel import ViT, train, test


def load_data():
    trans_train = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(45).
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trans_valid = Compose([
        Resize(224),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = CIFAR10(root='./cifar10', train=True, download=True, transform=trans_train)
    test_set = CIFAR10(root='./cifar10', train=False, download=False, transform=trans_valid)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--mlp_size', type=int, default=128)
    parser.add_argument('--gpu_index', type=int, default=0)

    args = parser.parse_args()

    gpu_index = args.gpu_index
    lr = 1e-4
    max_epoch = 500
    dim = args.dim
    layers = args.layers
    heads = args.heads
    hidden_size = args.hidden_size
    mlp_size = args.mlp_size

    train_loader, test_loader = load_data()

    device = torch.device("cuda", gpu_index)

    net = ViT(
        image_size=(224, 224),
        patch_size=(32, 32),
        num_classes=10,
        dim=dim,
        layers=layers,
        heads=heads,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dropout=0.,
        emb_dropout=0.,
        device=[device]
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch * len(train_loader), eta_min=1e-5)
    writer = SummaryWriter(log_dir=f'runs/cif10_rotate_head{heads}_layer{layers}_dim{dim}_ep500')

    with torch.no_grad():
        tensor = torch.rand(1, 3, 224, 224).to(device)
        writer.add_graph(net, input_to_model=tensor)

    for i in range(max_epoch):
        train(net, optimizer, scheduler, [device], i, train_loader, writer)

        if i % 5 == 0:
            test(net, [device], i, test_loader, writer)

        if i % 50 == 0 and i != 0:
            torch.save(net.state_dict(), f'models/cif10_head{heads}_layer{layers}_dim{dim}_ep{i}.pth')


if __name__ == '__main__':
    main()
