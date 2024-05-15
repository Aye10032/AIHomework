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

    train_sample = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_set)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=4, sampler=train_sample)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, sampler=test_sample)

    return train_loader, test_loader


def main() -> None:
    dist.init_process_group(backend="nccl")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    lr = 1e-4
    max_epoch = 500
    dim = 512
    layers = 6
    heads = 8
    hidden_size = 64
    mlp_size = 128

    train_loader, test_loader = load_data()

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

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
    ddp_net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch * len(train_loader), eta_min=1e-5)
    writer = SummaryWriter(log_dir=f'runs/cif10_lr{lr}_head{heads}_layer{layers}_dim{dim}_ep500')

    with torch.no_grad():
        tensor = torch.rand(1, 3, 224, 224).to(device)
        writer.add_graph(net, input_to_model=tensor)

    for i in range(max_epoch):
        train(net, optimizer, scheduler, [device], i, train_loader, writer)

        if i % 5 == 0:
            test(net, [device], i, test_loader, writer)

        if i % 50 == 0 and i != 0 and local_rank == 0:
            torch.save(net.state_dict(), f'models/cif10_head{heads}_layer{layers}_dim{dim}_ep{i}.pth')


if __name__ == '__main__':
    main()
