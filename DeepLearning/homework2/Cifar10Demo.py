import argparse

import evaluate
import torch
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
from accelerate import Accelerator, DataLoaderConfiguration

from VitModel import ViT, train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=7)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--mlp_size', type=int, default=128)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--scheduler', action='store_true', default=False)

    args = parser.parse_args()

    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)

    lr = 1e-4
    max_epoch = 500
    dim = args.dim
    layers = args.layers
    heads = args.heads
    hidden_size = args.hidden_size
    mlp_size = args.mlp_size

    trans_train = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(15),
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

    labels = test_set.classes

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    net = ViT(
        image_size=(224, 224),
        patch_size=(32, 32),
        num_classes=10,
        dim=dim,
        layers=layers,
        heads=heads,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dropout=0,
        emb_dropout=0,
    )
    net = accelerator.prepare_model(net)

    train_loader = accelerator.prepare_data_loader(train_loader)
    test_loader = accelerator.prepare_data_loader(test_loader)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = accelerator.prepare_optimizer(optimizer)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch * len(train_loader), eta_min=1e-5)
        scheduler = accelerator.prepare_scheduler(scheduler)
    else:
        scheduler = None

    writer = SummaryWriter(log_dir=f'more/cif10_head{heads}_layer{layers}_dim{dim}_hidden{hidden_size}_mlp{mlp_size}')
    train_metric = evaluate.load('accuracy')
    test_metric = evaluate.combine(['accuracy', 'confusion_matrix'])

    for i in range(max_epoch):
        train(net, optimizer, scheduler, accelerator, i, train_loader, writer, train_metric)

        if i % 5 == 0:
            test(net, accelerator, i, test_loader, writer, test_metric, labels)

        # if i % 50 == 0 and i != 0:
        #     torch.save(net.state_dict(), f'models/cif10_head{heads}_layer{layers}_dim{dim}_ep{i}.pth')


if __name__ == '__main__':
    main()
