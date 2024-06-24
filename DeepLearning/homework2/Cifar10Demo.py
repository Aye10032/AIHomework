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
    ElasticTransform,
    ColorJitter,
    ToTensor,
    Normalize,
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
    parser.add_argument('--hidden_size', type=int, default=72)
    parser.add_argument('--mlp_size', type=int, default=256)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--lr', type=int, default=1e-4)

    args = parser.parse_args()

    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)

    lr = args.lr
    max_epoch = 900
    dim = args.dim
    layers = args.layers
    heads = args.heads
    hidden_size = args.hidden_size
    mlp_size = args.mlp_size

    trans_train = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(90),
        ColorJitter(0.5, 0.5, 0.5),
        ToTensor(),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    trans_valid = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_set = CIFAR10(root='./cifar10', train=True, download=True, transform=trans_train)
    test_set = CIFAR10(root='./cifar10', train=False, download=False, transform=trans_valid)

    labels = test_set.classes

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, persistent_workers=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, persistent_workers=True, num_workers=4)

    net = ViT(
        image_size=(224, 224),
        patch_size=(32, 32),
        num_classes=10,
        dim=dim,
        layers=layers,
        heads=heads,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        pool='cls',
        dropout=0.05,
        emb_dropout=0.05,
    )
    net = accelerator.prepare_model(net)

    train_loader = accelerator.prepare_data_loader(train_loader)
    test_loader = accelerator.prepare_data_loader(test_loader)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = accelerator.prepare_optimizer(optimizer)
    if args.scheduler:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=max_epoch * len(train_loader),
        #     eta_min=1e-6
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            0.8,
            10,
        )
        scheduler = accelerator.prepare_scheduler(scheduler)
    else:
        scheduler = None

    model_name = f'cif10_head{heads}_layer{layers}_dim{dim}_hidden{hidden_size}_mlp{mlp_size}_{lr}+'
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    train_metric = evaluate.load('accuracy')
    test_metric = evaluate.combine(['accuracy', 'confusion_matrix'])

    best_acc = 0.
    for i in range(max_epoch + 1):
        train(net, optimizer, scheduler, accelerator, i, train_loader, writer, train_metric)

        if i % 2 == 0:
            best_acc = test(net, accelerator, i, test_loader, writer, test_metric, best_acc, model_name)


if __name__ == '__main__':
    main()
