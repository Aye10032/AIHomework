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
from tqdm import tqdm

from VitModel import ViT, ChannelSelection


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
    output_embed = torch.empty((0, 10))
    target_embeds = torch.empty(0)

    train_loader.sampler.set_epoch(epoch)

    if dist.get_rank() == 0:
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    else:
        loop = enumerate(train_loader)

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

        if dist.get_rank() == 0:
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total)

        if batch_idx <= 3:
            output_embed = torch.cat((output_embed, outputs.clone().cpu()), 0)
            target_embeds = torch.cat((target_embeds, targets.data.clone().cpu()), 0)

    writer.add_scalar('lr', schedule.get_last_lr()[0], epoch)

    if dist.get_rank() == 0:
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


def test(net: nn.Module, device: list[torch.device], epoch: int, test_loader: DataLoader, writer: SummaryWriter):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        if dist.get_rank() == 0:
            loop = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test {epoch}')
        else:
            loop = enumerate(test_loader)

        for batch_idx, (inputs, targets) in loop:
            inputs: Tensor
            targets: Tensor

            inputs, targets = inputs.to(device[0]), targets.to(device[-1])
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if dist.get_rank() == 0:
                loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=100. * correct / total)

        if dist.get_rank() == 0:
            writer.add_scalar('Test/loss', test_loss / (batch_idx + 1), epoch)
            writer.add_scalar('Test/acc', 100. * correct / total, epoch)
            writer.flush()


def load_data():
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

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, sampler=train_sample)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, sampler=test_sample)

    return train_loader, test_loader


def main() -> None:
    dist.init_process_group(backend="nccl")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    lr = 1e-4
    max_epoch = 500
    dim = 512
    layers = 7
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
            torch.save(net.state_dict(), f'models/cifar10_head6_layer3_dim256_ep{i}.pth')


if __name__ == '__main__':
    main()
