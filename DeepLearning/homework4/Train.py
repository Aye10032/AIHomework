from dataclasses import asdict

import torch.nn
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Data import TransData, DataType
from Model import Transformer
from Config import *


def collate_fn(batch: list[Tensor]):
    src_batch = []
    target_batch = []

    for src, target in batch:
        src_batch.append(src)
        target_batch.append(target)

    pad_src = pad_sequence(src_batch, True, PAD_IDX)
    pad_target = pad_sequence(target_batch, True, PAD_IDX)

    return pad_src, pad_target


def train(
        model: nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        dataloader: DataLoader,
        writer: SummaryWriter,
        epoch: int
):
    total = 0
    total_loss = 0.
    total_correct = 0.

    model.train()
    for index, (src, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'epoch {epoch}/{EPOCH}'):
        src: Tensor = src.cuda()
        target: Tensor = target.cuda()
        real_target = target[:, 1:]
        input_target = target[:, :-1]

        outputs: Tensor = model(src, input_target)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), real_target.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1
        total_correct += torch.eq(outputs.argmax(-1).long(), real_target).sum().item() / real_target.flatten().shape[0]

    writer.add_scalar('train/loss', total_loss / total, epoch)
    writer.add_scalar('train/acc', total_correct / total, epoch)
    writer.flush()


def valid(
        model: nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        dataloader: DataLoader,
        writer: SummaryWriter,
        epoch: int
):
    total = 0
    total_loss = 0.
    total_correct = 0.

    model.eval()
    for index, (src, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'valid {epoch}/{EPOCH}'):
        src: Tensor = src.cuda()
        target: Tensor = target.cuda()
        real_target = target[:, 1:]
        input_target = target[:, :-1]

        outputs: Tensor = model(src, input_target)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), real_target.flatten())

        total_loss += loss.item()
        total += 1
        total_correct += torch.eq(outputs.argmax(-1).long(), real_target).sum().item() / real_target.flatten().shape[0]

    writer.add_scalar('valid/loss', total_loss / total, epoch)
    writer.add_scalar('valid/acc', total_correct / total, epoch)
    writer.flush()


def main() -> None:
    train_set = TransData('data', DataType.TRAIN)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    valid_set = TransData('data', DataType.VALID)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    config = ModelConfig.default_config()
    net = Transformer(**asdict(config)).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(net.parameters(), LR)

    writer = SummaryWriter(
        log_dir=f"runs/head{config.head}_layer{config.num_layers}_emb{config.emb_size}_hidden{config.hidden_size}_mlp{config.ffw_size}_{EPOCH}")

    with torch.no_grad():
        test_data, test_target = next(iter(train_loader))
        test_data, test_target = test_data.cuda(), test_target.cuda()
        writer.add_graph(net, input_to_model=[test_data, test_target])
        print(net)

    for epoch in range(EPOCH):
        train(net, criterion, optimizer, train_loader, writer, epoch)
        valid(net, criterion, valid_loader, writer, epoch)


if __name__ == '__main__':
    main()
