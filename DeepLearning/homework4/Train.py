from dataclasses import asdict

import torch.nn
from loguru import logger
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
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
        epoch: int
):
    total = 0
    total_loss = 0.

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

    logger.debug(total_loss / total)


def main() -> None:
    train_set = TransData('data', DataType.TRAIN)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)

    config = asdict(ModelConfig.default_config())
    net = Transformer(**config).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(net.parameters(), LR)

    print(net)

    for epoch in range(EPOCH):
        train(net, criterion, optimizer, train_loader, epoch)


if __name__ == '__main__':
    main()
