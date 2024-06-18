from dataclasses import asdict

import evaluate
import numpy as np
import torch.nn
from accelerate import Accelerator, DataLoaderConfiguration
from evaluate import EvaluationModule
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Data import TransData, DataType
from Model import Transformer
from Config import *


def train(
        model: nn.Module,
        accelerator: Accelerator,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        dataloader: DataLoader,
        metric: EvaluationModule,
        writer: SummaryWriter,
        epoch: int
):
    train_loss = []

    model.train()
    for index, (src, target) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f'Epoch {epoch}({accelerator.device})',
            disable=not accelerator.is_local_main_process
    ):
        src: Tensor
        target: Tensor
        real_target = target[:, 1:]
        input_target = target[:, :-1]

        outputs: Tensor = model(src, input_target)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), real_target.flatten())

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        train_loss.append(loss.item())
        predicted = outputs.argmax(-1)

        global_predictions, global_targets = accelerator.gather_for_metrics((predicted.flatten(), real_target.flatten()))
        metric.add_batch(predictions=global_predictions, references=global_targets)

    global_train_loss = accelerator.gather_for_metrics(train_loss)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        result = metric.compute()

        writer.add_scalar('Train/loss', np.mean(global_train_loss), epoch)
        writer.add_scalar('Train/acc', 100. * result['accuracy'], epoch)

        writer.flush()


def valid(
        model: nn.Module,
        accelerator: Accelerator,
        criterion: torch.nn.CrossEntropyLoss,
        dataloader: DataLoader,
        metric: EvaluationModule,
        writer: SummaryWriter,
        epoch: int
):
    valid_loss = []

    model.eval()
    for index, (src, target) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f'Valid {epoch}({accelerator.device})',
            disable=not accelerator.is_local_main_process
    ):
        src: Tensor
        target: Tensor
        real_target = target[:, 1:]
        input_target = target[:, :-1]

        outputs: Tensor = model(src, input_target)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), real_target.flatten())

        valid_loss.append(loss.item())
        predicted = outputs.argmax(-1).long()

        global_predictions, global_targets = accelerator.gather_for_metrics((predicted.flatten(), real_target.flatten()))
        metric.add_batch(predictions=global_predictions, references=global_targets)

    global_valid_loss = accelerator.gather_for_metrics(valid_loss)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        result = metric.compute()

        writer.add_scalar('Train/loss', np.mean(global_valid_loss), epoch)
        writer.add_scalar('Train/acc', 100. * result['accuracy'], epoch)

        writer.flush()


def main() -> None:
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)

    config = ModelConfig.default_config()
    net = Transformer(**asdict(config))
    net = accelerator.prepare_model(net)

    train_set = TransData('data', DataType.TRAIN)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    train_loader = accelerator.prepare_data_loader(train_loader)

    valid_set = TransData('data', DataType.VALID)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True)
    valid_loader = accelerator.prepare_data_loader(valid_loader)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(net.parameters(), LR)
    optimizer = accelerator.prepare_optimizer(optimizer)

    writer = SummaryWriter(
        log_dir=f"runs/head{config.head}_layer{config.num_layers}_emb{config.emb_size}_hidden{config.hidden_size}_mlp{config.ffw_size}_{EPOCH}")

    # with torch.no_grad():
    #     test_data, test_target = next(iter(train_loader))
    #     test_data, test_target = test_data.cuda(), test_target.cuda()
    #     writer.add_graph(net, input_to_model=[test_data, test_target])
    #     print(net)
    metric = evaluate.load('accuracy')

    for epoch in range(EPOCH):
        train(net, accelerator, criterion, optimizer, train_loader, metric, writer, epoch)
        valid(net, accelerator, criterion, valid_loader, metric, writer, epoch)


if __name__ == '__main__':
    main()
