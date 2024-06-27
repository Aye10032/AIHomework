from dataclasses import asdict

import evaluate
import numpy as np
import torch.nn
from accelerate import Accelerator, DataLoaderConfiguration
from evaluate import EvaluationModule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

from Data import TransData, DataType, collate_fn
from Model import Transformer
from Config import *


def train(
        model: nn.Module,
        accelerator: Accelerator,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        scheduler: ReduceLROnPlateau,
        dataloader: DataLoader,
        acc_metric: EvaluationModule,
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
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss.append(loss.item())
        predicted = outputs.argmax(-1)

        global_predictions, global_targets = accelerator.gather_for_metrics((predicted.flatten(), real_target.flatten()))
        acc_metric.add_batch(predictions=global_predictions, references=global_targets)

    global_train_loss = accelerator.gather_for_metrics(train_loss)
    if not accelerator.optimizer_step_was_skipped:
        scheduler.step(np.mean(global_train_loss), epoch)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        acc_result = acc_metric.compute()

        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        writer.add_scalar('Train/loss', np.mean(global_train_loss), epoch)
        writer.add_scalar('Train/acc', 100. * acc_result['accuracy'], epoch)

        writer.flush()


def valid(
        model: nn.Module,
        accelerator: Accelerator,
        criterion: torch.nn.CrossEntropyLoss,
        dataloader: DataLoader,
        acc_metric: EvaluationModule,
        best_acc: float,
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
        acc_metric.add_batch(predictions=global_predictions, references=global_targets)

    global_valid_loss = accelerator.gather_for_metrics(valid_loss)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        acc_result = acc_metric.compute()
        best_acc = max(best_acc, 100. * acc_result['accuracy'])

        writer.add_scalar('Valid/loss', np.mean(global_valid_loss), epoch)
        writer.add_scalar('Valid/acc', best_acc, epoch)

        writer.flush()

    return best_acc


def main() -> None:
    dataloader_config = DataLoaderConfiguration(split_batches=True, dispatch_batches=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)

    config = ModelConfig.default_config()
    net = Transformer(**asdict(config))
    net = accelerator.prepare_model(net)

    train_set = TransData('data', DataType.TRAIN, accelerator.is_local_main_process)
    train_loader = DataLoader(train_set, batch_size=64, num_workers=32, persistent_workers=True, shuffle=False, collate_fn=collate_fn)
    train_loader = accelerator.prepare_data_loader(train_loader)

    valid_set = TransData('data', DataType.VALID, accelerator.is_local_main_process)
    valid_loader = DataLoader(valid_set, batch_size=64, num_workers=4, persistent_workers=True, shuffle=False, collate_fn=collate_fn)
    valid_loader = accelerator.prepare_data_loader(valid_loader)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(net.parameters(), LR)
    optimizer = accelerator.prepare_optimizer(optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=5,
        cooldown=2,
        min_lr=1e-7
    )
    scheduler = accelerator.prepare_scheduler(scheduler)

    writer = SummaryWriter(
        log_dir=f"runs/head{config.head}_layer{config.num_layers}_emb{config.emb_size}_hidden{config.hidden_size}_mlp{config.ffw_size}_{EPOCH}")

    # if accelerator.is_local_main_process:
    #     with torch.no_grad():
    #         test_data, test_target = next(iter(train_loader))
    #         test_data, test_target = test_data.to(accelerator.device), test_target.to(accelerator.device)
    #         writer.add_graph(net, input_to_model=[test_data, test_target], use_strict_trace=False)
    acc_metric = evaluate.load('accuracy')

    best_acc = 0.
    for epoch in range(EPOCH):
        train(net, accelerator, criterion, optimizer, scheduler, train_loader, acc_metric, writer, epoch)
        new_acc = valid(net, accelerator, criterion, valid_loader, acc_metric, best_acc, writer, epoch)

        accelerator.wait_for_everyone()
        if new_acc > best_acc:
            logger.info('save best model')
            unwrap_model = accelerator.unwrap_model(net)
            accelerator.save_model(unwrap_model, 'model', '1GB', True)
            best_acc = new_acc


if __name__ == '__main__':
    main()
