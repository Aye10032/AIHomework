import argparse
import os
from collections import OrderedDict
from glob import glob

import albumentations as albu
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import losses
from dataset import Dataset
from model import UNet
from utils import AverageMeter, str2bool
from utils import dice_coef

LOSS_NAMES = losses.__all__
LOSS_NAMES.append(['BCELoss', 'DICELoss', 'IoULoss'])


# 解析命令行参数并返回配置信息。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='UNet', type=str)

    # 迭代多少轮
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # batch_size
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # loss
    parser.add_argument('--loss', default='BCELoss',
                        choices=LOSS_NAMES,
                        help='loss define')

    # dataset
    parser.add_argument('--dataset', default='MITO',
                        help='dataset name, ER or MITO')
    parser.add_argument('--img_ext', default='.tif',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.tif',
                        help='mask file extension')
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD', 'AdamW', 'RMSProp'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD', 'AdamW', 'RMSProp']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--num_workers', default=0, type=int)
    config = parser.parse_args()

    return config


# 定义训练过程。使用训练数据集进行训练，计算损失函数和评估指标，并更新模型参数。
def train(train_loader, model, criterion, optimizer):
    # 初始化平均指标（损失和dice）的AverageMeter对象
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}
    # 将模型设置为训练模式
    model.train()
    # 进度条可视化
    pbar = tqdm(total=len(train_loader))
    # 遍历训练数据加载器，获取输入和目标数据。
    for input, target, _ in train_loader:
        # 将输入和目标数据移至GPU
        input = input.cuda()
        target = target.cuda()

        # compute output
        # 计算模型的输出和损失
        output = model(input)

        loss = criterion(output, target)
        # To do: you should change more metric to evaluate the results, including DICE, dice, Hausdorff Distance
        dice = dice_coef(output > 0.5, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        # 更新平均指标（损失和dice）的值
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        # 根据当前训练进度，更新并显示进度条的后缀信息，包括平均损失和平均dice
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            dice = dice_coef(output > 0.5, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    # 解析命令行参数
    config = vars(parse_args())
    exp_dir = os.path.join(config['name'], f"{config['dataset']}-{config['loss']}-{config['optimizer']}-lr{config['lr']}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(f'{exp_dir}/config.yml', 'w') as f:
        yaml.dump(config, f)
    # define loss function (criterion)
    criterion = losses.__dict__[config['loss']]().cuda()

    # create model，建模
    model = UNet(n_channels=1, n_classes=1)
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSProp':
        optimizer = optim.RMSprop(params, lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    else:
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    # 根据配置参数config中的学习率调度器类型和参数设置创建学习率调度器对象scheduler：
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    if config['dataset'] == 'ER':
        train_num, val_num, test_num = 157, 28, 38
    elif config['dataset'] == 'MITO':
        train_num, val_num, test_num = 165, 8, 10
    else:
        train_num, val_num, test_num = 0, 0, 0
    # Data loading code
    train_img_ids = glob(os.path.join('data', config['dataset'] + '_dataset', 'train', 'images', '*' + config['img_ext']))
    val_img_ids = glob(os.path.join('data', config['dataset'] + '_dataset', 'val', 'images', '*' + config['img_ext']))
    # 提取文件名
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    # 数据增强：
    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),  # 按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),

    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
    ])
    # 构建数据集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('data', config['dataset'] + '_dataset', 'train', 'images'),
        mask_dir=os.path.join('data', config['dataset'] + '_dataset', 'train', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data', config['dataset'] + '_dataset', 'val', 'images'),
        mask_dir=os.path.join('data', config['dataset'] + '_dataset', 'val', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('dice', []),
        ('val_loss', []),
        ('val_dice', []),
    ])
    writer = SummaryWriter(log_dir=f"runs/{config['dataset']}_{config['optimizer']}_lr{config['lr']}_{config['loss']}")

    best_dice = 0
    trigger = 0  # 计数器

    with torch.no_grad():
        test_data = next(iter(train_loader))
        writer.add_graph(model, test_data[0].cuda())

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        scheduler.step(train_log['loss'])
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('train/loss', train_log['loss'], epoch)
        writer.add_scalar('train/dice', train_log['dice'], epoch)
        writer.add_scalar('valid/loss', val_log['loss'], epoch)
        writer.add_scalar('valid/dice', val_log['dice'], epoch)
        writer.flush()

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{exp_dir}/log.csv', index=False)
        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'{exp_dir}/model.pth')
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
