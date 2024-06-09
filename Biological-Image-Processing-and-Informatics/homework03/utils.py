import argparse

import torch
from torch import Tensor
from torch.types import Number


def dice_coef(output, target):
    smooth = 1e-5

    intersection = torch.logical_and(output, target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return dice.item()


def iou_coef(output: Tensor, target: Tensor) -> Number:
    smooth = 1e-5

    intersection = torch.logical_and(output, target).sum()
    union = torch.logical_or(output, target).sum()
    result = (intersection + smooth) / (union + smooth)

    return result.item()


def acc_coef(output: Tensor, target: Tensor) -> Number:
    total = output.view(-1).size(0)
    accuracy = torch.eq(output, target).sum() / total

    return accuracy.item()


def specificity_coef(output: Tensor, target: Tensor) -> Number:
    """

    :param output:
    :param target:
    :return:
    """
    smooth = 1e-5

    tn = torch.logical_and(torch.logical_not(output), torch.logical_not(target)).sum()
    fn = torch.logical_and(output, torch.logical_not(target)).sum()
    tnr = (tn + smooth) / (tn + fn + smooth)

    return tnr.item()


def sensitivity_coef(output: Tensor, target: Tensor) -> Number:
    """

    :param output:
    :param target:
    :return:
    """
    smooth = 1e-5

    tp = torch.logical_and(output, target).sum()
    fp = torch.logical_and(output, torch.logical_not(target)).sum()
    tpr = (tp + smooth) / (tp + fp + smooth)

    return tpr.item()


def hausdorff_distance_coef(output: Tensor, target: Tensor) -> Number:
    """
    计算两张图片的Hausdorff Distance

    :param output: 网络预测值，仅有0或1两个值
    :param target: 实际掩码，仅有0或1两个值
    :return: Hausdorff Distance
    """

    output = output.squeeze(1).float()
    target = target.squeeze(1).float()

    distance_matrix = torch.cdist(output, target, p=2)

    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)

    return value.max(1)[0].mean().item()


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
