import argparse
import torch
from torch import Tensor
from torch.types import Number


def dice_coef(output, target):
    smooth = 1e-5

    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


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
