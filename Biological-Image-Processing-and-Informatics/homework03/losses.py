import torch
import torch.nn as nn

__all__ = ['BCELoss', 'DICELoss', 'IoULoss']

from torch import Tensor


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        loss = self.bce_loss(input, target)
        return loss


class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, smooth=1e-8):
        intersection = 2 * torch.sum(input * target, dim=(1, 2, 3)) + smooth
        union = torch.sum(input, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + smooth
        loss = (1 - intersection / union).mean()
        return loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, input, target, smooth=1e-8):
        intersection = torch.sum(input * target, dim=(1, 2, 3)) + smooth
        union = torch.sum(input, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection + smooth

        loss = (1 - intersection / union).mean()
        return loss
