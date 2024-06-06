import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss

__all__ = ['BCELoss', 'DICELoss', 'IoULoss']


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        bceloss = nn.BCELoss()
        return bceloss(input, target)


class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input, target, ep=1e-8):
        # you should add the code to compute DICELoss
        intersection = 2 * torch.sum(input * target) + ep
        union = torch.sum(input) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, ep=1e-8):
        # you should add the code to compute IoULoss
        intersection = torch.logical_and(input, target).sum()
        union = torch.logical_or(input, target).sum()
        iou = (intersection + ep) / (union + ep)
        return 1 - iou
