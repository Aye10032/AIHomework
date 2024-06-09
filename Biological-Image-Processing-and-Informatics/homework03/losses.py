import torch
import torch.nn as nn

__all__ = ['BCELoss', 'DICELoss', 'IoULoss']


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

    def forward(self, input, target, ep=1e-8):
        # you should add the code to compute DICELoss
        intersection = 2 * torch.sum(input * target, dim=(1, 2, 3)) + ep
        union = torch.sum(input, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + ep
        loss = (1 - intersection / union).mean()
        return loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, input, target, ep=1e-8):
        # you should add the code to compute IoULoss
        intersection = torch.sum(input * target, dim=(1, 2, 3)) + ep
        union = torch.sum(input, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection + ep

        loss = (1 - intersection / union).mean()
        return loss
