import torch.nn as nn

__all__ = ['BCELoss', 'DICELoss', 'IoULoss']


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bceloss = nn.BCELoss()
        return bceloss(input, target)


class DICELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # you should add the code to compute DICELoss
        return 0


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # you should add the code to compute IoULoss
        return 0
