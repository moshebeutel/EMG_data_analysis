import torch
import torch.nn.functional as F
import torch.nn as nn


class CircularPadLastDim(nn.Module):
    def __init__(self, pad_before=1, pad_after=1):
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.pad = (self.pad_after, self.pad_before, 0, 0)

    def forward(self, x):
        assert x.dim() in [4,5]
        if x.dim() == 4:
            return F.pad(x, pad=self.pad, mode='circular')
        if x.dim() == 5:
            return torch.concat([F.pad(x[i], pad=self.pad, mode='circular') for i in range(x.size(0))], dim=0)

