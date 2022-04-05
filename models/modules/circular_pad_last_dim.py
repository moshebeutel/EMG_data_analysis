import torch
import torch.nn.functional as F
import torch.nn as nn


class CircularPadLastDim(nn.Module):
    def __init__(self, pad_before=1, pad_after=1):
        super(CircularPadLastDim, self).__init__()
        self._pad_before = pad_before
        self._pad_after = pad_after
        self.pad = (self._pad_after, self._pad_before, 0, 0)

    def __str__(self):
        return f'CircularPadLastDim(\n  (pad): {self.pad}'

    def forward(self, x):
        assert x.dim() in [4, 5]
        if x.ndim == 4:
            padded = F.pad(x, pad=self.pad, mode='circular')
        if x.ndim == 5:
            padded = torch.concat([F.pad(x[i], pad=self.pad, mode='circular').unsqueeze(dim=0)
                                   for i in range(x.size(0))], dim=0)
        assert x.ndim == padded.ndim, f'pad changed number of dims from {x.ndim} to {padded.ndim}'
        assert padded.shape[-1] == x.shape[-1] + sum(self.pad), \
            f'pad did not pad correctly - shape before {x.shape} shape after {padded.shape} '
        return padded


if __name__ == '__main__':
    Cin = 3
    T = 4
    W = 3
    H = 8
    Cout = 2
    batch = 2
    input = torch.arange(batch * Cin * T * W * H).float().reshape(batch, Cin, T, W, H)
    pad = CircularPadLastDim()
    padded = pad(input)
    print(padded.shape)
