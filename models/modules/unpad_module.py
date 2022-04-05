import torch
import torch.nn as nn


class UnpadLastDimModule(nn.Module):
    def __init__(self, unpad_before: int = 1, unpad_after: int = 1):
        super(UnpadLastDimModule, self).__init__()
        self._unpad_before = unpad_before
        self._unpad_after = unpad_after

    def forward(self, x):
        assert x.shape[-1] > self._unpad_before + self._unpad_after, f'Expected input with' \
                                                                     f' at least' \
                                                                     f' {self._unpad_after + self._unpad_before + 1}' \
                                                                     f' elements at last dim - got {x.shape[-1]}'

        return transpose(transpose(x)[self._unpad_before:-self._unpad_after])


def transpose(x: torch.Tensor):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))
