import torch.nn as nn
from models.modules import circular_pad_last_dim, unpad_module, identity_module


class Conv2dCustomPad(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple = 3, stride: int or tuple = 1,
                 pad_before=1, pad_after=1, remove_oversized=False, unpad_before=1, unpad_after=1):
        super(Conv2dCustomPad, self).__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride)
        self._pad = circular_pad_last_dim.CircularPadLastDim(pad_before=pad_before, pad_after=pad_after)

        self._post_proc = unpad_module.UnpadLastDimModule(unpad_before=unpad_before, unpad_after=unpad_after) \
            if remove_oversized else identity_module.IdentityModule()

    def forward(self, x):
        assert x.ndim == 4, f'Expected input tensor to be 4D - got {x.ndim}'
        padded_x = self._pad(x)
        assert padded_x.size(3) == x.size(3) + sum(self._pad.pad),\
            f'pad did not pad correctly - shape before {x.shape} shape after {padded_x.shape} '
        conv = self._conv(padded_x)
        post_proc = self._post_proc(conv)
        return post_proc


if __name__ == '__main__':
    import torch
    Cin = 3
    W = 3
    H = 8
    Cout = 2
    batch = 2
    input = torch.arange(batch*Cin*W*H).float().reshape(batch,Cin, W, H)
    custom_conv = Conv2dCustomPad(Cin, Cout, kernel_size=3, pad_before=2, pad_after=2, remove_oversized=True)
    output = custom_conv(input)
    print(input.shape, output.shape)

