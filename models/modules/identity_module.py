import torch.nn as nn


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()
        pass

    def forward(self, x):
        return x
