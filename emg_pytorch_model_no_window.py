import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FeatureEmgConvnet(nn.Module):
    def __init__(self, number_of_class, channels=4, W=3, H=8):
        super(FeatureEmgConvnet, self).__init__()

        self._reshape = lambda x: x.reshape(-1, channels, W, H)
        self._pad_before_last_dim_constant = lambda x: \
            F.pad(x, pad=(0, 0, 1, 1), mode='constant')
        self._pad_last_dim_circular = lambda x: \
            torch.concat([x[:, :, :, -1].reshape(-1, channels, W + 2, 1),
                          x, x[:, :, :, 0].reshape(-1, channels, W + 2, 1)], axis=3)
        self._conv1 = nn.Conv2d(4, 32, kernel_size=(2, 2))
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(.5)

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(2, 2))
        self._pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(.5)

        self.flatten = lambda x: x.view(-1, 64)
        self._fc1 = nn.Linear(64, 64)
        self._batch_norm3 = nn.BatchNorm1d(64)
        self._prelu3 = nn.PReLU(64)
        self._dropout3 = nn.Dropout(.5)
        self._output = nn.Linear(64, number_of_class)
        self._output_softmax = nn.Softmax(dim=1)
        self.initialize_weights()

        print(self)

        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # print(x)
        x = self._reshape(x)
        # print(x)
        x = self._pad_before_last_dim_constant(x)
        # print(x)
        x = self._pad_last_dim_circular(x)
        # print(x)
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # print(conv1.shape)
        pool1 = self._pool1(conv1)
        # print(pool1.shape)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        # print(conv2.shape)
        pool2 = self._pool2(conv2)
        # print(pool2.shape)
        flatten_tensor = self.flatten(pool2)  # pool2.view(-1, 1024) if self.enhanced else pool1.view(-1, 1024)
        # print(flatten_tensor.shape)
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        # print(fc1.shape)
        # fc1 = self._dropout3(self._prelu3(self._fc1(flatten_tensor)))
        output = self._output(fc1)
        # print(output.shape)
        output = self._output_softmax(output)
        # print(output.shape)
        return output


class OneDimCircularPadding(nn.Module):
    def __init__(self, pad=((0, 0), (1, 1))):
        super().__init__()
        self._pad = pad

    def forward(self, x):
        x_np = x.numpy()
        x_np = x_np.copy().squeeze()
        x_np = np.pad(x_np, self._pad, mode='wrap')
        x_np = np.pad(x_np, self._pad[::-1], mode='constant', constant_values=0)
        x_np = x_np.reshape((1, 1) + x_np.shape)
        return torch.from_numpy(x_np)


if __name__ == '__main__':
    x_input = torch.arange(2 * 24 * 4).float().reshape(2, 4, 24)
    net = FeatureEmgConvnet(10)
    output = net(x_input)
    print(torch.max(output, dim=1).indices)
