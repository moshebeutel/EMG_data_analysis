import torch
import torch.nn as nn
import numpy as np


class RawEmgConvnet(nn.Module):
    def __init__(self, number_of_class, window_size=30, shrink_to_one_raw=False, enhanced=False):
        super(RawEmgConvnet, self).__init__()
        self.enhanced = enhanced
        self._shrink_to_one_raw = shrink_to_one_raw
        if shrink_to_one_raw:
            self._shrink = lambda x: x.reshape(-1, window_size, 3, 8).median(axis=-2). \
                values.reshape(-1, 1, window_size, 8)
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(128, 3), padding=(0, 1), padding_mode='circular')
        self._pool1 = nn.MaxPool2d(kernel_size=(128, 4))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(.3)

        if enhanced:
            self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 2))
            self._pool2 = nn.MaxPool2d(kernel_size=(1, 2))
            self._batch_norm2 = nn.BatchNorm2d(64)
            self._prelu2 = nn.PReLU(64)
            self._dropout2 = nn.Dropout2d(.3)

        self.flatten = lambda x: x.view(-1, 896)
        self._fc1 = nn.Linear(896, 64)
        self._batch_norm3 = nn.BatchNorm1d(64)
        self._prelu3 = nn.PReLU(64)
        self._dropout3 = nn.Dropout(.3)
        self._output = nn.Linear(64, number_of_class)
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
        x = self._shrink(x) if self._shrink_to_one_raw else x
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(self._one_dim_pad(x)))))
        # print(conv1.shape)
        pool1 = self._pool1(conv1)
        # print(pool1.shape)
        if self.enhanced:
            conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
            pool2 = self._pool2(conv2)
        flatten_tensor = self.flatten(pool2)  # pool2.view(-1, 1024) if self.enhanced else pool1.view(-1, 1024)
        if flatten_tensor.size(0) != x.size(0):
            print()
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output


class RawEmgWindowConvnet(nn.Module):
    def __init__(self, number_of_classes, window_size, enhanced=False):
        super(RawEmgWindowConvnet, self).__init__()
        self.enhanced = enhanced
        # self._one_dim_pad = OneDimCircularPadding()
        self._conv1 = nn.Conv3d(window_size, 32, kernel_size=(3, 3, 5))
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(.5)

        if enhanced:
            self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 2))
            self._pool2 = nn.MaxPool2d(kernel_size=(1, 2))
            self._batch_norm2 = nn.BatchNorm2d(64)
            self._prelu2 = nn.PReLU(64)
            self._dropout2 = nn.Dropout2d(.5)

        self.flatten = lambda x: x.view(-1, 192)
        self._fc1 = nn.Linear(192, 128)
        self._batch_norm3 = nn.BatchNorm1d(128)
        self._prelu3 = nn.PReLU(128)
        self._dropout3 = nn.Dropout(.5)
        self._output = nn.Linear(128, number_of_classes)
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
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(self._one_dim_pad(x)))))
        # print(conv1.shape)
        pool1 = self._pool1(conv1)
        # print(pool1.shape)
        if self.enhanced:
            conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
            pool2 = self._pool2(conv2)
        flatten_tensor = self.flatten(pool1)  # pool2.view(-1, 1024) if self.enhanced else pool1.view(-1, 1024)
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        # fc1 = self._dropout3(self._prelu3(self._fc1(flatten_tensor)))
        output = self._output(fc1)
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
