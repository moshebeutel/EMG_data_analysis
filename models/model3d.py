import torch
import torch.nn as nn
import numpy as np


class RawEmg3DConvnet(nn.Module):
    def __init__(self, number_of_classes, window_size=1280, depthwise_multiplier=32):
        super(RawEmg3DConvnet, self).__init__()
        self._reshape = lambda x: x.reshape(-1, 1, window_size, 3, 8)
        self._conv1 = nn.Conv3d(1, depthwise_multiplier, groups=1, kernel_size=(128, 3, 3))
        self._pool1 = nn.MaxPool3d(kernel_size=(int(window_size/10), 1, 3))
        self._batch_norm1 = nn.BatchNorm3d(depthwise_multiplier)
        self._prelu1 = nn.PReLU(depthwise_multiplier)
        self._dropout1 = nn.Dropout2d(.3)

        self._conv2 = nn.Conv3d(depthwise_multiplier, 2 * depthwise_multiplier, groups=1, kernel_size=(3, 1, 2))
        self._pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1))
        self._batch_norm2 = nn.BatchNorm3d(2*depthwise_multiplier)
        self._prelu2 = nn.PReLU(2*depthwise_multiplier)
        self._dropout2 = nn.Dropout2d(.3)

        self.flatten = lambda x: x.view(-1, 128)
        self._fc1 = nn.Linear(128, 64)
        self._batch_norm3 = nn.BatchNorm1d(64)
        self._prelu3 = nn.PReLU(64)
        self._dropout3 = nn.Dropout(.3)
        self._output = nn.Linear(64, number_of_classes)
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
        x = self._reshape(x)
        # print('reshape', x.shape)
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # print('conv1', conv1.shape)
        pool1 = self._pool1(conv1)
        # print('pool1', pool1.shape)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        # print('conv2', conv2.shape)
        pool2 = self._pool2(conv2)
        # print('pool2', pool2.shape)
        flatten_tensor = self.flatten(pool2)
        # print('flatten_tensor', flatten_tensor.shape)
        # if flatten_tensor.size(0) != x.size(0):
        #     print()
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        # print('fc1', fc1.shape)
        output = self._output(fc1)
        return output
