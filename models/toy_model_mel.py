import torch.nn as nn


class ConvBlock2D(nn.Module):
    """2D Conv Layer with BatchNorm, MaxPool and ReLU"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, pool_length=4,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_length),
        )

    def forward(self, x):
        return self.conv_block(x)


class ToyModelMel(nn.Module):
    """
    Implementation of a toy model
        corresponding to the K2C2 Network in https://arxiv.org/pdf/1609.04243v3.pdf with 2 conv layers only
    """

    def __init__(self):
        super(ToyModelMel, self).__init__()
        self.toy_model_mel = nn.Sequential(
            ConvBlock2D(
                in_channels=3,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_length=(2, 4),
            ),
            ConvBlock2D(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_length=(4, 4),
            ),
            nn.AvgPool2d((16, 10)),
            nn.Flatten(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.toy_model_mel(x)
