import torch.nn as nn


class ConvBlock1D(nn.Module):
    """1D Conv Layer with BatchNorm, MaxPool and ReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pool_length=4,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_length),
        )

    def forward(self, x):
        return self.conv_block(x)


class ToyModel(nn.Module):
    """
    Implementation of a toy model
        corresponding to the M3 Network in https://arxiv.org/pdf/1610.00087v1.pdf
    """

    def __init__(self):
        super(ToyModel, self).__init__()
        self.toy_model = nn.Sequential(
            ConvBlock1D(
                in_channels=1,
                out_channels=256,
                kernel_size=80,
                stride=4,
                padding=40,
                pool_length=4,
            ),
            ConvBlock1D(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_length=4,
            ),
            nn.AvgPool1d(500),
            nn.Flatten(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.toy_model(x)
