from typing import Optional, Union

from torch import nn


class ConvBNAct(nn.Module):
    """
    Regular Convolution-BatchNormalization-Activation layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[Union[str, int]] = None,
        groups: int = 1,
        bias: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        super(ConvBNAct, self).__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=kernel_size // 2 if padding is None else padding,
            bias=bias,
        )

        bn = nn.BatchNorm2d(out_channels)

        if activation is None:
            self.net = nn.Sequential(conv, bn)
        else:
            self.net = nn.Sequential(conv, bn, activation)

    def forward(self, x):
        return self.net(x)
