"""
Explanation: https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/
"""

from torch import nn

from modules import ConvBNAct


class DepthwiseSeparableConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        bias=True,
    ):
        super().__init__()

        self.depthwise_conv = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            activation=None,
            bias=bias,
        )

        self.pointwise_conv = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
