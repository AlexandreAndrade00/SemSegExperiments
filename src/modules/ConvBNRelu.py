from torch import nn


class ConvBNRelu(nn.Module):
    """
    Regular Convolution-BatchNormalization-ReLU layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=True,
    ):
        super(ConvBNRelu, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.net = nn.Sequential(self.conv, self.bn, self.relu)

    def forward(self, x):
        return self.net(x)
