"""
Paper: https://arxiv.org/abs/2104.13188
Source: https://github.com/MichaelFan01/STDC-Seg

License:
MIT License

Copyright (c) 2021 Mingyuan Fan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
from torch import nn

from modules import ConvBNRelu


class STDCNet(nn.Module):
    """
    STDCNet Backbone implementation

    Args:
        base(int, optional): base channels. Default: 64.
        layers(list, optional): layers numbers list. It determines STDC block numbers of STDCNet's stage3\4\5. Default: [2, 2, 2] (STDC1).
        block_num(int,optional): block_num of features block. Default: 4.
        type(str,optional): feature fusion method "cat"/"add". Default: "cat".
        in_channels (int, optional): The channels of input image. Default: 3.
    """

    def __init__(
        self,
        base=64,
        layers=[2, 2, 2],
        block_num=4,
        type="cat",
        in_channels=3,
    ):
        super(STDCNet, self).__init__()

        if type == "cat":
            block = STDCCatBottleneck
        elif type == "add":
            block = STDCAddBottleneck

        self.layers = layers

        self.feat_channels: list[int] = [base // 2, base, base * 4, base * 8, base * 16]

        self.features = self._make_layers(in_channels, base, layers, block_num, block)

        self.init_weight()

    def forward(self, x):
        """
        forward function for feature extract.
        """
        out_feats = []

        x = self.features[0](x)
        out_feats.append(x)

        x = self.features[1](x)
        out_feats.append(x)

        idx = [
            [2, 2 + self.layers[0]],
            [2 + self.layers[0], 2 + sum(self.layers[0:2])],
            [2 + sum(self.layers[0:2]), 2 + sum(self.layers)],
        ]

        for start_idx, end_idx in idx:
            for i in range(start_idx, end_idx):
                x = self.features[i](x)

            out_feats.append(x)

        return out_feats

    def _make_layers(self, in_channels, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(in_channels, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 1)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            2,
                        )
                    )
                else:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 2)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            1,
                        )
                    )

        return nn.Sequential(*features)


class STDCAddBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=3, stride=1):
        super(STDCAddBottleneck, self).__init__()

        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = nn.ModuleList()

        self.stride = stride

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels // 2,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_channels // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(in_channels),
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_channels),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_channels, out_channels // 2, kernel_size=1)
                )
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // 2,
                        out_channels // 2,
                        stride=stride,
                    )
                )
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // 2,
                        out_channels // 4,
                        stride=stride,
                    )
                )
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // int(math.pow(2, idx)),
                        out_channels // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // int(math.pow(2, idx)),
                        out_channels // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)

            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class STDCCatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=3, stride=1):
        super(STDCCatBottleneck, self).__init__()

        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = nn.ModuleList()

        self.stride = stride

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels // 2,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_channels // 2),
            )

            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_channels, out_channels // 2, kernel_size=1)
                )
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // 2,
                        out_channels // 2,
                        stride=stride,
                    )
                )
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // 2,
                        out_channels // 4,
                        stride=stride,
                    )
                )
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // int(math.pow(2, idx)),
                        out_channels // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_channels // int(math.pow(2, idx)),
                        out_channels // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []

        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)

        out_list.insert(0, out1)

        out = torch.cat(out_list, axis=1)

        return out
