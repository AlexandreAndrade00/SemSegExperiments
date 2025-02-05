"""
Paper: https://arxiv.org/abs/2204.02681v1
Source: https://github.com/PaddlePaddle/PaddleSeg

License:
copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn.functional as F
from torch import nn

from modules import ConvBNRelu


class SPPM(nn.Module):
    """
    Simple Pyramid Pooling Module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        bin_sizes=(1, 3),
        align_corners=False,
    ):
        super().__init__()

        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, inter_channels, size) for size in bin_sizes]
        )

        self.conv_out = ConvBNRelu(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2D(output_size=size)

        conv = ConvBNRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="valid",
        )

        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)

            x = F.interpolate(
                x, input_shape, mode="bilinear", align_corners=self.align_corners
            )

            if out is None:
                out = x
            else:
                out += x

        return self.conv_out(out)
