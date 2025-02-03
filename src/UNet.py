import torch
from torch import nn
from torch.nn.functional import pad


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        second_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.net = nn.Sequential(first_conv, nn.BatchNorm2d(out_channels), nn.ReLU(), second_conv,
                                 nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class EncoderConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.net = nn.Sequential(max_pool, DoubleConvolution(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class DecoderConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        up_sample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.up_sample_net = up_sample
        self.net = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_sample_net(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat((x2, x1), 1)

        return self.net(x)


class OutputConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        result = self.output_conv(x)

        diff_y = x.size()[2] - result.size()[2]
        diff_x = x.size()[3] - result.size()[3]

        result = pad(result, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2])

        return result


class UNet(nn.Module):
    def __init__(self, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = DoubleConvolution(3, 64, *args, **kwargs)
        self.encode1 = EncoderConvolution(64, 128, *args, **kwargs)
        self.encode2 = EncoderConvolution(128, 256, *args, **kwargs)
        self.encode3 = EncoderConvolution(256, 512, *args, **kwargs)
        self.encode4 = EncoderConvolution(512, 1024 // 2, *args, **kwargs)
        self.decode1 = DecoderConvolution(1024, 512 // 2, *args, **kwargs)
        self.decode2 = DecoderConvolution(512, 256 // 2, *args, **kwargs)
        self.decode3 = DecoderConvolution(256, 128 // 2, *args, **kwargs)
        self.decode4 = DecoderConvolution(128, 64, *args, **kwargs)
        self.output = OutputConvolution(64, out_channels, *args, **kwargs)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)
        x6 = self.decode1(x5, x4)
        x7 = self.decode2(x6, x3)
        x8 = self.decode3(x7, x2)
        x9 = self.decode4(x8, x1)

        return self.output(x9)
