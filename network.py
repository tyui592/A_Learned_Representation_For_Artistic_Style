"""Network Code."""

import torch
import torch.nn as nn


class CIN(nn.Module):
    """Conditional Instance Norm."""

    def __init__(self, num_style, ch):
        """Init with number of style and channel."""
        super(CIN, self).__init__()
        self.normalize = nn.InstanceNorm2d(ch, affine=False)
        self.offset = nn.Parameter(0.01 * torch.randn(1, num_style, ch))
        self.scale = nn.Parameter(1 + 0.01 * torch.randn(1, num_style, ch))

    def forward(self, x, style_codes):
        """Forward func."""
        b, c, h, w = x.size()

        x = self.normalize(x)

        gamma = torch.sum(self.scale * style_codes, dim=1).view(b, c, 1, 1)
        beta = torch.sum(self.offset * style_codes, dim=1).view(b, c, 1, 1)

        x = x * gamma + beta

        return x.view(b, c, h, w)


class ConvWithCIN(nn.Module):
    """Convolution layer with CIN."""

    def __init__(self, num_style, in_ch, out_ch, stride, activation, ksize):
        """Init."""
        super(ConvWithCIN, self).__init__()
        self.padding = nn.ReflectionPad2d(ksize // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride)

        self.cin = CIN(num_style, out_ch)

        # activatoin
        if activation == "relu":
            self.activation = nn.ReLU()

        elif activation == "linear":
            self.activation = lambda x: x

    def forward(self, x, style_codes):
        """Forward func."""
        x = self.padding(x)
        x = self.conv(x)
        x = self.cin(x, style_codes)
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    """ResidualBlock."""

    def __init__(self, num_style, in_ch, out_ch):
        """Init."""
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvWithCIN(num_style, in_ch, out_ch, 1, "relu", 3)
        self.conv2 = ConvWithCIN(num_style, out_ch, out_ch, 1, "linear", 3)

    def forward(self, x, style_codes):
        """Forward func."""
        out = self.conv1(x, style_codes)
        out = self.conv2(out, style_codes)

        return x + out


class UpsamleBlock(nn.Module):
    """Upsampling Bloack."""

    def __init__(self, num_style, in_ch, out_ch):
        """Init."""
        super(UpsamleBlock, self).__init__()
        self.conv = ConvWithCIN(num_style, in_ch, out_ch, 1, "relu", 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, style_codes):
        """Forward func."""
        x = self.upsample(x)
        x = self.conv(x, style_codes)

        return x


class StyleTransferNetwork(nn.Module):
    """Style Transfer Network."""

    def __init__(self, num_style=16):
        """Init."""
        super(StyleTransferNetwork, self).__init__()
        self.conv1 = ConvWithCIN(num_style,  3, 32, 1, 'relu', 9)
        self.conv2 = ConvWithCIN(num_style, 32, 64, 2, 'relu', 3)
        self.conv3 = ConvWithCIN(num_style, 64, 128, 2, 'relu', 3)

        self.residual1 = ResidualBlock(num_style, 128, 128)
        self.residual2 = ResidualBlock(num_style, 128, 128)
        self.residual3 = ResidualBlock(num_style, 128, 128)
        self.residual4 = ResidualBlock(num_style, 128, 128)
        self.residual5 = ResidualBlock(num_style, 128, 128)

        self.upsampling1 = UpsamleBlock(num_style, 128, 64)
        self.upsampling2 = UpsamleBlock(num_style, 64, 32)

        self.conv4 = ConvWithCIN(num_style, 32, 3, 1, 'linear', 9)

    def forward(self, x, style_codes):
        """Forward func."""
        x = self.conv1(x, style_codes)
        x = self.conv2(x, style_codes)
        x = self.conv3(x, style_codes)

        x = self.residual1(x, style_codes)
        x = self.residual2(x, style_codes)
        x = self.residual3(x, style_codes)
        x = self.residual4(x, style_codes)
        x = self.residual5(x, style_codes)

        x = self.upsampling1(x, style_codes)
        x = self.upsampling2(x, style_codes)

        x = self.conv4(x, style_codes)

        return x
