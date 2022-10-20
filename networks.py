import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict


def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'S':
            L.append(nn.Sigmoid())
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)


class DWTForward(nn.Conv2d):
    def __init__(self, in_channels=64):
        super(DWTForward, self).__init__(in_channels, in_channels*4, 2, 2,
                                  groups=in_channels, bias=False)
        weight = torch.tensor([[[[0.5,  0.5], [ 0.5,  0.5]]],
                               [[[0.5,  0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [ 0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5,  0.5]]]],
                              dtype=torch.get_default_dtype()
                             ).repeat(in_channels, 1, 1, 1)# / 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


class DWTInverse(nn.ConvTranspose2d):
    def __init__(self, in_channels=64):
        super(DWTInverse, self).__init__(in_channels, in_channels//4, 2, 2,
                                  groups=in_channels//4, bias=False)
        weight = torch.tensor([[[[0.5,  0.5], [ 0.5,  0.5]]],
                               [[[0.5,  0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [ 0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5,  0.5]]]],
                              dtype=torch.get_default_dtype()
                             ).repeat(in_channels//4, 1, 1, 1)# * 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# -------------------------------------------------------
# Spatial Attention Layer (SA)
# -------------------------------------------------------
class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return x * out


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding=padding, bias=bias, mode=mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        self.ca = CALayer(out_channels, reduction)
        self.sa = SALayer()

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        res = self.sa(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                       bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))

        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x
