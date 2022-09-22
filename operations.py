from turtle import forward

from numpy import pad
import torch
import torch.nn as nn


class Zero(nn.Module):
    def __init__(self, stride) -> None:
        super(Zero, self).__init__()
        self.stride = stride
    
    def forward(self, x):
        if (self.stride == 1):
            return x.mul(0.)
        return x[:,:,::self.stride, ::self.stride].mul(0.)

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine = True) -> None:
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(C_out, affine = affine),
        )
    
    def forward(self, x):
        return self.op(x)

class DilConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine = True) -> None:
        super(DilConvBlock, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_in, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = C_in, bias = False),
            nn.Conv2d(C_in, C_out, kernel_size = 1, padding = 0, bias = False),
            nn.BatchNorm2d(C_out, affine = affine)
        )

    def forward(self, x):
        return self.op(x)

class SepConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine = True) -> None:
        super(SepConvBlock, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_in, kernel_size = kernel_size, stride = stride, padding = padding, groups = C_in, bias = False),
            nn.Conv2d(C_in, C_in, kernel_size = 1, padding = 0, bias = False),
            nn.BatchNorm2d(C_in, affine = affine),
            nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_in, kernel_size = kernel_size, stride = 1, padding = padding, groups = C_in, bias = False),
            nn.Conv2d(C_in, C_out, kernel_size = 1, padding = 0, bias = False),
            nn.BatchNorm2d(C_out, affine = affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine = True) -> None:
        super(FactorizedReduce, self).__init__()

        self.conv1 = nn.Conv2d(C_in, C_out//2, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.conv2 = nn.Conv2d(C_in, C_out//2, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.bn = nn.BatchNorm2d(C_out, affine = affine)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x[:,:,1:,1:])
        out = torch.concat([x1,x2], dim = 1)
        out = self.bn(out)
        out = self.relu(out)
        return out


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C,stride,affine: nn.AvgPool2d(kernel_size = 3, stride = stride, padding = 1),
    'max_pool_3x3': lambda C,stride,affine: nn.MaxPool2d(kernel_size = 3, stride = stride, padding = 1),
    'skip_connect': lambda C,stride,affine: Identity() if stride == 1 else FactorizedReduce(C_in=C, C_out=C, affine = affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConvBlock(C, C, 3, stride, 1, affine = affine),
    'sep_conv_5x5': lambda C,stride,affine: SepConvBlock(C,C,5,stride,2,affine = affine),
    'sep_conv_7x7': lambda C,stride,affine: SepConvBlock(C,C,7,stride,3,affine = affine),
    'dil_conv_3x3': lambda C,stride,affine: DilConvBlock(C,C,3,stride,2,2,affine = affine), # padding - 2, dilation - 2
    'dil_conv_5x5': lambda C,stride,affine: DilConvBlock(C,C,5,stride,4,2,affine = affine), # padding - 4, dilation - 2
}

