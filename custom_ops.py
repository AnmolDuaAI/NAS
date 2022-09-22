import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1,
        bias = False, theta = 0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation,
        groups = groups, bias = bias)
        self.theta = theta


    def forward(self, x):
        out_normal = self.conv(x)
        if (math.fabs(self.theta - 0.0) < 1e-8):
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:,:,None,None]
            out_diff = F.conv2d(input = x, weight = kernel_diff, bias = self.conv.bias, stride = self.conv.stride, padding = 0, groups = self.conv.groups)
            return out_normal - self.theta * out_diff
    

class Conv3d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1,
        bias = False, theta = 0.7):
        super(Conv3d_cd, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if (math.fabs(self.theta - 0.0) < 1e-8):
            return out_normal
        else:
            [C_out, C_in, depth, kernel_size, kernel_size] = self.conv.weight.shape
            w_t0 = self.conv.weight[:,:,0,:,:]
            w_t2 = self.conv.weight[:,:,2,:,:]
            w_t0 = w_t0.sum(2).sum(2)
            w_t2 = w_t2.sum(2).sum(2)
            kernel_diff = torch.zeros((C_out, C_in, depth, 1, 1))
            kernel_diff[:,:,1,:,:] = (w_t0 + w_t2)[:,:,None,None]
            out_diff = F.conv3d(input = x, weight = kernel_diff, bias = self.conv.bias, stride = self.conv.stride, padding = (1,0,0), groups = self.conv.groups)
            return out_normal - self.theta * out_diff


def debug():

    in_2d = torch.rand(1,3,128,32,32)

    res = Conv3d_cd(
        in_channels=3,
        out_channels=16,
    )(in_2d)

    print (res.size())

if (__name__ == "__main__"):
    debug()