import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', in_norm_learnable=False, groupnormnum=8, convGroup=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            if in_norm_learnable == True:
                self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(norm_dim, affine=False)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=convGroup)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



class Pose_encoder(nn.Module):
    def __init__(self, n_downsample=3, input_dim=18, dim=32, norm='in', activ='relu', pad_type='reflect'):
        super(Pose_encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)]
        # downsampling blocks
        self.model += [ChannelAttention(32)]
        self.model += [Conv2dBlock(32,16, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)]
        self.model += [ChannelAttention(16)]
        self.model += [Conv2dBlock(16, 8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)]
        self.model += [ChannelAttention(8)]
        self.model += [Conv2dBlock(8, 4, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x







# c=Pose_encoder().cuda()
# x=torch.zeros((10,19,256,172)).cuda()
# x = c(x)
# print(x.shape)


