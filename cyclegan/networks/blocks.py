import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())#注册tensor到parameter中，可以参与模型训练，被更改值
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)#2021-4-20 shape[1:],从第二维到最后一维

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2) #2021-4-19 list的扩展机制。[1]*2 = [1,1],[1,-1]+[1,1]=[1,-1,1,1]
            #重构x,全部扩充到第二个维度： 
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
     zero = nn.ZeroPad2d,
  reflect = nn.ReflectionPad2d,
replicate = nn.ReplicationPad2d)

conv_dict = dict(
   conv2d = nn.Conv2d,
 deconv2d = nn.ConvTranspose2d)

norm_dict = dict(
     none = lambda x: lambda x: x,
 spectral = lambda x: lambda x: x,
    batch = nn.BatchNorm2d,
 instance = nn.InstanceNorm2d,
    layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
      relu = lambda: nn.ReLU(inplace=True),
     lrelu = lambda: nn.LeakyReLU(0.2, inplace=True),
     prelu = lambda: nn.PReLU(),
      selu = lambda: nn.SELU(inplace=True),
      tanh = lambda: nn.Tanh())


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral": self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]() #2021-04-18 这个最后的()很重要。呼应了之前字典中的lambda

    def forward(self, x):
         return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ=activ, pad=pad)]
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ='none', pad=pad)]
        self.model = nn.Sequential(*block)

    def forward(self, x): return self.model(x) + x #2021-4-20 跳层连接


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm='none', activ='relu'):
        super(FullyConnectedBlock, self).__init__()

        self.fc = nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral": self.fc = spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x): return self.activ(self.norm(self.fc(x)))


