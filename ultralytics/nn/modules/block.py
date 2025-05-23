# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ultralytics.utils.torch_utils import fuse_conv_and_bn
import math
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad,Pinwheel_shapedConv,GSConv,Pinwheel_shapedConv
from .transformer import TransformerBlock
from ..Addmodules import *
import numpy as np
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "SimSPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))





#分组SPPCSPC 分组后参数量和计算量与原本差距不大，不知道效果怎么样
# class SPPF(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
#         super(SPPF, self).__init__()
#         c_ = int(2 * c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1, g=4)
#         self.cv2 = Conv(c1, c_, 1, 1, g=4)
#         self.cv3 = Conv(c_, c_, 3, 1, g=4)
#         self.cv4 = Conv(c_, c_, 1, 1, g=4)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#         self.cv5 = Conv(4 * c_, c_, 1, 1, g=4)
#         self.cv6 = Conv(c_, c_, 3, 1, g=4)
#         self.cv7 = Conv(2 * c_, c2, 1, 1, g=4)

#     def forward(self, x):
#         x1 = self.cv4(self.cv3(self.cv1(x)))
#         y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
#         y2 = self.cv2(x)
#         return self.cv7(torch.cat((y1, y2), dim=1))


# class LSKA(nn.Module):
#     # Large-Separable-Kernel-Attention
#     def __init__(self, dim, k_size=7):
#         super().__init__()
 
#         self.k_size = k_size
 
#         if k_size == 7:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
#         elif k_size == 11:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
#         elif k_size == 23:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
#         elif k_size == 35:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
#         elif k_size == 41:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
#         elif k_size == 53:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)
 
#         self.conv1 = nn.Conv2d(dim, dim, 1)
 
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0h(x)
#         attn = self.conv0v(attn)
#         attn = self.conv_spatial_h(attn)
#         attn = self.conv_spatial_v(attn)
#         attn = self.conv1(attn)
#         return u * attn   
# class SPPF_LSKA(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.lska = LSKA(c_ * 4, k_size=11)

#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

# class SPPF(nn.Module):
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2
#         Conv = BaseConv
#         self.cv1 = GSConv(c1, c_, 1, 1)
#         self.cv2 = GSConv(c_ * 6, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.am = nn.AdaptiveMaxPool2d(1)
#         self.aa = nn.AdaptiveAvgPool2d(1)
#     def forward(self, x):
#         x = self.cv1(x)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             y1 = self.m(x)
#             y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2),self.am(x).expand_as(x),self.am(x).expand_as(x)), 1) )
    
# class SPPF(nn.Module):原版2
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2((torch.cat((x, y1, y2, self.m(y2)),1)))

class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvModulOperationSpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size=3, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)        
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x
    
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
#  CASAtt   
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)
    
class CASAtt(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out
# 
# GCSA
class GCSA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
#     
# ELGCA
class ELGCA(nn.Module):
    """
    Efficient local global context aggregation module
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dwconv = nn.Conv2d(dim//2, dim//2, 3, padding=1, groups=dim//2)
        self.qkvl = nn.Conv2d(dim//2, (dim//4)*self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.act = nn.GELU()
    def forward(self, x):
        B, C, H, W = x.shape
        
        x1, x2 = torch.split(x, [C//2, C//2], dim=1)
        # apply depth-wise convolution on half channels
        x1 = self.act(self.dwconv(x1))
        # linear projection of other half before computing attention
        x2 = self.act(self.qkvl(x2))
        x2 = x2.reshape(B, self.heads, C//4, H, W)
        
        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:,-3, :, :, :]
        q = self.pool_q(q)
        k = self.pool_k(k)
        
        v = x2[:,-2,:,:,:].flatten(2)
        lfeat = x2[:,-1,:,:,:]
        
        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1,2))
        qk = torch.softmax(qk, dim=1).transpose(1,2)
        x2 = torch.matmul(qk, v).reshape(B, C//4, H, W)
        
        x = torch.cat([x1, lfeat, x2], dim=1)
        return x
# 

# SHSA
class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)
    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.pre_norm = GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init = 0))
        
    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))
        return x
# 
# MCA
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()
    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std
class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))
    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        out = out.expand_as(x)
        return x * out
class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()
        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)
        return x_out
#     
# TiedSELayer
class TiedSELayer(nn.Module):
    '''Tied Block Squeeze and Excitation Layer'''
    def __init__(self, channel, B=1, reduction=16):
        super(TiedSELayer, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channel = channel//B
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y
# 
# PSA
class PSA_1(nn.Module):
    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S
        self.convs=[]
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))
        self.se_blocks=[]
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, h, w = x.size()
        #Step1:SPC module
        SPC_out=x.view(b,self.S,c//self.S,h,w) #bs,s,ci,h,w
        for idx,conv in enumerate(self.convs):
            SPC_out[:,idx,:,:,:]=conv(SPC_out[:,idx,:,:,:])
        #Step2:SE weight
        se_out=[]
        for idx,se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:,idx,:,:,:]))
        SE_out=torch.stack(se_out,dim=1)
        SE_out=SE_out.expand_as(SPC_out)
        #Step3:Softmax
        softmax_out=self.softmax(SE_out)
        #Step4:SPA
        PSA_out=SPC_out*softmax_out
        PSA_out=PSA_out.view(b,-1,h,w)
        return PSA_out
# 
# 
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)
def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        RepConv(inp, oup, kernel_size=3, stride=stride, padding=None, groups=1, map_k=3),
        #conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )
def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
# 
# 
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 
    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g #(w + g) #self.gate(x, w) 
# 
# 
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)
        return x 
# 

# 
class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5
        attn1 = self.conv1(attn1) # b, dim/2, h, w
        attn2 = self.conv2(attn2) # b, dim/2, h, w
        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w
        agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w
        ch_attn1 = self.global_pool(attn) # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)
        a1,a2 =  a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size,dim // 2,1,1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)
        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)
        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()
        return x * attn
# 
# 
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x
        
# 注：作者未开源空间注意力代码，以下代码由《微信公众号：AI缝合术》提供.

        
class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.Spatial_Att = ConvModulOperationSpatialAttention(channels)
    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        x_out2 = self.Spatial_Att(x_out1)
        return x_out2
#
# 
class Attention_oc(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention_oc, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.GroupNorm(1, attention_channel)  # 或 InstanceNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention
        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention
        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def update_temperature(self, temperature):
        self.temperature = temperature
    @staticmethod
    def skip(_):
        return 1.0
    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention
    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention
    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention
    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention_oc(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)
    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output
    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output
    def forward(self, x):
        return self._forward_impl(x)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = ODConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ODConv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
class OCBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(OCBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
# 
# 
class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        # 初始化函数，plane是输入和输出特征图的通道数，norm_layer是归一化层（默认为BatchNorm2d）
        super(FullyAttentionalBlock, self).__init__()
        # 定义两个全连接层，conv1和conv2
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        
        # 定义卷积层 + 归一化层 + 激活函数（ReLU）
        self.conv = nn.Sequential(
            nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),  # 卷积操作
            norm_layer(plane),  # 归一化层
            nn.ReLU()  # ReLU激活函数
        )
        
        # 定义softmax操作，用于计算关系矩阵
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化可学习的参数gamma，用于调整最终的输出
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 前向传播过程，x为输入的特征图，形状为 (batch_size, channels, height, width)
        batch_size, _, height, width = x.size()
        
        # 对输入张量进行排列和变形，获取水平和垂直方向的特征
        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)  # 水平方向特征
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)  # 垂直方向特征
        
        # 对输入张量分别在水平方向和垂直方向进行池化，并通过全连接层进行编码
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())  # 水平方向编码
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())  # 垂直方向编码
        
        # 计算水平方向和垂直方向的关系矩阵
        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))  # 计算水平方向的关系
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))  # 计算垂直方向的关系
        
        # 计算经过softmax后的关系矩阵
        full_relation_h = self.softmax(energy_h)  # 水平方向的关系
        full_relation_w = self.softmax(energy_w)  # 垂直方向的关系
        
        # 通过矩阵乘法和关系矩阵，对特征进行加权和增强
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)  # 水平方向的增强
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)  # 垂直方向的增强
        
        # 将水平和垂直方向的增强特征进行融合，并加上原始输入特征
        out = self.gamma * (full_aug_h + full_aug_w) + x
        
        # 通过卷积层进行进一步的特征处理
        out = self.conv(out)
        
        return out  # 返回处理后的特征图
# 
# 
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
        
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc // 2, inc // 2, k=(5,1), g=inc // 2)
#         self.conv2_fan = Conv(inc // 2, inc // 2, k=(1,5), g=inc // 2)
#         self.conv3 = Conv(inc // 4, inc // 4, k=(7,1), g=inc // 4)
#         self.conv3_fan = Conv(inc // 4, inc // 4, k=(1,7), g=inc // 4)
#         self.conv4 = Conv(inc, inc, 1)
    
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out = self.conv2_fan(conv2_out)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)
#         conv3_out = self.conv3_fan(conv3_out)
        
        
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         out = self.conv4(out) + x   #残差链接去了不太行
#         return out
class CoreModule(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        super(CoreModule, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        self.edgeconvs = nn.ModuleList([])

        self.recon_trunk = common.ResidualBlock_noBN(nf=features, at='prelu')

        self.conv3x3 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        EC_combination = ['conv1-sobelx', 'conv1-sobely', 'conv1-laplacian']
        for i in range(len(EC_combination)):
            self.edgeconvs.append(nn.Sequential(
                common.EdgeConv(EC_combination[i], features, features),
            ))

        self.conv_reduce = nn.Conv2d(features * len(EC_combination), features, kernel_size=1, padding=0)
        self.sa = SpatialAttention()

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        based_x = x

        out = self.recon_trunk(based_x)

        for i, edgeconv in enumerate(self.edgeconvs):
            fea = edgeconv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        feas = self.conv_reduce(feas)
        feas = self.sa(feas) * feas

        feas_f = torch.cat([out.unsqueeze_(dim=1), feas.unsqueeze_(dim=1)], dim=1)
        fea_f_U = torch.sum(feas_f, dim=1)

        fea_f_s = fea_f_U.mean(-1).mean(-1)
        fea_f_z = self.fc(fea_f_s)
        for i, fc in enumerate(self.fcs):
            vector_f = fc(fea_f_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors_f = vector_f
            else:
                attention_vectors_f = torch.cat([attention_vectors_f, vector_f], dim=1)
        attention_vectors_f = self.softmax(attention_vectors_f)
        attention_vectors_f = attention_vectors_f.unsqueeze(-1).unsqueeze(-1)
        fea_v_out = (feas_f * attention_vectors_f).sum(dim=1)

        return fea_v_out


# # Edge-guided Residual Attention Block (EGRAB)
class EGRAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(EGRAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)

        modules_body.append(CoreModule(n_feat, M=2, G=8, r=2, stride=1, L=32))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),  # 缩小卷积核
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return self.conv(torch.cat([avg_out, max_out], dim=1))

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias = False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

# class GEM(nn.Module):
#     def __init__(self, sync_bn=False, input_channels=256):
#         super(GEM, self).__init__()
#         self.input_channels = input_channels
#         if sync_bn == True:
#             BatchNorm1d = SynchronizedBatchNorm1d
#             BatchNorm2d = SynchronizedBatchNorm2d
#         else:
#             BatchNorm1d = nn.BatchNorm1d
#             BatchNorm2d = nn.BatchNorm2d
#         self.edge_aggregation_func = nn.Sequential(
#             nn.Linear(4, 1),
#             BatchNorm1d(1),
#             nn.ReLU(inplace=True),
#         )
#         self.vertex_update_func = nn.Sequential(
#             nn.Linear(2 * input_channels, input_channels // 2),
#             BatchNorm1d(input_channels // 2),
#             nn.ReLU(inplace=True),
#         )
#         self.edge_update_func = nn.Sequential(
#             nn.Linear(2 * input_channels, input_channels // 2),
#             BatchNorm1d(input_channels // 2),
#             nn.ReLU(inplace=True),
#         )
#         self.update_edge_reduce_func = nn.Sequential(
#             nn.Linear(4, 1),
#             BatchNorm1d(1),
#             nn.ReLU(inplace=True),
#         )
#         self.final_aggregation_layer = nn.Sequential(
#             nn.Conv2d(input_channels + input_channels // 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm2d(input_channels),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, input):
#         x = input
#         B, C, H, W = x.size()
#         vertex = input
#         edge = torch.stack(
#             (
#                 torch.cat((input[:,:,-1:], input[:,:,:-1]), dim=2),
#                 torch.cat((input[:,:,1:], input[:,:,:1]), dim=2),
#                 torch.cat((input[:,:,:,-1:], input[:,:,:,:-1]), dim=3),
#                 torch.cat((input[:,:,:,1:], input[:,:,:,:1]), dim=3)
#             ), dim=-1
#         ) * input.unsqueeze(dim=-1)
#         aggregated_edge = self.edge_aggregation_func(
#             edge.reshape(-1, 4)
#         ).reshape((B, C, H, W))
#         cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1)
#         update_vertex = self.vertex_update_func(
#             cat_feature_for_vertex.permute(0, 2, 3, 1).reshape((-1, 2 * self.input_channels))
#         ).reshape((B, H, W, self.input_channels // 2)).permute(0, 3, 1, 2)
#         # output = self.final_aggregation_layer(update_vertex)
#         cat_feature_for_edge = torch.cat(
#             (
#                 torch.stack((vertex, vertex, vertex, vertex), dim=-1),
#                 edge
#             ), dim=1
#         ).permute(0, 2, 3, 4, 1).reshape((-1, 2 * self.input_channels))
#         update_edge = self.edge_update_func(cat_feature_for_edge).reshape((B, H, W, 4, C//2)).permute(0, 4, 1, 2, 3).reshape((-1, 4))
#         update_edge_converted = self.update_edge_reduce_func(update_edge).reshape((B, C//2, H, W))
#         update_feature = update_vertex * update_edge_converted
#         output = self.final_aggregation_layer(
#             torch.cat((x, update_feature), dim=1)
#         )
#         return output
# #     改进7
# class PMSFA(nn.Module):
#     def __init__(self, inc,sync_bn=False) -> None:
#         super().__init__()
        
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
#         self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
#         self.conv4 = Conv(inc, inc, 1)
#         self.gem = GEM(sync_bn=sync_bn, input_channels=inc)
    
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)
        
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         graph_enhanced = self.gem(out)
#         out = self.conv4(graph_enhanced) + x
#         return out
# class GEM(nn.Module):
#     def __init__(self, sync_bn=False, input_channels=256):
#         super().__init__()  # 使用更简洁的super初始化
#         self.input_channels = input_channels
        
#         # 参数初始化改进
#         bn_momentum = 0.01  # 增加BatchNorm动量参数
#         if sync_bn:
#             BatchNorm1d = lambda c: SynchronizedBatchNorm1d(c, momentum=bn_momentum)
#             BatchNorm2d = lambda c: SynchronizedBatchNorm2d(c, momentum=bn_momentum)
#         else:
#             BatchNorm1d = lambda c: nn.BatchNorm1d(c, momentum=bn_momentum)
#             BatchNorm2d = lambda c: nn.BatchNorm2d(c, momentum=bn_momentum)
        
#         # 增强参数初始化
#         def init_linear(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
        
#         # 改进的边聚合结构
#         self.edge_aggregation_func = nn.Sequential(
#             nn.Linear(4, 16),  # 增加中间维度
#             BatchNorm1d(16),
#             nn.ReLU(inplace=True),
#             nn.Linear(16, 1)  # 分层聚合
#         ).apply(init_linear)
        
#         # 顶点更新增加残差连接
#         self.vertex_update_func = nn.Sequential(
#             nn.Linear(2 * input_channels, input_channels),
#             BatchNorm1d(input_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_channels, input_channels // 2)  # 分层处理
#         ).apply(init_linear)
        
#         # 边更新结构改进
#         self.edge_update_func = nn.Sequential(
#             nn.Linear(2 * input_channels, input_channels),
#             BatchNorm1d(input_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_channels, input_channels // 2)
#         ).apply(init_linear)
        
#         # 最终聚合层增强
#         self.final_aggregation_layer = nn.Sequential(
#             nn.Conv2d(input_channels + input_channels//2, input_channels*2, 3, padding=1),
#             BatchNorm2d(input_channels*2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_channels*2, input_channels, 1)
#         )
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         identity = x  # 保留原始输入
        
#         # 改进的边特征计算
#         with torch.no_grad():  # 冻结位移操作梯度
#             shifted = [
#                 torch.roll(x, shifts=(-1, 0)),  # 使用torch.roll简化代码
#                 torch.roll(x, shifts=(1, 0)),
#                 torch.roll(x, shifts=(0, -1)),
#                 torch.roll(x, shifts=(0, 1))]
        
#         edge = torch.stack(shifted, dim=-1) * x.unsqueeze(-1)
        
#         # 分阶段边聚合
#         edge_features = edge.view(B*C*H*W, 4)
#         aggregated_edge = self.edge_aggregation_func(edge_features)
#         aggregated_edge = aggregated_edge.view(B, C, H, W)
        
#         # 顶点更新增加跳跃连接
#         vertex_input = torch.cat([x, aggregated_edge], dim=1).permute(0,2,3,1).contiguous()
#         vertex_features = self.vertex_update_func(vertex_input.view(-1, 2*C))
#         update_vertex = vertex_features.view(B, H, W, -1).permute(0,3,1,2)
        
#         # 边特征动态更新
#         edge_input = torch.cat([
#             x.unsqueeze(-1).expand(-1,-1,-1,-1,4),
#             edge
#         ], dim=1).permute(0,2,3,4,1).contiguous()
        
#         edge_features = self.edge_update_func(edge_input.view(-1, 2*C))
#         edge_features = edge_features.view(B, H, W, 4, -1).permute(0,4,1,2,3)
#         update_edge = torch.mean(edge_features, dim=-1)  # 简化聚合方式
        
#         # 特征融合增强
#         combined = torch.cat([identity, update_vertex * update_edge], dim=1)
#         return self.final_aggregation_layer(combined) + identity  # 增加残差连接

# class PMSFA(nn.Module):
#     def __init__(self, inc, sync_bn=False):
#         super().__init__()
#         # 通道分割比例调整
#         # 改进的通道分割函数
#         def safe_split(channels, ratio):
#             split = max(1, round(channels * ratio))  # 使用round代替int取整
#             split = split if split % 2 == 0 else split + 1  # 确保偶数
#             return min(split, channels)  # 不超过总通道数
            
#         # 动态调整分组数逻辑
#         def get_groups(channels):
#             return max(1, channels // 2) if channels >= 2 else 1  # 当通道数不足时自动调整分组数
            
#         # 第一层分割
#         self.split1 = safe_split(inc, 0.4)
#         # 第二层分割
#         self.split2 = safe_split(self.split1, 0.5)
        
#         # 修正卷积层定义
#         self.conv1 = nn.Sequential(
#             Conv(inc, inc, 3),
#             nn.BatchNorm2d(inc),
#             nn.ReLU(inplace=True)
#         )
        
#         self.conv2 = nn.Sequential(
#             Conv(self.split1, self.split1, 5, 
#                 g=get_groups(self.split1)),  # 动态调整分组数
#             nn.BatchNorm2d(self.split1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.conv3 = nn.Sequential(
#             Conv(self.split2, self.split2, 7,
#                 g=get_groups(self.split2)),  # 动态调整分组数
#             nn.BatchNorm2d(self.split2),
#             nn.ReLU(inplace=True)
#         )
        
#         # 动态通道调整
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(inc, inc//4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inc//4, inc, 1),
#             nn.Sigmoid()
#         )
        
#         self.gem = GEM(sync_bn=sync_bn, input_channels=inc)
#         self.conv4 = Conv(inc, inc, 1)
        
#     def forward(self, x):
#         identity = x
        
#         # 第一阶段带通道注意力
#         conv1_out = self.conv1(x)
#         ca = self.channel_attention(conv1_out)
#         conv1_out = conv1_out * ca
        
#         # 动态通道分割
#         split1 = int(conv1_out.size(1)*0.4)
#         conv1_out1, conv1_out2 = torch.split(conv1_out, [split1, conv1_out.size(1)-split1], dim=1)
        
#         # 第二阶段特征处理
#         conv2_out = self.conv2(conv1_out1)
#         split2 = int(conv2_out.size(1)*0.5)
#         conv2_out1, conv2_out2 = torch.split(conv2_out, [split2, conv2_out.size(1)-split2], dim=1)
        
#         # 第三阶段特征处理
#         conv3_out = self.conv3(conv2_out1)
        
#         # 特征融合改进
#         fused = torch.cat([
#             F.interpolate(conv3_out, scale_factor=2, mode='nearest'),  # 空间对齐
#             conv2_out2,
#             F.max_pool2d(conv1_out2, kernel_size=2)  # 匹配空间维度
#         ], dim=1)
        
#         # 图增强模块
#         graph_out = self.gem(fused)
        
#         # 最终融合
#         out = self.conv4(graph_out)
        
#         # 改进的残差连接
#         return torch.sigmoid(out) * identity + out  # 门控残差连接
# 改进6 0.60
# class EfficientEdgeEnhancer(nn.Module):
#     """高效边缘增强模块，参数量减少50%但保持感受野"""
#     def __init__(self, in_dim):
#         super().__init__()
#         self.edge_extract = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim//2, 3, padding=1, groups=in_dim//2),  # 分组卷积
#             nn.Conv2d(in_dim//2, in_dim, 1),  # 通道恢复
#             nn.BatchNorm2d(in_dim),
#             nn.PReLU()
#         )
#         self.edge_enhance = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),  # 深度可分离卷积
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         edge = self.edge_extract(x)
#         return x * (1 + self.edge_enhance(edge))  # 残差增强
# class PMSFA(nn.Module):  # 尺寸没变
#     def __init__(self, in_channels=64, growth_rate=32):   #in_channels=64, growth_rate=32
#         super(PMSFA, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)  # 64-》32
#         self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)  # 96->32
#         self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)  # 128->32
#         self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, in_channels, kernel_size=1)  # 160->64




#     def forward(self, x):
#         out1 = torch.relu(self.conv1(x))  # 32
#         out2 = torch.relu(self.conv2(torch.cat([x, out1], 1)))  # cat:96   out2:32
#         out3 = torch.relu(self.conv3(torch.cat([x, out1, out2], 1)))  # cat:128  out3:32
#         out4 = torch.relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))  # cat:160  out4:64
#         out = out4 + x

#         return out
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        
#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))
# class PMSFA(nn.Module):
#     def __init__(self, inc):
#         super().__init__()
#         self.inc = inc
        
#         # 确保最小通道数
#         min_channel = max(inc // 16, 8)  # 增加通道数下限
        
#         # 路径一：密集残差路径
#         self.dense_path = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(inc, inc//2, 3, padding=1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.Conv2d(inc//2 + inc, min_channel*4, 3, padding=1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.Conv2d(min_channel*4 + inc, min_channel*2, 3, padding=1),
#                 nn.ReLU(inplace=True)
#             ),
#             DepthwiseSeparableConv(min_channel*2 + inc, inc)
#         ])
        
#         # 路径二：金字塔多尺度路径
#         self.pyramid_path = nn.ModuleList([
#             Conv(inc, inc//2, 3, g=max(1, inc//8)),
#             ChannelAttention(inc//2),
#             Conv(inc//2, min_channel*4, 5, g=max(1, inc//16)),
#             SpatialAttention(),
#             Conv(min_channel*4, min_channel*2, 7, g=max(1, min_channel//4)),
#             ChannelAttention(min_channel*2)
#         ])
        
#         # 双路径融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(inc + min_channel*2, inc, 1),
#             nn.BatchNorm2d(inc),
#             nn.ReLU(inplace=True)
#         )
        
#         # 门控残差连接
#         self.gate = nn.Sequential(
#             nn.Conv2d(inc, inc, 3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         identity = x
        
#         # 密集路径
#         d1 = self.dense_path[0](x)
#         d2 = self.dense_path[1](torch.cat([x, d1], 1))
#         d3 = self.dense_path[2](torch.cat([x, d2], 1))
#         dense_out = self.dense_path[3](torch.cat([x, d3], 1))
        
#         # 金字塔路径
#         p1 = self.pyramid_path[0](x)
#         p1 = self.pyramid_path[1](p1) * p1
#         p2 = self.pyramid_path[2](p1)
#         p2 = self.pyramid_path[3](p2) * p2
#         p3 = self.pyramid_path[4](p2)
#         p3 = self.pyramid_path[5](p3) * p3
#         pyramid_out = F.interpolate(p3, scale_factor=2, mode='bilinear')
        
#         # 双路径融合
#         fused = self.fusion(torch.cat([dense_out, pyramid_out], 1))
        
#         # 门控残差
#         gate = self.gate(fused)
#         return gate * identity + fused

# # 改进的ChannelAttention
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, ratio=4):
#         super().__init__()
#         # 确保ratio值有效
#         ratio = max(1, min(ratio, channel))
        
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         # 修正卷积层参数
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, max(1, channel//ratio), kernel_size=1),  # 添加kernel_size
#             nn.ReLU(),
#             nn.Conv2d(max(1, channel//ratio), channel, kernel_size=1),  # 明确参数名称
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
#         return self.fc(avg_out + max_out)
class DRB(nn.Module):  # 尺寸没变
    def __init__(self, in_channels, growth_rate=32):   #in_channels=64, growth_rate=32
        super(DRB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)  # 64-》32
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)  # 96->32
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)  # 128->32
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, kernel_size=1)  # 160->32

        self.depthwise = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192)
        self.pointwise = nn.Conv2d(192, in_channels, 1)

    def forward(self, x):
        out1 = torch.relu(self.conv1(x))  # 32
        out2 = torch.relu(self.conv2(torch.cat([x, out1], 1)))  # cat:96   out2:32
        out3 = torch.relu(self.conv3(torch.cat([x, out1, out2], 1)))  # cat:128  out3:32
        out4 = torch.relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))  # cat:160  out4:32
        out5 = torch.relu(self.pointwise(self.depthwise(torch.cat([x, out1, out2, out3, out4], 1))))  # cat:192->32->64

        out = out5 + x


        return out
class PMSFA_OR(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3)
        self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
        self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
        self.conv4 = Conv(inc, inc, 1)
    
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        conv3_out = self.conv3(conv2_out_1)
        
        out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv4(out) + x
        return out 
# class PMSFA(nn.Module):
#     def __init__(self, in_channels=64, growth_rate=32):
#         super().__init__()
        
#         # DRB路径（密集残差路径）
#         self.drb_conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
#         self.drb_conv2 = nn.Conv2d(in_channels+growth_rate, growth_rate, 3, padding=1)
#         self.drb_conv3 = nn.Conv2d(in_channels+2*growth_rate, growth_rate, 3, padding=1)
#         self.drb_conv4 = nn.Conv2d(in_channels+3*growth_rate, growth_rate, 1)
        
#         # 深度可分离卷积
#         self.drb_depthwise = nn.Conv2d(in_channels+4*growth_rate, 
#                                       in_channels+4*growth_rate,
#                                       3, padding=1, 
#                                       groups=in_channels+4*growth_rate)
#         self.drb_pointwise = nn.Conv2d(in_channels+4*growth_rate, in_channels, 1)

#         # PMSFA路径（渐进多尺度路径）
#         self.pmsfa_conv1 = Conv(in_channels, in_channels, 3)
#         self.pmsfa_conv2 = Conv(in_channels//2, in_channels//2, 5, g=in_channels//2)
#         self.pmsfa_conv3 = Conv(in_channels//4, in_channels//4, 7, g=in_channels//4)
#         self.pmsfa_conv4 = Conv(in_channels, in_channels, 1)

#     def forward(self, x):
#         # DRB路径处理
#         d1 = torch.relu(self.drb_conv1(x))
#         d2 = torch.relu(self.drb_conv2(torch.cat([x, d1], 1)))
#         d3 = torch.relu(self.drb_conv3(torch.cat([x, d1, d2], 1)))
#         d4 = torch.relu(self.drb_conv4(torch.cat([x, d1, d2, d3], 1)))
        
#         # 深度可分离卷积融合
#         drb_cat = torch.cat([x, d1, d2, d3, d4], 1)
#         drb_out = self.drb_pointwise(torch.relu(self.drb_depthwise(drb_cat)))

#         # PMSFA路径处理
#         p1 = self.pmsfa_conv1(x)
#         p1_1, p1_2 = p1.chunk(2, 1)
#         p2 = self.pmsfa_conv2(p1_1)
#         p2_1, p2_2 = p2.chunk(2, 1)
#         p3 = self.pmsfa_conv3(p2_1)
        
#         # 多尺度特征融合
#         pmsfa_cat = torch.cat([p3, p2_2, p1_2], 1)
#         pmsfa_out = self.pmsfa_conv4(pmsfa_cat)

#         # 双路径融合 + 残差连接
#         return drb_out + pmsfa_out + x
class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix,self).__init__()
        self.shift_size = shift_size
        
    def forward(self,x): #x的张量 [B,C,H,W]
        x1,x2,x3,x4=x.chunk(4, dim=1)
        x1 = torch.roll(x1,self.shift_size,dims=2) #[::1.:]
        x2 = torch.roll(x2,self.shift_size,dims=2)#[::,:-1,:]
        x3 = torch.roll(x3,self.shift_size,dims=3) #[:,:,:,1:]
        x4 = torch.roll(x4,self.shift_size,dims=3) #[::,::-1]
        x= torch.cat([x1,x2,x3,x4],1)
        return x
# class AdaptiveShift(nn.Module):
#     def __init__(self, inc):
#         super().__init__()
#         # 定义多个可能的 shift_size
#         self.shift_sizes = [1, 2, 3]  # 可扩展为动态生成或从配置中读取
#         # 初始化多个 Shift_channel_mix 模块
#         self.shifts = nn.ModuleList([Shift_channel_mix(s) for s in self.shift_sizes])
 
#     def forward(self, x):
#         # 根据通道数动态选择位移大小
#         shift_idx = min(x.shape[1] // 64, len(self.shift_sizes) - 1)  # 简单启发式规则
#         return self.shifts[shift_idx](x)

# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
        
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
#         self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
#         self.conv4 = Conv(inc, inc, 1)
    
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)
        
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         out = self.conv4(out) + x
#         return out
# class PMSFA(nn.Module): #0.636也可以
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 初始化核心卷积层
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         # 新增分层位移模块
#         self.shift1 = Shift_channel_mix(shift_size=1)  # 初级位移
#         self.shift2 = Shift_channel_mix(shift_size=2)  # 中级位移
#         self.shift3 = Shift_channel_mix(shift_size=3)  # 高级位移

#     def forward(self, x):
#         # 第一级处理
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
        
#         # 第二级处理（带位移增强）
#         x_high = self.shift1(x_high)
#         x_high = self.conv2(x_high)
#         x_mid, x_high = x_high.chunk(2, dim=1)
        
#         # 第三级处理（带位移增强）
#         x_mid = self.shift2(x_mid)
#         x_mid = self.conv3(x_mid)
        
#         # 特征重组与融合
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.shift3(out)  # 最终位移增强
#         return self.conv4(out) + x

# class PMSFA(nn.Module): #0.634
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 初始化核心卷积层
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
#         self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
#         self.conv4 = Conv(inc, inc, 1)
 
#         # 新增分层位移模块（使用 AdaptiveShift）
#         self.adaptive_shift = AdaptiveShift(inc)  # 自适应位移模块
 
#     def forward(self, x):
#         # 第一级处理
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
 
#         # 第二级处理（带位移增强）
#         x_high = self.adaptive_shift(x_high)  # 使用自适应位移
#         x_high = self.conv2(x_high)
#         x_mid, x_high = x_high.chunk(2, dim=1)
 
#         # 第三级处理（带位移增强）
#         x_mid = self.adaptive_shift(x_mid)  # 使用自适应位移
#         x_mid = self.conv3(x_mid)
 
#         # 特征重组与融合
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.adaptive_shift(out)  # 最终位移增强
#         return self.conv4(out) + x
#//////开了最大池化，exp36 0.87,0.742,0.82,0.636，不开最大池化0.859，0.751，0.835，0.636
# class EffectiveSEModule(nn.Module):
#     def __init__(self, channels, add_maxpool=False):
#         super(EffectiveSEModule, self).__init__()
#         self.add_maxpool = add_maxpool
#         self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.gate = nn.Hardsigmoid()
 
#     def forward(self, x):
#         x_se = x.mean((2, 3), keepdim=True)
#         if self.add_maxpool:
#             x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
#         x_se = self.fc(x_se)
#         return x * self.gate(x_se)
    
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         self.shift1 = Shift_channel_mix(shift_size=1)
#         self.shift2 = Shift_channel_mix(shift_size=2)
#         self.shift3 = Shift_channel_mix(shift_size=3)
 
#         # 添加 EffectiveSEModule
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)  # 可选：在最终输出前使用最大池化
 
#     def forward(self, x):
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
        
#         x_high = self.shift1(x_high)
#         x_high = self.conv2(x_high)
#         x_high = self.se1(x_high)  # 在 conv2 后应用 SE 模块
#         x_mid, x_high = x_high.chunk(2, dim=1)
        
#         x_mid = self.shift2(x_mid)
#         x_mid = self.conv3(x_mid)
#         x_mid = self.se2(x_mid)  # 在 conv3 后应用 SE 模块
        
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.shift3(out)
#         out = self.se_final(out)  # 在最终输出前应用 SE 模块
#         return self.conv4(out) + x
# //////
# ////4.21 17.36
# class Shift_channel_mix(nn.Module):
#     def __init__(self):
#         super(Shift_channel_mix, self).__init__()
#         # 不再需要 shift_size 参数
 
#     def compute_shift_size(self, x):
#         # 一个简单的启发式规则来计算 shift_size
#         # 例如，基于通道数或空间维度
#         return min(x.shape[1] // 64, 3)  # 确保 shift_size 不超过 3
 
#     def forward(self, x):  # x 的张量 [B, C, H, W]
#         shift_size = self.compute_shift_size(x)
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
#         x1 = torch.roll(x1, shift_size, dims=2)
#         x2 = torch.roll(x2, shift_size, dims=2)
#         x3 = torch.roll(x3, shift_size, dims=3)
#         x4 = torch.roll(x4, shift_size, dims=3)
#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         return x
    
# class EffectiveSEModule(nn.Module):
#     def __init__(self, channels, add_maxpool=False):
#         super(EffectiveSEModule, self).__init__()
#         self.add_maxpool = add_maxpool
#         self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.gate = nn.Hardsigmoid()
 
#     def forward(self, x):
#         x_se = x.mean((2, 3), keepdim=True)
#         if self.add_maxpool:
#             x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
#         x_se = self.fc(x_se)
#         return x * self.gate(x_se)
    
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
 
#         self.shift1 = Shift_channel_mix()
#         self.shift2 = Shift_channel_mix()
#         self.shift3 = Shift_channel_mix()
 
#         # 添加 EffectiveSEModule
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)  # 可选：在最终输出前使用最大池化
 
#     def forward(self, x):
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
 
#         x_high = self.shift1(x_high)
#         x_high = self.conv2(x_high)
#         x_high = self.se1(x_high)  # 在 conv2 后应用 SE 模块
#         x_mid, x_high = x_high.chunk(2, dim=1)
 
#         x_mid = self.shift2(x_mid)
#         x_mid = self.conv3(x_mid)
#         x_mid = self.se2(x_mid)  # 在 conv3 后应用 SE 模块
 
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.shift3(out)
#         out = self.se_final(out)  # 在最终输出前应用 SE 模块
#         return self.conv4(out) + x
# //////
# 上面跑完跑这个
# 
# class ImprovedShiftChannelMix(nn.Module):0.626
#     def __init__(self, channels, shift_range=3):
#         super(ImprovedShiftChannelMix, self).__init__()
#         self.channels = channels
#         self.shift_range = shift_range
        
#         # 自适应 shift_size 参数
#         self.shift_size_h = nn.Parameter(torch.zeros(1))  # 水平方向移位
#         self.shift_size_v = nn.Parameter(torch.zeros(1))  # 垂直方向移位
        
#         # 用于混合相邻特征的卷积核
#         self.mix_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels//4)
        
#         # 归一化层
#         self.norm = nn.InstanceNorm2d(channels)
        
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # 计算实际移位量（限制在合理范围内）
#         shift_h = int(torch.sigmoid(self.shift_size_h) * self.shift_range)
#         shift_v = int(torch.sigmoid(self.shift_size_v) * self.shift_range)
        
#         # 分割通道
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
        
#         # 水平正向和负向移位
#         x1RolledPos = torch.roll(x1, shifts=shift_h, dims=3)
#         x1RolledNeg = torch.roll(x1, shifts=-shift_h, dims=3)
#         x1Mixed = (x1 + x1RolledPos + x1RolledNeg) / 3
        
#         # 水平正向和负向移位 + 垂直正向和负向移位
#         x2RolledHPos = torch.roll(x2, shifts=shift_h, dims=3)
#         x2RolledHNeg = torch.roll(x2, shifts=-shift_h, dims=3)
#         x2RolledVPos = torch.roll(x2, shifts=shift_v, dims=2)
#         x2RolledVNeg = torch.roll(x2, shifts=-shift_v, dims=2)
#         x2Mixed = (x2 + x2RolledHPos + x2RolledHNeg + x2RolledVPos + x2RolledVNeg) / 5
        
#         # 垂直正向和负向移位
#         x3RolledPos = torch.roll(x3, shifts=shift_v, dims=2)
#         x3RolledNeg = torch.roll(x3, shifts=-shift_v, dims=2)
#         x3Mixed = (x3 + x3RolledPos + x3RolledNeg) / 3
        
#         # 混合方向移位
#         x4RolledH = torch.roll(x4, shifts=shift_h, dims=3)
#         x4RolledV = torch.roll(x4, shifts=shift_v, dims=2)
#         x4Mixed = (x4 + x4RolledH + x4RolledV) / 3
        
#         # 合并通道并通过卷积进行进一步特征混合
#         x = torch.cat([x1Mixed, x2Mixed, x3Mixed, x4Mixed], dim=1)
#         x = self.mix_conv(x)
#         x = self.norm(x)
        
#         return x

# class AdaptiveShiftChannelMix(nn.Module):0.637当前最好改进46
#     def __init__(self, channels, max_shift=3):
#         super(AdaptiveShiftChannelMix, self).__init__()
#         self.max_shift = max_shift
        
#         # 自适应移位量参数（初始化为1，保持原始行为）
#         self.shift_size_h = nn.Parameter(torch.ones(1))  # 水平方向
#         self.shift_size_v = nn.Parameter(torch.ones(1))  # 垂直方向
        
#         # 自适应方向参数（初始化为0，表示正向）
#         self.direction_h = nn.Parameter(torch.zeros(1))  # 水平方向
#         self.direction_v = nn.Parameter(torch.zeros(1))  # 垂直方向
        
#     def forward(self, x):
#         shift_size_h = torch.sigmoid(self.shift_size_h) * self.max_shift
#         shift_size_v = torch.sigmoid(self.shift_size_v) * self.max_shift
        
#         direction_h = torch.round(torch.sigmoid(self.direction_h)) * 2 - 1  # -1 或 1
#         direction_v = torch.round(torch.sigmoid(self.direction_v)) * 2 - 1  # -1 或 1
        
#         # 分割通道
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
        
#         # 水平正向移位
#         x1 = torch.roll(x1, shifts=int(shift_size_h * direction_h), dims=2)
        
#         # 水平负向移位
#         x2 = torch.roll(x2, shifts=int(shift_size_h * -direction_h), dims=2)
        
#         # 垂直正向移位
#         x3 = torch.roll(x3, shifts=int(shift_size_v * direction_v), dims=3)
        
#         # 垂直负向移位
#         x4 = torch.roll(x4, shifts=int(shift_size_v * -direction_v), dims=3)
        
#         # 合并通道
#         x = torch.cat([x1, x2, x3, x4], dim=1)
        
#         return x

#4. PMSFA改进  7   0.635
# class AttentionAdaptiveShift(nn.Module):
#     def __init__(self, channels, max_shift=3):
#         super().__init__()
#         self.max_shift = max_shift
        
#         # 注意力模块
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, kernel_size=1),
#             nn.GELU(),
#             nn.Conv2d(channels // 4, 4, kernel_size=1),  # 为每个通道组生成注意力权重
#             nn.Sigmoid()
#         )
        
#         # 移位参数（与之前相同）
#         self.shift_size_h = nn.Parameter(torch.ones(1))
#         self.shift_size_v = nn.Parameter(torch.ones(1))
#         self.direction_h = nn.Parameter(torch.zeros(1))
#         self.direction_v = nn.Parameter(torch.zeros(1))
    
#     def forward(self, x):
#         attn = self.attention(x)  # [B, 4, H, W] 注意力权重
        
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
#         a1, a2, a3, a4 = attn.chunk(4, dim=1)
        
#         # 计算移位参数
#         shift_h = torch.sigmoid(self.shift_size_h) * self.max_shift
#         dir_h = torch.tanh(self.direction_h)
#         shift_v = torch.sigmoid(self.shift_size_v) * self.max_shift
#         dir_v = torch.tanh(self.direction_v)
        
#         # 应用移位和注意力
#         x1 = a1 * torch.roll(x1, shifts=int(shift_h * dir_h), dims=2)
#         x2 = a2 * torch.roll(x2, shifts=int(shift_h * -dir_h), dims=2)
#         x3 = a3 * torch.roll(x3, shifts=int(shift_v * dir_v), dims=3)
#         x4 = a4 * torch.roll(x4, shifts=int(shift_v * -dir_v), dims=3)
        
#         return torch.cat([x1, x2, x3, x4], dim=1)

# 4,0.627
# class MultiDirectionAdaptiveShift(nn.Module):
#     def __init__(self, channels, max_shift=3):
#         super().__init__()
#         self.max_shift = max_shift
        
#         # 水平移位参数
#         self.shift_size_h = nn.Parameter(torch.ones(1))
#         self.direction_h = nn.Parameter(torch.zeros(1))
        
#         # 垂直移位参数
#         self.shift_size_v = nn.Parameter(torch.ones(1))
#         self.direction_v = nn.Parameter(torch.zeros(1))
    
#     def forward(self, x):
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
        
#         # 水平移位
#         shift_h = torch.sigmoid(self.shift_size_h) * self.max_shift
#         dir_h = torch.tanh(self.direction_h)
#         x1 = torch.roll(x1, shifts=int(shift_h * dir_h), dims=2)
#         x2 = torch.roll(x2, shifts=int(shift_h * -dir_h), dims=2)
        
#         # 垂直移位
#         shift_v = torch.sigmoid(self.shift_size_v) * self.max_shift
#         dir_v = torch.tanh(self.direction_v)
#         x3 = torch.roll(x3, shifts=int(shift_v * dir_v), dims=3)
#         x4 = torch.roll(x4, shifts=int(shift_v * -dir_v), dims=3)
        
#         # 额外的对角线移位
#         x1 = torch.roll(x1, shifts=int(shift_h * 0.5), dims=3)  # 水平+垂直
#         x4 = torch.roll(x4, shifts=int(shift_v * 0.5), dims=2)  # 垂直+水平
        
#         return torch.cat([x1, x2, x3, x4], dim=1)
# 2,0.627
# class AdaptiveShiftChannelMix(nn.Module):
#     def __init__(self, channels, max_shift=3):
#         super().__init__()
#         self.max_shift = max_shift
        
#         # 每个通道组的移位参数
#         self.shift_size_h1 = nn.Parameter(torch.ones(1))  # x1 的水平移位
#         self.shift_size_h2 = nn.Parameter(torch.ones(1))  # x2 的水平移位
#         self.shift_size_v3 = nn.Parameter(torch.ones(1))  # x3 的垂直移位
#         self.shift_size_v4 = nn.Parameter(torch.ones(1))  # x4 的垂直移位
        
#         # 每个通道组的方向参数
#         self.direction_h1 = nn.Parameter(torch.zeros(1))
#         self.direction_h2 = nn.Parameter(torch.zeros(1))
#         self.direction_v3 = nn.Parameter(torch.zeros(1))
#         self.direction_v4 = nn.Parameter(torch.zeros(1))
    
#     def forward(self, x):
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
        
#         # x1: 水平正向移位
#         shift_h1 = torch.sigmoid(self.shift_size_h1) * self.max_shift
#         dir_h1 = torch.tanh(self.direction_h1)
#         x1 = torch.roll(x1, shifts=int(shift_h1 * dir_h1), dims=2)
        
#         # x2: 水平负向移位
#         shift_h2 = torch.sigmoid(self.shift_size_h2) * self.max_shift
#         dir_h2 = torch.tanh(self.direction_h2)
#         x2 = torch.roll(x2, shifts=int(shift_h2 * -dir_h2), dims=2)
        
#         # x3: 垂直正向移位
#         shift_v3 = torch.sigmoid(self.shift_size_v3) * self.max_shift
#         dir_v3 = torch.tanh(self.direction_v3)
#         x3 = torch.roll(x3, shifts=int(shift_v3 * dir_v3), dims=3)
        
#         # x4: 垂直负向移位
#         shift_v4 = torch.sigmoid(self.shift_size_v4) * self.max_shift
#         dir_v4 = torch.tanh(self.direction_v4)
#         x4 = torch.roll(x4, shifts=int(shift_v4 * -dir_v4), dims=3)
        
#         return torch.cat([x1, x2, x3, x4], dim=1)
# 5.  0.632
# class GatedAdaptiveShift(nn.Module):
#     def __init__(self, channels, max_shift=3):
#         super().__init__()
#         self.max_shift = max_shift
#         self.conv = nn.Conv2d(channels, 4, kernel_size=1)  # 为每个通道组生成门控信号
        
#         # 移位参数
#         self.shift_size_h = nn.Parameter(torch.ones(1))
#         self.shift_size_v = nn.Parameter(torch.ones(1))
#         self.direction_h = nn.Parameter(torch.zeros(1))
#         self.direction_v = nn.Parameter(torch.zeros(1))
    
#     def forward(self, x):
#         gates = torch.sigmoid(self.conv(x))  # [B, 4, H, W] 门控信号
        
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
#         g1, g2, g3, g4 = gates.chunk(4, dim=1)
        
#         # 计算移位参数
#         shift_h = torch.sigmoid(self.shift_size_h) * self.max_shift
#         dir_h = torch.tanh(self.direction_h)
#         shift_v = torch.sigmoid(self.shift_size_v) * self.max_shift
#         dir_v = torch.tanh(self.direction_v)
        
#         # 应用移位和门控
#         x1 = g1 * torch.roll(x1, shifts=int(shift_h * dir_h), dims=2)
#         x2 = g2 * torch.roll(x2, shifts=int(shift_h * -dir_h), dims=2)
#         x3 = g3 * torch.roll(x3, shifts=int(shift_v * dir_v), dims=3)
#         x4 = g4 * torch.roll(x4, shifts=int(shift_v * -dir_v), dims=3)
        
#         return torch.cat([x1, x2, x3, x4], dim=1)
# 斜着走
# class AdvancedAdaptiveShift(nn.Module):
#     def __init__(self, channels, max_shift=3):
#         super().__init__()
#         self.max_shift = max_shift
        
#         # 前置卷积和激活
#         self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
#         self.pre_act = nn.GELU()
        
#         # 注意力模块（输出4个分组，每个分组2个方向）
#         self.spatial_attn = nn.Sequential(
#             nn.Conv2d(channels, 8, kernel_size=1),  # 每个分组两个注意力图（水平和垂直）
#             nn.Sigmoid()
#         )
        
#         # 通道注意力（每个分组独立）
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // 4, kernel_size=1),
#             nn.GELU(),
#             nn.Conv2d(channels // 4, channels * 2, kernel_size=1),  # 每个分组两个通道注意力
#             nn.Sigmoid()
#         )
        
#         # 移位参数（每个通道组独立学习水平和垂直方向）
#         self.shift_params = nn.ParameterList([
#             nn.Parameter(torch.randn(4)),  # 4个分组，每个分组有h_shift, h_dir, v_shift, v_dir
#             nn.Parameter(torch.randn(4)),
#             nn.Parameter(torch.randn(4)),
#             nn.Parameter(torch.randn(4))
#         ])
        
#         # 后置卷积和激活
#         self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)
#         self.post_act = nn.GELU()
    
#     def forward(self, x):
#         x = self.pre_conv(x)
#         x = self.pre_act(x + self._shift_mix(x))  # 残差连接
#         x = self.post_conv(x)
#         return self.post_act(x)
    
#     def _shift_mix(self, x):
#         B, C, H, W = x.shape
        
#         # 获取空间注意力（8通道：4分组×2方向）
#         spatial_attn = self.spatial_attn(x)  # [B, 8, H, W]
#         s_attn = spatial_attn.chunk(4, dim=1)  # 分成4组，每组2通道（水平/垂直）
        
#         # 通道注意力（C×2）
#         channel_attn = self.channel_attn(x)  # [B, 2C, 1, 1]
#         c_attn = channel_attn.chunk(4, dim=1)  # 分成4组，每组C/2通道
        
#         # 分割输入特征
#         x_groups = x.chunk(4, dim=1)
        
#         shifted_groups = []
#         for i in range(4):
#             x_group = x_groups[i]
#             s_h, s_v = s_attn[i].chunk(2, dim=1)  # 水平和垂直空间注意力
#             c_h, c_v = c_attn[i].chunk(2, dim=1)  # 水平和垂直通道注意力
            
#             # 获取当前分组的移位参数
#             params = self.shift_params[i]
#             h_shift = torch.sigmoid(params[0]) * self.max_shift
#             h_dir = torch.tanh(params[1])
#             v_shift = torch.sigmoid(params[2]) * self.max_shift
#             v_dir = torch.tanh(params[3])
            
#             # 应用水平和垂直移位
#             shifted_h = torch.roll(x_group, shifts=int(h_shift * h_dir), dims=2)
#             shifted_v = torch.roll(x_group, shifts=int(v_shift * v_dir), dims=3)
            
#             # 组合两个方向的移位（注意力加权求和）
#             combined = s_h * c_h * shifted_h + s_v * c_v * shifted_v
#             shifted_groups.append(combined)
        
#         return torch.cat(shifted_groups, dim=1)
#9
class AdvancedAdaptiveShift(nn.Module):
    def __init__(self, channels, max_shift=3):
        super().__init__()
        self.max_shift = max_shift
        
        # 前置卷积和激活
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.pre_act = nn.GELU()
        
        # 注意力模块
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 移位参数（每个通道组独立学习）
        self.shift_size_h1 = nn.Parameter(torch.ones(1))
        self.shift_size_h2 = nn.Parameter(torch.ones(1))
        self.shift_size_v3 = nn.Parameter(torch.ones(1))
        self.shift_size_v4 = nn.Parameter(torch.ones(1))
        
        self.direction_h1 = nn.Parameter(torch.zeros(1))
        self.direction_h2 = nn.Parameter(torch.zeros(1))
        self.direction_v3 = nn.Parameter(torch.zeros(1))
        self.direction_v4 = nn.Parameter(torch.zeros(1))
        
        # 后置卷积和激活
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.post_act = nn.GELU()
    
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_act(x + self._shift_mix(x))  # 残差连接
        x = self.post_conv(x)
        return self.post_act(x)
    
    def _shift_mix(self, x):
        # 获取各种注意力权重
        spatial_attn = self.spatial_attn(x)  # [B, 4, H, W]
        channel_attn = self.channel_attn(x)  # [B, C, 1, 1]
        
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        s_a1, s_a2, s_a3, s_a4 = spatial_attn.chunk(4, dim=1)
        c_a1, c_a2, c_a3, c_a4 = channel_attn.chunk(4, dim=1)
        
        # 计算每个通道组的移位参数
        shift_h1 = torch.sigmoid(self.shift_size_h1) * self.max_shift
        dir_h1 = torch.tanh(self.direction_h1)
        
        shift_h2 = torch.sigmoid(self.shift_size_h2) * self.max_shift
        dir_h2 = torch.tanh(self.direction_h2)
        
        shift_v3 = torch.sigmoid(self.shift_size_v3) * self.max_shift
        dir_v3 = torch.tanh(self.direction_v3)
        
        shift_v4 = torch.sigmoid(self.shift_size_v4) * self.max_shift
        dir_v4 = torch.tanh(self.direction_v4)
        
        # 应用移位和注意力
        x1 = s_a1 * c_a1 * torch.roll(x1, shifts=int(shift_h1 * dir_h1), dims=2)
        x2 = s_a2 * c_a2 * torch.roll(x2, shifts=int(shift_h2 * -dir_h2), dims=2)
        x3 = s_a3 * c_a3 * torch.roll(x3, shifts=int(shift_v3 * dir_v3), dims=3)
        x4 = s_a4 * c_a4 * torch.roll(x4, shifts=int(shift_v4 * -dir_v4), dims=3)
        
        return torch.cat([x1, x2, x3, x4], dim=1)
class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()
 
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)

# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 基础结构
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
#         self.shift1 = AdvancedAdaptiveShift(inc//2)
#         self.shift2 = AdvancedAdaptiveShift(inc//4)
#         self.shift3 = AdvancedAdaptiveShift(inc)
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)

#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
    
#         conv1_out_1 = self.shift1(conv1_out_1)
#         conv1_out_1 = self.conv2(conv1_out_1)
#         conv2_out = self.se1(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
            
#         conv2_out_1 = self.shift2(conv2_out_1)
#         conv2_out_1 = self.conv3(conv2_out_1)
#         conv3_out = self.se2(conv2_out_1)
            
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         out = self.shift3(out)
#         out = self.se_final(out)
   
#         return self.conv4(out) + x
# class PMSFA(nn.Module):
#     def __init__(self, inc):
#         super().__init__()
#         # 始终启用高级模块，通过动态对齐保证通道合法性
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         # 直接初始化高级模块
#         self.shift1 = AdvancedAdaptiveShift(inc//2)
#         self.shift2 = AdvancedAdaptiveShift(inc//4)
#         self.shift3 = AdvancedAdaptiveShift(inc)
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x_high, x_low = x1.chunk(2, dim=1)
        
#         x_high = self.shift1(x_high)
#         x_high = self.conv2(x_high)
#         x_high = self.se1(x_high)
#         x_mid, x_high = x_high.chunk(2, dim=1)
        
#         x_mid = self.shift2(x_mid)
#         x_mid = self.conv3(x_mid)
#         x_mid = self.se2(x_mid)
        
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.shift3(out)
#         out = self.se_final(out)
        
#         return self.conv4(out) + x
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 基础结构
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
#         # 增强的决策参数（增加缩放因子和偏置）
#         self.decision_bias = nn.Parameter(torch.tensor([3.0]))  # 初始偏置控制激活阈值-3.0初始靠近简易分支，3.0初始靠近高级分支

#         # 标记需要延迟初始化的模块
#         self.advanced_modules_initialized = False
#         self._register_advanced_modules(inc)  # 注册模块但不初始化

#     def _register_advanced_modules(self, inc):
#         """注册模块占位符但不实际初始化参数"""
#         # 使用 nn.ModuleDict 管理高级模块
#         self.advanced = nn.ModuleDict({
#             "shift1": nn.Identity(),  # 占位符
#             "shift2": nn.Identity(),
#             "shift3": nn.Identity(),
#             "se1": nn.Identity(),
#             "se2": nn.Identity(),
#             "se_final": nn.Identity()
#         })
#         # 保存初始化所需参数
#         self.inc = inc

#     def _init_advanced_modules(self):
#         """按需初始化高级模块"""
#         if not self.advanced_modules_initialized:
#             # 替换占位符为实际模块
#             self.advanced["shift1"] = AdvancedAdaptiveShift(self.inc//2)
#             self.advanced["shift2"] = AdvancedAdaptiveShift(self.inc//4)
#             self.advanced["shift3"] = AdvancedAdaptiveShift(self.inc)
#             self.advanced["se1"] = EffectiveSEModule(self.inc//2)
#             self.advanced["se2"] = EffectiveSEModule(self.inc//4)
#             self.advanced["se_final"] = EffectiveSEModule(self.inc)
#             self.advanced_modules_initialized = True
#     def _get_decision(self, x):
#         """改进的决策函数"""
#         # 增强的复杂度计算（标准化后的相对能量）
#         B, C, H, W = x.shape
#         channel_energy = x.pow(2).mean(dim=[2,3])  # [B, C]
#         spatial_variation = x.std(dim=[2,3]).mean(dim=1)  # [B]
#         complexity = (channel_energy.mean(dim=1) * spatial_variation).detach()  # [B]
        
#         # 自适应决策公式
#         decision_logit = complexity * 1.0 + self.decision_bias
#         decision = torch.sigmoid(decision_logit).mean()
#         print(decision)
#         # 直通估计器保持梯度
#         hard_decision = (decision > 0.5).float()
#         return hard_decision - decision.detach() + decision

#     def forward(self, x):
#         decision = self._get_decision(x)
#         # 动态初始化逻辑
#         if decision >= 0.5 and not self.advanced_modules_initialized:
#             self._init_advanced_modules()  # 触发延迟初始化

#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        
#         if decision >= 0.5 and self.advanced_modules_initialized:
#             # 高级分支
#             print("True")
#             conv1_out_1 = self.advanced["shift1"](conv1_out_1)
#             conv2_out = self.conv2(conv1_out_1)
#             conv2_out = self.advanced["se1"](conv2_out)
#             conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
            
#             conv2_out_1 = self.advanced["shift2"](conv2_out_1)
#             conv3_out = self.conv3(conv2_out_1)
#             conv3_out = self.advanced["se2"](conv3_out)
            
#             out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#             out = self.advanced["shift3"](out)
#             out = self.advanced["se_final"](out)
#         else:
#             # 简化分支
#             print("False")
#             conv2_out = self.conv2(conv1_out_1)
#             conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#             conv3_out = self.conv3(conv2_out_1)
#             out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         return self.conv4(out) + x
    
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 基础结构保持不变
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
# #         # 隐藏初始化高级模块参数
# #         self.shift1 = AdvancedAdaptiveShift(inc//2)
# #         self.shift2 = AdvancedAdaptiveShift(inc//4)
# #         self.shift3 = AdvancedAdaptiveShift(inc)
# #         self.se1 = EffectiveSEModule(inc//2)
# #         self.se2 = EffectiveSEModule(inc//4)
# #         self.se_final = EffectiveSEModule(inc)
        
# #         # 动态决策参数（每个实例独立学习）
# #         self.decision_param = nn.Parameter(torch.tensor([-3.]))  # 初始偏向简化分支

# #     def _get_decision(self, x):
# #         """轻量级决策函数"""
# #         # 计算特征复杂度指标（约0.01%计算量）
# #         complexity = x.abs().mean(dim=[1,2,3]).detach()  # [B]
# #         decision = (complexity * torch.sigmoid(self.decision_param)).mean()  # 可学习决策
        
# #         # 直通估计器实现二值化
# #         hard_decision = (decision > 0.5).float()
# #         return hard_decision - decision.detach() + decision  # 前向硬决策，反向软梯度

#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        
#         # decision = self._get_decision(x_high)
#         # 动态选择单一分支执行
# #         if decision >= 0.5:  # 高级分支
# #             print("True")
# #             x_high = self.shift1(x_high)
# #             x_high = self.conv2(x_high)
# #             x_high = self.se1(x_high)
# #             x_mid, x_high = x_high.chunk(2, dim=1)
            
# #             x_mid = self.shift2(x_mid)
# #             x_mid = self.conv3(x_mid)
# #             x_mid = self.se2(x_mid)
            
# #             out = torch.cat([x_mid, x_high, x_low], dim=1)
# #             out = self.shift3(out)
# #             out = self.se_final(out)
#         # else:  # 简化分支
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        
#         return self.conv4(out) + x
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
        
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
#         self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
#         self.conv4 = Conv(inc, inc, 1)
    
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)
        
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         out = self.conv4(out) + x
#         return out
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 基础结构保持不变
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         # 隐藏初始化高级模块参数
#         self.shift1 = AdvancedAdaptiveShift(inc//2)
#         self.shift2 = AdvancedAdaptiveShift(inc//4)
#         self.shift3 = AdvancedAdaptiveShift(inc)
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)
        
#         # 动态决策参数（每个实例独立学习）
#         self.decision_param = nn.Parameter(torch.tensor([-3.]))  # 初始偏向简化分支

#     def _get_decision(self, x):
#         """轻量级决策函数"""
#         # 计算特征复杂度指标（约0.01%计算量）
#         complexity = x.abs().mean(dim=[1,2,3]).detach()  # [B]
#         decision = (complexity * torch.sigmoid(self.decision_param)).mean()  # 可学习决策
        
#         # 直通估计器实现二值化
#         hard_decision = (decision > 0.5).float()
#         return hard_decision - decision.detach() + decision  # 前向硬决策，反向软梯度

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x_high, x_low = x1.chunk(2, dim=1)
        
#         decision = self._get_decision(x_high)
        
#         # 动态选择单一分支执行
#         if decision >= 0.5:  # 高级分支
#             x_high = self.shift1(x_high)
#             x_high = self.conv2(x_high)
#             x_high = self.se1(x_high)
#             x_mid, x_high = x_high.chunk(2, dim=1)
            
#             x_mid = self.shift2(x_mid)
#             x_mid = self.conv3(x_mid)
#             x_mid = self.se2(x_mid)
            
#             out = torch.cat([x_mid, x_high, x_low], dim=1)
#             out = self.shift3(out)
#             out = self.se_final(out)
#         else:  # 简化分支
#             x_high = self.conv2(x_high)
#             x_mid, x_high = x_high.chunk(2, dim=1)
#             x_mid = self.conv3(x_mid)
#             out = torch.cat([x_mid, x_high, x_low], dim=1)
        
#         return self.conv4(out) + x
    
# class PMSFA(nn.Module):
#     def __init__(self, inc, use_shift_se=True) -> None:  # 新增 use_shift_se 参数，默认True表示启用
#         super().__init__()
#         self.use_shift_se = use_shift_se  # 控制是否启用高级模块
        
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         if self.use_shift_se:  # 只有启用时才初始化这些模块
#             self.shift1 = AdvancedAdaptiveShift(inc//2)
#             self.shift2 = AdvancedAdaptiveShift(inc//4)
#             self.shift3 = AdvancedAdaptiveShift(inc)
#             self.se1 = EffectiveSEModule(inc//2)
#             self.se2 = EffectiveSEModule(inc//4)
#             self.se_final = EffectiveSEModule(inc)

#     def forward(self, x):
# #         如果复现不了原来的0.644，就是这里，原版残差的是x，0.644残差的是x1(残差的是经过conv1后的)就是下面的
# # x1
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
        
# #         PMSFA的原版 x
#         # x1 = self.conv1(x)
#         # x_high, x_low = x1.chunk(2, dim=1)
        
#         if self.use_shift_se:  # 启用时执行完整流程
#             x_high = self.shift1(x_high)
#             x_high = self.conv2(x_high)
#             x_high = self.se1(x_high)
#             x_mid, x_high = x_high.chunk(2, dim=1)
            
#             x_mid = self.shift2(x_mid)
#             x_mid = self.conv3(x_mid)
#             x_mid = self.se2(x_mid)
            
#             out = torch.cat([x_mid, x_high, x_low], dim=1)
#             out = self.shift3(out)
#             out = self.se_final(out)
#         else:  # 不启用时执行简化流程
#             x_high = self.conv2(x_high)
#             x_mid, x_high = x_high.chunk(2, dim=1)
    
#             x_mid = self.conv3(x_mid)
            
#             out = torch.cat([x_mid, x_high, x_low], dim=1)
#         return self.conv4(out) + x
    
# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         self.shift1 = AdvancedAdaptiveShift(inc//2)
#         self.shift2 = AdvancedAdaptiveShift(inc//4)
#         self.shift3 =AdvancedAdaptiveShift(inc)
 
#         # 添加 EffectiveSEModule
#         self.se1 = EffectiveSEModule(inc//2)
#         self.se2 = EffectiveSEModule(inc//4)
#         self.se_final = EffectiveSEModule(inc)  # 可选：在最终输出前使用最大池化
 
#     def forward(self, x):
#         x = self.conv1(x)
#         x_high, x_low = x.chunk(2, dim=1)
        
#         x_high = self.shift1(x_high)
#         x_high = self.conv2(x_high)
#         x_high = self.se1(x_high)  # 在 conv2 后应用 SE 模块
#         x_mid, x_high = x_high.chunk(2, dim=1)
        
#         x_mid = self.shift2(x_mid)
#         x_mid = self.conv3(x_mid)
#         x_mid = self.se2(x_mid)  # 在 conv3 后应用 SE 模块
        
#         out = torch.cat([x_mid, x_high, x_low], dim=1)
#         out = self.shift3(out)
#         out = self.se_final(out)  # 在最终输出前应用 SE 模块
#         return self.conv4(out) + x
# /////
# class PMSFA(nn.Module):#4.16日22.38分当前跑的 0.643
#     def __init__(self, in_channels, growth_rate=32):
#         super(PMSFA, self).__init__()

#         # PMSFA部分
#         self.conv1 = Conv(in_channels, in_channels, k=3)
#         self.conv2 = Conv(in_channels // 2, in_channels // 2, k=5, g=in_channels // 2)
#         self.conv3 = Conv(in_channels // 4, in_channels // 4, k=7, g=in_channels // 4)


#         # DRB部分
#         self.conv1_dr = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
#         self.conv2_dr = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
#         self.conv3_dr = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
#         self.conv4_dr = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, kernel_size=1)

#         # 深度可分离卷积
#         self.depthwise = nn.Conv2d(in_channels + 4 * growth_rate, in_channels + 4 * growth_rate, kernel_size=3, stride=1, padding=1, groups=in_channels + 4 * growth_rate)
#         self.pointwise = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, kernel_size=1)

#     def forward(self, x):
#         # PMSFA部分
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
#         conv3_out = self.conv3(conv2_out_1)

#         # PMSFA特征融合
#         pmsfa_out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)

#         # DRB部分
#         out1_dr = torch.relu(self.conv1_dr(pmsfa_out))
#         out2_dr = torch.relu(self.conv2_dr(torch.cat([pmsfa_out, out1_dr], dim=1)))
#         out3_dr = torch.relu(self.conv3_dr(torch.cat([pmsfa_out, out1_dr, out2_dr], dim=1)))
#         out4_dr = torch.relu(self.conv4_dr(torch.cat([pmsfa_out, out1_dr, out2_dr, out3_dr], dim=1)))

#         # 深度可分离卷积
#         out_dr = torch.cat([pmsfa_out, out1_dr, out2_dr, out3_dr, out4_dr], dim=1)
#         out_dr = torch.relu(self.pointwise(self.depthwise(out_dr)))

#         # 最终输出
#         out = out_dr + x
#         return out

# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 保持原始卷积分支结构
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         # 高效改进点
#         self.edge_pre = EfficientEdgeEnhancer(inc)
#         self.edge_mid = EfficientEdgeEnhancer(inc//2)
#         self.core_att = nn.Sequential(
#             nn.Conv2d(inc, inc//4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inc//4, inc, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 分层边缘增强
#         x = self.edge_pre(x)  # 输入级增强
        
#         conv1_out = self.conv1(x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        
#         conv2_out = self.edge_mid(self.conv2(conv1_out_1))  # 中间层增强
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        
#         conv3_out = self.conv3(conv2_out_1)
        
#         # 轻量注意力融合
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         attn = self.core_att(out)
#         return self.conv4(out * attn) + x
    
# 0.824 0.764 0.829 0.635改进4   
# class LightEdgeEnhancer(nn.Module):
#     """轻量版边缘增强模块，参数量减少75%"""
#     def __init__(self, in_dim):
#         super().__init__()
#         self.edge_conv = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim//4, 1),  # 通道压缩
#             nn.Conv2d(in_dim//4, in_dim//4, 3, padding=1, groups=in_dim//4),  # 深度可分离卷积
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_dim//4, in_dim, 1),
#             nn.Sigmoid()
#         )
#         self.pool = nn.AvgPool2d(3, stride=1, padding=1)
    
#     def forward(self, x):
#         identity = x
#         edge = self.pool(x) - x  # 反向边缘提取
#         return identity + self.edge_conv(edge)

# class PMSFA(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#         # 保持原始卷积分支
#         self.conv1 = Conv(inc, inc, k=3)
#         self.conv2 = Conv(inc//2, inc//2, k=5, g=inc//2)
#         self.conv3 = Conv(inc//4, inc//4, k=7, g=inc//4)
#         self.conv4 = Conv(inc, inc, 1)
        
#         # 轻量化改进点
#         self.edge_enhancer = LightEdgeEnhancer(inc)  # 共享增强模块
#         self.eca = nn.Sequential(  # 替换原核心模块为ECA注意力
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(inc, inc, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 单次边缘增强
#         edge_x = self.edge_enhancer(x)
        
#         # 主干处理
#         conv1_out = self.conv1(edge_x)
#         conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        
#         conv2_out = self.conv2(conv1_out_1)
#         conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        
#         conv3_out = self.conv3(conv2_out_1)
        
#         # 注意力融合
#         out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
#         attn = self.eca(out)
#         return self.conv4(out * attn) + x
class PMSFA(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3)
        self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
        self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
        self.conv4 = Conv(inc, inc, 1)
    
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        conv3_out = self.conv3(conv2_out_1)
        
        out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv4(out) + x
        return out
class Bottleneck_gai(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # self.full=FullyAttentionalBlock(c2) 不行 
        # self.lia=LocalAttention(c2)不行
        # self.sge=SpatialGroupEnhance(c2)不行
        # self.ocbam=OCBAM(c2,reduction_ratio=16,pool_types=['avg', 'max'], no_spatial=False)
        # self.CAS = CASAtt(c2)还是不行
        # self.gcsa = GCSA(c2)不行  train1346    0.614
        # self.IEL = IEL(c2) 不行   2391620  train1347  0.626
        # self.ConvSpAtt=ConvModulOperationSpatialAttention(c2)   train1348  和上面差不多160轮才0.60
        # # self.LRSA = LRSA(c2)   train1349
# 再不行就bottleneck改卷积，光加注意力没用
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k_gai(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(PMSFA(c_) for _ in range(n)))
class C3k2_gai(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False,e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_gai(self.c, self.c, 2, shortcut,g) if c3k else PMSFA(self.c) for _ in range(n)
        )
# class C3k_gai(C3):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

#     def __init__(self, c1, c2, n=1, shortcut=True,use_shift_se=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(PMSFA(c_,use_shift_se) for _ in range(n)))
# class C3k2_gai(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, c3k=False,use_shift_se=False,e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k_gai(self.c, self.c, 2, shortcut,use_shift_se,g) if c3k else PMSFA(self.c,use_shift_se) for _ in range(n)
#         )
class SPPF(nn.Module):#原版
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # self.IEL = IEL(c2)
        # self.ConvSpAtt=ConvModulOperationSpatialAttention(c2)
        # self.LRSA = LRSA(c2)
        # self.CAS = CASAtt(c2)
        # self.gcsa = GCSA(c2)
        # self.shsa = SHSA(c2,c2,c2)
        # self.MCA=MCALayer(c2)
        # self.TiedSELayer=TiedSELayer(c2)
        # self.psa=PSA_1(c2,8)#运行不了
        # self.lia=LocalAttention(c2)
        # self.sge=SpatialGroupEnhance(c2)
        # self.ksfa=KernelSelectiveFusionAttention(c2)
        # self.nam=NAM(c2)
        # self.ocbam=OCBAM(c2,reduction_ratio=16,pool_types=['avg', 'max'], no_spatial=False)
        # self.full=FullyAttentionalBlock(c2)
        
        
    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2((torch.cat(y, 1)))
# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.

#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(c_*4, c_*4, 1, 1)
#         self.cv2 = Conv(c_ * 5, c2, 1, 1)
#         self.dcv = nn.Conv2d(c1,c2, 3, 1, dilation=2)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         y = [self.cv1(x)]
#         y.extend(self.m(y[-1]) for _ in range(3))
#         return self.cv2(torch.cat((self.cv3(torch.cat(y, 1)),self.dcv(x)),1))

# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.

#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.cv3 = Conv(c_, c_, 1, 1)
#         self.avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.max = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(c_ * 2, c_ * 4, 1),  # 1x1卷积，特征融合
#             nn.GELU(),  # GELU激活函数
#             # nn.ReLU(True),  # 可选：ReLU激活函数
#             nn.Conv2d(c_ * 4, c2, 1)  # 1x1卷积，恢复维度
#         )

#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         indenity = x
#         x = self.cv1(x)
#         y11 = self.max(x)
#         y12 = self.max(y11)
#         y21 = self.avg(x)
#         y22 = self.avg(y21)
#         # x1 = self.upsample(self.cv3(x))
#         # return self.cv2(torch.cat(self.cv2((torch.cat((x, y21, y22, self.avg(y22)),1))),
#         #                           self.cv2((torch.cat((x, y11, y12, self.max(y12)),1))),
#         #                           1))
#         x=torch.cat(((torch.cat((x, y21, y22, self.avg(y22)),1)),(torch.cat((x, y11, y12, self.max(y12)),1))),1)
#         x=self.mlp(x)
#         return indenity+x
#         # x1=self.upsample(self.cv3(self.avg_pool(x)))
#         # return self.cv2(torch.cat(y1,1))

class SimAM(torch.nn.Module):
    def __init__(self, channels = None,out_channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y) 
       
class SPPFCSPC(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
 
    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class Bottleneck_IEL(nn.Module):
    """增强版IEL Bottleneck，优化计算效率与特征融合能力"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5, ffn_ratio=1.25):
        super().__init__()
        # 基础参数配置
        self.c_ = int(c2 * e)  # 中间通道数
        self.add = shortcut and c1 == c2
        
        # 轻量化基础卷积组
        self.base_conv = nn.Sequential(
            nn.Conv2d(c1, self.c_, k[0], padding=k[0]//2, bias=False),
            nn.BatchNorm2d(self.c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_, c2, k[1], padding=k[1]//2, groups=g, bias=False),
            nn.BatchNorm2d(c2)
        )
        
        # 高效IEL增强模块
        hidden_dim = int(c2 * ffn_ratio)
        self.iel = nn.Sequential(
            nn.Conv2d(c2, hidden_dim, 1, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 
                     padding=1, groups=hidden_dim//4, bias=False),  # 分组深度卷积
            nn.SiLU(),
            nn.Conv2d(hidden_dim, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        
        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 基础特征提取
        x_base = self.base_conv(x)
        
        # IEL特征增强 (残差结构内置)
        x_iel = x_base + self.iel(x_base)
        
        # 最终残差连接
        return x + x_iel if self.add else x_iel
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention_LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
      
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim=36, mlp_dim=96,heads=1):
        super().__init__()
     

        self.layer = nn.ModuleList([
                PreNorm(dim, Attention_LRSA(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps=16):
        ps=16
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x
class Bottleneck_LRSA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.lrsa = LRSA(dim=c2)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.lrsa(self.cv2(self.cv1(x))) if self.add else self.lrsa(self.cv2(self.cv1(x)))  

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class C3k_LRSA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_LRSA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
class C3k2_LRSA(C2f):
    """C3k2 with LRSA-enhanced bottlenecks."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_LRSA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_LRSA(self.c, self.c, shortcut, g) for _ in range(n)
        )

class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# class OutlookAttention(nn.Module):
#     """
#     Implementation of outlook attention
#     --dim: hidden dim
#     --num_heads: number of heads
#     --kernel_size: kernel size in each window for outlook attention
#     return: token features after outlook attention
#     """

#     def __init__(self, dim, num_heads=8, kernel_size=3, padding=1, stride=1,
#                  qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         head_dim = dim // num_heads
#         self.num_heads = num_heads
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.scale = qk_scale or head_dim ** -0.5

#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
#         self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
#         self.id = nn.Identity()

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         B, H, W, C = x.shape

#         v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

#         h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
#         v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
#                                    self.kernel_size * self.kernel_size,
#                                    h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

#         attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#         attn = self.attn(attn).reshape(
#             B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
#                self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
#         attn = attn * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
#             B, C * self.kernel_size * self.kernel_size, h * w)
#         x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
#                    padding=self.padding, stride=self.stride)

#         x = self.proj(x.permute(0, 2, 3, 1))
#         x = self.proj_drop(x)  # B, H, W, C
#         x = x.permute(0, 3, 1, 2)  # B, C, H, W
#         return x
class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        # self.attn = OutlookAttention(c, num_heads)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
