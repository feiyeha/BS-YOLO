import torch
import torch.nn as nn
import math
from .conv import Conv

import torch
import torch.nn as nn
import math

# class ELA(nn.Module):
#     """Constructs an Efficient Local Attention module.
#     Args:
#         channel: Number of channels of the input feature map
#         kernel_size: Adaptive selection of kernel size
#     """

#     def __init__(self, channel, kernel_size=7):
#         super(ELA, self).__init__()

#         self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=kernel_size // 2,
#                               groups=channel, bias=False)
#         self.gn = nn.GroupNorm(16, channel)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         B, C, H, W = x.size()

#         x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
#         x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
#         x_h = self.sigmoid(self.gn(self.conv(x_h))).view(B, C, H, 1)
#         x_w = self.sigmoid(self.gn(self.conv(x_w))).view(B, C, 1, W)

#         return x * x_h * x_w
    
# class channel_att(nn.Module):
#     def __init__(self, channel, b=1, gamma=2):
#         super(channel_att, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.mlp = MLP(in_features=channel)  # 初始化 MLP，输入维度为 channel

#     def forward(self, x):
#         B, C, H, W = x.shape  # 获取输入张量的形状
#         y = self.avg_pool(x)  # 形状: [B, C, 1, 1]
#         y = y.squeeze(-1).squeeze(-1)  # 形状: [B, C]
#         y = y.unsqueeze(1)  # 形状: [B, 1, C]
#         # 调用 MLP
#         y = self.mlp(y, H, W)  # 形状: [B, 1, C]
#         y = y.transpose(1, 2).unsqueeze(-1)  # 形状: [B, C, 1, 1]
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# class CPAM(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, ch=256):
#         super().__init__()
#         self.channel_att = channel_att(ch)
#         self.local_att = ELA(ch)

#     def forward(self, x):
#         input1 = x
#         input2 = self.channel_att(input1)
#         # 加权融合
#         x = input1 + input2
#         x = self.local_att(x)
#         return x
    
# class ELA(nn.Module):
#     def __init__(self, in_channels, phi='L'):
#         super(ELA, self).__init__()
#         '''
#         ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
#         ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
#         ELA-L 特别适合大型网络。
#         '''
#         Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
#         groups = {'T': in_channels, 'B': in_channels, 'S': in_channels//8, 'L': in_channels//8}[phi]
#         num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
#         pad = Kernel_size//2
#         self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
#         self.GN = nn.GroupNorm(num_groups, in_channels)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, input):
#         b, c, h, w = input.size()
#         x_h = torch.mean(input, dim=3, keepdim=True).view(b,c,h)
#         x_w = torch.mean(input, dim=2, keepdim=True).view(b,c,w)
#         x_h = self.con1(x_h)    # [b,c,h]
#         x_w = self.con1(x_w)    # [b,c,w]
#         x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)   # [b, c, h, 1]
#         x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)   # [b, c, 1, w]
#         return x_h * x_w * input

import math
import torch
import torch.nn as nn
# 0.653 train 39
class ELA(nn.Module):
    """统一参数初始化版本的ELA模块（语法修正版）"""
    
    def __init__(self, channel, b=1, gamma=2):
        super(ELA, self).__init__()
        
        # 动态卷积核尺寸（修复括号匹配问题）
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))  # 修正括号
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        # 通道注意力分支（保持结构）
        self.ch_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),
            nn.Conv1d(channel, channel, 
                     kernel_size=kernel_size,
                     padding=(kernel_size-1)//2,
                     groups=channel, 
                     bias=False),
            nn.Sigmoid(),
            nn.Unflatten(2, (1,1))
        )  # 确保Sequential闭合
        
        # 空间注意力（修正padding计算）
        self.spatial_conv = nn.Conv1d(
            channel, channel,
            kernel_size=kernel_size,
            padding=(kernel_size-1)*2//2,  # 空洞卷积的正确计算
            dilation=2,
            groups=channel,
            bias=False)
        
        # 参数初始化（保持正确性），就加了参数权重，改进版本结合通道注意力和空间注意力的加权和，或者引入可学习的参数来调整不同注意力的贡献。
        self.ch_weight = nn.Parameter(torch.zeros(1))
        self.sp_weight = nn.Parameter(torch.zeros(1))
        self.res_weight = nn.Parameter(torch.zeros(1))
        
        self.gn = nn.GroupNorm(max(1, channel//16), channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        identity = x  # 保留原始特征
        
        # ========== 通道注意力分支 ==========
        ch_att = self.ch_att(x)  # [B,C,1,1]
        
        # ========== 空间注意力分支 ==========
        # 水平方向处理
        x_h = x.mean(dim=3)       # [B,C,H]
        h_att = self.spatial_conv(x_h)  # [B,C,H]
        h_att = self.gn(h_att)
        h_att = self.sigmoid(h_att).view(B, C, H, 1)  # [B,C,H,1]
        
        # 垂直方向处理
        x_w = x.mean(dim=2)       # [B,C,W]
        w_att = self.spatial_conv(x_w)  # [B,C,W]
        w_att = self.gn(w_att)
        w_att = self.sigmoid(w_att).view(B, C, 1, W)  # [B,C,1,W]
        
        # ========== 注意力融合 ==========
        ch_coef = self.ch_weight.sigmoid()  # [0~1]
        sp_coef = self.sp_weight.sigmoid()  # [0~1]
        att_mask = ch_coef * ch_att + sp_coef * (h_att * w_att)
        
        # ========== 残差连接 ==========
        res_coef = self.res_weight.sigmoid()  # [0~1]
        return x * att_mask + res_coef * identity