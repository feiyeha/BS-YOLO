import torch
import torch.nn as nn

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
        identity = x
        
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
    
