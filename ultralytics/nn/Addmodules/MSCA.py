######################  MSCAAttention ####     START   by  AI&CV  ###############################
import torch
import torch.nn as nn
from torch.nn import functional as F

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations ),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.dilconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                              dilation=2, groups=dim)
        
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv3_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        
        self.conv4 = nn.Conv2d(dim, dim, 1)
        
        self.SE1 = oneConv(dim, dim,1,0,1)
        self.SE2 = oneConv(dim, dim,1,0,1)
        self.SE3 = oneConv(dim, dim,1,0,1)
        self.SE4 = oneConv(dim, dim,1,0,1)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
 
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
         
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.dilconv(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.dilconv(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.dilconv(attn_2)
        
        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        
        y0_weight = self.SE1(self.gap(attn_0))
        y1_weight = self.SE2(self.gap(attn_1))
        y2_weight = self.SE3(self.gap(attn_2))
        y3_weight = self.SE4(self.gap(attn_3))
        
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        
        x_att = y0_weight*attn_0+y1_weight*attn_1+y2_weight*attn_2+y3_weight*attn_3
 
        attn = self.conv4(x_att)
 
        return attn * u


# if __name__ == '__main__':
# #定义输入张量的形状为B，C，H，W
#     input = torch.randn(1,64,32,32)
# #创建一个 MSCA 模块实例
#     msca =MSCAAttention(dim=64)
# # 执行前向传播
#     output = msca(input)
# #打印输入和输出的形状
#     print('input_size:',input.size())
#     print('output_size:',output.size())






###################### MSCAAttention  ####     end   by  AI&CV  ###############################
