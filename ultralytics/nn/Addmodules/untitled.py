class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始化两个批归一化层
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # 定义不同大小的卷积层
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5卷积，反射填充
        # 膨胀率3的分组卷积，用于捕捉不同范围的特征
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # 简单像素注意力机制
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积用于维度变换
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 分组卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 通道注意力机制
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            # nn.ReLU(True),  # 可选：ReLU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 像素注意力机制
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积降维
            nn.GELU(),  # GELU激活函数
            # nn.ReLU(True),  # 可选：ReLU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积升维
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 多层感知机（MLP），用于特征融合
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，特征融合
            nn.GELU(),  # GELU激活函数
            # nn.ReLU(True),  # 可选：ReLU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，恢复维度
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 第二个1x1卷积，特征融合
            nn.GELU(),  # GELU激活函数
            # nn.ReLU(True),  # 可选：ReLU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，恢复维度
        )

    def forward(self, x):
        # 第一个分支
        identity = x  # 保存输入x作为残差连接
        x = self.norm1(x)  # 批归一化
        x = self.conv1(x)  # 1x1卷积
        x = self.conv2(x)  # 5x5卷积
        # 融合不同膨胀率的分组卷积输出
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)  # MLP进行特征融合
        x = identity + x  # 残差连接

        # 第二个分支，包含注意力机制
        identity = x  # 保存上一步的输出作为新的残差连接
        x = self.norm2(x)  # 批归一化
        # 融合简单像素注意力、通道注意力和像素注意力
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)  # 第二个MLP进行特征融合
        x = identity + x  # 残差连接
        return x

class EdgeEnhancer(nn.Module):
    def init(self, in_dim, norm=nn.BatchNorm2d, act=nn.GELU):
        super().init()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x):
        '''
        经过平均池化  降低细节
        原图减去 弱化细节的图  突出细节
        再将这些细节加到原图上 就增签了原图的细节
        '''
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
